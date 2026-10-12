"""El worker: corre el grafo en segundo plano, en un proceso aparte de la API.

Patrón: la API **no** ejecuta el grafo. Crea el job en Redis (`PENDING`), lo
encola en Arq y responde con el `job_id`. Este proceso toma la tarea de la cola
y la corre:

    ejecutar_tarea(job_id)          PENDING ──► RUNNING ──► DONE | FAILED | AWAITING_APPROVAL
    reanudar_tarea(job_id, decisión) PENDING ──► RUNNING ──► DONE | FAILED

Separar los procesos tiene tres consecuencias buscadas:

- El event loop de la API nunca ejecuta trabajo de CPU del grafo (BM25,
  chequeos, serialización): solo atiende HTTP y habla con Redis.
- El worker escala por su cuenta: `MAX_TAREAS_CONCURRENTES` corutinas por
  proceso, y tantos procesos como haga falta (`docker compose up --scale
  worker=3`). Arq reparte la cola.
- Un reinicio de la API no corta ninguna ejecución, y un reinicio del worker no
  pierde los jobs pausados ni las conversaciones: su estado está en los
  checkpoints de Redis (`thread_id = conversacion_id`).

Al arrancar, el worker abre el **RAG** (`app.rag.abrir_recuperador`): conecta
con Pinecone, verifica el índice y lo reindexa solo si el corpus cambió.

Manejo de errores (lo que pide la consigna): **toda** salida anormal de una
tarea termina en `FAILED` con el tipo y el mensaje del error. Excepción del
grafo o de un LLM → `except Exception`. Timeout de Arq o apagado del worker →
`except CancelledError`. Muerte del proceso sin aviso → el job deja de latir y
`AlmacenJobs.obtener` lo marca al leerlo. Las tareas usan `max_tries=1`:
reintentar a ciegas un grafo que ya gastó tokens (o que ya escribió en la
base) no es una decisión que deba tomar la cola.

Uso:
    python -m app.worker
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from typing import Any

from arq import func
from arq.connections import RedisSettings
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from openinference.semconv.trace import OpenInferenceSpanKindValues
from redis.asyncio import Redis

from app import config
from app.fuentes import db
from app.graph import abrir_checkpointer, config_ejecucion, construir_grafo, entrada_de_turno
from app.hitl import codigos_reintentables
from app.jobs import AlmacenJobs, EstadoJob, JobNoEncontrado, ahora_iso
from app.observability import cerrar_observabilidad, contexto_de_job, iniciar_observabilidad, trazar
from app.rag import abrir_recuperador

log = logging.getLogger("nimbus.worker")


# --- La corrida del grafo ----------------------------------------------------


def _resumen_salida(salida: dict[str, Any]) -> Any:
    """Lo que se ve como `output` de la traza raíz en Phoenix."""
    if salida.get("interrupcion") is not None:
        return {"pausa": "AWAITING_APPROVAL", "solicitud": salida["interrupcion"]}
    return salida.get("respuesta_final", "")


@trazar(
    "tarea_nimbus",
    tipo=OpenInferenceSpanKindValues.AGENT,
    entrada=lambda grafo, jobs, job_id, conversacion_id, entrada, fase: (
        entrada["messages"][0].content if fase == "ejecucion" else {"reanudacion": entrada.resume}
    ),
    salida=_resumen_salida,
)
async def _correr_grafo(
    grafo: CompiledStateGraph,
    jobs: AlmacenJobs,
    job_id: str,
    conversacion_id: str,
    entrada: dict[str, Any] | Command,
    fase: str,
) -> dict[str, Any]:
    """Corre (o reanuda) el grafo y devuelve el estado final, o la interrupción.

    Es la raíz de la traza en Phoenix: todos los spans de LangChain (nodos,
    subgrafos ReAct, herramientas, llamadas al LLM) quedan colgados de este.

    `astream(stream_mode="updates")` en vez de `ainvoke`: cada nodo que termina
    es un **latido** del job y queda en su recorrido, así el cliente ve el
    progreso y un worker colgado se detecta.
    """
    ejecucion = config_ejecucion(thread_id=conversacion_id, job_id=job_id)
    async for actualizacion in grafo.astream(entrada, config=ejecucion, stream_mode="updates"):
        for nodo in actualizacion:
            if nodo != "__interrupt__":
                await jobs.latido(job_id, nodo)

    instantanea = await grafo.aget_state(ejecucion)
    valores = dict(instantanea.values)
    valores["interrupcion"] = instantanea.interrupts[0].value if instantanea.interrupts else None
    return valores


async def correr_job(
    grafo: CompiledStateGraph,
    jobs: AlmacenJobs,
    job_id: str,
    *,
    decision: dict[str, Any] | None = None,
) -> str:
    """Lleva un job de PENDING a un estado final (o a la pausa). Devuelve ese estado.

    Es la lógica del worker sin Arq de por medio: los tests la llaman directo.
    """
    fase = "reanudacion" if decision is not None else "ejecucion"

    try:
        marca = "iniciado_en" if fase == "ejecucion" else "reanudado_en"
        tomado = await jobs.transicionar(
            job_id, (EstadoJob.PENDING,), EstadoJob.RUNNING, nodo_actual="", **{marca: ahora_iso()}
        )
    except JobNoEncontrado:
        log.warning("job %s: no existe (¿expiró?), se descarta la tarea", job_id)
        return "NO_EXISTE"
    if not tomado:
        # Otro worker ya lo tomó, o es un reintento de la cola sobre un job
        # que ya terminó. Correrlo de nuevo duplicaría tokens y efectos.
        actual = await jobs.obtener(job_id)
        log.warning("job %s: no está PENDING (%s), se ignora", job_id, actual and actual["estado"])
        return str(actual and actual["estado"])

    job = await jobs.obtener(job_id)
    assert job is not None
    # Jobs creados antes de que existieran las conversaciones: su hilo era el job.
    conversacion_id = job.get("conversacion_id") or job_id
    turno = job.get("turno") or 1
    entrada: dict[str, Any] | Command = (
        Command(resume=decision) if decision is not None else entrada_de_turno(job["pregunta"], turno)
    )
    inicio = asyncio.get_running_loop().time()
    log.info("job %s (conversación %s, turno %s): %s", job_id, conversacion_id, turno, fase)

    try:
        with contexto_de_job(job_id, conversacion_id=conversacion_id, turno=turno, fase=fase):
            final = await _correr_grafo(grafo, jobs, job_id, conversacion_id, entrada, fase)
    except asyncio.CancelledError:
        # Timeout de Arq o apagado del worker. Se marca y se re-lanza: una
        # cancelación tragada deja a Arq esperando una tarea que no termina.
        await jobs.fallar(job_id, "Cancelada", f"La tarea se canceló (timeout de {config.TIMEOUT_TAREA_SEGUNDOS} s o apagado del worker).")
        raise
    except Exception as error:  # noqa: BLE001 - cualquier error del grafo es un FAILED
        log.exception("job %s: falló", job_id)
        await jobs.fallar(job_id, type(error).__name__, str(error) or repr(error))
        return EstadoJob.FAILED

    duracion = round(asyncio.get_running_loop().time() - inicio, 3)
    # La duración de cada fase por separado: la pausa humana no es latencia
    # del sistema y no tiene que inflarla.
    campo_duracion = "duracion_segundos" if fase == "ejecucion" else "duracion_reanudacion_segundos"

    if final["interrupcion"] is not None:
        await jobs.transicionar(
            job_id,
            (EstadoJob.RUNNING,),
            EstadoJob.AWAITING_APPROVAL,
            aprobacion_pendiente=final["interrupcion"],
            **{campo_duracion: duracion},
        )
        log.info("job %s: esperando aprobación humana", job_id)
        return EstadoJob.AWAITING_APPROVAL

    await jobs.transicionar(
        job_id,
        (EstadoJob.RUNNING,),
        EstadoJob.DONE,
        respuesta=final.get("respuesta_final", ""),
        task_completed=bool(final.get("task_completed", False)),
        resultado_accion=final.get("resultado_accion"),
        **{campo_duracion: duracion},
    )
    log.info("job %s: DONE en %.1f s", job_id, duracion)
    return EstadoJob.DONE


# --- Tareas de Arq -----------------------------------------------------------


async def ejecutar_tarea(ctx: dict[str, Any], job_id: str) -> str:
    return await correr_job(ctx["grafo"], ctx["jobs"], job_id)


async def reanudar_tarea(ctx: dict[str, Any], job_id: str, decision: dict[str, Any]) -> str:
    return await correr_job(ctx["grafo"], ctx["jobs"], job_id, decision=decision)


async def al_iniciar(ctx: dict[str, Any]) -> None:
    configurar_logging()
    iniciar_observabilidad("worker")
    if not config.DB_NEGOCIO.exists():
        conteo = await db.sembrar()
        log.info("Base de negocio sembrada en %s: %s", config.DB_NEGOCIO, conteo)
    # La tabla de códigos reintentables se lee del disco una vez (lru_cache):
    # mejor acá, en un hilo, que la primera vez que la pida un nodo del grafo.
    await asyncio.to_thread(codigos_reintentables)

    pila = AsyncExitStack()
    ctx["pila"] = pila
    redis = Redis.from_url(config.REDIS_URL, decode_responses=True)
    pila.push_async_callback(redis.aclose)
    ctx["jobs"] = AlmacenJobs(redis)
    # El RAG antes que el grafo: si Pinecone o las claves fallan, el worker no
    # arranca (y lo dice), en vez de aceptar jobs que van a fallar todos.
    recuperador = await pila.enter_async_context(abrir_recuperador(redis))
    checkpointer = await pila.enter_async_context(abrir_checkpointer())
    ctx["grafo"] = construir_grafo(checkpointer=checkpointer)
    await redis.set(f"{config.PREFIJO_REDIS}:worker:rag_modo", recuperador.modo)
    log.info(
        "Worker listo: modelo %s/%s (supervisor %s), RAG %s, hasta %d tareas a la vez.",
        config.PROVEEDOR,
        config.MODELO,
        config.MODELO_SUPERVISOR,
        recuperador.modo,
        config.MAX_TAREAS_CONCURRENTES,
    )


async def al_apagar(ctx: dict[str, Any]) -> None:
    jobs: AlmacenJobs | None = ctx.get("jobs")
    if jobs is not None:
        # `/health` deja de informar el RAG hasta que un worker vuelva a arrancar.
        await jobs.redis.delete(f"{config.PREFIJO_REDIS}:worker:rag_modo")
    pila: AsyncExitStack | None = ctx.get("pila")
    if pila is not None:
        await pila.aclose()
    cerrar_observabilidad()


class WorkerSettings:
    """Configuración que lee Arq (`arq app.worker.WorkerSettings` también sirve)."""

    functions = [
        func(ejecutar_tarea, name="ejecutar_tarea", max_tries=1),
        func(reanudar_tarea, name="reanudar_tarea", max_tries=1),
    ]
    on_startup = al_iniciar
    on_shutdown = al_apagar
    redis_settings = RedisSettings.from_dsn(config.REDIS_URL)
    max_jobs = config.MAX_TAREAS_CONCURRENTES
    job_timeout = config.TIMEOUT_TAREA_SEGUNDOS
    max_tries = 1
    #: El resultado de Arq es solo el estado final; el job completo está en
    #: nuestro hash. Una hora alcanza para depurar.
    keep_result = 3600


def configurar_logging() -> None:
    config.configurar_consola()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # El cliente HTTP del proveedor loguea cada request en INFO: es ruido.
    for ruidoso in ("httpx", "openai", "anthropic", "pinecone", "redisvl"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def main() -> None:
    from arq.worker import run_worker

    configurar_logging()
    # En Windows el event loop no soporta `add_signal_handler`: se apaga con
    # Ctrl+C igual, pero sin el apagado ordenado de Arq.
    run_worker(WorkerSettings, handle_signals=sys.platform != "win32")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
