"""API REST del sistema: consultas de soporte resueltas por el grafo multi-agente.

    POST /tasks                       → crea el turno, lo encola y devuelve job_id + conversacion_id (202)
    GET  /tasks/{job_id}              → estado del turno (polling); nunca espera al grafo
    POST /tasks/{job_id}/approve      → aprueba o rechaza la acción crítica pendiente (202)
    GET  /tasks                       → últimos jobs (filtrable por estado)
    GET  /conversations/{id}          → la conversación, leída del checkpoint de LangGraph
    GET  /health                      → la API, Redis y el checkpointer responden

**Ningún endpoint ejecuta el grafo ni espera a un LLM.** Todo lo que hacen es
leer y escribir Redis con clientes asíncronos (`redis.asyncio`, el
`AsyncRedisSaver` de LangGraph) y encolar en Arq, que también es asíncrono: el
event loop nunca se bloquea, y un `POST` responde en milisegundos aunque el
grafo tarde un minuto.

Uso:
    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app import config
from app.cola import Cola, ColaArq
from app.jobs import (
    AlmacenJobs,
    ConversacionNoEncontrada,
    ConversacionOcupada,
    EstadoJob,
    JobNoEncontrado,
)
from app.schemas import (
    PATRON_ID,
    Conversacion,
    CrearTarea,
    DecisionAprobacion,
    ErrorAPI,
    Job,
    Mensaje,
    Salud,
    TareaEncolada,
)
from app.state import AUTORES_DE_RESPUESTA

log = logging.getLogger("nimbus.api")


@dataclass
class Recursos:
    jobs: AlmacenJobs
    cola: Cola
    #: Para leer conversaciones. `None` = el endpoint responde 503.
    checkpointer: BaseCheckpointSaver | None = None


@asynccontextmanager
async def recursos_de_produccion() -> AsyncIterator[Recursos]:
    from app.graph import abrir_checkpointer

    async with AsyncExitStack() as pila:
        redis = Redis.from_url(config.REDIS_URL, decode_responses=True)
        pila.push_async_callback(redis.aclose)
        pool = await create_pool(RedisSettings.from_dsn(config.REDIS_URL))
        pila.push_async_callback(pool.aclose)
        checkpointer = await pila.enter_async_context(abrir_checkpointer())
        yield Recursos(jobs=AlmacenJobs(redis), cola=ColaArq(pool), checkpointer=checkpointer)


def mensajes_de_checkpoint(mensajes: list[BaseMessage], incluir_trabajo: bool) -> list[Mensaje]:
    """Los mensajes del hilo, en el formato de la API."""
    salida: list[Mensaje] = []
    for mensaje in mensajes:
        if mensaje.type == "human":
            salida.append(Mensaje(rol="usuario", autor="usuario", contenido=str(mensaje.content)))
        elif isinstance(mensaje, AIMessage):
            autor = mensaje.name or "asistente"
            if autor in AUTORES_DE_RESPUESTA or incluir_trabajo:
                salida.append(Mensaje(rol="asistente", autor=autor, contenido=str(mensaje.content)))
    return salida


_E404 = {404: {"model": ErrorAPI}}
_E409 = {409: {"model": ErrorAPI}}
_E503 = {503: {"model": ErrorAPI}}


def crear_app(
    abrir_recursos: Callable[[], "AsyncIterator[Recursos]"] = recursos_de_produccion,  # type: ignore[assignment]
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with abrir_recursos() as recursos:  # type: ignore[attr-defined]
            app.state.recursos = recursos
            yield

    app = FastAPI(
        title="Nimbus Pagos — The Intelligence System",
        version="1.0.0",
        description=(
            "Soporte técnico de Nimbus Pagos resuelto por un grafo multi-agente (LangGraph): un "
            "supervisor delega en un investigador (RAG híbrido Pinecone + BM25 sobre la "
            "documentación de nimbus-sdk, y la base de pagos) y en un analista (validación y "
            "cálculos). Corre en segundo plano; las conversaciones persisten en Redis y las "
            "acciones con efectos quedan en pausa hasta una aprobación humana.\n\n"
            "Flujo: `POST /tasks` → polling de `GET /tasks/{job_id}` → (si queda en "
            "`AWAITING_APPROVAL`) `POST /tasks/{job_id}/approve`. Para seguir la conversación, "
            "otro `POST /tasks` con el mismo `conversacion_id`."
        ),
        lifespan=lifespan,
    )

    def recursos(request: Request) -> Recursos:
        return request.app.state.recursos

    async def job_o_404(request: Request, job_id: str) -> dict:
        job = await recursos(request).jobs.obtener(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id} (o ya expiró).")
        return job

    @app.post(
        "/tasks",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=TareaEncolada,
        responses={**_E404, **_E409, **_E503},
        summary="Encolar una consulta (un turno de una conversación)",
    )
    async def crear_tarea(cuerpo: CrearTarea, request: Request, response: Response) -> TareaEncolada:
        r = recursos(request)
        try:
            job = await r.jobs.crear(cuerpo.pregunta, cuerpo.conversacion_id)
        except ConversacionNoEncontrada as error:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"No existe la conversación {error} (o ya expiró)."
            ) from error
        except ConversacionOcupada as error:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"La conversación tiene un turno sin terminar ({error.job_id}, {error.estado}). "
                "Esperá a que termine, o aprobalo/rechazalo si está en AWAITING_APPROVAL.",
            ) from error

        try:
            await r.cola.encolar_ejecucion(job["job_id"])
        except (RedisError, OSError) as error:
            # El job existe pero nadie lo va a correr: se marca FAILED para que
            # quien lo consulte no espere para siempre (y libera la conversación).
            await r.jobs.fallar(job["job_id"], "NoSePudoEncolar", str(error))
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No se pudo encolar la tarea.") from error

        url = str(request.url_for("obtener_tarea", job_id=job["job_id"]))
        response.headers["Location"] = url
        return TareaEncolada(
            job_id=job["job_id"],
            conversacion_id=job["conversacion_id"],
            turno=job["turno"],
            estado=job["estado"],
            url_estado=url,
            url_conversacion=str(request.url_for("obtener_conversacion", conversacion_id=job["conversacion_id"])),
        )

    @app.get("/tasks/{job_id}", response_model=Job, responses=_E404, summary="Estado de un turno")
    async def obtener_tarea(job_id: str, request: Request) -> Job:
        return Job.desde_almacen(await job_o_404(request, job_id))

    @app.post(
        "/tasks/{job_id}/approve",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=Job,
        responses={**_E404, **_E409, **_E503},
        summary="Aprobar o rechazar la acción crítica pendiente",
    )
    async def aprobar(job_id: str, decision: DecisionAprobacion, request: Request) -> Job:
        r = recursos(request)
        await job_o_404(request, job_id)
        datos = decision.model_dump()
        try:
            # Atómico: si dos personas aprueban a la vez, solo una pasa.
            pasado = await r.jobs.transicionar(
                job_id, (EstadoJob.AWAITING_APPROVAL,), EstadoJob.PENDING, aprobacion=datos
            )
        except JobNoEncontrado as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.") from error
        if not pasado:
            actual = await r.jobs.obtener(job_id)
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"El job está en {actual and actual['estado']}: solo se aprueba en AWAITING_APPROVAL.",
            )
        try:
            await r.cola.encolar_reanudacion(job_id, datos)
        except (RedisError, OSError) as error:
            await r.jobs.fallar(job_id, "NoSePudoEncolar", str(error))
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No se pudo encolar la reanudación.") from error
        return Job.desde_almacen(await job_o_404(request, job_id))

    @app.get("/tasks", response_model=list[Job], summary="Últimos jobs")
    async def listar(
        request: Request,
        estado: EstadoJob | None = None,
        limite: int = Query(20, ge=1, le=100),
    ) -> list[Job]:
        return [Job.desde_almacen(j) for j in await recursos(request).jobs.listar(limite, estado)]

    @app.get(
        "/conversations/{conversacion_id}",
        response_model=Conversacion,
        responses={**_E404, **_E503},
        summary="Una conversación, leída del checkpoint",
    )
    async def obtener_conversacion(
        request: Request,
        conversacion_id: str,
        incluir_trabajo: bool = Query(
            False, description="Además de preguntas y respuestas, los pasos del supervisor y los informes."
        ),
    ) -> Conversacion:
        if not _es_id(conversacion_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la conversación {conversacion_id}.")
        r = recursos(request)
        if r.checkpointer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No hay checkpointer configurado.")

        registro = await r.jobs.obtener_conversacion(conversacion_id)
        tupla = await r.checkpointer.aget_tuple({"configurable": {"thread_id": conversacion_id}})
        if registro is None and tupla is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"No existe la conversación {conversacion_id} (o ya expiró)."
            )

        mensajes: list[Any] = []
        if tupla is not None:
            mensajes = tupla.checkpoint.get("channel_values", {}).get("messages", [])
        jobs = []
        for job_id in (registro or {}).get("job_ids", []):
            job = await r.jobs.obtener(job_id)
            if job is not None:
                jobs.append(Job.desde_almacen(job))

        return Conversacion(
            conversacion_id=conversacion_id,
            creada_en=(registro or {}).get("creada_en"),
            actualizada_en=(registro or {}).get("actualizada_en"),
            turnos=(registro or {}).get("turnos", 0),
            job_activo=(registro or {}).get("job_activo"),
            mensajes=mensajes_de_checkpoint(mensajes, incluir_trabajo),
            jobs=jobs,
        )

    @app.get("/health", response_model=Salud, responses=_E503, summary="La API y sus dependencias responden")
    async def salud(request: Request) -> Salud:
        r = recursos(request)
        try:
            await r.jobs.redis.ping()
            rag = await r.jobs.redis.get(f"{config.PREFIJO_REDIS}:worker:rag_modo")
        except (RedisError, OSError) as error:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Redis no responde: {error}") from error
        return Salud(
            api="ok",
            redis="ok",
            checkpointer="ok" if r.checkpointer is not None else "no configurado",
            rag=rag or "sin worker",
        )

    return app


def _es_id(valor: str) -> bool:
    import re

    return re.fullmatch(PATRON_ID, valor) is not None


app = crear_app()
