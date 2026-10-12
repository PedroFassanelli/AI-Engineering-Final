"""La API y el worker, de punta a punta, sin Docker ni red.

- Redis: `fakeredis` (mismo protocolo, en memoria).
- Cola: `ColaFalsa`, que anota lo encolado. El test decide cuándo "corre el
  worker" llamando a `correr_job`, que es exactamente lo que hace la tarea de
  Arq.
- Grafo: el real, con `LLMFalso` por rol y un `InMemorySaver`.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

import fakeredis
import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from redis.exceptions import ConnectionError as RedisConnectionError

from app.fuentes import db
from app.graph import construir_grafo
from app.jobs import AlmacenJobs, EstadoJob
from app.main import Recursos, crear_app
from app.worker import correr_job
from tests.dobles import LLMFalso, propuesta
from tests.test_grafo import modelos
from tests.test_hitl import PREGUNTA_ACCION, supervisor_feliz

PREGUNTA = "¿Qué código falla más en Bazar Pampa y qué porcentaje de los pagos representa?"


class ColaFalsa:
    def __init__(self) -> None:
        self.ejecuciones: list[str] = []
        self.reanudaciones: list[tuple[str, dict[str, Any]]] = []
        self.rota = False

    async def encolar_ejecucion(self, job_id: str) -> None:
        if self.rota:
            raise RedisConnectionError("redis caído")
        self.ejecuciones.append(job_id)

    async def encolar_reanudacion(self, job_id: str, decision: dict[str, Any]) -> None:
        if self.rota:
            raise RedisConnectionError("redis caído")
        self.reanudaciones.append((job_id, decision))


class LLMQueFalla(LLMFalso):
    async def _agenerate(self, *args, **kwargs):
        raise RuntimeError("el proveedor devolvió 500")


@pytest.fixture
def saver():
    """El checkpointer compartido por la API (que lee conversaciones) y el grafo."""
    return InMemorySaver()


@pytest.fixture
async def entorno(saver):
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    jobs = AlmacenJobs(redis, prefijo="test", ttl_segundos=60, sin_latido_segundos=30)
    cola = ColaFalsa()

    @asynccontextmanager
    async def recursos():
        yield Recursos(jobs=jobs, cola=cola, checkpointer=saver)

    app = crear_app(recursos)
    async with app.router.lifespan_context(app):
        transporte = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transporte, base_url="http://api") as cliente:
            yield cliente, jobs, cola
    await redis.aclose()


def grafo(planificador: LLMFalso | None = None, **otros):
    m = modelos(supervisor_feliz(), **otros)
    if planificador is not None:
        m.planificador = planificador
    return construir_grafo(m, checkpointer=InMemorySaver())


# --- Encolar sin bloquear ---------------------------------------------------


async def test_post_encola_y_devuelve_el_job_id_sin_correr_el_grafo(entorno):
    cliente, _, cola = entorno
    respuesta = await cliente.post("/tasks", json={"pregunta": PREGUNTA})

    assert respuesta.status_code == 202
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "PENDING"
    assert cola.ejecuciones == [cuerpo["job_id"]]
    assert respuesta.headers["location"].endswith(f"/tasks/{cuerpo['job_id']}")

    estado = (await cliente.get(f"/tasks/{cuerpo['job_id']}")).json()
    assert estado["estado"] == "PENDING" and estado["pregunta"] == PREGUNTA


async def test_cinco_posts_concurrentes_dan_cinco_jobs_distintos(entorno):
    cliente, _, cola = entorno
    respuestas = await asyncio.gather(
        *(cliente.post("/tasks", json={"pregunta": f"{PREGUNTA} #{i}"}) for i in range(5))
    )
    ids = {r.json()["job_id"] for r in respuestas}
    assert all(r.status_code == 202 for r in respuestas)
    assert len(ids) == 5 and set(cola.ejecuciones) == ids


async def test_validacion_y_404(entorno):
    cliente, _, _ = entorno
    assert (await cliente.post("/tasks", json={"pregunta": "?"})).status_code == 422
    assert (await cliente.get("/tasks/no-existe")).status_code == 404


# --- El worker actualiza el estado ------------------------------------------


async def test_el_worker_lleva_el_job_a_done_con_su_recorrido(entorno):
    cliente, jobs, cola = entorno
    job_id = (await cliente.post("/tasks", json={"pregunta": PREGUNTA})).json()["job_id"]

    assert await correr_job(grafo(), jobs, job_id) == EstadoJob.DONE

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["estado"] == "DONE"
    assert estado["respuesta"] == "Respuesta final."
    assert estado["nodos"] == [
        "supervisor", "investigador", "supervisor", "analista", "supervisor", "sintetizador", "planificador"
    ]
    assert estado["iniciado_en"] and estado["terminado_en"] and estado["duracion_segundos"] is not None


async def test_una_excepcion_del_grafo_deja_el_job_en_failed(entorno):
    cliente, jobs, _ = entorno
    job_id = (await cliente.post("/tasks", json={"pregunta": PREGUNTA})).json()["job_id"]
    m = modelos([])
    m.supervisor = LLMQueFalla()
    roto = construir_grafo(m, checkpointer=InMemorySaver())

    assert await correr_job(roto, jobs, job_id) == EstadoJob.FAILED

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["estado"] == "FAILED"
    assert estado["error"] == {"tipo": "RuntimeError", "mensaje": "el proveedor devolvió 500"}
    assert estado["terminado_en"]


async def test_una_cancelacion_deja_el_job_en_failed_y_se_propaga(entorno):
    cliente, jobs, _ = entorno
    job_id = (await cliente.post("/tasks", json={"pregunta": PREGUNTA})).json()["job_id"]

    class LLMColgado(LLMFalso):
        async def _agenerate(self, *args, **kwargs):
            await asyncio.sleep(3600)

    m = modelos([])
    m.supervisor = LLMColgado()
    tarea = asyncio.create_task(correr_job(construir_grafo(m, checkpointer=InMemorySaver()), jobs, job_id))
    await asyncio.sleep(0.2)
    tarea.cancel()  # lo mismo que hace Arq al vencer `job_timeout`
    with pytest.raises(asyncio.CancelledError):
        await tarea

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["estado"] == "FAILED" and estado["error"]["tipo"] == "Cancelada"


async def test_un_job_running_sin_latidos_pasa_a_failed_al_leerlo(entorno):
    cliente, jobs, _ = entorno
    job_id = (await cliente.post("/tasks", json={"pregunta": PREGUNTA})).json()["job_id"]
    await jobs.transicionar(job_id, (EstadoJob.PENDING,), EstadoJob.RUNNING)
    await jobs.redis.hset(jobs.clave(job_id), "latido", str(time.time() - 120))  # el worker "murió"

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["estado"] == "FAILED" and estado["error"]["tipo"] == "WorkerSinLatido"


async def test_si_no_se_puede_encolar_responde_503_y_el_job_no_queda_colgado(entorno):
    cliente, jobs, cola = entorno
    cola.rota = True
    respuesta = await cliente.post("/tasks", json={"pregunta": PREGUNTA})
    assert respuesta.status_code == 503
    [job] = await jobs.listar()
    assert job["estado"] == "FAILED" and job["error"]["tipo"] == "NoSePudoEncolar"


async def test_un_reintento_de_la_cola_no_vuelve_a_correr_un_job_terminado(entorno):
    cliente, jobs, _ = entorno
    job_id = (await cliente.post("/tasks", json={"pregunta": PREGUNTA})).json()["job_id"]
    g = grafo()
    await correr_job(g, jobs, job_id)
    assert await correr_job(g, jobs, job_id) == EstadoJob.DONE  # no hizo nada: ya estaba DONE


# --- Human-in-the-loop por la API ---------------------------------------------


async def _job_en_pausa(cliente, jobs) -> tuple[str, Any]:
    job_id = (await cliente.post("/tasks", json={"pregunta": PREGUNTA_ACCION})).json()["job_id"]
    g = grafo(LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402", "pay_10403", "pay_10405"])]))
    assert await correr_job(g, jobs, job_id) == EstadoJob.AWAITING_APPROVAL
    return job_id, g


async def test_flujo_hitl_completo(entorno):
    cliente, jobs, cola = entorno
    job_id, g = await _job_en_pausa(cliente, jobs)

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["estado"] == "AWAITING_APPROVAL"
    assert estado["aprobacion_pendiente"]["pago_ids"] == ["pay_10402", "pay_10403", "pay_10405"]
    assert estado["respuesta"] is None
    assert await db.listar_reintentos() == []

    respuesta = await cliente.post(
        f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "ana", "comentario": "dale"}
    )
    assert respuesta.status_code == 202 and respuesta.json()["estado"] == "PENDING"
    [(reanudado, decision)] = cola.reanudaciones
    assert reanudado == job_id and decision["revisor"] == "ana"

    assert await correr_job(g, jobs, job_id, decision=decision) == EstadoJob.DONE

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["estado"] == "DONE"
    assert estado["resultado_accion"]["ejecutada"] is True
    assert "aprobada por ana" in estado["respuesta"]
    assert estado["nodos"][-2:] == ["aprobacion_humana", "ejecutar_accion"]
    assert estado["duracion_reanudacion_segundos"] is not None
    assert len(await db.listar_reintentos(job_id)) == 3


async def test_aprobar_dos_veces_da_409_y_encola_una_sola_reanudacion(entorno):
    cliente, jobs, cola = entorno
    job_id, _ = await _job_en_pausa(cliente, jobs)
    cuerpo = {"aprobado": True, "revisor": "ana"}
    a, b = await asyncio.gather(
        cliente.post(f"/tasks/{job_id}/approve", json=cuerpo),
        cliente.post(f"/tasks/{job_id}/approve", json=cuerpo),
    )
    assert sorted([a.status_code, b.status_code]) == [202, 409]
    assert len(cola.reanudaciones) == 1


async def test_no_se_aprueba_un_job_que_no_esta_en_pausa(entorno):
    cliente, _, _ = entorno
    job_id = (await cliente.post("/tasks", json={"pregunta": PREGUNTA})).json()["job_id"]
    respuesta = await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "x"})
    assert respuesta.status_code == 409
    assert (await cliente.post("/tasks/nada/approve", json={"aprobado": True, "revisor": "x"})).status_code == 404


async def test_rechazo_por_la_api(entorno):
    cliente, jobs, cola = entorno
    job_id, g = await _job_en_pausa(cliente, jobs)
    await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "revisor": "ana", "comentario": "no"})
    await correr_job(g, jobs, job_id, decision=cola.reanudaciones[0][1])

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["estado"] == "DONE" and estado["resultado_accion"]["ejecutada"] is False
    assert await db.listar_reintentos() == []


async def test_listar_y_health(entorno):
    cliente, jobs, _ = entorno
    for i in range(3):
        await cliente.post("/tasks", json={"pregunta": f"{PREGUNTA} {i}"})
    assert len((await cliente.get("/tasks")).json()) == 3
    assert (await cliente.get("/tasks", params={"estado": "DONE"})).json() == []
    assert (await cliente.get("/health")).json() == {
        "api": "ok",
        "redis": "ok",
        "checkpointer": "ok",
        "rag": "sin worker",
    }
