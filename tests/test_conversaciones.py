"""Conversaciones de varios turnos sobre el checkpointer.

Lo que se prueba:

- el turno 2 corre sobre el **mismo hilo** que el 1 y ve su historial;
- lo que es de un turno (contribuciones, decisiones, contador de pasos) se
  reinicia: el supervisor del turno 2 no hereda la rúbrica del 1;
- la conversación sobrevive a que se descarte el grafo (lo que pasa al
  reiniciar el worker): otro grafo con el mismo checkpointer la retoma;
- un turno a la vez por conversación (409), también mientras se espera una
  aprobación humana;
- `GET /conversations/{id}` lee los mensajes del checkpoint.

El checkpointer es un `InMemorySaver`, que cumple el mismo contrato que el
`AsyncRedisSaver` de producción. Que los checkpoints sobrevivan a un reinicio
**real** de Redis se prueba con Docker: `scripts/pruebas_sistema.py --reiniciar`.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import Modelos, config_ejecucion, construir_grafo
from app.jobs import EstadoJob
from app.state import historial_conversacion, solicitud_actual, turnos_previos
from app.worker import correr_job
from tests.dobles import LLMFalso, propuesta, sin_accion
from tests.test_api import entorno, saver  # noqa: F401 - fixtures
from tests.test_grafo import guion_analista, guion_investigador
from tests.test_hitl import PREGUNTA_ACCION, supervisor_feliz

PREGUNTA_1 = "¿Qué código falla más en Bazar Pampa y qué porcentaje de los pagos representa?"
PREGUNTA_2 = "¿Y ese código es reintentable según la documentación?"


def supervisor_dos_turnos() -> list[AIMessage]:
    return supervisor_feliz() + supervisor_feliz()


def modelos_dos_turnos(planificador: LLMFalso | None = None) -> Modelos:
    return Modelos(
        supervisor=LLMFalso(guion=supervisor_dos_turnos()),
        investigador=LLMFalso(guion=guion_investigador() + guion_investigador()),
        analista=LLMFalso(guion=guion_analista() + guion_analista()),
        sintetizador=LLMFalso(guion=[
            AIMessage(content="Respuesta 1: E4090 es el más frecuente."),
            AIMessage(content="Respuesta 2: E4090 no es reintentable."),
        ]),
        planificador=planificador or LLMFalso(guion=[sin_accion(), sin_accion()]),
    )


async def _turno(cliente, pregunta: str, conversacion_id: str | None = None) -> dict:
    cuerpo = {"pregunta": pregunta}
    if conversacion_id:
        cuerpo["conversacion_id"] = conversacion_id
    respuesta = await cliente.post("/tasks", json=cuerpo)
    assert respuesta.status_code == 202, respuesta.text
    return respuesta.json()


# --- Historial ----------------------------------------------------------------


def test_turnos_previos_toma_pregunta_y_respuesta_de_cada_turno_anterior():
    mensajes = [
        HumanMessage(content="p1"),
        AIMessage(content="→ investigador", name="supervisor"),
        AIMessage(content="informe", name="investigador"),
        AIMessage(content="r1", name="sintetizador"),
        AIMessage(content="Acción ejecutada.", name="sistema"),
        HumanMessage(content="p2"),
        AIMessage(content="→ investigador", name="supervisor"),
    ]
    assert turnos_previos(mensajes) == [{"pregunta": "p1", "respuesta": "r1\n\nAcción ejecutada."}]
    estado = {"messages": mensajes}
    assert solicitud_actual(estado) == "p2"  # type: ignore[arg-type]
    historial = historial_conversacion(estado, max_turnos=3)  # type: ignore[arg-type]
    assert "p1" in historial and "r1" in historial and "informe" not in historial
    solo_preguntas = historial_conversacion(estado, max_turnos=3, con_respuestas=False)  # type: ignore[arg-type]
    assert "p1" in solo_preguntas and "r1" not in solo_preguntas


def test_primer_turno_no_tiene_historial():
    assert historial_conversacion({"messages": [HumanMessage(content="p1")]}, max_turnos=3) == ""  # type: ignore[arg-type]


# --- Dos turnos sobre el mismo hilo --------------------------------------------


async def test_el_segundo_turno_ve_el_primero_y_arranca_de_cero(entorno, saver):  # noqa: F811
    cliente, jobs, _ = entorno
    m = modelos_dos_turnos()
    grafo = construir_grafo(m, checkpointer=saver)

    t1 = await _turno(cliente, PREGUNTA_1)
    assert t1["turno"] == 1
    assert await correr_job(grafo, jobs, t1["job_id"]) == EstadoJob.DONE

    t2 = await _turno(cliente, PREGUNTA_2, t1["conversacion_id"])
    assert t2["turno"] == 2 and t2["conversacion_id"] == t1["conversacion_id"]
    assert await correr_job(grafo, jobs, t2["job_id"]) == EstadoJob.DONE

    # El supervisor del turno 2 recibió el turno 1 como historial...
    contexto_t2 = m.supervisor.texto_de_llamada(3)
    assert PREGUNTA_1 in contexto_t2 and "Respuesta 1" in contexto_t2
    assert f"(turno actual)\n{PREGUNTA_2}" in contexto_t2
    # ...pero no sus contribuciones: arranca sin trabajo hecho y con el presupuesto entero.
    assert "(todavía ninguna)" in contexto_t2
    assert "Decisión 1 de" in contexto_t2
    # El investigador también ve la pregunta anterior, para resolver "ese código".
    assert PREGUNTA_1 in m.investigador.texto_de_llamada(4)

    final = (await grafo.aget_state(config_ejecucion(thread_id=t1["conversacion_id"]))).values
    assert final["turno"] == 2
    assert final["pasos_supervisor"] == 3
    assert len(final["contribuciones"]) == 2  # las del turno 2, no 4
    assert final["respuesta_final"] == "Respuesta 2: E4090 no es reintentable."

    conversacion = (await cliente.get(f"/conversations/{t1['conversacion_id']}")).json()
    assert conversacion["turnos"] == 2 and conversacion["job_activo"] is None
    assert [(x["rol"], x["contenido"][:11]) for x in conversacion["mensajes"]] == [
        ("usuario", PREGUNTA_1[:11]),
        ("asistente", "Respuesta 1"),
        ("usuario", PREGUNTA_2[:11]),
        ("asistente", "Respuesta 2"),
    ]
    assert [j["turno"] for j in conversacion["jobs"]] == [1, 2]

    con_trabajo = (await cliente.get(
        f"/conversations/{t1['conversacion_id']}", params={"incluir_trabajo": True}
    )).json()
    assert {"supervisor", "investigador", "analista"} <= {x["autor"] for x in con_trabajo["mensajes"]}


async def test_la_conversacion_sobrevive_a_descartar_el_grafo(entorno, saver):  # noqa: F811
    """Lo que pasa al reiniciar el worker: el grafo nuevo retoma desde el checkpointer."""
    cliente, jobs, _ = entorno
    t1 = await _turno(cliente, PREGUNTA_1)
    await correr_job(construir_grafo(modelos_dos_turnos(), checkpointer=saver), jobs, t1["job_id"])

    m = modelos_dos_turnos()
    m.supervisor = LLMFalso(guion=supervisor_dos_turnos()[3:])
    nuevo = construir_grafo(m, checkpointer=saver)  # "otro proceso"
    t2 = await _turno(cliente, PREGUNTA_2, t1["conversacion_id"])
    assert await correr_job(nuevo, jobs, t2["job_id"]) == EstadoJob.DONE
    assert PREGUNTA_1 in m.supervisor.texto_de_llamada(0)


# --- Un turno a la vez ------------------------------------------------------------


async def test_no_se_abre_un_turno_mientras_el_anterior_no_termino(entorno, saver):  # noqa: F811
    cliente, jobs, _ = entorno
    t1 = await _turno(cliente, PREGUNTA_1)
    ocupada = await cliente.post("/tasks", json={"pregunta": PREGUNTA_2, "conversacion_id": t1["conversacion_id"]})
    assert ocupada.status_code == 409 and t1["job_id"] in ocupada.json()["detail"]

    await correr_job(construir_grafo(modelos_dos_turnos(), checkpointer=saver), jobs, t1["job_id"])
    assert (await cliente.post(
        "/tasks", json={"pregunta": PREGUNTA_2, "conversacion_id": t1["conversacion_id"]}
    )).status_code == 202


async def test_dos_turnos_concurrentes_en_la_misma_conversacion_solo_uno_entra(entorno, saver):  # noqa: F811
    cliente, jobs, _ = entorno
    t1 = await _turno(cliente, PREGUNTA_1)
    await correr_job(construir_grafo(modelos_dos_turnos(), checkpointer=saver), jobs, t1["job_id"])
    cuerpo = {"pregunta": PREGUNTA_2, "conversacion_id": t1["conversacion_id"]}
    a, b = await asyncio.gather(cliente.post("/tasks", json=cuerpo), cliente.post("/tasks", json=cuerpo))
    assert sorted([a.status_code, b.status_code]) == [202, 409]


async def test_un_turno_que_falla_libera_la_conversacion(entorno):  # noqa: F811
    cliente, jobs, _ = entorno
    t1 = await _turno(cliente, PREGUNTA_1)
    await jobs.fallar(t1["job_id"], "Prueba", "falló")
    assert (await cliente.post(
        "/tasks", json={"pregunta": PREGUNTA_2, "conversacion_id": t1["conversacion_id"]}
    )).status_code == 202


async def test_una_aprobacion_pendiente_ocupa_la_conversacion(entorno, saver):  # noqa: F811
    cliente, jobs, cola = entorno
    m = modelos_dos_turnos(
        planificador=LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402", "pay_10403", "pay_10405"])])
    )
    grafo = construir_grafo(m, checkpointer=saver)
    t1 = await _turno(cliente, PREGUNTA_ACCION)
    assert await correr_job(grafo, jobs, t1["job_id"]) == EstadoJob.AWAITING_APPROVAL

    bloqueada = await cliente.post("/tasks", json={"pregunta": PREGUNTA_2, "conversacion_id": t1["conversacion_id"]})
    assert bloqueada.status_code == 409 and "AWAITING_APPROVAL" in bloqueada.json()["detail"]

    await cliente.post(f"/tasks/{t1['job_id']}/approve", json={"aprobado": True, "revisor": "ana"})
    assert await correr_job(grafo, jobs, t1["job_id"], decision=cola.reanudaciones[0][1]) == EstadoJob.DONE

    conversacion = (await cliente.get(f"/conversations/{t1['conversacion_id']}")).json()
    # El resultado de la acción queda en la conversación, como mensaje del sistema.
    assert conversacion["mensajes"][-1]["autor"] == "sistema"
    assert "aprobada por ana" in conversacion["mensajes"][-1]["contenido"]
    assert (await cliente.post(
        "/tasks", json={"pregunta": PREGUNTA_2, "conversacion_id": t1["conversacion_id"]}
    )).status_code == 202


# --- Validación ------------------------------------------------------------------


async def test_conversacion_inexistente_o_invalida(entorno):  # noqa: F811
    cliente, _, _ = entorno
    otra = "0" * 32
    assert (await cliente.post("/tasks", json={"pregunta": PREGUNTA_1, "conversacion_id": otra})).status_code == 404
    assert (await cliente.post("/tasks", json={"pregunta": PREGUNTA_1, "conversacion_id": "x"})).status_code == 422
    assert (await cliente.get(f"/conversations/{otra}")).status_code == 404
    assert (await cliente.get("/conversations/no-es-un-id")).status_code == 404
