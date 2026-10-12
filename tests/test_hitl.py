"""Human-in-the-loop: la acción crítica no se ejecuta sin una aprobación explícita.

El grafo se compila con un `InMemorySaver` (el mismo contrato que el
`AsyncRedisSaver` de producción): `interrupt()` necesita un checkpointer, y lo
que se prueba acá es la semántica de la pausa, no Redis.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.fuentes import db
from app.graph import config_ejecucion, construir_grafo, entrada_de_turno
from app.hitl import AccionNoAprobada, ejecutar_accion, evaluar_criticidad, validar_propuesta
from tests.dobles import LLMFalso, decision, propuesta
from tests.test_grafo import guion_analista, guion_investigador, modelos

PREGUNTA_ACCION = (
    "¿Qué código falla más en Bazar Pampa? Y reintentá los pagos fallidos por ese código."
)


def supervisor_feliz() -> list[AIMessage]:
    return [
        decision("investigador", "Buscá el comercio, sus pagos y el código."),
        decision("analista", "Calculá el porcentaje."),
        decision("finalizar", cobertura=True),
    ]


async def correr_hasta_pausa(planificador: LLMFalso, job_id: str = "job-1"):
    grafo = construir_grafo(
        modelos(supervisor_feliz(), planificador=planificador), checkpointer=InMemorySaver()
    )
    ejecucion = config_ejecucion(thread_id=job_id)
    await grafo.ainvoke(entrada_de_turno(PREGUNTA_ACCION), config=ejecucion)
    return grafo, ejecucion


# --- Validación de la propuesta ---------------------------------------------


async def test_solo_sobreviven_pagos_consultados_y_fallidos():
    grafo, ejecucion = await correr_hasta_pausa(
        LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402", "pay_10401", "pay_99999", "pay_10405", "pay_10403"])])
    )
    accion = (await grafo.aget_state(ejecucion)).values["accion_propuesta"]
    assert accion["pago_ids"] == ["pay_10402", "pay_10405", "pay_10403"]
    assert accion["monto_total_centavos"] == 890000 + 2150000 + 890000
    assert accion["comercio"] == "Bazar Pampa"
    assert any("pay_10401" in d and "aprobado" in d for d in accion["descartados"])
    assert any("pay_99999" in d for d in accion["descartados"])


def test_toda_accion_con_efectos_es_critica_y_dice_por_que():
    razones = evaluar_criticidad(
        {"tipo": "programar_reintentos", "comercio_id": 1, "comercio": "x", "pago_ids": ["a"],
         "monto_total_centavos": 100, "moneda": "ARS", "motivo": "", "criticidad": [], "descartados": []}
    )
    assert len(razones) == 1 and razones[0].startswith("Efecto secundario")


def test_una_propuesta_sin_pagos_validos_no_es_accion():
    assert validar_propuesta("programar_reintentos", ["pay_inventado"], "m", {"contribuciones": []}) is None


# --- La pausa ---------------------------------------------------------------


async def test_una_accion_critica_pausa_el_grafo_antes_de_escribir():
    grafo, ejecucion = await correr_hasta_pausa(
        LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402", "pay_10403", "pay_10405"])])
    )
    instantanea = await grafo.aget_state(ejecucion)

    assert instantanea.next == ("aprobacion_humana",)
    [pausa] = instantanea.interrupts
    assert pausa.value["pago_ids"] == ["pay_10402", "pay_10403", "pay_10405"]
    assert "Bazar Pampa" in pausa.value["resumen"]
    assert pausa.value["criticidad"]
    # La consulta ya está respondida, pero no se escribió nada en la base.
    assert instantanea.values["respuesta_final"] == "Respuesta final."
    assert await db.listar_reintentos() == []


async def test_aprobada_ejecuta_la_accion_y_lo_dice_en_la_respuesta():
    grafo, ejecucion = await correr_hasta_pausa(
        LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402", "pay_10403"])])
    )
    final = await grafo.ainvoke(
        Command(resume={"aprobado": True, "revisor": "ana", "comentario": "ok"}), config=ejecucion
    )

    assert final["aprobacion"]["aprobado"] is True
    assert final["resultado_accion"]["ejecutada"] is True
    filas = await db.listar_reintentos("job-1")
    assert [f["pago_id"] for f in filas] == ["pay_10402", "pay_10403"]
    assert all(f["aprobado_por"] == "ana" for f in filas)
    assert final["respuesta_final"].startswith("Respuesta final.")
    assert "aprobada por ana" in final["respuesta_final"]


async def test_rechazada_no_escribe_nada():
    grafo, ejecucion = await correr_hasta_pausa(
        LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402"])])
    )
    final = await grafo.ainvoke(
        Command(resume={"aprobado": False, "revisor": "ana", "comentario": "esperar al banco"}),
        config=ejecucion,
    )
    assert final["resultado_accion"]["ejecutada"] is False
    assert "esperar al banco" in final["respuesta_final"]
    assert await db.listar_reintentos() == []


@pytest.mark.parametrize("respuesta", ["si", {"comentario": "sin decidir"}, {"aprobado": "true"}, {"revisor": "x"}])
async def test_una_respuesta_ambigua_cuenta_como_rechazo(respuesta):
    grafo, ejecucion = await correr_hasta_pausa(
        LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402"])])
    )
    final = await grafo.ainvoke(Command(resume=respuesta), config=ejecucion)
    assert final["aprobacion"]["aprobado"] is False
    assert await db.listar_reintentos() == []


async def test_una_consulta_sin_accion_termina_sin_pausa():
    grafo = construir_grafo(modelos(supervisor_feliz()), checkpointer=InMemorySaver())
    ejecucion = config_ejecucion(thread_id="job-2")
    final = await grafo.ainvoke(entrada_de_turno("¿Qué código falla más en Bazar Pampa?"), config=ejecucion)
    instantanea = await grafo.aget_state(ejecucion)
    assert instantanea.next == () and not instantanea.interrupts
    assert final["accion_propuesta"] is None


async def test_sin_pagos_fallidos_el_planificador_no_llama_al_modelo():
    investigador = LLMFalso(guion=[
        guion_investigador()[0],
        AIMessage(content="Solo encontré el comercio."),
    ])
    planificador = LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402"])])
    grafo = construir_grafo(
        modelos(
            [decision("investigador", "Buscá el comercio."), decision("finalizar", cobertura=True)],
            investigador=investigador,
            planificador=planificador,
        )
    )
    final = await grafo.ainvoke(entrada_de_turno(PREGUNTA_ACCION), config=config_ejecucion())
    assert planificador.llamadas == []
    assert final["accion_propuesta"] is None


async def test_ejecutar_accion_se_niega_sin_aprobacion():
    with pytest.raises(AccionNoAprobada):
        await ejecutar_accion({"accion_propuesta": {"pago_ids": ["x"]}, "aprobacion": None}, {})


async def test_reanudar_dos_veces_no_duplica_el_efecto():
    """`INSERT OR IGNORE` sobre (pago_id, job_id): la escritura es idempotente."""
    grafo, ejecucion = await correr_hasta_pausa(
        LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402"])])
    )
    final = await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "ana"}), config=ejecucion)
    await ejecutar_accion(final, ejecucion)
    assert len(await db.listar_reintentos("job-1")) == 1


async def test_quien_aprueba_ve_si_la_documentacion_dice_que_no_es_reintentable():
    grafo, ejecucion = await correr_hasta_pausa(
        LLMFalso(guion=[propuesta("programar_reintentos", ["pay_10402", "pay_10406"])])
    )
    [pausa] = (await grafo.aget_state(ejecucion)).interrupts
    assert pausa.value["codigos_error"] == ["E4090", "E4290"]
    [advertencia] = [r for r in pausa.value["criticidad"] if r.startswith("ADVERTENCIA")]
    assert "E4090" in advertencia and "E4290" not in advertencia  # E4290 sí es reintentable
