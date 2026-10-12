"""El grafo completo, con un `LLMFalso` por rol.

Cubre lo que pide la consigna de punta a punta: la delegación
supervisor → investigador → analista → síntesis, el aislamiento de contexto de
cada especialista, el registro de quién aportó qué, y que ningún guion de
modelo —por insistente que sea— pueda dejar al grafo en un bucle.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app import config
from app.graph import Modelos, config_ejecucion, construir_grafo, diagrama_mermaid, entrada_de_turno
from tests.dobles import LLMFalso, decision, pedido, sin_accion
from tests.test_chequeos import PAGOS_104

PREGUNTA = "¿Qué código falla más en Bazar Pampa y qué porcentaje de los pagos representa?"

CONCLUSION_INVESTIGADOR = (
    "El código más frecuente es E4090: conflicto de idempotencia "
    "[03-manejo-de-errores.md]. Datos del comercio 104."
)


def guion_investigador() -> list[AIMessage]:
    return [
        pedido("buscar_comercio", {"nombre": "bazar pampa"}, "i1"),
        pedido("consultar_pagos", {"comercio_id": 104}, "i2"),
        pedido("buscar_documentacion", {"consulta": "E4090"}, "i3"),
        AIMessage(content=CONCLUSION_INVESTIGADOR),
    ]


def guion_analista() -> list[AIMessage]:
    return [
        pedido("perfilar_pagos", {"pagos": PAGOS_104}, "a1"),
        pedido("porcentaje", {"parte": 3, "total": 7}, "a2"),
        pedido("porcentaje", {"parte": 3930000, "total": 6680000}, "a3"),
        AIMessage(content="E4090: 3 de 7 pagos (42.86%); 3,930,000 de 6,680,000 centavos (58.83%)."),
    ]


def modelos(supervisor: list[AIMessage], **otros: LLMFalso) -> Modelos:
    return Modelos(
        supervisor=LLMFalso(guion=supervisor),
        investigador=otros.get("investigador", LLMFalso(guion=guion_investigador())),
        analista=otros.get("analista", LLMFalso(guion=guion_analista())),
        sintetizador=otros.get("sintetizador", LLMFalso(guion=[AIMessage(content="Respuesta final.")])),
        planificador=otros.get("planificador", LLMFalso(guion=[sin_accion()])),
    )


async def correr(m: Modelos) -> dict:
    return await construir_grafo(m).ainvoke(entrada_de_turno(PREGUNTA), config=config_ejecucion())


# --- El flujo que pide la consigna ------------------------------------------


async def test_flujo_completo_investigador_analista_sintesis():
    m = modelos([
        decision("investigador", "Identificá el código más frecuente y conseguí los pagos."),
        decision("analista", "Calculá qué porcentaje de los pagos representan los fallidos por ese código."),
        decision("finalizar", cobertura=True),
    ])
    final = await correr(m)

    assert [d["destino"] for d in final["decisiones"]] == ["investigador", "analista", "sintetizador"]
    assert [c["agente"] for c in final["contribuciones"]] == ["investigador", "analista"]
    assert all(c["problemas"] == [] for c in final["contribuciones"])
    assert final["task_completed"] is True
    assert final["respuesta_final"] == "Respuesta final."
    assert final["pasos_supervisor"] == 3


async def test_cada_contribucion_registra_agente_instruccion_y_evidencia():
    m = modelos([
        decision("investigador", "Conseguí los pagos."),
        decision("analista", "Calculá el porcentaje."),
        decision("finalizar", cobertura=True),
    ])
    final = await correr(m)
    investigacion, analisis = final["contribuciones"]

    assert investigacion["instruccion"] == "Conseguí los pagos."
    assert [e["herramienta"] for e in investigacion["evidencia"]] == [
        "buscar_comercio", "consultar_pagos", "buscar_documentacion",
    ]
    assert investigacion["evidencia"][1]["resultado"]["cantidad"] == 7
    assert analisis["evidencia"][2]["resultado"]["porcentaje"] == 58.83


async def test_la_conversacion_legible_nombra_a_cada_agente():
    m = modelos([
        decision("investigador", "a"), decision("analista", "b"), decision("finalizar", cobertura=True),
    ])
    final = await correr(m)
    nombres = [getattr(msg, "name", None) for msg in final["messages"]]
    assert nombres == [None, "supervisor", "investigador", "supervisor", "analista", "supervisor", "sintetizador"]


# --- Aislamiento de contexto ------------------------------------------------


async def test_el_analista_recibe_datos_crudos_y_no_las_conclusiones_del_investigador():
    analista = LLMFalso(guion=guion_analista())
    m = modelos(
        [decision("investigador", "a"), decision("analista", "Calculá."), decision("finalizar", cobertura=True)],
        analista=analista,
    )
    await correr(m)
    visto = analista.texto_de_llamada(0)

    assert "pay_10405" in visto  # los registros crudos
    assert PREGUNTA in visto  # la solicitud, para saber qué cuantificar
    assert "idempotencia" not in visto  # ni la conclusión ni la documentación
    assert "buscar_documentacion" not in visto


async def test_el_supervisor_no_ve_la_evidencia_cruda():
    supervisor = LLMFalso(guion=[
        decision("investigador", "a"), decision("analista", "b"), decision("finalizar", cobertura=True),
    ])
    m = modelos([])
    m.supervisor = supervisor
    await correr(m)
    segunda = supervisor.texto_de_llamada(1)
    assert CONCLUSION_INVESTIGADOR in segunda
    assert "pay_10405" not in segunda


async def test_cada_especialista_arranca_con_un_historial_nuevo():
    investigador = LLMFalso(guion=[*guion_investigador(), *guion_investigador()])
    m = modelos(
        [decision("investigador", "Primera."), decision("investigador", "Corregí la cita."),
         decision("finalizar", cobertura=True)],
        investigador=investigador,
    )
    await correr(m)
    # La primera llamada de la segunda ronda es la quinta del modelo.
    segunda_ronda = investigador.llamadas[4]
    assert "Corregí la cita." in segunda_ronda[-1].content
    assert not any("Primera." in str(msg.content) for msg in segunda_ronda)


# --- Condiciones de parada --------------------------------------------------


async def test_un_supervisor_que_nunca_finaliza_se_corta_por_rondas(monkeypatch):
    monkeypatch.setattr(config, "MAX_RONDAS_POR_AGENTE", 2)
    investigador = LLMFalso(guion=[AIMessage(content="No encontré nada.")], ciclar=True)
    sintetizador = LLMFalso(guion=[AIMessage(content="Parcial.")])
    m = modelos(
        [decision("investigador", f"Buscá de nuevo, intento {i}") for i in range(10)],
        investigador=investigador, sintetizador=sintetizador,
    )
    final = await correr(m)

    assert [d["destino"] for d in final["decisiones"]] == ["investigador", "investigador", "sintetizador"]
    assert "rondas" in final["decisiones"][-1]["guarda"]
    assert final["task_completed"] is False
    # El sintetizador se entera del cierre forzado y de por qué.
    entrada = sintetizador.texto_de_llamada(0)
    assert "Cierre forzado" in entrada and "No usó ninguna herramienta" in entrada


async def test_instruccion_repetida_corta_antes_de_agotar_rondas():
    m = modelos([decision("investigador", "Buscá E4090"), decision("investigador", "Buscá E4090")])
    final = await correr(m)
    assert [d["destino"] for d in final["decisiones"]] == ["investigador", "sintetizador"]
    assert "repetida" in final["decisiones"][-1]["guarda"]


async def test_limite_de_pasos_cierra_sin_consultar_al_modelo(monkeypatch):
    monkeypatch.setattr(config, "MAX_PASOS_SUPERVISOR", 2)
    monkeypatch.setattr(config, "MAX_RONDAS_POR_AGENTE", 10)
    supervisor = LLMFalso(guion=[decision("investigador", f"intento {i}") for i in range(10)])
    investigador = LLMFalso(guion=[AIMessage(content="Nada.")], ciclar=True)
    m = modelos([], investigador=investigador)
    m.supervisor = supervisor
    final = await correr(m)

    assert len(supervisor.llamadas) == 2
    assert final["pasos_supervisor"] == 3
    assert "Límite de 2 decisiones" in final["decisiones"][-1]["guarda"]


async def test_salida_invalida_del_supervisor_cae_en_la_decision_de_respaldo():
    m = modelos([AIMessage(content="no es una decisión"), decision("finalizar", cobertura=True)])
    final = await correr(m)
    primera = final["decisiones"][0]
    assert primera["destino"] == "investigador"
    assert primera["motivo"] == "Decisión de respaldo ante una salida inválida."


async def test_bucle_de_herramientas_del_especialista_queda_como_problema(monkeypatch):
    monkeypatch.setattr(config, "LIMITE_RECURSION_ESPECIALISTA", 4)
    investigador = LLMFalso(guion=[pedido("buscar_comercio", {"nombre": "x"}, "b")], ciclar=True)
    m = modelos([decision("investigador", "a"), decision("finalizar")], investigador=investigador)
    final = await correr(m)
    [contribucion] = final["contribuciones"]
    assert any("posible bucle" in p for p in contribucion["problemas"])


# --- Diagrama ---------------------------------------------------------------


def test_diagrama_tiene_la_topologia_jerarquica():
    mermaid = diagrama_mermaid()
    for arista in (
        "__start__ --> supervisor",
        "supervisor -.-> investigador",
        "supervisor -.-> analista",
        "supervisor -.-> sintetizador",
        "investigador --> supervisor",
        "analista --> supervisor",
        "sintetizador --> planificador",
        "planificador -.-> aprobacion_humana",
        "aprobacion_humana -.-> ejecutar_accion",
        "ejecutar_accion --> __end__",
    ):
        assert arista in mermaid


async def test_el_sintetizador_no_recibe_las_lineas_con_cifras_tachadas():
    investigador = LLMFalso(guion=[
        *guion_investigador()[:-1],
        AIMessage(content=f"{CONCLUSION_INVESTIGADOR}\nTotal sin cobrar: 5,000,000 ARS."),
    ])
    sintetizador = LLMFalso(guion=[AIMessage(content="ok")])
    m = modelos(
        [decision("investigador", "a"), decision("analista", "b"), decision("finalizar", cobertura=True)],
        investigador=investigador, sintetizador=sintetizador,
    )
    final = await correr(m)
    assert "[cifra descartada]" in final["contribuciones"][0]["contenido"]
    entrada = str(sintetizador.llamadas[0][-1].content)  # el mensaje, sin el prompt de sistema
    assert CONCLUSION_INVESTIGADOR in entrada
    assert "Total sin cobrar" not in entrada and "Limitaciones" not in entrada


# --- Limitaciones inventadas -------------------------------------------------


def test_quitar_limitaciones_inventadas_recorta_desde_el_encabezado():
    from app.agents.synthesizer import quitar_limitaciones_inventadas

    texto = "E4090 es un conflicto de idempotencia.\n\n**Limitaciones:** el cálculo le corresponde a otro agente."
    assert quitar_limitaciones_inventadas(texto) == "E4090 es un conflicto de idempotencia."
    assert quitar_limitaciones_inventadas("Sin sección.") == "Sin sección."


async def test_sin_limitaciones_en_la_entrada_no_hay_limitaciones_en_la_respuesta():
    inventa = LLMFalso(guion=[AIMessage(content="Respuesta.\n\nLimitaciones: no se puede programar nada.")])
    m = modelos(
        [decision("investigador", "a"), decision("analista", "b"), decision("finalizar", cobertura=True)],
        sintetizador=inventa,
    )
    final = await correr(m)
    assert final["task_completed"] is True
    assert final["respuesta_final"] == "Respuesta."


async def test_las_limitaciones_reales_se_conservan():
    """Un cierre forzado sí trae limitaciones: esas no se tocan."""
    sintetizador = LLMFalso(guion=[AIMessage(content="Respuesta.\n\nLimitaciones: el analista no intervino.")])
    m = modelos([decision("investigador", "a"), decision("finalizar")], sintetizador=sintetizador)
    final = await correr(m)
    assert final["task_completed"] is False
    assert "Limitaciones" in final["respuesta_final"]
