"""Las guardas del supervisor y la rúbrica, sin compilar el grafo."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app import config
from app.agents.supervisor import (
    DecisionSupervisor,
    aplicar_guardas,
    armar_contexto_supervisor,
    enrutar,
    evaluar_respaldo,
)
from app.state import Contribucion, Decision


def propuesta(siguiente: str, instruccion: str = "hacé algo") -> DecisionSupervisor:
    return DecisionSupervisor(
        evaluacion="-", cobertura=False, coherencia=True,
        siguiente=siguiente, instruccion=instruccion, motivo="-",
    )


def aporte(agente: str, problemas: list[str] | None = None, ronda: int = 1) -> Contribucion:
    return Contribucion(
        agente=agente, ronda=ronda, instruccion="", contenido="informe",
        evidencia=[], problemas=problemas or [], avisos=[],
    )


def decidido(destino: str, instruccion: str) -> Decision:
    return Decision(
        paso=1, propuesta=destino, destino=destino, instruccion=instruccion,
        motivo="-", rubrica={}, guarda=None,
    )


def estado(contribuciones=(), decisiones=()) -> dict:
    return {
        "messages": [HumanMessage(content="pregunta")],
        "contribuciones": list(contribuciones),
        "decisiones": list(decisiones),
        "pasos_supervisor": len(decisiones),
    }


RUBRICA_INCOMPLETA = {"cobertura": False, "coherencia": True, "respaldo": True}
RUBRICA_COMPLETA = {"cobertura": True, "coherencia": True, "respaldo": True}

# --- Rúbrica verificable ----------------------------------------------------


def test_respaldo_no_se_cumple_sin_contribuciones():
    assert evaluar_respaldo(estado()) is False


def test_respaldo_mira_solo_la_ultima_ronda_de_cada_agente():
    corregido = estado([aporte("investigador", ["mal"]), aporte("investigador", [], ronda=2)])
    assert evaluar_respaldo(corregido) is True
    empeorado = estado([aporte("investigador"), aporte("analista", ["cifra sin respaldo"])])
    assert evaluar_respaldo(empeorado) is False


def test_el_contexto_del_supervisor_informa_el_criterio_automatico():
    texto = armar_contexto_supervisor(estado([aporte("investigador", ["cita inventada"])]))
    assert "respaldo: NO se cumple" in texto
    assert "cita inventada" in texto


# --- Guardas ----------------------------------------------------------------


def test_sin_guardas_la_propuesta_pasa_tal_cual():
    destino, instruccion, guarda = aplicar_guardas(
        propuesta("investigador", "Buscá E4090"), estado(), RUBRICA_INCOMPLETA
    )
    assert (destino, instruccion, guarda) == ("investigador", "Buscá E4090", None)


def test_finalizar_va_al_sintetizador():
    assert aplicar_guardas(propuesta("finalizar", ""), estado(), RUBRICA_INCOMPLETA) == (
        "sintetizador", "", None,
    )


def test_suficiencia_corta_el_refinamiento_de_un_agente_que_ya_trabajo():
    ya_trabajaron = estado([aporte("investigador"), aporte("analista")])
    destino, _, guarda = aplicar_guardas(propuesta("analista", "Detallá más"), ya_trabajaron, RUBRICA_COMPLETA)
    assert destino == "sintetizador"
    assert "no se refina" in guarda


def test_suficiencia_no_impide_la_primera_intervencion_de_un_agente():
    """Caso real: el modelo marcó cobertura=True y a la vez pidió al analista,
    que nunca había intervenido. Cerrar ahí dejaba la pregunta sin cifras."""
    destino, _, guarda = aplicar_guardas(
        propuesta("analista", "Calculá el monto"), estado([aporte("investigador")]), RUBRICA_COMPLETA
    )
    assert (destino, guarda) == ("analista", None)


def test_dependencia_redirige_al_analista_sin_datos():
    destino, instruccion, guarda = aplicar_guardas(
        propuesta("analista", "Calculá el porcentaje"), estado(), RUBRICA_INCOMPLETA
    )
    assert destino == "investigador"
    assert "Calculá el porcentaje" in instruccion
    assert "no tiene datos" in guarda


def test_limite_de_rondas_por_agente(monkeypatch):
    monkeypatch.setattr(config, "MAX_RONDAS_POR_AGENTE", 2)
    ya_dos = estado([aporte("investigador"), aporte("investigador", ronda=2)])
    destino, _, guarda = aplicar_guardas(propuesta("investigador", "otra vez"), ya_dos, RUBRICA_INCOMPLETA)
    assert destino == "sintetizador"
    assert "2 rondas" in guarda


def test_instruccion_repetida_corta_el_bucle():
    previo = estado([aporte("investigador")], [decidido("investigador", "Buscá  E4090")])
    destino, _, guarda = aplicar_guardas(propuesta("investigador", "buscá e4090"), previo, RUBRICA_INCOMPLETA)
    assert destino == "sintetizador"
    assert "repetida" in guarda


@pytest.mark.parametrize("destino", ["investigador", "analista", "sintetizador"])
def test_enrutar_devuelve_el_destino_decidido(destino):
    assert enrutar({"next_agent": destino}) == destino
