"""Chequeos deterministas sobre las contribuciones de los especialistas.

Los casos de `test_procedencia_*` reproducen la primera corrida real del
proyecto (README → "Lo que salió mal"): el investigador inventó un total, el
supervisor lo copió en la instrucción y el analista calculó sobre él.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.agents.analyst_agent import chequeo_escala_de_montos, chequeo_procedencia_de_cifras
from app.agents.base import (
    cifras_sin_respaldo,
    lecturas_por_token,
    montos_en_centavos,
    montos_mal_escalados,
    numeros_en,
    respaldado,
)
from app.agents.research_agent import chequeo_citas_verificables, chequeo_pagos_incompletos, depurar_cifras
from app.fuentes.db import PAGOS
from app.state import Contribucion

PAGOS_104 = [
    {"pago_id": p[0], "comercio_id": p[1], "monto_centavos": p[2], "moneda": p[3],
     "estado": p[4], "codigo_error": p[5]}
    for p in PAGOS if p[1] == 104
]


def contribucion(agente, contenido, evidencia, instruccion="") -> Contribucion:
    return Contribucion(
        agente=agente, ronda=1, instruccion=instruccion, contenido=contenido,
        evidencia=evidencia, problemas=[], avisos=[],
    )


def investigacion(contenido="Datos del comercio 104.") -> Contribucion:
    return contribucion(
        "investigador",
        contenido,
        [{"herramienta": "consultar_pagos", "argumentos": {"comercio_id": 104},
          "resultado": {"comercio_id": 104, "cantidad": 7, "pagos": PAGOS_104}}],
    )


def estado_con(*contribuciones: Contribucion) -> dict:
    return {"messages": [HumanMessage(content="¿?")], "contribuciones": list(contribuciones)}


PERFIL = {
    "por_moneda": {"ARS": {"total": {"cantidad": 7, "monto_centavos": 6680000},
                           "por_codigo_error": {"E4090": {"cantidad": 3, "monto_centavos": 3930000}}}}
}

# --- Lectura de números -----------------------------------------------------


def test_lecturas_conserva_el_token_y_admite_las_dos_convenciones():
    [(token, lecturas)] = lecturas_por_token("total: 6.680.000")
    assert token == "6.680.000" and 6680000.0 in lecturas
    assert {58.83} <= numeros_en("58,83 %") and {58.83} <= numeros_en("58.83%")


def test_no_lee_digitos_pegados_a_identificadores():
    assert numeros_en("pay_10402 falló con E4090") == set()


def test_respaldado_admite_redondeo_y_conversion_de_centavos():
    assert respaldado(58.8, {58.83})
    assert respaldado(39300, {3930000})
    assert respaldado(100, set())  # el 100 de "× 100"
    assert not respaldado(5000000, {6680000})


def test_cifras_sin_respaldo_informa_el_texto_original():
    assert cifras_sin_respaldo("el total es 5,000,000", {6680000}) == ["5,000,000"]


# --- Investigador -----------------------------------------------------------


def test_citas_inventadas_se_detectan_con_o_sin_corchetes():
    c = contribucion(
        "investigador",
        "Ver [03-manejo-de-errores.md] y también Fuente: 04-webhooks-y-conciliacion.md",
        [{"herramienta": "buscar_documentacion", "argumentos": {"consulta": "E4090"},
          "resultado": {"resultados": [{"fuente": "03-manejo-de-errores.md"}]}}],
    )
    [problema] = chequeo_citas_verificables(c, estado_con())
    assert "04-webhooks-y-conciliacion.md" in problema
    assert "03-manejo" not in problema


def test_total_calculado_por_el_investigador_se_tacha_y_se_avisa():
    c = investigacion("El comercio tiene 7 pagos y el total es de 5,000,000 centavos.")
    contenido, [aviso] = depurar_cifras(c, estado_con())
    assert contenido == "El comercio tiene 7 pagos y el total es de [cifra descartada] centavos."
    assert "5,000,000" in aviso


def test_montos_copiados_de_los_datos_no_se_tocan():
    texto = "Fallidos E4090: 890,000, 890,000 y 2,150,000 centavos (21.500,00 ARS)."
    assert depurar_cifras(investigacion(texto), estado_con()) == (texto, [])


def test_monto_en_centavos_presentado_como_moneda_se_tacha():
    c = investigacion("El pago pay_10405 fue de 2,150,000 ARS.")
    contenido, [aviso] = depurar_cifras(c, estado_con())
    assert "[cifra descartada] ARS" in contenido
    assert "centavos como si fueran unidades" in aviso


# --- Escala de montos -------------------------------------------------------


def test_montos_mal_escalados_distingue_centavos_de_unidades():
    centavos = {6300000.0, 3150000.0}
    texto = "6,300,000 CLP sin cobrar (63.000 CLP). Cada uno: CLP 3.150.000; en unidades 31,500.00 CLP."
    assert montos_mal_escalados(texto, centavos) == ["6,300,000", "3.150.000"]


def test_montos_en_centavos_recorre_la_estructura():
    assert montos_en_centavos({"a": [{"monto_centavos": 5}, {"x": {"monto_centavos": 7}}]}) == {5.0, 7.0}


def test_analista_con_centavos_como_moneda_es_problema_bloqueante():
    analisis = contribucion(
        "analista", "Monto sin cobrar por E4090 = 3,930,000 ARS",
        [{"herramienta": "perfilar_pagos", "argumentos": {"pagos": PAGOS_104}, "resultado": PERFIL}],
    )
    [problema] = chequeo_escala_de_montos(analisis, estado_con(investigacion()))
    assert "3,930,000" in problema


# --- Analista: procedencia --------------------------------------------------


def test_procedencia_acepta_una_cadena_completa_desde_los_datos():
    analisis = contribucion(
        "analista",
        "3 de 7 pagos (42.86%). Monto: 3,930,000 de 6,680,000 centavos = 58.83%.",
        [
            {"herramienta": "perfilar_pagos", "argumentos": {"pagos": PAGOS_104}, "resultado": PERFIL},
            {"herramienta": "porcentaje", "argumentos": {"parte": 3, "total": 7},
             "resultado": {"porcentaje": 42.86}},
            {"herramienta": "porcentaje", "argumentos": {"parte": 3930000, "total": 6680000},
             "resultado": {"porcentaje": 58.83}},
        ],
    )
    assert chequeo_procedencia_de_cifras(analisis, estado_con(investigacion())) == []


def test_procedencia_rechaza_valores_dictados_por_la_instruccion():
    """La instrucción del supervisor no respalda nada, aunque el número esté ahí."""
    analisis = contribucion(
        "analista",
        "4,000,000 sobre 5,000,000 centavos: 80.00%.",
        [
            {"herramienta": "perfilar_pagos", "argumentos": {"pagos": PAGOS_104}, "resultado": PERFIL},
            {"herramienta": "porcentaje", "argumentos": {"parte": 4000000, "total": 5000000},
             "resultado": {"porcentaje": 80.0}},
        ],
        instruccion="Usá el total de 5,000,000 centavos y 4,000,000 de fallidos.",
    )
    problemas = chequeo_procedencia_de_cifras(analisis, estado_con(investigacion()))
    assert any("`porcentaje` recibió valores" in p for p in problemas)
    assert any("80.00" in p for p in problemas)


def test_procedencia_sin_datos_del_investigador_no_respalda_nada():
    analisis = contribucion(
        "analista", "Total: 6,680,000.",
        [{"herramienta": "calcular", "argumentos": {"expresion": "6680000"},
          "resultado": {"resultado": 6680000}}],
    )
    assert chequeo_procedencia_de_cifras(analisis, estado_con()) != []


# --- Pagos listados ---------------------------------------------------------


def test_listar_parte_de_los_fallidos_de_un_codigo_es_problema():
    incompleta = investigacion("Fallidos por E4090: pay_10402 y pay_10405.")
    [problema] = chequeo_pagos_incompletos(incompleta, estado_con(incompleta))
    assert "E4090" in problema and "pay_10403" in problema


def test_listar_todos_o_ninguno_no_es_problema():
    completa = investigacion("Fallidos por E4090: pay_10402, pay_10403 y pay_10405; por E4290: pay_10406.")
    assert chequeo_pagos_incompletos(completa, estado_con(completa)) == []
    sin_ids = investigacion("El código más frecuente es E4090.")
    assert chequeo_pagos_incompletos(sin_ids, estado_con(sin_ids)) == []
