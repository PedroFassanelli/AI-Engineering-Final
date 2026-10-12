"""Herramientas de los dos especialistas: argumentos, salidas validadas y fuentes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.rag.esquemas import ResultadoBusqueda
from app.tools.analysis_tools import calcular, evaluar_expresion, perfilar_pagos, porcentaje
from app.tools.esquemas import (
    ResultadoBuscarComercio,
    ResultadoCalcular,
    ResultadoConsultarPagos,
    ResultadoPerfilarPagos,
    ResultadoPorcentaje,
    SalidaHerramienta,
)
from app.tools.research_tools import buscar_comercio, buscar_documentacion, consultar_pagos

# --- Documentación (RAG) ----------------------------------------------------


async def test_buscar_documentacion_encuentra_el_codigo_en_el_catalogo():
    resultado = await buscar_documentacion.ainvoke({"consulta": "E4090"})
    fuentes = [r["fuente"] for r in resultado["resultados"]]
    assert "03-manejo-de-errores.md" in fuentes
    assert all("E4090" in r["texto"] for r in resultado["resultados"])


async def test_buscar_documentacion_trae_metadatos_para_citar():
    resultado = await buscar_documentacion.ainvoke({"consulta": "idempotencia de los cobros", "k": 2})
    primero = resultado["resultados"][0]
    assert set(primero) >= {"fuente", "seccion", "categoria", "posicion", "rango_lexico", "texto"}
    assert len(resultado["resultados"]) == 2
    assert resultado["modo"] == "lexico"  # en los tests el RAG corre sin Pinecone


async def test_buscar_documentacion_sin_coincidencias_devuelve_sugerencia():
    resultado = await buscar_documentacion.ainvoke({"consulta": "zzzz qqqq"})
    assert "error" in resultado and "sugerencia" in resultado
    assert ResultadoBusqueda.model_validate(resultado).error


async def test_buscar_documentacion_valida_los_argumentos():
    with pytest.raises(ValidationError):
        await buscar_documentacion.ainvoke({"consulta": "E4090", "k": 50})


# --- Base de datos ----------------------------------------------------------


async def test_buscar_comercio_ignora_acentos():
    resultado = await buscar_comercio.ainvoke({"nombre": "cafe tortoni"})
    assert [c["comercio_id"] for c in resultado["comercios"]] == [102, 103]


async def test_consultar_pagos_devuelve_todos_sin_recortar():
    resultado = await consultar_pagos.ainvoke({"comercio_id": 104})
    assert resultado["cantidad"] == 7
    assert sum(p["codigo_error"] == "E4090" for p in resultado["pagos"]) == 3
    # `codigo_error: null` es un dato de cada pago: no se omite al serializar.
    assert all("codigo_error" in p for p in resultado["pagos"])


async def test_consultar_pagos_de_comercio_inexistente_sugiere_buscarlo():
    resultado = await consultar_pagos.ainvoke({"comercio_id": 999})
    assert "buscar_comercio" in resultado["sugerencia"]
    assert "pagos" not in resultado


async def test_consultar_pagos_rechaza_ids_invalidos():
    with pytest.raises(ValidationError):
        await consultar_pagos.ainvoke({"comercio_id": -1})


# --- Análisis ---------------------------------------------------------------


async def _pagos_bazar_pampa() -> list[dict]:
    return (await consultar_pagos.ainvoke({"comercio_id": 104}))["pagos"]


async def test_perfilar_pagos_agrega_por_estado_y_por_codigo():
    perfil = await perfilar_pagos.ainvoke({"pagos": await _pagos_bazar_pampa()})
    ars = perfil["por_moneda"]["ARS"]
    assert ars["total"] == {"cantidad": 7, "monto_centavos": 6_680_000}
    assert ars["por_codigo_error"]["E4090"] == {"cantidad": 3, "monto_centavos": 3_930_000}
    # El código más frecuente va primero.
    assert next(iter(ars["por_codigo_error"])) == "E4090"


async def test_perfilar_pagos_informa_registros_invalidos_y_sigue():
    pagos = await _pagos_bazar_pampa()
    pagos[0] = {**pagos[0], "estado": "fallido", "codigo_error": None}  # fallido sin código
    pagos[1] = {**pagos[1], "monto_centavos": 129.9}  # float en vez de centavos
    perfil = await perfilar_pagos.ainvoke({"pagos": pagos})
    assert perfil["registros_validos"] == 5
    assert {i["pago_id"] for i in perfil["invalidos"]} == {"pay_10401", "pay_10402"}


async def test_perfilar_pagos_no_suma_monedas_distintas():
    pagos = [
        {"pago_id": "pay_1", "monto_centavos": 100, "moneda": "ARS", "estado": "aprobado"},
        {"pago_id": "pay_2", "monto_centavos": 900, "moneda": "CLP", "estado": "aprobado"},
    ]
    perfil = await perfilar_pagos.ainvoke({"pagos": pagos})
    assert set(perfil["por_moneda"]) == {"ARS", "CLP"}
    assert perfil["por_moneda"]["ARS"]["total"]["monto_centavos"] == 100


async def test_perfilar_pagos_sin_ningun_valido_devuelve_error():
    resultado = await perfilar_pagos.ainvoke({"pagos": [{"pago_id": "x"}]})
    assert "error" in resultado and "sugerencia" in resultado


@pytest.mark.parametrize(
    ("expresion", "esperado"),
    [("(890000 + 2150000) / 100", 30400.0), ("2 ** 10", 1024), ("-3 + 5", 2), ("58,83 * 2", 117.66)],
)
async def test_calcular_aritmetica(expresion, esperado):
    resultado = await calcular.ainvoke({"expresion": expresion})
    assert resultado["resultado"] == pytest.approx(esperado)


@pytest.mark.parametrize(
    "expresion", ["__import__('os').system('echo hola')", "abs(-1)", "x + 1", "10 ** 1000"]
)
async def test_calcular_rechaza_todo_lo_que_no_es_aritmetica(expresion):
    assert "error" in await calcular.ainvoke({"expresion": expresion})


def test_evaluar_expresion_no_usa_eval():
    with pytest.raises(ValueError):
        evaluar_expresion("[].__class__")


async def test_calcular_division_por_cero():
    assert (await calcular.ainvoke({"expresion": "1 / 0"}))["error"] == "División por cero."


async def test_porcentaje():
    assert (await porcentaje.ainvoke({"parte": 3, "total": 7}))["porcentaje"] == 42.86
    assert (await porcentaje.ainvoke({"parte": 3_930_000, "total": 6_680_000}))["porcentaje"] == 58.83


async def test_porcentaje_detecta_parte_mayor_que_total():
    assert "invertidos" in (await porcentaje.ainvoke({"parte": 8, "total": 7}))["sugerencia"]


# --- Contratos de salida -------------------------------------------------------


async def test_cada_herramienta_devuelve_su_contrato():
    """Lo que ve el modelo valida contra el esquema de salida de cada herramienta."""
    pagos = await _pagos_bazar_pampa()
    casos = [
        (ResultadoBusqueda, await buscar_documentacion.ainvoke({"consulta": "E4290 límite de tasa"})),
        (ResultadoBuscarComercio, await buscar_comercio.ainvoke({"nombre": "pampa"})),
        (ResultadoConsultarPagos, await consultar_pagos.ainvoke({"comercio_id": 104})),
        (ResultadoPerfilarPagos, await perfilar_pagos.ainvoke({"pagos": pagos})),
        (ResultadoCalcular, await calcular.ainvoke({"expresion": "1 + 1"})),
        (ResultadoPorcentaje, await porcentaje.ainvoke({"parte": 1, "total": 4})),
    ]
    for modelo, salida in casos:
        modelo.model_validate(salida)


def test_un_error_sin_sugerencia_no_es_una_salida_valida():
    with pytest.raises(ValidationError):
        SalidaHerramienta(error="algo falló")


def test_consultar_pagos_exige_cantidad_coherente():
    with pytest.raises(ValidationError):
        ResultadoConsultarPagos(comercio_id=1, cantidad=3, pagos=[])
