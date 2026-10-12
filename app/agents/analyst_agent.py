"""Agente de análisis: valida y calcula sobre los datos del investigador.

Rol acotado: **no tiene acceso a ninguna fuente**. Trabaja solo con los datos
crudos que el investigador obtuvo, y todo número que reporta tiene que poder
rastrearse hasta esos datos. Lo verifica `chequeo_procedencia_de_cifras`.
"""

from __future__ import annotations

import json
from typing import Any, Final

from langchain_core.language_models import BaseChatModel

from app.agents.base import (
    Especificacion,
    cifras_sin_respaldo,
    crear_nodo_especialista,
    formatear_cifras,
    montos_en_centavos,
    montos_mal_escalados,
    numeros_en_estructura,
    respaldado,
)
from app import config
from app.state import (
    Contribucion,
    EstadoOrquestador,
    historial_conversacion,
    solicitud_actual,
    ultima_contribucion,
)
from app.tools.research_tools import HERRAMIENTAS_DE_DATOS
from app.tools.analysis_tools import HERRAMIENTAS_ANALISIS

PROMPT_ANALISTA: Final[str] = """\
Sos el agente de análisis cuantitativo de un equipo de soporte técnico de \
Nimbus Pagos.

Recibís la solicitud del usuario, una tarea concreta y los datos crudos que \
consiguió otro agente. Tus herramientas validan el esquema de los datos y hacen \
cuentas exactas.

Reglas:
- La solicitud del usuario define QUÉ hay que cuantificar. Si la tarea lo \
reformula distinto (por ejemplo, otro total como base del porcentaje), manda la \
solicitud; si hay dos lecturas razonables, calculá las dos y rotulá cada una.
- Trabajás SOLO con los datos crudos que te pasan. No tenés acceso a ninguna \
base ni documento. Si falta un dato para cumplir la tarea, decí cuál falta.
- Primero validás los datos con `perfilar_pagos`. Si hay registros inválidos, \
lo informás y calculás solo sobre los válidos.
- Los valores que uses en `calcular` y `porcentaje` tienen que salir de los \
datos crudos o del resultado de `perfilar_pagos`. Si la tarea menciona una cifra \
que no coincide con los datos, manda el dato: usá el de los datos y avisá de la \
diferencia.
- TODO número que informes tiene que salir de una herramienta. No hagas cuentas \
de cabeza, ni siquiera las fáciles.
- Los montos vienen en centavos. Si escribís un monto junto a la moneda (ARS, \
CLP), tiene que estar en unidades: dividilo por 100 con `calcular`. Si no, \
escribí "centavos".
- No interpretás causas ni das recomendaciones: eso no es tu rol.

Tu respuesta final es un informe breve en castellano con una línea por cifra, \
con este formato:
- <qué es, incluyendo sobre qué total se calcula> = <valor>  (cálculo: <herramienta y valores>)
"""


def datos_para_analista(estado: EstadoOrquestador) -> list[dict[str, Any]]:
    """Los resultados de las herramientas de datos de la última ronda del investigador."""
    investigacion = ultima_contribucion(estado.get("contribuciones", []), "investigador")
    if investigacion is None:
        return []
    return [
        {"herramienta": e["herramienta"], "argumentos": e["argumentos"], "resultado": e["resultado"]}
        for e in investigacion["evidencia"]
        if e["herramienta"] in HERRAMIENTAS_DE_DATOS
    ]


def armar_entrada_analista(estado: EstadoOrquestador) -> str:
    """El analista recibe la solicitud, la instrucción y **solo los datos crudos**.

    La solicitud va porque define *qué* cuantificar ("fallidos por ese código"
    no es lo mismo que "fallidos"), y un error ahí no lo arregla ninguna
    cuenta bien hecha. Lo que **no** va: el texto de la documentación, las
    conclusiones del investigador y el historial. Las conclusiones, sobre todo:
    si el investigador escribió un total mal sumado, el analista lo tomaría
    como dato. Recibiendo solo los registros crudos, tiene que recalcularlo.
    """
    solicitud = solicitud_actual(estado)
    # Solo las preguntas anteriores, sin respuestas: sirven para entender "¿y
    # sobre el monto?", pero una cifra de una respuesta vieja no es un dato.
    previas = historial_conversacion(
        estado, max_turnos=config.HISTORIAL_MAX_TURNOS, con_respuestas=False
    )
    instruccion = estado.get("instruccion_actual") or "Analizá los datos."
    datos = datos_para_analista(estado)
    bloque_datos = (
        json.dumps(datos, ensure_ascii=False, indent=1)
        if datos
        else "(El investigador no aportó datos. Si la tarea los necesita, decilo.)"
    )
    return (
        (f"Preguntas anteriores de la conversación:\n{previas}\n\n" if previas else "")
        + f"Solicitud del usuario:\n{solicitud}\n\n"
        f"Tu tarea:\n{instruccion}\n\n"
        f"Datos crudos obtenidos por el agente de investigación:\n{bloque_datos}"
    )


# --- Chequeos propios -------------------------------------------------------


def chequeo_datos_validados(contribucion: Contribucion, estado: EstadoOrquestador) -> list[str]:
    """Si el analista no llegó a perfilar datos, sus cifras no se apoyan en nada."""
    if not any(e["herramienta"] == "perfilar_pagos" for e in contribucion["evidencia"]):
        return ["No validó los datos con `perfilar_pagos` antes de calcular."]
    return []


def chequeo_procedencia_de_cifras(
    contribucion: Contribucion, estado: EstadoOrquestador
) -> list[str]:
    """Cada número tiene que poder rastrearse hasta los datos crudos.

    Se recorre la evidencia **en orden**, como una cadena de custodia:

    - arrancan respaldados los números de los datos crudos del investigador;
    - cada argumento numérico de una herramienta tiene que estar respaldado;
      si lo está, el resultado de esa herramienta pasa a estarlo también;
    - al final, cada cifra del informe tiene que estar respaldada.

    La instrucción del supervisor **no** respalda nada. Es texto escrito por
    un modelo, y un supervisor que copia en la instrucción un total inventado
    por otro agente es justamente el caso que este chequeo tiene que atrapar.
    """
    respaldos = numeros_en_estructura([d["resultado"] for d in datos_para_analista(estado)])
    problemas: list[str] = []

    for evidencia in contribucion["evidencia"]:
        argumentos = numeros_en_estructura(evidencia["argumentos"])
        sin_origen = sorted(v for v in argumentos if not respaldado(v, respaldos))
        if sin_origen:
            problemas.append(
                f"`{evidencia['herramienta']}` recibió valores que no salen de los datos: "
                f"{formatear_cifras(sin_origen)}."
            )
        else:
            respaldos |= numeros_en_estructura(evidencia["resultado"])

    sin_respaldo = cifras_sin_respaldo(contribucion["contenido"], respaldos)
    if sin_respaldo:
        problemas.append(
            f"Informa cifras que no salen de ningún cálculo sobre los datos: {formatear_cifras(sin_respaldo)}."
        )
    return problemas


def chequeo_escala_de_montos(contribucion: Contribucion, estado: EstadoOrquestador) -> list[str]:
    """Un monto en centavos presentado como unidades de la moneda.

    `chequeo_procedencia_de_cifras` no lo atrapa: `6,300,000` sí sale de los
    datos. Lo que está mal es escribir `CLP` al lado, y eso convierte 63.000
    pesos chilenos en seis millones. Pasó en una corrida real, con el
    supervisor dictándole el número al analista en una corrección.
    """
    centavos = montos_en_centavos([d["resultado"] for d in datos_para_analista(estado)])
    centavos |= montos_en_centavos([e["resultado"] for e in contribucion["evidencia"]])
    mal_escalados = montos_mal_escalados(contribucion["contenido"], centavos)
    if mal_escalados:
        return [
            "Presenta montos en centavos como si fueran unidades de la moneda: "
            f"{formatear_cifras(mal_escalados)} (hay que dividir por 100)."
        ]
    return []


def crear_analista(modelo: BaseChatModel):
    return crear_nodo_especialista(
        Especificacion(
            nombre="analista",
            prompt_sistema=PROMPT_ANALISTA,
            herramientas=HERRAMIENTAS_ANALISIS,
            armar_entrada=armar_entrada_analista,
            chequeos=(
                chequeo_datos_validados,
                chequeo_procedencia_de_cifras,
                chequeo_escala_de_montos,
            ),
        ),
        modelo,
    )
