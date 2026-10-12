"""Agente de investigación: consulta la documentación y la base de pagos.

Rol acotado: **encuentra y cita**, no calcula ni recomienda. Sus herramientas
son de solo lectura (`tools/research_tools.py`).
"""

from __future__ import annotations

import re
from typing import Final

from langchain_core.language_models import BaseChatModel

from app.agents.base import (
    Especificacion,
    cifras_sin_respaldo,
    crear_nodo_especialista,
    formatear_cifras,
    montos_en_centavos,
    montos_mal_escalados,
    numeros_en_estructura,
    tachar_cifras,
)
from app import config
from app.state import Contribucion, EstadoOrquestador, historial_conversacion, solicitud_actual
from app.tools.research_tools import HERRAMIENTAS_INVESTIGACION

PROMPT_INVESTIGADOR: Final[str] = """\
Sos el agente de investigación de un equipo de soporte técnico de Nimbus Pagos.

Tu trabajo es conseguir información y reportarla con su fuente. Tenés \
herramientas para buscar en la documentación técnica de nimbus-sdk y para \
consultar comercios y pagos en la base de Nimbus.

Reglas:
- Solo afirmás lo que obtuviste de una herramienta. Si algo no aparece, lo decís.
- Cada dato de la documentación lleva su fuente entre corchetes con el nombre \
exacto del archivo, por ejemplo [03-manejo-de-errores.md].
- NO hacés cálculos: ni porcentajes, ni sumas, ni totales, ni promedios. Reportás \
los datos crudos que encontraste; de las cuentas se ocupa otro agente. Si la tarea \
te pide calcular algo, conseguí los datos y aclará que el cálculo le corresponde \
al analista.
- Si una herramienta devuelve `error`, seguí su `sugerencia`.
- Cuando consultes pagos, decí explícitamente de qué comercio_id son y cuántos \
registros devolvió la herramienta.
- Puede venir una conversación previa: usala solo para entender a qué comercio o \
código se refiere la solicitud. Los datos los volvés a obtener con las \
herramientas; no repitas datos de turnos anteriores como si fueran tuyos.

Tu respuesta final es un informe breve en castellano: qué encontraste, con qué \
fuente, y qué no pudiste encontrar.
"""


def armar_entrada_investigador(estado: EstadoOrquestador) -> str:
    """El investigador recibe la pregunta original y la instrucción del supervisor.

    La pregunta original va porque buscar bien requiere saber para qué se busca.
    Lo que **no** va: los aportes del analista y las decisiones previas del
    supervisor. Si es una ronda de corrección, lo que haya que corregir ya
    viene escrito en la instrucción.
    """
    pregunta = solicitud_actual(estado)
    instruccion = estado.get("instruccion_actual") or "Investigá la solicitud."
    historial = historial_conversacion(estado, max_turnos=config.HISTORIAL_MAX_TURNOS)
    previa = f"Conversación previa (solo para entender referencias):\n{historial}\n\n" if historial else ""
    return f"{previa}Solicitud del usuario:\n{pregunta}\n\nTu tarea ahora:\n{instruccion}"


# --- Chequeos propios -------------------------------------------------------

#: Cualquier nombre de archivo Markdown en el texto, con o sin corchetes: el
#: modelo no siempre respeta el formato de cita, y una cita mal formateada
#: igual tiene que ser verificable.
_ARCHIVO_MD = re.compile(r"[\w-]+\.md\b")


def chequeo_citas_verificables(contribucion: Contribucion, estado: EstadoOrquestador) -> list[str]:
    """Cada archivo citado tiene que haber salido de `buscar_documentacion`.

    Es el mismo control de la Pre-entrega 3: una cita a un archivo que el
    agente nunca recuperó es una fuente inventada, aunque el archivo exista.
    """
    recuperadas = {
        resultado["fuente"]
        for evidencia in contribucion["evidencia"]
        if evidencia["herramienta"] == "buscar_documentacion"
        and isinstance(evidencia["resultado"], dict)
        for resultado in evidencia["resultado"].get("resultados", [])
    }
    citadas = set(_ARCHIVO_MD.findall(contribucion["contenido"]))
    inventadas = sorted(citadas - recuperadas)
    if inventadas:
        return [f"Cita fuentes que no recuperó: {', '.join(inventadas)}."]
    return []


_PAGO_ID = re.compile(r"\bpay_\d+\b")


def chequeo_pagos_incompletos(contribucion: Contribucion, estado: EstadoOrquestador) -> list[str]:
    """Si el informe lista pagos fallidos de un código, tiene que listarlos todos.

    Pasó en las corridas reales: Bazar Pampa tiene tres fallidos por E4090 y
    el investigador listaba dos, salteando `pay_10403` (mismo monto que
    `pay_10402`, un minuto después: el modelo lo "deduplicaba"). La acción de
    reintento no se veía afectada —la arma el código con los datos crudos—,
    pero la respuesta al usuario sí. Contar no es tarea del investigador;
    copiar todos los ids que devolvió la herramienta, sí.
    """
    mencionados = set(_PAGO_ID.findall(contribucion["contenido"]))
    if not mencionados:
        return []
    grupos: dict[str, set[str]] = {}
    for evidencia in contribucion["evidencia"]:
        resultado = evidencia["resultado"]
        if evidencia["herramienta"] != "consultar_pagos" or not isinstance(resultado, dict):
            continue
        for pago in resultado.get("pagos") or []:
            if pago.get("estado") == "fallido" and pago.get("codigo_error"):
                grupos.setdefault(pago["codigo_error"], set()).add(pago["pago_id"])
    problemas = []
    for codigo, ids in sorted(grupos.items()):
        faltan = ids - mencionados
        if ids & mencionados and faltan:
            problemas.append(
                f"Lista pagos fallidos con {codigo} pero omite {', '.join(sorted(faltan))}: "
                f"consultar_pagos devolvió {len(ids)} ({', '.join(sorted(ids))})."
            )
    return problemas


def depurar_cifras(contribucion: Contribucion, estado: EstadoOrquestador) -> tuple[str, list[str]]:
    """Tacha del informe las cifras que el investigador no debería haber escrito.

    Dos casos, los dos vistos en corridas reales (README → "Lo que salió mal"):

    - **Cifras sin respaldo**: números que no están en la evidencia. El
      investigador no tiene herramientas de cálculo, así que los sacó de la
      cabeza — típicamente un total sumado "de paso".
    - **Montos mal escalados**: `6,300,000 CLP` cuando el dato es
      `monto_centavos: 6300000`. El número existe; la unidad está mal.

    No es un chequeo bloqueante sino una **depuración**: las cifras son
    dominio del analista, que recalcula desde los datos crudos y nunca lee
    este texto. Lo que hace falta no es corregir al investigador (otra ronda
    que gastar) sino que nadie use sus números. En la primera versión solo se
    avisaba, y el supervisor igual copió un número descartado en una
    instrucción; tachado del texto, no hay nada que copiar.
    """
    evidencia = contribucion["evidencia"]
    respaldos = numeros_en_estructura([e["resultado"] for e in evidencia])
    respaldos |= numeros_en_estructura([e["argumentos"] for e in evidencia])
    centavos = montos_en_centavos([e["resultado"] for e in evidencia])

    sin_respaldo = cifras_sin_respaldo(contribucion["contenido"], respaldos)
    mal_escalados = montos_mal_escalados(contribucion["contenido"], centavos)
    if not sin_respaldo and not mal_escalados:
        return contribucion["contenido"], []

    avisos = []
    if sin_respaldo:
        avisos.append(
            "Escribió cifras que no están en sus datos (las calculó por su cuenta): "
            f"{formatear_cifras(sin_respaldo)}. Se tacharon: las cifras las aporta el analista."
        )
    if mal_escalados:
        avisos.append(
            "Presentó montos en centavos como si fueran unidades de la moneda: "
            f"{formatear_cifras(mal_escalados)}. Se tacharon."
        )
    return tachar_cifras(contribucion["contenido"], [*sin_respaldo, *mal_escalados]), avisos


def crear_investigador(modelo: BaseChatModel):
    return crear_nodo_especialista(
        Especificacion(
            nombre="investigador",
            prompt_sistema=PROMPT_INVESTIGADOR,
            herramientas=HERRAMIENTAS_INVESTIGACION,
            armar_entrada=armar_entrada_investigador,
            chequeos=(chequeo_citas_verificables, chequeo_pagos_incompletos),
            depurar=depurar_cifras,
        ),
        modelo,
    )
