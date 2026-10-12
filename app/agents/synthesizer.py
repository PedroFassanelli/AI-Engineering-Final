"""El sintetizador: la fase final, única salida del grafo hacia `END`.

No investiga ni calcula. Integra la **última ronda** de cada especialista en una
respuesta para la persona, atribuyendo cada parte a quien la aportó, y declara
lo que quedó sin resolver en vez de disimularlo.
"""

from __future__ import annotations

import re
from typing import Any, Final

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app import config
from app.state import (
    ESPECIALISTAS,
    EstadoOrquestador,
    historial_conversacion,
    solicitud_actual,
    ultima_contribucion,
)

PROMPT_SINTETIZADOR: Final[str] = """\
Sos quien redacta la respuesta final de un equipo de soporte técnico de Nimbus \
Pagos. Recibís la solicitud del usuario y los informes de dos especialistas: un \
investigador (documentación y datos) y un analista (validación y cálculos).

Reglas:
- Usás SOLO lo que está en los informes. No agregás datos, cifras ni fuentes.
- Las explicaciones técnicas vienen del investigador, con la misma cita entre \
corchetes que trae su informe (por ejemplo [03-manejo-de-errores.md]). Citá un \
archivo solo al lado de la información que sale de él.
- Las cifras (totales, montos, porcentajes) vienen SOLO del informe del \
analista. Ignorá cualquier total o porcentaje que aparezca en el informe del \
investigador. Si el analista no intervino, no des cifras calculadas.
- El analista informa cada cifra como "<qué es> = <valor>". Usá esa descripción \
tal cual, sin resumirla: dice sobre qué total está calculada la cifra, y un \
porcentaje con el total equivocado es un error aunque el número sea correcto.
- Si dos informes se contradicen, el dato crudo del investigador manda sobre la \
interpretación, y la cuenta del analista manda sobre una cifra mencionada al pasar.
- Si la entrada trae una sección de limitaciones o problemas abiertos, los decís \
explícitamente al final, en una sección "Limitaciones". No los escondas. Si la \
entrada no la trae, tu respuesta tampoco. Esa sección contiene SOLO lo que trae la \
entrada: no le agregues nada propio.
- Las aclaraciones de un especialista sobre lo que le corresponde a otro rol \
("el cálculo le corresponde al analista", "no puedo programar reintentos") no son \
limitaciones ni información para el usuario: no las repitas.
- No enumeres pagos uno por uno salvo que la solicitud lo pida.

- Si la solicitud pide ejecutar una acción (por ejemplo, programar reintentos de pagos), NO digas que la hiciste, que la vas a hacer ni que no se puede (tampoco como limitación): esa parte la resuelve otro paso del sistema, con aprobación humana, y su resultado se agrega después de tu respuesta. Si los informes traen un dato relevante para esa decisión (por ejemplo, si la documentación marca el error como reintentable o no), decilo como dato, con su cita.

- Puede venir una lista de preguntas anteriores de la conversación: es contexto \
para entender la solicitud actual (por ejemplo "¿y el otro comercio?"). \
Respondé SOLO la solicitud actual, con los informes de este turno.

Formato: castellano rioplatense, claro y directo. Un párrafo por cada parte de \
la solicitud; sin títulos salvo "Limitaciones" si hace falta.
"""


def _sin_lineas_tachadas(texto: str) -> str:
    """Saca las líneas donde la depuración tachó una cifra.

    Esas líneas hablaban de números, y los números son del analista. Si se
    dejaban con la marca `[cifra descartada]`, el modelo la interpretaba como
    un dato faltante y agregaba una "limitación" que no existía.
    """
    return "\n".join(l for l in texto.splitlines() if "[cifra descartada]" not in l)


#: El encabezado de la sección, con o sin markdown: "Limitaciones:",
#: "**Limitaciones**", "### Limitaciones y problemas abiertos".
_SECCION_LIMITACIONES = re.compile(r"^\W*limitaciones\b", re.IGNORECASE | re.MULTILINE)


def quitar_limitaciones_inventadas(texto: str) -> str:
    """Recorta una sección "Limitaciones" que el modelo escribió por su cuenta.

    Solo se llama cuando la entrada **no** traía limitaciones. El prompt ya lo
    prohíbe, pero en las corridas reales `gpt-4o-mini` igual agregaba cosas
    como "No se puede programar el reintento en esta respuesta" o "los cálculos
    le corresponden a otro agente": aclaraciones de rol que el usuario no tiene
    por qué leer, presentadas como si faltara algo. Como en el resto del
    sistema, lo que se puede verificar no se deja librado al modelo.
    """
    coincidencia = _SECCION_LIMITACIONES.search(texto)
    if coincidencia is None:
        return texto
    return texto[: coincidencia.start()].rstrip()


def armar_entrada_sintetizador(estado: EstadoOrquestador) -> str:
    solicitud = solicitud_actual(estado)
    previas = historial_conversacion(
        estado, max_turnos=config.HISTORIAL_MAX_TURNOS, con_respuestas=False
    )
    contribuciones = estado.get("contribuciones", [])

    informes = []
    limitaciones: list[str] = []
    for agente in ESPECIALISTAS:
        ultima = ultima_contribucion(contribuciones, agente)
        if ultima is None:
            # Con la rúbrica completa, que un especialista no haya intervenido
            # quiere decir que no hacía falta (una consulta de documentación no
            # necesita al analista). Solo es una limitación si el cierre fue forzado.
            if not estado.get("task_completed", False):
                limitaciones.append(f"El {agente} no intervino.")
            continue
        informes.append(
            f"### Informe del {agente} (ronda {ultima['ronda']})\n{_sin_lineas_tachadas(ultima['contenido'])}"
        )
        limitaciones.extend(f"{agente}: {p}" for p in ultima["problemas"])

    decisiones = estado.get("decisiones", [])
    if decisiones and not estado.get("task_completed", False):
        if decisiones[-1]["guarda"]:
            limitaciones.append(f"Cierre forzado: {decisiones[-1]['guarda']}")
        fallidos = [c for c, ok in decisiones[-1]["rubrica"].items() if not ok]
        if fallidos:
            limitaciones.append(
                "Criterios de validación que no se cumplieron: " + ", ".join(fallidos)
            )

    entrada = (
        (f"## Preguntas anteriores de la conversación\n{previas}\n\n" if previas else "")
        + f"## Solicitud\n{solicitud}\n\n"
        f"{chr(10).join(informes) if informes else '(no hay informes)'}"
    )
    # Sin limitaciones, la sección directamente no existe. Mandarla con
    # "(ninguna)" hacía que el modelo escribiera "Limitaciones: ninguna" en la
    # respuesta, pese a que el prompt le pide omitirla.
    if limitaciones:
        entrada += "\n\n## Limitaciones y problemas abiertos\n" + "\n".join(
            f"- {l}" for l in limitaciones
        )
    return entrada


def crear_sintetizador(modelo: BaseChatModel):
    sistema = SystemMessage(content=PROMPT_SINTETIZADOR)

    async def sintetizador(estado: EstadoOrquestador) -> dict[str, Any]:
        entrada = armar_entrada_sintetizador(estado)
        respuesta = await modelo.ainvoke([sistema, HumanMessage(content=entrada)])
        contenido = respuesta.content
        if isinstance(contenido, list):  # bloques de contenido (Anthropic)
            contenido = "".join(b.get("text", "") for b in contenido if isinstance(b, dict))
        texto = str(contenido).strip()
        if "## Limitaciones y problemas abiertos" not in entrada:
            texto = quitar_limitaciones_inventadas(texto) or texto
        return {
            "respuesta_final": texto,
            "messages": [AIMessage(content=texto, name="sintetizador")],
        }

    return sintetizador
