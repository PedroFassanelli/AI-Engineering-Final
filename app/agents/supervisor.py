"""El supervisor: router del grafo y validador de los especialistas.

Cada vez que corre responde una sola pregunta: *dada la solicitud y lo que
aportó cada especialista, ¿quién interviene ahora, o ya se puede cerrar?*

La decisión tiene tres piezas, y la separación es el punto central del diseño:

1. **Una rúbrica híbrida.** Dos criterios son semánticos (¿está respondida
   cada parte de la pregunta?, ¿los agentes se contradicen?) y los juzga el
   modelo. El tercero —¿cada cifra y cada cita tiene respaldo?— es verificable,
   y lo calcula el código a partir de los chequeos de los especialistas. El
   modelo no ve los datos crudos, así que pedirle que juzgue respaldo es
   pedirle que adivine; en la primera versión lo hacía, y adivinaba mal.
2. **El modelo propone.** Con salida estructurada (`DecisionSupervisor`):
   evalúa sus dos criterios, elige el siguiente paso y redacta la instrucción.
3. **Las guardas disponen.** Funciones deterministas (`aplicar_guardas`) que
   pueden pisar la propuesta: suficiencia, dependencia de datos, límite de
   rondas por agente, instrucciones repetidas y límite de pasos. Son las que
   garantizan que el grafo termina sin depender de que el modelo "se dé
   cuenta" de que está en un bucle.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app import config
from app.state import (
    ESPECIALISTAS,
    Contribucion,
    Decision,
    Destino,
    EstadoOrquestador,
    historial_conversacion,
    rondas_de,
    solicitud_actual,
    ultima_contribucion,
)
from app.tools.research_tools import HERRAMIENTAS_DE_DATOS

# --- Rúbrica ----------------------------------------------------------------

#: Los criterios que juzga el modelo.
CRITERIOS_SEMANTICOS: Final[dict[str, str]] = {
    "cobertura": (
        "Cada parte de la solicitud está respondida por alguna contribución. Si la "
        "solicitud pide identificar algo, explicarlo, decir cómo se resuelve y "
        "cuantificarlo, hacen falta las cuatro cosas."
    ),
    "coherencia": (
        "Las contribuciones no se contradicen entre sí: mismo comercio, mismo código, "
        "mismas cantidades. Una contradicción se resuelve pidiendo una corrección, no eligiendo."
    ),
}

#: El criterio que calcula el código. Se describe en el prompt para que el
#: modelo sepa que existe y qué significa, pero no lo evalúa.
CRITERIO_VERIFICABLE: Final[tuple[str, str]] = (
    "respaldo",
    "Cada cifra de un especialista sale de una herramienta sobre los datos, y cada "
    "archivo citado fue efectivamente recuperado. Lo verifican los chequeos "
    "automáticos: se cumple si la última ronda de cada especialista no tiene problemas.",
)

RUBRICA: Final[tuple[str, ...]] = (*CRITERIOS_SEMANTICOS, CRITERIO_VERIFICABLE[0])


def evaluar_respaldo(estado: EstadoOrquestador) -> bool:
    """El criterio verificable: la última ronda de cada especialista que
    intervino pasó todos sus chequeos. Sin ninguna contribución, no hay nada
    respaldado."""
    contribuciones = estado.get("contribuciones", [])
    ultimas = [ultima_contribucion(contribuciones, a) for a in ESPECIALISTAS]
    presentes = [c for c in ultimas if c is not None]
    return bool(presentes) and all(not c["problemas"] for c in presentes)


class DecisionSupervisor(BaseModel):
    """Salida estructurada del supervisor.

    El orden de los campos es deliberado: primero la evaluación y los
    criterios, después la decisión. El modelo genera en ese orden, así que
    decide *habiendo escrito* por qué.
    """

    evaluacion: str = Field(
        description="Dos o tres oraciones: qué aportó cada especialista y qué falta."
    )
    cobertura: bool = Field(description=CRITERIOS_SEMANTICOS["cobertura"])
    coherencia: bool = Field(description=CRITERIOS_SEMANTICOS["coherencia"])
    siguiente: Literal["investigador", "analista", "finalizar"] = Field(
        description="Quién interviene ahora, o 'finalizar' si la rúbrica se cumple."
    )
    instruccion: str = Field(
        description=(
            "Instrucción concreta y autosuficiente para el especialista elegido: qué hacer "
            "y, si es una corrección, qué estuvo mal. Vacía si 'finalizar'."
        )
    )
    motivo: str = Field(description="Una oración: por qué este paso y no otro.")


PROMPT_SUPERVISOR: Final[str] = """\
Sos el supervisor de un equipo de dos agentes especialistas que resuelven \
consultas de soporte técnico de Nimbus Pagos. No respondés vos: decidís quién \
trabaja y cuándo el trabajo está completo.

Especialistas:
- investigador: busca en la documentación técnica de nimbus-sdk y consulta \
comercios y pagos en la base de datos. Reporta datos con fuente. NO calcula.
- analista: valida el esquema de los datos que consiguió el investigador y hace \
cálculos exactos (totales, porcentajes). NO tiene acceso a ninguna fuente: solo \
ve la solicitud, tu instrucción y los datos crudos del investigador.

Rúbrica de validación:
{rubrica}

Vos evaluás `cobertura` y `coherencia`. El criterio `respaldo` NO lo evaluás vos: \
lo calculan los chequeos automáticos y te lo informo en cada turno. Si dice que \
se cumple, confiá en él: vos no ves los datos crudos y no podés verificar una \
cifra mejor que ellos.

Cómo decidir:
- Si los tres criterios se cumplen, elegí 'finalizar'. No pidas rondas extra \
para "confirmar" o "detallar" algo que ya está bien.
- Si un criterio falla, mandá al especialista que puede arreglarlo, con una \
instrucción que diga exactamente qué corregir. No repitas una instrucción que ya \
diste.
- El analista depende del investigador: nunca lo mandes antes de que existan \
los datos que necesita. Pero en cuanto existan (lo ves en "Datos crudos que \
recibiría el analista"), cualquier total, suma o porcentaje que falte es trabajo \
del analista, no del investigador.
- Nunca le pidas cálculos al investigador (ni totales, ni sumas): no tiene \
herramientas para eso. Las cuentas son siempre del analista.
- NO escribas cifras en las instrucciones. El analista saca los números de los \
datos crudos; si le dictás un número, puede ser uno equivocado. Decile QUÉ \
calcular, no con qué valores.
- Al mandar al analista, decí exactamente qué cuantificar usando las mismas \
palabras de la solicitud.
- Si los chequeos marcan problemas en una contribución, pedile la corrección al \
agente que la hizo, citando el problema. Los avisos NO requieren corrección: \
el sistema ya descarta lo que señalan (por ejemplo, las cifras que escriba el \
investigador no se usan nunca; las cifras son siempre del analista).
- Quién manda en cada tipo de dato, para resolver contradicciones: en hechos \
(qué código, qué significa, cómo se resuelve) manda el investigador; en cifras \
(cantidades, montos, porcentajes) manda SIEMPRE el analista. Las cifras que el \
investigador escribió por su cuenta ya vienen tachadas como [cifra descartada]: \
no las uses para contradecir al analista.
- Si la solicitud pide EJECUTAR una acción sobre los pagos (por ejemplo, programar reintentos), ningún especialista la ejecuta: la resuelve una fase posterior del sistema, con aprobación humana. Para `cobertura`, esa parte cuenta como cubierta en cuanto el investigador consultó los pagos del comercio involucrado. No mandes a nadie a "ejecutar" ni a "programar" nada.
- Puede haber una conversación previa. Sirve SOLO para entender a qué se refiere \
la solicitud actual ("ese comercio", "esos pagos", "¿y el otro?"). Cuando la \
solicitud use una referencia así, resolvela y escribí en la instrucción el dato \
explícito (nombre del comercio, código de error). Lo que se respondió en turnos \
anteriores NO cuenta como contribución de este turno: para `cobertura` valen solo \
las contribuciones de este turno, porque los datos se vuelven a consultar.
"""


def _formatear_rubrica() -> str:
    lineas = [f"- {nombre}: {descripcion}" for nombre, descripcion in CRITERIOS_SEMANTICOS.items()]
    lineas.append(f"- {CRITERIO_VERIFICABLE[0]} (automático): {CRITERIO_VERIFICABLE[1]}")
    return "\n".join(lineas)


def _resumen_datos(contribucion: Contribucion) -> str:
    """Qué datos crudos le quedarían disponibles al analista, en una línea.

    Sin esto, el supervisor solo ve el texto del investigador; si ahí los
    totales aparecen tachados, concluye que "faltan datos" y vuelve a mandar
    al investigador —que no puede calcular— en vez de al analista, que sí.
    Pasó en una corrida real (README → "Lo que salió mal").
    """
    partes = []
    for evidencia in contribucion["evidencia"]:
        resultado = evidencia["resultado"]
        if evidencia["herramienta"] in HERRAMIENTAS_DE_DATOS and isinstance(resultado, dict):
            if "error" in resultado:
                continue
            argumentos = ", ".join(f"{k}={v}" for k, v in evidencia["argumentos"].items())
            partes.append(
                f"{evidencia['herramienta']}({argumentos}) → {resultado.get('cantidad', '?')} registros"
            )
    return "; ".join(partes) if partes else "ninguno (el analista no tendría con qué trabajar)"


def armar_contexto_supervisor(estado: EstadoOrquestador) -> str:
    """Lo que ve el supervisor: la solicitud, las conclusiones y los chequeos.

    No ve la evidencia cruda (los JSON de las herramientas): evaluar si el
    trabajo está completo no requiere releer cada pago, y meterlos acá
    multiplicaría el costo de cada decisión por la cantidad de datos.
    """
    solicitud = solicitud_actual(estado)
    historial = historial_conversacion(estado, max_turnos=config.HISTORIAL_MAX_TURNOS)
    contribuciones = estado.get("contribuciones", [])
    paso = estado.get("pasos_supervisor", 0) + 1

    bloques = []
    for c in contribuciones:
        herramientas = ", ".join(e["herramienta"] for e in c["evidencia"]) or "ninguna"
        problemas = "\n".join(f"  - {p}" for p in c["problemas"]) or "  (ninguno)"
        avisos = "\n".join(f"  - {a}" for a in c["avisos"]) or "  (ninguno)"
        bloque = (
            f"### {c['agente']} — ronda {c['ronda']}\n"
            f"Instrucción recibida: {c['instruccion']}\n"
            f"Herramientas usadas: {herramientas}\n"
            f"Conclusión:\n{c['contenido'] or '(vacía)'}\n"
            f"Problemas (bloqueantes) detectados por los chequeos automáticos:\n{problemas}\n"
            f"Avisos (no bloqueantes, no requieren corrección):\n{avisos}"
        )
        if c["agente"] == "investigador":
            bloque += f"\nDatos crudos que recibiría el analista: {_resumen_datos(c)}"
        bloques.append(bloque)

    respaldo = "SE CUMPLE" if evaluar_respaldo(estado) else "NO se cumple"
    presupuesto = ", ".join(
        f"{a}: {rondas_de(contribuciones, a)}/{config.MAX_RONDAS_POR_AGENTE} rondas usadas"
        for a in ESPECIALISTAS
    )
    previa = f"## Conversación previa (solo contexto)\n{historial}\n\n" if historial else ""
    return (
        f"{previa}"
        f"## Solicitud del usuario (turno actual)\n{solicitud}\n\n"
        f"## Contribuciones\n{chr(10).join(bloques) if bloques else '(todavía ninguna)'}\n\n"
        f"## Criterio automático\nrespaldo: {respaldo}\n\n"
        f"## Presupuesto\nDecisión {paso} de {config.MAX_PASOS_SUPERVISOR}. {presupuesto}."
    )


# --- Guardas deterministas --------------------------------------------------


def aplicar_guardas(
    propuesta: DecisionSupervisor, estado: EstadoOrquestador, rubrica: dict[str, bool]
) -> tuple[Destino, str, str | None]:
    """Convierte la propuesta del modelo en un destino real.

    Devuelve `(destino, instruccion, guarda)`. `guarda` es `None` si la
    propuesta pasó tal cual, o la explicación de por qué se la pisó.

    El orden importa. La suficiencia va primero: si la rúbrica completa está en
    verde, no hay corrección que justifique otra vuelta. La dependencia va antes
    que el límite de rondas porque redirigir al investigador (que en ese caso
    tiene todas sus rondas disponibles) es mejor que cerrar.
    """
    contribuciones = estado.get("contribuciones", [])
    decisiones = estado.get("decisiones", [])

    if propuesta.siguiente == "finalizar":
        return "sintetizador", "", None

    # 1. Suficiencia estricta: con la rúbrica completa, no se **refina**. Solo
    # corta rondas extra de un agente que ya intervino; si el modelo manda a
    # uno que nunca trabajó, está diciendo que falta algo, y su `cobertura=True`
    # es la parte inconsistente de la propuesta, no el destino.
    if all(rubrica.values()) and rondas_de(contribuciones, propuesta.siguiente) > 0:
        return (
            "sintetizador",
            "",
            f"La rúbrica se cumple completa: no se refina a {propuesta.siguiente}, se cierra.",
        )

    destino: Destino = propuesta.siguiente
    instruccion = propuesta.instruccion.strip()

    # 2. Dependencia: el analista sin datos no puede hacer nada útil.
    if destino == "analista" and rondas_de(contribuciones, "investigador") == 0:
        return (
            "investigador",
            f"Conseguí los datos crudos que hacen falta para esta tarea de análisis: {instruccion}",
            "El analista no tiene datos todavía: se redirige al investigador.",
        )

    # 3. Presupuesto por agente.
    if rondas_de(contribuciones, destino) >= config.MAX_RONDAS_POR_AGENTE:
        return (
            "sintetizador",
            "",
            f"{destino} ya usó sus {config.MAX_RONDAS_POR_AGENTE} rondas: se cierra con lo que hay.",
        )

    # 4. Sin progreso: misma instrucción al mismo agente que la vez anterior.
    anteriores = [d for d in decisiones if d["destino"] == destino]
    if anteriores and _normalizar(anteriores[-1]["instruccion"]) == _normalizar(instruccion):
        return (
            "sintetizador",
            "",
            f"Instrucción repetida para {destino}: no hay progreso posible, se cierra.",
        )

    return destino, instruccion, None


def _normalizar(texto: str) -> str:
    return " ".join(texto.lower().split())


def decision_de_respaldo(estado: EstadoOrquestador, error: str) -> DecisionSupervisor:
    """Qué hacer si la salida del modelo no se pudo interpretar.

    Sin contribuciones, lo razonable es empezar investigando; con alguna, es
    cerrar con lo que hay. No se reintenta la misma llamada: un modelo que no
    devolvió el esquema una vez, con el mismo prompt, suele no devolverlo
    tampoco la segunda.
    """
    sin_nada = not estado.get("contribuciones")
    return DecisionSupervisor(
        evaluacion=f"La salida del supervisor no se pudo interpretar ({error}).",
        cobertura=False,
        coherencia=False,
        siguiente="investigador" if sin_nada else "finalizar",
        instruccion="Investigá la solicitud del usuario." if sin_nada else "",
        motivo="Decisión de respaldo ante una salida inválida.",
    )


# --- Nodo -------------------------------------------------------------------


def crear_supervisor(modelo: BaseChatModel):
    """Construye el nodo supervisor.

    `include_raw=True` hace que un error de parseo vuelva como dato
    (`parsing_error`) en vez de como excepción, y así cae en
    `decision_de_respaldo` en lugar de tumbar el grafo.
    """
    estructurado = modelo.with_structured_output(DecisionSupervisor, include_raw=True)
    sistema = SystemMessage(content=PROMPT_SUPERVISOR.format(rubrica=_formatear_rubrica()))

    async def supervisor(estado: EstadoOrquestador) -> dict[str, Any]:
        paso = estado.get("pasos_supervisor", 0) + 1
        respaldo = evaluar_respaldo(estado)

        if paso > config.MAX_PASOS_SUPERVISOR:
            # Condición de parada dura: ni siquiera se consulta al modelo.
            destino: Destino = "sintetizador"
            instruccion = ""
            guarda: str | None = (
                f"Límite de {config.MAX_PASOS_SUPERVISOR} decisiones alcanzado: "
                "se cierra sin consultar al modelo."
            )
            propuesto, motivo = "finalizar", "Límite de pasos."
            rubrica = {"cobertura": False, "coherencia": False, "respaldo": respaldo}
        else:
            salida = await estructurado.ainvoke(
                [sistema, HumanMessage(content=armar_contexto_supervisor(estado))]
            )
            propuesta = salida.get("parsed")
            if propuesta is None:
                propuesta = decision_de_respaldo(estado, str(salida.get("parsing_error")))
            rubrica = {
                "cobertura": propuesta.cobertura,
                "coherencia": propuesta.coherencia,
                "respaldo": respaldo,
            }
            destino, instruccion, guarda = aplicar_guardas(propuesta, estado, rubrica)
            propuesto, motivo = propuesta.siguiente, propuesta.motivo

        decision = Decision(
            paso=paso,
            propuesta=propuesto,
            destino=destino,
            instruccion=instruccion,
            motivo=motivo,
            rubrica=rubrica,
            guarda=guarda,
        )

        # "Completa" = se cierra con la rúbrica entera en verde. Un cierre
        # forzado por límite de rondas o de pasos también va a síntesis, pero
        # con `task_completed=False`, y el sintetizador lo declara.
        completa = destino == "sintetizador" and all(rubrica.values())

        resumen = f"→ {destino}" + (f": {instruccion}" if instruccion else "")
        if guarda:
            resumen += f"  [guarda: {guarda}]"

        return {
            "next_agent": destino,
            "instruccion_actual": instruccion,
            "task_completed": completa,
            "decisiones": [decision],
            "pasos_supervisor": 1,
            "messages": [AIMessage(content=resumen, name="supervisor")],
        }

    return supervisor


def enrutar(estado: EstadoOrquestador) -> Literal["investigador", "analista", "sintetizador"]:
    """Arista condicional del supervisor.

    El retorno es un `Literal` con los nombres exactos de los nodos, como pide
    la consigna. No decide nada: la decisión ya la tomó el nodo (modelo +
    guardas) y quedó en `next_agent`. Separarlo así hace que el ruteo sea
    testeable sin compilar el grafo.
    """
    return estado["next_agent"]
