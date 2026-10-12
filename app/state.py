"""Esquema del estado compartido del orquestador.

El estado es la única vía de comunicación entre nodos: ningún agente le habla a
otro directamente. Por eso su forma decide dos cosas que la consigna pide
explícitamente:

1. **Trazabilidad**: cada aporte queda registrado como una `Contribucion` con
   el nombre del agente, la ronda, la instrucción que recibió y la evidencia
   cruda que produjo. Siempre se puede responder "¿quién dijo esto y a partir
   de qué dato?".
2. **Sin pérdida de contexto**: los campos que acumulan (`contribuciones`,
   `decisiones`, `pasos_supervisor`) usan un reducer. Un nodo devuelve solo su
   aporte y LangGraph lo concatena; ningún nodo puede pisar por accidente lo
   que escribió otro.

Los campos que se **reemplazan** (`next_agent`, `instruccion_actual`,
`task_completed`, `respuesta_final`) son, a propósito, los que tienen un único
escritor: el supervisor los tres primeros, el sintetizador el último.

Conversaciones de varios turnos
-------------------------------
El `thread_id` del checkpointer es la **conversación**, no la consulta: cada
turno nuevo se corre sobre el mismo hilo y encuentra el estado del anterior.
Eso parte el estado en dos:

- lo que **persiste** entre turnos: `messages` (preguntas, informes y
  respuestas, con su reducer `add_messages`). De ahí sale el historial que
  permite entender "¿y el otro comercio?" o "programá el reintento de esos".
- lo que es **de un turno**: contribuciones, decisiones, contador de pasos,
  acción propuesta. Si se arrastraran, el supervisor del turno 2 vería la
  rúbrica del turno 1 en verde y cerraría sin trabajar, y el límite de pasos
  ya estaría gastado. Al entrar cada turno, `app.graph.entrada_de_turno` los
  reinicia con `Overwrite`, que pisa el valor saltándose el reducer.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import MessagesState

# --- Nombres de nodos -------------------------------------------------------
#
# Son `Literal` y no `str` para que un typo ("investigadr") sea un error del
# chequeo de tipos y no un KeyError en tiempo de ejecución, en el medio de una
# corrida que ya gastó tokens.

Especialista = Literal["investigador", "analista"]

#: A dónde puede mandar el supervisor. `sintetizador` es la fase final: una vez
#: ahí, el grafo no vuelve atrás.
Destino = Literal["investigador", "analista", "sintetizador"]

ESPECIALISTAS: tuple[Especialista, ...] = ("investigador", "analista")


class Evidencia(TypedDict):
    """Una llamada a herramienta hecha por un especialista, con su resultado."""

    herramienta: str
    argumentos: dict[str, Any]
    resultado: Any


class Contribucion(TypedDict):
    """Lo que aportó un especialista en una intervención.

    `contenido` es el texto que escribió el agente; `evidencia` es lo que
    devolvieron sus herramientas. Se guardan los dos porque sirven a lectores
    distintos: el supervisor evalúa el `contenido`, el analista recibe la
    `evidencia` del investigador, y los chequeos automáticos comparan uno
    contra otra.
    """

    agente: Especialista
    #: 1 para la primera intervención de ese agente, 2 para la corrección, etc.
    ronda: int
    instruccion: str
    contenido: str
    evidencia: list[Evidencia]
    #: Hallazgos **bloqueantes** de los chequeos deterministas (ver
    #: `agents/base.py`): mientras la última ronda de un agente tenga alguno,
    #: el criterio `respaldo` de la rúbrica no se cumple.
    problemas: list[str]
    #: Hallazgos **no bloqueantes**: algo que el agente hizo fuera de su rol y
    #: que el resto del sistema descarta (p. ej. cifras escritas por el
    #: investigador, que no se usan porque los números son del analista).
    avisos: list[str]


class Decision(TypedDict):
    """Una decisión del supervisor, tal como quedó después de las guardas."""

    paso: int
    #: Lo que propuso el modelo (`finalizar` incluido).
    propuesta: str
    #: A dónde se fue efectivamente. Difiere de `propuesta` cuando actuó una guarda.
    destino: Destino
    instruccion: str
    motivo: str
    #: Resultado de la rúbrica: criterio -> cumple.
    rubrica: dict[str, bool]
    #: Si una guarda determinista pisó la propuesta del modelo, cuál y por qué.
    guarda: str | None


class AccionPropuesta(TypedDict):
    """Una acción con efecto secundario, ya validada contra la evidencia.

    La propone el modelo (`planificador`), pero los `pago_ids` y el monto los
    fija el código: solo sobreviven pagos que el investigador efectivamente
    consultó y que figuran como fallidos.
    """

    tipo: Literal["programar_reintentos"]
    comercio_id: int
    comercio: str
    pago_ids: list[str]
    monto_total_centavos: int
    moneda: str
    codigos_error: list[str]
    motivo: str
    #: Por qué el sistema la clasificó como crítica (ver `app/hitl.py`).
    criticidad: list[str]
    #: Pagos que propuso el modelo y el código descartó, con el motivo.
    descartados: list[str]


class Aprobacion(TypedDict):
    """La respuesta humana que reanuda el grafo."""

    aprobado: bool
    revisor: str
    comentario: str | None
    decidido_en: str


class ResultadoAccion(TypedDict):
    ejecutada: bool
    detalle: str
    pago_ids: list[str]


class EstadoOrquestador(MessagesState):
    """Estado del grafo.

    Hereda de `MessagesState`, que aporta `messages` con el reducer
    `add_messages`. Ahí queda la conversación **legible**: la pregunta, un
    mensaje por especialista (con `name=` del agente) y la respuesta final.
    Lo que los agentes necesitan para trabajar no viaja por `messages` sino por
    `contribuciones`: ver "Contaminación de contexto" en el README.
    """

    #: Número de turno dentro de la conversación (1, 2, ...).
    turno: int

    # --- Control de flujo (escribe solo el supervisor) ----------------------

    #: Próximo nodo. Lo lee la arista condicional del supervisor.
    next_agent: Destino
    #: Instrucción concreta para el próximo especialista.
    instruccion_actual: str

    # --- Registro (se acumula) ----------------------------------------------

    contribuciones: Annotated[list[Contribucion], operator.add]
    decisiones: Annotated[list[Decision], operator.add]
    #: Contador de pasos del supervisor. Es la condición de parada dura.
    pasos_supervisor: Annotated[int, operator.add]

    # --- Cierre ---------------------------------------------------------------

    #: Lo escribe el supervisor: `True` solo si cerró por decisión propia con la
    #: rúbrica completa, `False` si el cierre lo forzó una guarda.
    task_completed: bool
    #: Lo escribe el sintetizador (y lo completa `ejecutar_accion`).
    respuesta_final: str

    # --- Acciones con efectos secundarios (Pre-entrega 7) -------------------

    #: `None` si la solicitud era solo de consulta.
    accion_propuesta: AccionPropuesta | None
    #: Lo escribe el nodo `aprobacion_humana` al reanudarse.
    aprobacion: Aprobacion | None
    resultado_accion: ResultadoAccion | None


#: Quiénes escriben la respuesta de un turno en `messages`: el sintetizador, y
#: el sistema cuando agrega el resultado de una acción aprobada o rechazada.
AUTORES_DE_RESPUESTA: frozenset[str] = frozenset({"sintetizador", "sistema"})


def solicitud_actual(estado: EstadoOrquestador) -> str:
    """La pregunta del turno en curso: el **último** mensaje humano.

    Hasta la Pre-entrega 7 cada hilo tenía una sola pregunta y se tomaba el
    primero. Con conversaciones, el primero es la pregunta del turno 1.
    """
    for mensaje in reversed(estado.get("messages", [])):
        if mensaje.type == "human":
            return str(mensaje.content)
    return ""


def turnos_previos(mensajes: Sequence[BaseMessage]) -> list[dict[str, str]]:
    """Los turnos **anteriores** al actual, como `{pregunta, respuesta}`.

    Recorre `messages`: cada mensaje humano abre un turno; su respuesta es lo
    que escribieron el sintetizador y el sistema antes del humano siguiente.
    Los informes de los especialistas y los pasos del supervisor no cuentan:
    son el trabajo del turno, no lo que se le dijo a la persona.
    """
    turnos: list[dict[str, Any]] = []
    for mensaje in mensajes:
        if mensaje.type == "human":
            turnos.append({"pregunta": str(mensaje.content), "respuesta": []})
        elif (
            turnos
            and isinstance(mensaje, AIMessage)
            and getattr(mensaje, "name", None) in AUTORES_DE_RESPUESTA
        ):
            turnos[-1]["respuesta"].append(str(mensaje.content))
    previos = turnos[:-1]  # el último es el turno en curso
    return [{"pregunta": t["pregunta"], "respuesta": "\n\n".join(t["respuesta"])} for t in previos]


def historial_conversacion(
    estado: EstadoOrquestador,
    *,
    max_turnos: int,
    con_respuestas: bool = True,
    max_caracteres_respuesta: int = 1500,
) -> str:
    """El historial de la conversación, listo para meter en un prompt.

    Vacío si es el primer turno. `con_respuestas=False` deja solo las
    preguntas: es lo que ven el analista y el sintetizador, que tienen que
    trabajar con datos de **este** turno y no copiar cifras de una respuesta
    anterior.
    """
    previos = turnos_previos(estado.get("messages", []))[-max_turnos:] if max_turnos > 0 else []
    if not previos:
        return ""
    bloques = []
    for i, turno in enumerate(previos, start=1):
        bloque = f"Turno anterior {i} — el usuario preguntó:\n{turno['pregunta']}"
        if con_respuestas and turno["respuesta"]:
            respuesta = turno["respuesta"]
            if len(respuesta) > max_caracteres_respuesta:
                respuesta = respuesta[:max_caracteres_respuesta].rstrip() + " […]"
            bloque += f"\nY se le respondió:\n{respuesta}"
        bloques.append(bloque)
    return "\n\n".join(bloques)


def ultima_contribucion(
    contribuciones: list[Contribucion], agente: Especialista
) -> Contribucion | None:
    """La intervención más reciente de `agente`, o `None` si nunca intervino.

    Es la regla con la que se resuelven los conflictos **dentro** de un mismo
    agente: una ronda de corrección reemplaza a la anterior, no se promedia con
    ella.
    """
    for contribucion in reversed(contribuciones):
        if contribucion["agente"] == agente:
            return contribucion
    return None


def rondas_de(contribuciones: list[Contribucion], agente: Especialista) -> int:
    """Cuántas veces intervino `agente`."""
    return sum(1 for c in contribuciones if c["agente"] == agente)
