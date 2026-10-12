"""El grafo del orquestador: supervisor + especialistas + síntesis + HITL.

                     ┌──────────────► investigador ──┐
                     │                               │
    START ──► supervisor ◄───────────────────────────┤
                     │                               │
                     ├──────────────► analista ──────┘
                     │
                     └──────────────► sintetizador ──► planificador ──┬──► END
                                                                      │
                                         aprobacion_humana ◄──────────┘
                                         (interrupt)  │
                                                      ├──► ejecutar_accion ──► END
                                                      └──► END (rechazada)

La parte de arriba es el orquestador jerárquico de la Pre-entrega 6: todo pasa
por el supervisor y la única salida de esa fase es el sintetizador. Después
viene lo de la Pre-entrega 7:

- `planificador`: decide si la solicitud, además de una consulta, pide una
  acción con efectos (programar reintentos de pagos).
- `aprobacion_humana`: el punto de interrupción. Ver `app/hitl.py`.
- `ejecutar_accion`: la única escritura del sistema.

El grafo se compila con un **checkpointer**. En producción es un
`AsyncRedisSaver` (`abrir_checkpointer`): cada super-paso queda guardado en
Redis bajo `thread_id = conversacion_id`. Eso permite dos cosas:

- **Conversaciones que sobreviven a un reinicio.** El turno 2 de una
  conversación corre sobre el mismo hilo y encuentra los mensajes del turno 1,
  aunque entre medio se hayan reiniciado la API, el worker y el propio Redis
  (que persiste en disco con AOF).
- **Pausar en el `interrupt`**, apagar el proceso y reanudar en otro.

El `job_id` (un turno) viaja aparte, en `configurable["job_id"]`: es la clave
de idempotencia de la acción con efectos, que es por turno y no por conversación.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite

from app import config
from app.agents import (
    crear_analista,
    crear_investigador,
    crear_sintetizador,
    crear_supervisor,
    enrutar,
)
from app.agents.planner import crear_planificador
from app.hitl import aprobacion_humana, ejecutar_accion, ruta_tras_aprobacion, ruta_tras_planificador
from app.state import EstadoOrquestador


@dataclass
class Modelos:
    """Un chat model por rol.

    Separarlos permite usar un modelo más capaz en el supervisor (el único que
    decide flujo) y uno barato en los demás, y en los tests darle a cada rol
    su propio guion.
    """

    supervisor: BaseChatModel
    investigador: BaseChatModel
    analista: BaseChatModel
    sintetizador: BaseChatModel
    planificador: BaseChatModel

    @classmethod
    def desde_config(cls) -> "Modelos":
        especialistas = config.construir_llm()
        supervisor = (
            especialistas
            if config.MODELO_SUPERVISOR == config.MODELO
            else config.construir_llm(config.MODELO_SUPERVISOR)
        )
        return cls(
            supervisor=supervisor,
            investigador=especialistas,
            analista=especialistas,
            sintetizador=especialistas,
            planificador=especialistas,
        )


def construir_grafo(
    modelos: Modelos | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compila el grafo.

    Sin `checkpointer` el grafo funciona igual para consultas, pero no puede
    pausar: `interrupt()` necesita un lugar donde guardar el estado.
    """
    modelos = modelos or Modelos.desde_config()

    grafo = StateGraph(EstadoOrquestador)

    grafo.add_node("supervisor", crear_supervisor(modelos.supervisor))
    grafo.add_node("investigador", crear_investigador(modelos.investigador))
    grafo.add_node("analista", crear_analista(modelos.analista))
    grafo.add_node("sintetizador", crear_sintetizador(modelos.sintetizador))
    grafo.add_node("planificador", crear_planificador(modelos.planificador))
    grafo.add_node("aprobacion_humana", aprobacion_humana)
    grafo.add_node("ejecutar_accion", ejecutar_accion)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        enrutar,
        # `path_map` explícito: además de documentar los destinos posibles, es
        # lo que permite dibujar el diagrama sin ejecutar el grafo.
        {
            "investigador": "investigador",
            "analista": "analista",
            "sintetizador": "sintetizador",
        },
    )
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")

    grafo.add_edge("sintetizador", "planificador")
    grafo.add_conditional_edges(
        "planificador",
        ruta_tras_planificador,
        {"aprobacion_humana": "aprobacion_humana", END: END},
    )
    grafo.add_conditional_edges(
        "aprobacion_humana",
        ruta_tras_aprobacion,
        {"ejecutar_accion": "ejecutar_accion", END: END},
    )
    grafo.add_edge("ejecutar_accion", END)

    return grafo.compile(checkpointer=checkpointer)


@asynccontextmanager
async def abrir_checkpointer(redis_url: str | None = None) -> AsyncIterator[BaseCheckpointSaver]:
    """`AsyncRedisSaver` listo para usar (índices creados) y cerrado al salir.

    Necesita Redis 8 (o Redis Stack): el saver usa los módulos de búsqueda y
    JSON. Los checkpoints expiran con el mismo TTL que los jobs, y el TTL se
    renueva cada vez que se leen (una aprobación tardía no encuentra el
    checkpoint vencido mientras alguien siga consultando el job).
    """
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(
        redis_url or config.REDIS_URL,
        ttl={"default_ttl": config.TTL_CONVERSACION_SEGUNDOS / 60, "refresh_on_read": True},
    ) as saver:
        yield saver


def entrada_de_turno(pregunta: str, turno: int = 1) -> dict[str, Any]:
    """La entrada con la que arranca cada turno de una conversación.

    `messages` **acumula**: la pregunta nueva se agrega a la conversación que
    ya está en el checkpoint. Todo lo demás es de un turno y se reinicia. Los
    campos con reducer (`contribuciones`, `decisiones`, `pasos_supervisor`) se
    pisan con `Overwrite`: con el reducer `operator.add`, un `[]` se sumaría a
    lo del turno anterior en vez de reemplazarlo, y el supervisor del turno 2
    arrancaría con la rúbrica y el presupuesto del turno 1.

    En el primer turno el `Overwrite` no cambia nada (no hay nada que pisar),
    así que la misma entrada sirve para los dos casos.
    """
    return {
        "messages": [HumanMessage(content=pregunta)],
        "turno": turno,
        "contribuciones": Overwrite([]),
        "decisiones": Overwrite([]),
        "pasos_supervisor": Overwrite(0),
        "next_agent": "investigador",
        "instruccion_actual": "",
        "task_completed": False,
        "respuesta_final": "",
        "accion_propuesta": None,
        "aprobacion": None,
        "resultado_accion": None,
    }


def config_ejecucion(thread_id: str | None = None, job_id: str | None = None) -> dict[str, Any]:
    """Config de una corrida.

    `thread_id` (= `conversacion_id`) es la clave del checkpoint; `job_id` (=
    el turno) es la clave de idempotencia de `ejecutar_accion`.
    """
    ejecucion: dict[str, Any] = {"recursion_limit": config.limite_recursion_grafo()}
    if thread_id is not None:
        ejecucion["configurable"] = {"thread_id": thread_id, "job_id": job_id or thread_id}
    return ejecucion


def diagrama_mermaid(grafo: CompiledStateGraph | None = None) -> str:
    """El Mermaid del grafo, generado por LangGraph (no escrito a mano)."""
    return (grafo or construir_grafo_para_dibujar()).get_graph().draw_mermaid()


def construir_grafo_para_dibujar() -> CompiledStateGraph:
    """Compila el grafo con modelos de relleno, solo para obtener su forma.

    Dibujar no ejecuta nodos, así que no hace falta una API key; pero
    `create_react_agent` exige un chat model al construirse.
    """
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class _Relleno(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001 - firma de LangChain
            return self

    relleno = _Relleno(responses=[])
    return construir_grafo(Modelos(relleno, relleno, relleno, relleno, relleno))
