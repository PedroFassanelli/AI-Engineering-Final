"""Un LLM de mentira con guion fijo (el mismo doble de la Pre-entrega 5).

Testear el orquestador contra el modelo real es testear el modelo, no el
orquestador: la respuesta cambia entre corridas, cuesta plata y necesita una API
key. Lo que hay que verificar acá es otra cosa — que el supervisor enrute a
donde decidió, que las guardas corten los bucles, que cada especialista reciba
solo su contexto, que el estado acumule — y todo eso es determinista si los
modelos lo son.

Cada rol recibe su propio `LLMFalso`, así que el guion de uno no se mezcla con
el de otro, y `llamadas` registra exactamente lo que vio cada agente.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class LLMFalso(BaseChatModel):
    """Chat model determinista que sigue un guion de `AIMessage`."""

    guion: list[AIMessage] = Field(default_factory=list)

    #: Lo que recibió en cada llamada, en orden. Es la ventana al contexto real.
    llamadas: list[list[BaseMessage]] = Field(default_factory=list)

    #: Con `ciclar=True`, al agotarse el guion repite el último mensaje con ids
    #: nuevos para siempre: así se simula un modelo que no se detiene.
    ciclar: bool = False

    @property
    def _llm_type(self) -> str:
        return "falso"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "LLMFalso":
        """Se devuelve a sí mismo: el guion ya trae las llamadas a herramientas.

        Sirve tanto para `create_react_agent` como para
        `with_structured_output`, que por defecto se implementa atando el
        esquema como herramienta y parseando el `tool_call` resultante.
        """
        return self

    def _siguiente(self, mensajes: list[BaseMessage]) -> AIMessage:
        self.llamadas.append(list(mensajes))
        indice = len(self.llamadas) - 1
        if indice < len(self.guion):
            return self.guion[indice]
        if self.ciclar and self.guion:
            plantilla = self.guion[-1]
            return AIMessage(
                content=plantilla.content,
                tool_calls=[
                    {**llamada, "id": f"{llamada['id']}-v{indice}"}
                    for llamada in plantilla.tool_calls
                ],
            )
        return AIMessage(content="(el guion se agotó)")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._siguiente(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._siguiente(messages))])

    def texto_de_llamada(self, indice: int) -> str:
        """Todo el texto que recibió el modelo en la llamada `indice`."""
        return "\n".join(str(m.content) for m in self.llamadas[indice])


def pedido(nombre: str, argumentos: dict[str, Any], identificador: str) -> AIMessage:
    """Un `AIMessage` que pide una herramienta."""
    return AIMessage(
        content="",
        tool_calls=[{"name": nombre, "args": argumentos, "id": identificador}],
    )


def decision(
    siguiente: str,
    instruccion: str = "",
    *,
    cobertura: bool = False,
    coherencia: bool = True,
    identificador: str = "d",
) -> AIMessage:
    """Un `AIMessage` con la salida estructurada del supervisor."""
    return pedido(
        "DecisionSupervisor",
        {
            "evaluacion": "evaluación de prueba",
            "cobertura": cobertura,
            "coherencia": coherencia,
            "siguiente": siguiente,
            "instruccion": instruccion,
            "motivo": "motivo de prueba",
        },
        identificador,
    )


def propuesta(accion: str, pago_ids: list[str] | None = None, identificador: str = "p") -> AIMessage:
    """Un `AIMessage` con la salida estructurada del planificador."""
    return pedido(
        "PropuestaAccion",
        {"accion": accion, "pago_ids": pago_ids or [], "motivo": "motivo de prueba"},
        identificador,
    )


def sin_accion() -> AIMessage:
    return propuesta("ninguna")
