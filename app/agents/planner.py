"""El planificador: detecta si la solicitud pide **ejecutar** algo.

Corre después del sintetizador, cuando la parte de consulta ya está resuelta.
Le pregunta al modelo una sola cosa: ¿el usuario pidió una acción con efectos
sobre los pagos? Hoy hay una sola acción posible, `programar_reintentos`.

El modelo solo **propone**. Qué pagos se tocan y por cuánto lo fija el código
(`app/hitl.py → validar_propuesta`): solo sobreviven pagos que el investigador
efectivamente consultó y que están fallidos. Un `pago_id` inventado o uno
aprobado se descarta antes de llegar a la persona que aprueba.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app import config
from app.hitl import pagos_consultados, validar_propuesta
from app.state import EstadoOrquestador, historial_conversacion, solicitud_actual


class PropuestaAccion(BaseModel):
    """Salida estructurada del planificador."""

    accion: Literal["ninguna", "programar_reintentos"] = Field(
        description=(
            "'programar_reintentos' SOLO si el usuario pide explícitamente reintentar, "
            "reprocesar o volver a cobrar pagos. Si la solicitud es solo de consulta "
            "(qué pasó, por qué, cuánto), 'ninguna'."
        )
    )
    pago_ids: list[str] = Field(
        default_factory=list,
        description="Los pago_id exactos de la tabla que el usuario pidió reintentar. Vacío si 'ninguna'.",
    )
    motivo: str = Field(description="Una oración: qué pidió el usuario y por qué esos pagos.")


PROMPT_PLANIFICADOR: Final[str] = """\
Sos el planificador de acciones de un equipo de soporte de Nimbus Pagos. La \
consulta del usuario ya fue respondida. Tu único trabajo es decidir si, además, \
el usuario pidió EJECUTAR una acción sobre sus pagos.

La única acción disponible es `programar_reintentos`: volver a intentar el \
cobro de pagos fallidos. Tiene efectos reales (genera nuevos cobros), así que:
- Elegila solo si el pedido es explícito ("reintentá", "volvé a cobrar", \
"reprocesá", "programá el reintento").
- Preguntar por qué falló un pago, cuánto representa o cómo se resuelve NO es \
pedir un reintento.
- Usá solo pago_id de la tabla, y solo los que el usuario pidió (si acota por \
código de error, respetalo).
- Si la solicitud actual se refiere a la conversación previa ("reintentá esos \
pagos"), usá la conversación para saber cuáles. Un pedido de reintento de un \
turno ANTERIOR que ya se resolvió no cuenta: decidís solo sobre la solicitud actual.
"""


def armar_entrada_planificador(estado: EstadoOrquestador) -> str:
    solicitud = solicitud_actual(estado)
    historial = historial_conversacion(estado, max_turnos=config.HISTORIAL_MAX_TURNOS)
    filas = []
    for comercio in pagos_consultados(estado).values():
        for pago in comercio["pagos"]:
            if pago.get("estado") == "fallido":
                filas.append(
                    f"| {pago['pago_id']} | {comercio['nombre']} | {pago.get('codigo_error') or '-'} "
                    f"| {pago['monto_centavos']} {pago.get('moneda', '')} |"
                )
    tabla = "\n".join(["| pago_id | comercio | código | monto (centavos) |", "|---|---|---|---|", *filas])
    previa = f"## Conversación previa (solo contexto)\n{historial}\n\n" if historial else ""
    return f"{previa}## Solicitud del usuario (turno actual)\n{solicitud}\n\n## Pagos fallidos consultados\n{tabla}"


def crear_planificador(modelo: BaseChatModel):
    estructurado = modelo.with_structured_output(PropuestaAccion, include_raw=True)
    sistema = SystemMessage(content=PROMPT_PLANIFICADOR)

    async def planificador(estado: EstadoOrquestador) -> dict[str, Any]:
        consultados = pagos_consultados(estado)
        hay_fallidos = any(
            p.get("estado") == "fallido" for c in consultados.values() for p in c["pagos"]
        )
        if not hay_fallidos:
            # Sin pagos fallidos a la vista no hay nada que reintentar: ni
            # siquiera se consulta al modelo (una llamada menos por consulta).
            return {"accion_propuesta": None}

        salida = await estructurado.ainvoke(
            [sistema, HumanMessage(content=armar_entrada_planificador(estado))]
        )
        propuesta: PropuestaAccion | None = salida.get("parsed")
        if propuesta is None or propuesta.accion == "ninguna":
            # Una salida que no se pudo interpretar NUNCA se convierte en una
            # acción: ante la duda, el sistema no toca nada.
            return {"accion_propuesta": None}

        return {
            "accion_propuesta": validar_propuesta(
                propuesta.accion, propuesta.pago_ids, propuesta.motivo, estado
            )
        }

    return planificador
