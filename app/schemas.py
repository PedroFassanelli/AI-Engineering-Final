"""Modelos de entrada y salida de la API.

Todo lo que entra se valida acá antes de tocar Redis (un `pregunta` vacío o un
`conversacion_id` con forma inválida es un 422, no un job que falla en el
worker), y todo lo que sale se serializa a través de estos modelos
(`response_model`): la documentación de `/docs` es el contrato real.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.jobs import EstadoJob

#: Los ids que genera el sistema: uuid4 en hexadecimal.
PATRON_ID = r"^[0-9a-f]{32}$"


class CrearTarea(BaseModel):
    pregunta: str = Field(
        min_length=5,
        max_length=2000,
        description="La consulta de soporte, en lenguaje natural.",
        examples=[
            "El comercio Bazar Pampa viene teniendo pagos fallidos. ¿Cuál es el código de "
            "error que más se repite, qué significa y cómo se resuelve?"
        ],
    )
    conversacion_id: str | None = Field(
        default=None,
        pattern=PATRON_ID,
        description=(
            "Para continuar una conversación, el `conversacion_id` que devolvió el primer turno. "
            "Vacío = conversación nueva."
        ),
    )


class TareaEncolada(BaseModel):
    job_id: str
    conversacion_id: str
    turno: int = Field(ge=1)
    estado: EstadoJob
    url_estado: str = Field(description="Dónde consultar el estado del turno (polling).")
    url_conversacion: str = Field(description="Dónde ver la conversación completa.")


class DecisionAprobacion(BaseModel):
    aprobado: bool = Field(description="`true` ejecuta la acción; `false` la descarta.")
    revisor: str = Field(min_length=1, max_length=120, description="Quién decide. Queda registrado.")
    comentario: str | None = Field(default=None, max_length=1000)


class SolicitudAprobacion(BaseModel):
    """Lo que tiene que decidir una persona (lo arma `app/hitl.py`)."""

    tipo: Literal["programar_reintentos"]
    resumen: str
    comercio_id: int
    comercio: str
    pago_ids: list[str]
    monto_total_centavos: int = Field(ge=0)
    moneda: str
    codigos_error: list[str]
    motivo: str
    criticidad: list[str] = Field(description="Por qué la acción es crítica.")
    descartados: list[str] = Field(description="Pagos que propuso el modelo y el código descartó.")
    como_responder: str


class AprobacionRegistrada(BaseModel):
    aprobado: bool
    revisor: str
    comentario: str | None = None
    decidido_en: str | None = None


class ResultadoAccion(BaseModel):
    ejecutada: bool
    detalle: str
    pago_ids: list[str]


class ErrorJob(BaseModel):
    tipo: str
    mensaje: str


class Job(BaseModel):
    job_id: str
    conversacion_id: str | None = None
    turno: int | None = None
    estado: EstadoJob
    pregunta: str
    creado_en: str
    actualizado_en: str | None = None
    iniciado_en: str | None = None
    reanudado_en: str | None = None
    terminado_en: str | None = None
    #: Último nodo del grafo que terminó, y el recorrido completo hasta ahora.
    nodo_actual: str | None = None
    nodos: list[str] = Field(default_factory=list)
    #: Solo en AWAITING_APPROVAL: lo que la persona tiene que aprobar.
    aprobacion_pendiente: SolicitudAprobacion | None = None
    aprobacion: AprobacionRegistrada | None = None
    respuesta: str | None = None
    task_completed: bool | None = None
    resultado_accion: ResultadoAccion | None = None
    error: ErrorJob | None = None
    duracion_segundos: float | None = None
    duracion_reanudacion_segundos: float | None = None

    @classmethod
    def desde_almacen(cls, datos: dict[str, Any]) -> "Job":
        return cls.model_validate({k: v for k, v in datos.items() if k in cls.model_fields})


class Mensaje(BaseModel):
    """Un mensaje de la conversación, leído del checkpoint de LangGraph."""

    rol: Literal["usuario", "asistente"]
    #: `usuario`, `sintetizador`, `sistema` (resultado de una acción) o, con
    #: `incluir_trabajo=true`, `supervisor` / `investigador` / `analista`.
    autor: str
    contenido: str


class Conversacion(BaseModel):
    conversacion_id: str
    creada_en: str | None = None
    actualizada_en: str | None = None
    turnos: int = Field(ge=0)
    #: El turno que está en curso (o pausado esperando aprobación), si hay uno.
    job_activo: str | None = None
    #: Del checkpoint: es lo que el grafo "recuerda" de la conversación.
    mensajes: list[Mensaje]
    #: Los jobs de cada turno que siguen en Redis (un job vive menos que su conversación).
    jobs: list[Job]


class Salud(BaseModel):
    api: Literal["ok"]
    redis: Literal["ok"]
    checkpointer: Literal["ok", "no configurado"]
    #: Lo informa el worker al arrancar: `hibrido`, `lexico`, o `sin worker` si no arrancó.
    rag: str


class ErrorAPI(BaseModel):
    detail: str
