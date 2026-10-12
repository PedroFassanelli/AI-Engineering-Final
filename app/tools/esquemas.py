"""Contratos de **salida** de las herramientas de los agentes.

Hasta la Pre-entrega 7 los argumentos ya se validaban (`args_schema`), pero las
herramientas devolvían `dict[str, Any]` armados a mano: nada impedía que una
devolviera `cantidad` como string, o un `error` sin `sugerencia`, y el agente
recibía lo que saliera. Ahora cada herramienta construye su resultado con uno de
estos modelos, así que la forma de lo que ve el modelo está validada en el
mismo lugar que la de lo que recibe.

Convención heredada de la Pre-entrega 5, ahora forzada por el tipo: una
herramienta que falla no lanza una excepción; devuelve `error` + `sugerencia`
(la próxima acción concreta) y el agente se corrige dentro de su ciclo ReAct.
`SalidaHerramienta` rechaza un `error` sin `sugerencia`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SalidaHerramienta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str | None = None
    sugerencia: str | None = None

    @model_validator(mode="after")
    def _error_con_sugerencia(self) -> "SalidaHerramienta":
        if self.error and not self.sugerencia:
            raise ValueError("un resultado con `error` tiene que traer `sugerencia`")
        return self

    def como_dict(self) -> dict[str, Any]:
        """Lo que devuelve la herramienta: JSON plano, sin los campos vacíos de primer nivel.

        Solo se omiten los `None` del primer nivel (`error`, `sugerencia`, ...):
        adentro de los registros un `codigo_error: null` es un dato.
        """
        return {k: v for k, v in self.model_dump(mode="json").items() if v is not None}


# --- Investigación -------------------------------------------------------------


class Comercio(BaseModel):
    comercio_id: int = Field(gt=0)
    nombre: str
    plan: str
    pais: str = Field(pattern=r"^[A-Z]{2}$")
    alta: str


class ResultadoBuscarComercio(SalidaHerramienta):
    nombre: str
    comercios: list[Comercio] | None = None


EstadoPago = Literal["aprobado", "fallido", "pendiente", "reembolsado"]


class RegistroPago(BaseModel):
    """Un pago tal como sale de la base (sin agregar ni recortar)."""

    pago_id: str = Field(pattern=r"^pay_\d+$")
    comercio_id: int = Field(gt=0)
    monto_centavos: int
    moneda: str = Field(pattern=r"^[A-Z]{3}$")
    estado: EstadoPago
    codigo_error: str | None
    metodo: str
    creado_en: str


class ResultadoConsultarPagos(SalidaHerramienta):
    comercio_id: int
    cantidad: int | None = Field(default=None, ge=0)
    pagos: list[RegistroPago] | None = None

    @model_validator(mode="after")
    def _cantidad_coherente(self) -> "ResultadoConsultarPagos":
        if self.pagos is not None and self.cantidad != len(self.pagos):
            raise ValueError("`cantidad` tiene que ser el largo de `pagos`")
        return self


# --- Análisis ------------------------------------------------------------------


class Agregado(BaseModel):
    cantidad: int = Field(ge=0)
    monto_centavos: int = Field(ge=0)


class PerfilMoneda(BaseModel):
    total: Agregado
    por_estado: dict[str, Agregado]
    por_codigo_error: dict[str, Agregado]


class RegistroInvalido(BaseModel):
    indice: int = Field(ge=0)
    pago_id: Any = None
    errores: list[str]


class ResultadoPerfilarPagos(SalidaHerramienta):
    registros_recibidos: int | None = Field(default=None, ge=0)
    registros_validos: int | None = Field(default=None, ge=0)
    invalidos: list[RegistroInvalido] = Field(default_factory=list)
    por_moneda: dict[str, PerfilMoneda] | None = None


class ResultadoCalcular(SalidaHerramienta):
    expresion: str
    resultado: float | None = None


class ResultadoPorcentaje(SalidaHerramienta):
    parte: float | None = None
    total: float | None = None
    porcentaje: float | None = None
