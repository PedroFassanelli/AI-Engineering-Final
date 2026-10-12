"""Herramientas del agente de análisis.

Ninguna toca una fuente de datos: el analista trabaja **solo** con lo que le
pasa el supervisor desde las contribuciones del investigador. Si le falta un
dato, no puede ir a buscarlo; tiene que decirlo, y el supervisor decide si
vuelve a mandar al investigador. Esa dependencia es la que obliga al flujo
investigador → analista que pide la consigna.

Las tres herramientas cubren los tres tipos de análisis que menciona la
consigna para este rol: validación de esquemas (`perfilar_pagos`) y cálculos
matemáticos (`calcular`, `porcentaje`).
"""

from __future__ import annotations

import ast
import operator
from collections import defaultdict
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.tools.esquemas import (
    Agregado,
    PerfilMoneda,
    RegistroInvalido,
    ResultadoCalcular,
    ResultadoPerfilarPagos,
    ResultadoPorcentaje,
)

# --- Validación de esquema --------------------------------------------------


class Pago(BaseModel):
    """Contrato de un pago. Es lo que el analista le exige a los datos crudos."""

    pago_id: str = Field(pattern=r"^pay_\d+$")
    monto_centavos: int = Field(gt=0)
    moneda: str = Field(pattern=r"^[A-Z]{3}$")
    estado: Literal["aprobado", "fallido", "pendiente", "reembolsado"]
    codigo_error: str | None = Field(default=None, pattern=r"^E\d{4}$")

    @model_validator(mode="after")
    def _codigo_segun_estado(self) -> "Pago":
        # Un fallido sin código no se puede agrupar por causa; un aprobado con
        # código es un dato corrupto. Los dos casos distorsionan los totales.
        if self.estado == "fallido" and self.codigo_error is None:
            raise ValueError("un pago fallido tiene que traer codigo_error")
        if self.estado != "fallido" and self.codigo_error is not None:
            raise ValueError(f"un pago '{self.estado}' no puede traer codigo_error")
        return self


class ArgsPerfilarPagos(BaseModel):
    pagos: list[dict[str, Any]] = Field(
        description=(
            "La lista de pagos, copiada tal cual del campo `pagos` que devolvió "
            "consultar_pagos. Cada pago necesita pago_id, monto_centavos, moneda, "
            "estado y codigo_error."
        )
    )


@tool(args_schema=ArgsPerfilarPagos)
async def perfilar_pagos(pagos: list[dict[str, Any]]) -> dict[str, Any]:
    """Valida el esquema de una lista de pagos y devuelve sus agregados.

    Descarta (e informa) los registros que no cumplen el contrato, y sobre los
    válidos devuelve cantidad y monto en centavos: totales, por estado y por
    código de error. Los montos se agregan **por moneda**: sumar ARS con CLP
    no tiene sentido.
    """
    validos: list[Pago] = []
    invalidos: list[dict[str, Any]] = []
    for indice, crudo in enumerate(pagos):
        try:
            validos.append(Pago.model_validate(crudo))
        except ValidationError as exc:
            invalidos.append(
                {
                    "indice": indice,
                    "pago_id": crudo.get("pago_id"),
                    "errores": [e["msg"] for e in exc.errors()],
                }
            )

    if not validos:
        return ResultadoPerfilarPagos(
            error="Ningún registro cumple el esquema de pago.",
            invalidos=[RegistroInvalido.model_validate(i) for i in invalidos],
            sugerencia=(
                "Pasá la lista `pagos` exactamente como la devolvió consultar_pagos, "
                "sin resumir ni renombrar campos."
            ),
        ).como_dict()

    por_moneda: dict[str, PerfilMoneda] = {}
    for moneda in sorted({p.moneda for p in validos}):
        del_grupo = [p for p in validos if p.moneda == moneda]
        por_estado: dict[str, dict[str, int]] = defaultdict(lambda: {"cantidad": 0, "monto_centavos": 0})
        por_codigo: dict[str, dict[str, int]] = defaultdict(lambda: {"cantidad": 0, "monto_centavos": 0})
        for pago in del_grupo:
            por_estado[pago.estado]["cantidad"] += 1
            por_estado[pago.estado]["monto_centavos"] += pago.monto_centavos
            if pago.codigo_error:
                por_codigo[pago.codigo_error]["cantidad"] += 1
                por_codigo[pago.codigo_error]["monto_centavos"] += pago.monto_centavos
        por_moneda[moneda] = PerfilMoneda(
            total=Agregado(
                cantidad=len(del_grupo),
                monto_centavos=sum(p.monto_centavos for p in del_grupo),
            ),
            por_estado={k: Agregado(**v) for k, v in sorted(por_estado.items())},
            por_codigo_error={
                k: Agregado(**v)
                for k, v in sorted(por_codigo.items(), key=lambda kv: (-kv[1]["cantidad"], kv[0]))
            },
        )

    return ResultadoPerfilarPagos(
        registros_recibidos=len(pagos),
        registros_validos=len(validos),
        invalidos=[RegistroInvalido.model_validate(i) for i in invalidos],
        por_moneda=por_moneda,
    ).como_dict()


# --- Cálculo ----------------------------------------------------------------

_OPERADORES: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluar_expresion(expresion: str) -> float:
    """Evalúa aritmética pura recorriendo el AST.

    Nunca `eval()`: el texto lo escribe un modelo, y `eval` sobre texto de un
    modelo es ejecución de código arbitrario. Acá solo pasan números y los
    operadores de `_OPERADORES`; cualquier otra cosa (nombres, llamadas,
    atributos) es un `ValueError`.
    """

    def visitar(nodo: ast.AST) -> float:
        if isinstance(nodo, ast.Expression):
            return visitar(nodo.body)
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)) and not isinstance(nodo.value, bool):
            return nodo.value
        if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
            izquierda, derecha = visitar(nodo.left), visitar(nodo.right)
            if isinstance(nodo.op, ast.Pow) and abs(derecha) > 100:
                raise ValueError("exponente demasiado grande")
            return _OPERADORES[type(nodo.op)](izquierda, derecha)
        if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
            return _OPERADORES[type(nodo.op)](visitar(nodo.operand))
        raise ValueError(f"elemento no permitido: {type(nodo).__name__}")

    return visitar(ast.parse(expresion.replace(",", "."), mode="eval"))


class ArgsCalcular(BaseModel):
    expresion: str = Field(
        description="Expresión aritmética con números y + - * / // % ** y paréntesis. Ej: (890000 + 2150000) / 100"
    )


@tool(args_schema=ArgsCalcular)
async def calcular(expresion: str) -> dict[str, Any]:
    """Calcula una expresión aritmética de forma exacta.

    Usala para cualquier cuenta — sumas, diferencias, conversiones de centavos
    a unidades — en vez de hacerla de cabeza.
    """
    try:
        resultado = evaluar_expresion(expresion)
    except ZeroDivisionError:
        salida = ResultadoCalcular(expresion=expresion, error="División por cero.", sugerencia="Revisá el denominador.")
    except (ValueError, SyntaxError) as exc:
        salida = ResultadoCalcular(
            expresion=expresion,
            error=f"Expresión inválida: {exc}",
            sugerencia="Usá solo números, operadores aritméticos y paréntesis.",
        )
    else:
        salida = ResultadoCalcular(expresion=expresion, resultado=round(resultado, 6))
    return salida.como_dict()


class ArgsPorcentaje(BaseModel):
    parte: float = Field(description="El valor de la parte.")
    total: float = Field(description="El valor del total. Tiene que ser mayor que cero.")
    decimales: int = Field(default=2, ge=0, le=4, description="Decimales del resultado.")


@tool(args_schema=ArgsPorcentaje)
async def porcentaje(parte: float, total: float, decimales: int = 2) -> dict[str, Any]:
    """Calcula qué porcentaje representa `parte` sobre `total`."""
    if total <= 0:
        salida = ResultadoPorcentaje(
            error="El total tiene que ser mayor que cero.", sugerencia="Revisá qué pasaste como total."
        )
    elif parte > total:
        salida = ResultadoPorcentaje(
            error="La parte es mayor que el total.",
            sugerencia="Probablemente estén invertidos, o el total no incluye a la parte.",
        )
    else:
        salida = ResultadoPorcentaje(parte=parte, total=total, porcentaje=round(parte / total * 100, decimales))
    return salida.como_dict()


HERRAMIENTAS_ANALISIS = [perfilar_pagos, calcular, porcentaje]
