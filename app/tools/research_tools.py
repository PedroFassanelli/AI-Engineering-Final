"""Herramientas del agente de investigación.

Todas **leen**; ninguna calcula. La separación es deliberada: si el
investigador pudiera sacar porcentajes, lo haría "de paso" y en su cabeza, y el
analista quedaría validando números que nadie computó con una herramienta.

- `buscar_documentacion`: el **RAG** (Pinecone + BM25, ver `app/rag/`).
- `buscar_comercio` / `consultar_pagos`: la base de negocio (SQLite, `aiosqlite`).

Las tres son `async`, validan sus argumentos con `args_schema` y su salida con
los modelos de `app/tools/esquemas.py`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app import config
from app.fuentes import db
from app.rag import recuperador_actual
from app.rag.esquemas import ResultadoBusqueda
from app.tools.esquemas import (
    Comercio,
    RegistroPago,
    ResultadoBuscarComercio,
    ResultadoConsultarPagos,
)


class ArgsBuscarDocumentacion(BaseModel):
    consulta: str = Field(
        min_length=2,
        max_length=500,
        description=(
            "Qué buscar en la documentación. Sirven tanto términos literales (códigos "
            "de error como E4090, nombres de función como capturar_pago, encabezados "
            "como Idempotency-Key) como preguntas en lenguaje natural."
        ),
    )
    k: int = Field(default=config.RAG_TOP_K, ge=1, le=5, description="Cuántos fragmentos devolver (1 a 5).")


@tool(args_schema=ArgsBuscarDocumentacion)
async def buscar_documentacion(consulta: str, k: int = config.RAG_TOP_K) -> dict[str, Any]:
    """Busca en la documentación técnica oficial de nimbus-sdk (búsqueda híbrida).

    Devuelve los fragmentos más relevantes con su archivo de origen (`fuente`)
    y su sección, para que puedas citarlos. Usala para saber qué significa un
    código de error, si es reintentable, cómo se resuelve, o cómo funciona una
    parte de la API.
    """
    resultado = await recuperador_actual().buscar(consulta, k=k)
    if not resultado.resultados:
        resultado = ResultadoBusqueda(
            consulta=consulta,
            modo=resultado.modo,
            aviso=resultado.aviso,
            error="Ningún fragmento de la documentación coincide con esa consulta.",
            sugerencia=(
                "Probá con términos más literales (el código de error exacto, el nombre "
                "de la excepción o de la función) o reformulá la pregunta."
            ),
        )
    return resultado.como_dict()


class ArgsBuscarComercio(BaseModel):
    nombre: str = Field(
        min_length=2,
        max_length=120,
        description="Nombre o parte del nombre del comercio. No distingue acentos ni mayúsculas.",
    )


@tool(args_schema=ArgsBuscarComercio)
async def buscar_comercio(nombre: str) -> dict[str, Any]:
    """Busca comercios de Nimbus Pagos por nombre y devuelve su `comercio_id`.

    Necesitás el `comercio_id` para consultar los pagos de un comercio.
    """
    coincidencias = await db.buscar_comercios_por_nombre(nombre)
    if not coincidencias:
        salida = ResultadoBuscarComercio(
            nombre=nombre,
            error="No hay ningún comercio con ese nombre.",
            sugerencia="Probá con una sola palabra del nombre, sin el tipo de negocio.",
        )
    else:
        salida = ResultadoBuscarComercio(
            nombre=nombre, comercios=[Comercio.model_validate(c) for c in coincidencias]
        )
    return salida.como_dict()


class ArgsConsultarPagos(BaseModel):
    comercio_id: int = Field(gt=0, description="ID numérico del comercio, obtenido con buscar_comercio.")


@tool(args_schema=ArgsConsultarPagos)
async def consultar_pagos(comercio_id: int) -> dict[str, Any]:
    """Devuelve TODOS los pagos de un comercio, tal como están en la base.

    Cada pago trae `pago_id`, `monto_centavos` (entero, en centavos), `moneda`,
    `estado` (aprobado, fallido, pendiente, reembolsado), `codigo_error` (solo
    en los fallidos) y la fecha. Los datos se devuelven crudos, sin totales:
    los cálculos los hace el analista.
    """
    pagos = await db.listar_pagos(comercio_id)
    if not pagos:
        salida = ResultadoConsultarPagos(
            comercio_id=comercio_id,
            error="Ese comercio no tiene pagos o no existe.",
            sugerencia="Verificá el comercio_id con buscar_comercio.",
        )
    else:
        registros = [RegistroPago.model_validate(p) for p in pagos]
        salida = ResultadoConsultarPagos(comercio_id=comercio_id, cantidad=len(registros), pagos=registros)
    return salida.como_dict()


HERRAMIENTAS_INVESTIGACION = [buscar_documentacion, buscar_comercio, consultar_pagos]

#: Las herramientas cuyo resultado es **dato** (y por eso se le pasa al
#: analista). La documentación no está acá: el analista no la necesita para
#: calcular, y mandársela sería contaminación de contexto.
HERRAMIENTAS_DE_DATOS: frozenset[str] = frozenset({"consultar_pagos"})
