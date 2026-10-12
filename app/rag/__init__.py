"""RAG sobre la documentación de `nimbus-sdk`: Pinecone + BM25.

    corpus.py     carga, chunking por sección (+ tokens) y metadatos
    lexico.py     BM25 en memoria
    vectorial.py  Pinecone Serverless asíncrono + sincronización del índice
    hibrido.py    fusión RRF, modo degradado y spans RETRIEVER
    evaluacion.py Recall@k / MRR contra un golden set

El recuperador es un recurso del **proceso** (abre conexiones a Pinecone), así
que no se construye dentro de la herramienta: el worker lo abre al arrancar con
`abrir_recuperador` y lo registra acá; `buscar_documentacion` lo toma con
`recuperador_actual()`. Los tests registran uno léxico, sin red.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from redis.asyncio import Redis

from app import config
from app.rag.corpus import chunks_por_defecto
from app.rag.hibrido import RecuperadorHibrido

log = logging.getLogger("nimbus.rag")

_recuperador: RecuperadorHibrido | None = None


class RAGNoInicializado(RuntimeError):
    """Se buscó en la documentación antes de abrir el recuperador."""


def establecer_recuperador(recuperador: RecuperadorHibrido | None) -> None:
    global _recuperador
    _recuperador = recuperador


def recuperador_actual() -> RecuperadorHibrido:
    if _recuperador is None:
        raise RAGNoInicializado(
            "El recuperador no está abierto. El worker lo abre al arrancar "
            "(app.rag.abrir_recuperador); en un script, usá ese mismo context manager."
        )
    return _recuperador


@asynccontextmanager
async def abrir_recuperador(redis: Redis | None = None) -> AsyncIterator[RecuperadorHibrido]:
    """Construye el recuperador según `RAG_MODO`, lo registra y lo cierra al salir.

    En modo `hibrido` sincroniza el índice de Pinecone con el corpus local
    antes de devolverlo (necesita Redis para el manifiesto y el lock).
    """
    # Leer y partir los Markdown es I/O de disco y CPU: fuera del event loop.
    chunks = await asyncio.to_thread(chunks_por_defecto)

    if config.RAG_MODO == "lexico":
        log.warning("RAG_MODO=lexico: búsqueda solo con BM25 (sin Pinecone ni embeddings).")
        recuperador = RecuperadorHibrido(chunks)
        establecer_recuperador(recuperador)
        try:
            yield recuperador
        finally:
            establecer_recuperador(None)
        return

    if config.RAG_MODO != "hibrido":
        raise config.ConfiguracionInvalida(f"RAG_MODO={config.RAG_MODO!r}: usá 'hibrido' o 'lexico'.")
    if redis is None:
        raise ValueError("El modo híbrido necesita Redis para el manifiesto del índice.")

    from app.rag.vectorial import IndiceVectorial, sincronizar

    async with IndiceVectorial.abrir() as vectorial:
        manifiesto, reindexado = await sincronizar(vectorial, chunks, redis)
        log.info(
            "RAG híbrido listo: %s/%s, %d fragmentos, %s (%s).",
            config.INDEX_NAME,
            vectorial.namespace,
            manifiesto.total_chunks,
            manifiesto.fingerprint,
            "reindexado ahora" if reindexado else "índice al día, sin reindexar",
        )
        recuperador = RecuperadorHibrido(chunks, vectorial)
        establecer_recuperador(recuperador)
        try:
            yield recuperador
        finally:
            establecer_recuperador(None)
