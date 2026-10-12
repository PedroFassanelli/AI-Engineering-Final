"""La mitad vectorial del híbrido: Pinecone Serverless, 100 % asíncrono.

- Cliente: `AsyncPinecone` / `AsyncIndex` (SDK oficial, aiohttp por debajo).
  Ninguna llamada síncrona dentro de una corutina: ni `requests`, ni el
  cliente sync de Pinecone.
- Embeddings: `OpenAIEmbeddings.aembed_*` (cliente `AsyncOpenAI` por debajo).

Nombre del índice, namespace, nube, región y modelo salen de `app.config`, que
los lee del entorno. Nada de eso está escrito acá.

Sincronización (`sincronizar`)
------------------------------
El worker la corre al arrancar. Es idempotente:

1. Si el índice no existe, lo crea (Serverless, dimensión derivada del modelo).
   Si existe, verifica la dimensión y se planta si no coincide: un índice de
   Pinecone no se puede redimensionar, y escribir vectores de otro largo
   termina en un 400 o, peor, en un índice que mezcla dos espacios.
2. Compara el `Manifiesto` guardado en Redis con el de los fragmentos locales.
   Si coinciden y el namespace tiene la cantidad esperada de vectores, no hace
   nada (el arranque normal no gasta un solo embedding).
3. Si no, vacía el namespace y reindexa con ids deterministas.

Con varios workers arrancando a la vez (`--scale worker=3`), solo uno indexa:
toma un lock en Redis y los demás esperan a que aparezca el manifiesto nuevo.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from pydantic import ValidationError
from redis.asyncio import Redis

from app import config
from app.rag.corpus import hash_corpus
from app.rag.esquemas import Chunk, Manifiesto, MetadatosChunk

log = logging.getLogger("nimbus.rag")

#: Vectores por llamada de upsert. Pinecone acepta hasta 1000 o 2 MB; cada
#: vector de 1536 floats con su texto pesa ~15 KB.
LOTE_UPSERT = 100


class ErrorDeIndice(RuntimeError):
    """El índice remoto no es compatible con la configuración local."""


def clave_manifiesto(prefijo: str | None = None) -> str:
    return f"{prefijo or config.PREFIJO_REDIS}:rag:manifiesto"


def manifiesto_esperado(chunks: Sequence[Chunk]) -> Manifiesto:
    return Manifiesto(
        fingerprint=config.fingerprint_embeddings(),
        index_name=config.INDEX_NAME,
        namespace=config.PINECONE_NAMESPACE,
        chunk_max_tokens=config.CHUNK_MAX_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        total_chunks=len(chunks),
        corpus_hash=hash_corpus(chunks),
        documentos=sorted({c.metadatos.fuente for c in chunks}),
    )


def mismo_indice(a: Manifiesto | None, b: Manifiesto) -> bool:
    """¿Describen el mismo contenido indexado? (la fecha no cuenta)."""
    if a is None:
        return False
    campos = ("fingerprint", "index_name", "namespace", "total_chunks", "corpus_hash")
    return all(getattr(a, c) == getattr(b, c) for c in campos)


def construir_embeddings():
    """`OpenAIEmbeddings` del modelo configurado.

    Los embeddings son siempre de OpenAI, aunque el chat use Anthropic:
    Anthropic no publica una API de embeddings.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise config.ConfiguracionInvalida(
            "Falta OPENAI_API_KEY: el RAG híbrido la necesita para los embeddings. "
            "Completala en .env, o usá RAG_MODO=lexico para correr solo con BM25."
        )
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, Any] = {"model": config.EMBEDDING_MODEL}
    nativa = config.DIMENSIONES.get(config.EMBEDDING_MODEL)
    if nativa is not None and config.dimension_embeddings() != nativa:
        # Los text-embedding-3-* admiten truncar el vector (Matryoshka). Si se
        # pidió otra dimensión hay que decírselo a la API.
        kwargs["dimensions"] = config.dimension_embeddings()
    return OpenAIEmbeddings(**kwargs)


class IndiceVectorial:
    """Un namespace de un índice de Pinecone, con su modelo de embeddings."""

    def __init__(self, indice: Any, embeddings: Any, namespace: str | None = None) -> None:
        self.indice = indice
        self.embeddings = embeddings
        self.namespace = namespace or config.PINECONE_NAMESPACE

    # --- Ciclo de vida -------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def abrir(cls, api_key: str | None = None) -> AsyncIterator["IndiceVectorial"]:
        """Asegura el índice (creación + dimensión) y deja abierto el cliente de datos."""
        from pinecone import AsyncPinecone

        clave = api_key or os.getenv("PINECONE_API_KEY", "").strip()
        if not clave:
            raise config.ConfiguracionInvalida(
                "Falta PINECONE_API_KEY (https://app.pinecone.io → API Keys). "
                "Completala en .env, o usá RAG_MODO=lexico para correr solo con BM25."
            )
        embeddings = construir_embeddings()
        async with AsyncPinecone(api_key=clave) as pc:
            await asegurar_indice(pc)
            indice = await pc.index(name=config.INDEX_NAME)
            async with indice:
                yield cls(indice, embeddings)

    # --- Lectura ---------------------------------------------------------------

    async def embeber_consulta(self, texto: str) -> list[float]:
        return await self.embeddings.aembed_query(texto)

    async def buscar_vector(self, vector: Sequence[float], k: int) -> list[tuple[str, float, MetadatosChunk]]:
        """Los `k` vecinos más cercanos, con la metadata **re-validada**.

        Un vector cuya metadata no cumple `MetadatosChunk` (escrito por otra
        versión del código, o por otra cosa en el mismo namespace) se descarta
        con un aviso en el log en vez de entrar al ranking.
        """
        respuesta = await self.indice.query(
            vector=list(vector), top_k=k, namespace=self.namespace, include_metadata=True
        )
        resultados: list[tuple[str, float, MetadatosChunk]] = []
        for match in respuesta.matches:
            try:
                metadatos = MetadatosChunk.model_validate(match.metadata or {})
            except ValidationError as error:
                log.warning("Vector %s con metadata inválida, se ignora: %s", match.id, error)
                continue
            resultados.append((match.id, float(match.score), metadatos))
        return resultados

    async def contar(self) -> int:
        estadisticas = await self.indice.describe_index_stats()
        resumen = estadisticas.namespaces.get(self.namespace)
        return int(resumen.vector_count) if resumen is not None else 0

    # --- Escritura ---------------------------------------------------------------

    async def reemplazar(self, chunks: Sequence[Chunk]) -> None:
        """Vacía el namespace y sube todos los fragmentos con ids deterministas."""
        if await self.contar() > 0:
            await self.indice.delete(delete_all=True, namespace=self.namespace)

        vectores = await self.embeddings.aembed_documents([c.texto_para_embedding for c in chunks])
        registros = [
            {"id": c.id, "values": v, "metadata": c.metadatos.model_dump()}
            for c, v in zip(chunks, vectores, strict=True)
        ]
        for inicio in range(0, len(registros), LOTE_UPSERT):
            await self.indice.upsert(
                vectors=registros[inicio : inicio + LOTE_UPSERT],
                namespace=self.namespace,
                show_progress=False,
            )
        await self._esperar_conteo(len(chunks))

    async def _esperar_conteo(self, esperado: int, espera_maxima: float = 60.0) -> None:
        """Serverless es eventualmente consistente: el conteo tarda unos segundos."""
        limite = asyncio.get_running_loop().time() + espera_maxima
        while asyncio.get_running_loop().time() < limite:
            if await self.contar() == esperado:
                return
            await asyncio.sleep(2)
        log.warning("El namespace %s todavía no muestra %d vectores.", self.namespace, esperado)


async def asegurar_indice(pc: Any) -> None:
    """Crea el índice si no existe y verifica su dimensión si existe."""
    from pinecone import ServerlessSpec

    dimension = config.dimension_embeddings()
    if not await pc.has_index(config.INDEX_NAME):
        log.info("Creando el índice %s (dim %d, %s)", config.INDEX_NAME, dimension, config.PINECONE_METRICA)
        await pc.create_index(
            name=config.INDEX_NAME,
            dimension=dimension,
            metric=config.PINECONE_METRICA,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
            timeout=180,
        )
    remota = (await pc.describe_index(config.INDEX_NAME)).dimension
    if remota is not None and remota != dimension:
        raise ErrorDeIndice(
            f"El índice {config.INDEX_NAME!r} tiene dimensión {remota} y el modelo "
            f"{config.EMBEDDING_MODEL!r} produce {dimension}. Un índice no se redimensiona: "
            "usá otro INDEX_NAME, o volvé al modelo anterior."
        )


async def leer_manifiesto(redis: Redis, prefijo: str | None = None) -> Manifiesto | None:
    crudo = await redis.get(clave_manifiesto(prefijo))
    return Manifiesto.model_validate_json(crudo) if crudo else None


async def sincronizar(
    vectorial: IndiceVectorial,
    chunks: Sequence[Chunk],
    redis: Redis,
    *,
    prefijo: str | None = None,
    espera_maxima: float = 300.0,
) -> tuple[Manifiesto, bool]:
    """Deja Pinecone alineado con los fragmentos locales. Devuelve `(manifiesto, reindexó)`."""
    esperado = manifiesto_esperado(chunks)
    clave_lock = f"{clave_manifiesto(prefijo)}:lock"
    limite = asyncio.get_running_loop().time() + espera_maxima

    while True:
        actual = await leer_manifiesto(redis, prefijo)
        if mismo_indice(actual, esperado) and await vectorial.contar() == esperado.total_chunks:
            return actual, False  # type: ignore[return-value]

        token = uuid.uuid4().hex
        if await redis.set(clave_lock, token, nx=True, ex=int(espera_maxima)):
            try:
                log.info(
                    "Reindexando %s/%s: %d fragmentos (%s).",
                    config.INDEX_NAME, vectorial.namespace, len(chunks), esperado.fingerprint,
                )
                await vectorial.reemplazar(chunks)
                await redis.set(clave_manifiesto(prefijo), esperado.model_dump_json())
                return esperado, True
            finally:
                if await redis.get(clave_lock) == token:
                    await redis.delete(clave_lock)

        # Otro worker está indexando: se espera a que termine.
        if asyncio.get_running_loop().time() > limite:
            raise ErrorDeIndice("Otro proceso tiene el lock de indexación hace demasiado tiempo.")
        await asyncio.sleep(2)
