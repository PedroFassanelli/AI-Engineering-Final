"""El corpus de `nimbus-sdk`: carga, chunking y metadatos.

Es la misma documentación de la Pre-entrega 4 (cinco Markdown: guía de inicio,
referencia de API, catálogo de errores, webhooks y changelog). Este módulo es
la **única** fuente de los fragmentos: la ingesta a Pinecone y el índice BM25
en memoria se construyen sobre exactamente la misma lista, así las dos mitades
del híbrido indexan el mismo material.

Chunking en dos niveles
-----------------------
1. **Por sección de Markdown** (`##` / `###`). Cada sección de este corpus habla
   de una sola cosa (un código de error, un método, un evento de webhook), así
   que la sección es la unidad natural para recuperar y para citar. En la
   Pre-entrega 4 el corte era por tamaño (800 tokens), y un fragmento mezclaba
   tres secciones: al agente le llegaba la respuesta correcta rodeada de dos que
   no venían al caso, y pagaba esos tokens en cada consulta.
2. **Por tokens**, solo si una sección supera `CHUNK_MAX_TOKENS`. Se usa
   `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (la medida es en
   tokens, no en caracteres) con solape, y cada parte conserva el título de su
   sección.

Los encabezados que caen **dentro** de un bloque de código no cortan: el
changelog tiene comentarios `# Antes (2.x)` en ejemplos de Python que un split
ingenuo confundiría con títulos.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from app import config
from app.rag.esquemas import Chunk, MetadatosChunk


class CorpusVacio(RuntimeError):
    """No hay nada para indexar."""


@dataclass(frozen=True, slots=True)
class Seccion:
    fuente: str
    documento: str
    seccion: str
    texto: str


def categoria_de(nombre: str) -> str:
    """`03-manejo-de-errores.md` → `manejo de errores`.

    El prefijo numérico ordena los archivos en el disco pero no aporta nada como
    metadato de búsqueda.
    """
    tronco = Path(nombre).stem
    sin_prefijo = tronco.split("-", 1)[1] if tronco[:2].isdigit() and "-" in tronco else tronco
    return sin_prefijo.replace("-", " ").replace("_", " ").strip()


def dividir_en_secciones(nombre: str, markdown: str) -> list[Seccion]:
    """Parte un documento en sus secciones `##`/`###`, ignorando los bloques de código."""
    secciones: list[Seccion] = []
    documento = Path(nombre).stem
    actual = ""
    lineas: list[str] = []
    en_codigo = False

    def cerrar() -> None:
        cuerpo = "\n".join(lineas).strip()
        if cuerpo:
            secciones.append(Seccion(fuente=nombre, documento=documento, seccion=actual, texto=cuerpo))

    for linea in markdown.splitlines():
        if linea.startswith("```"):
            en_codigo = not en_codigo
        if not en_codigo and linea.startswith("# "):
            documento = linea[2:].strip()
            continue
        if not en_codigo and re.match(r"#{2,3} ", linea):
            cerrar()
            actual = linea.lstrip("#").strip()
            lineas = []
            continue
        lineas.append(linea)
    cerrar()
    return secciones


# --- Tokens ------------------------------------------------------------------


@lru_cache(maxsize=1)
def _codificador():
    import tiktoken

    return tiktoken.get_encoding(config.CODIFICACION_TOKENS)


def contar_tokens(texto: str) -> int:
    return len(_codificador().encode(texto))


def _partir_por_tokens(texto: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.CODIFICACION_TOKENS,
        chunk_size=config.CHUNK_MAX_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(texto)


# --- Construcción de los fragmentos -------------------------------------------


def cargar_markdown(directorio: Path | None = None) -> dict[str, str]:
    """`{archivo: contenido}` en orden estable (el hash del corpus depende del orden)."""
    directorio = directorio or config.DIR_DOCS
    archivos = sorted(directorio.glob("*.md"))
    contenido = {a.name: a.read_text(encoding="utf-8") for a in archivos}
    contenido = {nombre: texto for nombre, texto in contenido.items() if texto.strip()}
    if not contenido:
        raise CorpusVacio(f"No hay documentos Markdown en {directorio}.")
    return contenido


def construir_chunks(documentos: dict[str, str]) -> list[Chunk]:
    """Secciones → (si hace falta) partes por tokens → `Chunk` con metadatos validados."""
    chunks: list[Chunk] = []
    for fuente, markdown in documentos.items():
        piezas: list[tuple[Seccion, str, int, int]] = []
        for seccion in dividir_en_secciones(fuente, markdown):
            partes = (
                _partir_por_tokens(seccion.texto)
                if contar_tokens(seccion.texto) > config.CHUNK_MAX_TOKENS
                else [seccion.texto]
            )
            piezas.extend((seccion, texto, i + 1, len(partes)) for i, texto in enumerate(partes))

        for chunk_id, (seccion, texto, parte, total_partes) in enumerate(piezas):
            chunks.append(
                Chunk(
                    id=f"{fuente}::{chunk_id}",
                    metadatos=MetadatosChunk(
                        text=texto,
                        fuente=fuente,
                        documento=seccion.documento,
                        seccion=seccion.seccion,
                        categoria=categoria_de(fuente),
                        chunk_id=chunk_id,
                        total_chunks=len(piezas),
                        parte=parte,
                        total_partes=total_partes,
                        tokens=contar_tokens(texto),
                    ),
                )
            )
    return chunks


def hash_corpus(chunks: Sequence[Chunk]) -> str:
    """SHA-256 de los fragmentos (id + texto + metadatos), estable ante el orden.

    Se hashean los **fragmentos** y no los archivos: cambiar los parámetros de
    chunking cambia lo indexado aunque los documentos sean los mismos, y eso
    también obliga a reindexar.
    """
    hasher = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c.id):
        hasher.update(chunk.id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(chunk.metadatos.model_dump_json().encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


@lru_cache(maxsize=1)
def chunks_por_defecto() -> tuple[Chunk, ...]:
    """Los fragmentos del corpus del repositorio, construidos una vez por proceso."""
    return tuple(construir_chunks(cargar_markdown()))
