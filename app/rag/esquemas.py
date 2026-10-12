"""Contratos de datos del RAG, como modelos Pydantic.

Tres fronteras donde un dato cambia de manos y conviene validarlo:

- **Ingesta → Pinecone**: `MetadatosChunk` es exactamente lo que viaja dentro de
  cada vector. Si un campo falta o está fuera de rango, falla acá, antes de
  gastar un embedding, y no en una consulta tres días después.
- **Pinecone → recuperador**: la metadata que vuelve se re-valida con el mismo
  modelo. Un vector escrito por otra versión del código (o por otra entrega en
  el mismo índice) no entra al ranking como si fuera nuestro.
- **Recuperador → agente**: `ResultadoBusqueda` es la salida de la herramienta
  `buscar_documentacion`. El agente recibe siempre la misma forma, con la fuente
  de cada fragmento, para poder citar y para que las citas se puedan verificar.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.tools.esquemas import SalidaHerramienta


class MetadatosChunk(BaseModel):
    """Lo que viaja dentro de cada vector de Pinecone (y de cada entrada de BM25)."""

    #: El texto original completo: el índice es autosuficiente, no hace falta
    #: una base aparte para reconstruir lo que se recuperó.
    text: str = Field(min_length=1)
    fuente: str = Field(description="Archivo de origen, p. ej. 03-manejo-de-errores.md")
    documento: str = Field(description="Título del documento (su encabezado #)")
    seccion: str = Field(description="Encabezado ## / ### de la sección")
    categoria: str = Field(description="Etiqueta temática derivada del nombre del archivo")
    chunk_id: int = Field(ge=0, description="Posición del fragmento dentro del documento")
    total_chunks: int = Field(ge=1)
    #: Si una sección larga se partió por tokens: qué parte es y de cuántas.
    parte: int = Field(default=1, ge=1)
    total_partes: int = Field(default=1, ge=1)
    tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def _en_rango(self) -> "MetadatosChunk":
        if self.chunk_id >= self.total_chunks:
            raise ValueError(f"chunk_id={self.chunk_id} fuera de rango ({self.total_chunks} fragmentos)")
        if self.parte > self.total_partes:
            raise ValueError(f"parte={self.parte} fuera de rango ({self.total_partes} partes)")
        return self

    @property
    def titulo(self) -> str:
        """`Documento › Sección`, como se le muestra al agente."""
        return f"{self.documento} › {self.seccion}" if self.seccion else self.documento


class Chunk(BaseModel):
    """Un fragmento listo para indexar: id determinista + metadatos."""

    id: str = Field(pattern=r"^[\w.-]+::\d+$", description="<fuente>::<chunk_id>")
    metadatos: MetadatosChunk

    @property
    def texto_para_embedding(self) -> str:
        """El título va adelante del texto al embeber.

        Una sección que dice "Es reintentable: sí" no significa nada sin saber
        de qué código habla; con "Manejo de errores › E5030" adelante, sí. Es la
        forma más barata de *contextual retrieval*.
        """
        return f"{self.metadatos.titulo}\n\n{self.metadatos.text}"


class Manifiesto(BaseModel):
    """Huella de lo que quedó indexado en Pinecone.

    Responde la pregunta que un RAG híbrido no puede saltear: *¿la mitad
    vectorial (remota) y la mitad léxica (en memoria) indexan el mismo corpus?*
    BM25 se arma desde los archivos locales; si alguien edita un documento y no
    reindexa, las dos mitades buscan en universos distintos sin ningún error
    visible. El worker compara esta huella al arrancar y reindexa si cambió.
    """

    fingerprint: str = Field(description="proveedor:modelo:dimensión de los embeddings")
    index_name: str
    namespace: str
    chunk_max_tokens: int
    chunk_overlap: int
    total_chunks: int = Field(ge=1)
    corpus_hash: str = Field(min_length=64, max_length=64)
    documentos: list[str]
    indexado_en: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Salida de la búsqueda ---------------------------------------------------

ModoBusqueda = Literal["hibrido", "lexico", "vectorial"]


class FragmentoRecuperado(BaseModel):
    """Un fragmento del ranking final, con su procedencia y cómo llegó ahí."""

    posicion: int = Field(ge=1)
    id: str
    fuente: str
    seccion: str = Field(description="`Documento › Sección`")
    categoria: str
    #: Puntaje de la fusión (RRF ponderado). Solo sirve para ordenar.
    puntaje: float = Field(ge=0)
    #: Posición en cada ranking de origen (`None` = no apareció en ese ranking).
    #: Es lo que permite ver en una traza *qué mitad* del híbrido trajo cada cosa.
    rango_lexico: int | None = Field(default=None, ge=1)
    rango_vectorial: int | None = Field(default=None, ge=1)
    texto: str


class ResultadoBusqueda(SalidaHerramienta):
    """Salida de `buscar_documentacion` (un `error` sin `sugerencia` no valida)."""

    consulta: str
    modo: ModoBusqueda
    resultados: list[FragmentoRecuperado] = Field(default_factory=list)
    #: Si la búsqueda se degradó (p. ej. Pinecone no respondió y se usó solo
    #: BM25), acá se dice. El agente lo ve; la traza también.
    aviso: str | None = None
