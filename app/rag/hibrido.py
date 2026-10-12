"""Recuperador híbrido: BM25 (léxico) + Pinecone (vectorial), fusionados con RRF.

Por qué hacen falta los dos (la conclusión medida de la Pre-entrega 4): la
búsqueda vectorial entiende parafraseos ("¿cómo pruebo sin mover plata de
verdad?" encuentra la sección del *sandbox*), pero es mala con literales
(`E4090`, `capturar_pago`, `X-Nimbus-Signature`). BM25 es exactamente lo
contrario. Las preguntas de soporte mezclan los dos tipos.

Fusión: **Reciprocal Rank Fusion ponderado**, la misma que usa
`EnsembleRetriever`. Cada mitad aporta `peso / (c + posición)` por fragmento;
se suman y se ordena. Fusiona *posiciones*, no puntajes, así que no importa
que el puntaje de BM25 (sin techo) y el coseno (0 a 1) no sean comparables.
Se implementa acá en vez de usar `EnsembleRetriever` porque este es async de
punta a punta y deja registrado, por fragmento, de qué mitad vino.

Degradación: si Pinecone o la API de embeddings fallan en una consulta, la
búsqueda no se cae; responde con BM25 solo, `modo="lexico"` y un `aviso` que
ve el agente y queda en la traza. Una consulta de soporte con resultados
léxicos es mejor que un job en FAILED.

Observabilidad: cada búsqueda es un span `RETRIEVER` de OpenInference con los
documentos recuperados (id, texto, puntaje y metadatos), y adentro un span por
mitad y uno `EMBEDDING` para el vector de la consulta. En Phoenix se ve qué
trajo cada mitad y qué quedó en el ranking final.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from openinference.instrumentation import get_attributes_from_context
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app import config
from app.rag.esquemas import Chunk, FragmentoRecuperado, ModoBusqueda, ResultadoBusqueda
from app.rag.lexico import IndiceLexico
from app.observability import contexto_padre
from app.rag.vectorial import IndiceVectorial

log = logging.getLogger("nimbus.rag")

#: Caracteres de texto por fragmento que se devuelven al agente. Las secciones
#: son cortas; el recorte existe para que una larga no se coma la ventana.
MAX_CARACTERES_FRAGMENTO = 1500

_tracer = trace.get_tracer("nimbus-rag")


def fusionar_rrf(
    rankings: dict[str, Sequence[str]],
    pesos: dict[str, float],
    c: int = config.RRF_C,
) -> list[tuple[str, float, dict[str, int]]]:
    """RRF ponderado. Devuelve `(id, puntaje, {ranking: posición})` ordenado.

    Ante empate de puntaje gana el que apareció mejor en algún ranking, y
    después el id: el orden es determinista (los tests y las trazas lo agradecen).
    """
    puntajes: dict[str, float] = {}
    posiciones: dict[str, dict[str, int]] = {}
    for nombre, ids in rankings.items():
        peso = pesos.get(nombre, 0.0)
        if peso <= 0:
            # Un ranking apagado no aporta candidatos (con puntaje 0 igual
            # aparecerían al final): así se evalúa cada mitad por separado.
            continue
        for posicion, id_ in enumerate(ids, start=1):
            puntajes[id_] = puntajes.get(id_, 0.0) + peso / (c + posicion)
            posiciones.setdefault(id_, {})[nombre] = posicion
    return sorted(
        ((id_, puntaje, posiciones[id_]) for id_, puntaje in puntajes.items()),
        key=lambda t: (-t[1], min(t[2].values()), t[0]),
    )


def _registrar_documentos(span: Any, fragmentos: Sequence[tuple[str, str, float, dict[str, Any]]]) -> None:
    """`retrieval.documents.<i>.document.*`: así Phoenix muestra la tabla de documentos."""
    for i, (id_, contenido, puntaje, metadatos) in enumerate(fragmentos):
        base = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
        span.set_attribute(f"{base}.{DocumentAttributes.DOCUMENT_ID}", id_)
        span.set_attribute(f"{base}.{DocumentAttributes.DOCUMENT_CONTENT}", contenido)
        span.set_attribute(f"{base}.{DocumentAttributes.DOCUMENT_SCORE}", float(puntaje))
        span.set_attribute(
            f"{base}.{DocumentAttributes.DOCUMENT_METADATA}",
            json.dumps(metadatos, ensure_ascii=False),
        )


def _span(nombre: str, tipo: OpenInferenceSpanKindValues, entrada: str, *, raiz: bool = False):
    """Un span de OpenInference. El de `raiz=True` cuelga de la herramienta que llamó al RAG."""
    return _tracer.start_as_current_span(
        nombre,
        context=contexto_padre() if raiz else None,
        attributes={
            **dict(get_attributes_from_context()),  # sesión y metadata del turno
            SpanAttributes.OPENINFERENCE_SPAN_KIND: tipo.value,
            SpanAttributes.INPUT_VALUE: entrada,
        },
    )


class RecuperadorHibrido:
    """Las dos mitades sobre la misma lista de fragmentos.

    `vectorial=None` es el modo `lexico`: BM25 solo, sin Pinecone ni
    embeddings (`RAG_MODO=lexico`, y los tests).
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectorial: IndiceVectorial | None = None,
        *,
        peso_lexico: float = config.PESO_BM25,
        peso_vectorial: float = config.PESO_VECTORIAL,
        candidatos: int = config.RAG_CANDIDATOS,
    ) -> None:
        self.chunks = {c.id: c for c in chunks}
        self.lexico = IndiceLexico(chunks)
        self.vectorial = vectorial
        self.pesos = {"lexico": peso_lexico, "vectorial": peso_vectorial}
        self.candidatos = candidatos

    @property
    def modo(self) -> ModoBusqueda:
        return "hibrido" if self.vectorial is not None else "lexico"

    @property
    def fuentes(self) -> set[str]:
        return {c.metadatos.fuente for c in self.chunks.values()}

    # --- Cada mitad ------------------------------------------------------------

    def _ranking_lexico(self, consulta: str) -> list[str]:
        with _span("busqueda_lexica", OpenInferenceSpanKindValues.RETRIEVER, consulta) as span:
            resultados = self.lexico.buscar(consulta, self.candidatos)
            _registrar_documentos(
                span, [(c.id, c.metadatos.titulo, p, {"motor": "bm25"}) for c, p in resultados]
            )
            return [c.id for c, _ in resultados]

    async def _ranking_vectorial(self, consulta: str) -> list[str]:
        assert self.vectorial is not None
        with _span("busqueda_vectorial", OpenInferenceSpanKindValues.RETRIEVER, consulta) as span:
            with _span("embedding_consulta", OpenInferenceSpanKindValues.EMBEDDING, consulta) as emb:
                emb.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, config.EMBEDDING_MODEL)
                vector = await self.vectorial.embeber_consulta(consulta)
            resultados = await self.vectorial.buscar_vector(vector, self.candidatos)
            # Un id que no está en los fragmentos locales es un vector viejo que
            # el manifiesto no detectó: no puede entrar al ranking (no hay texto
            # local que devolver, y BM25 no lo conoce).
            conocidos = [(id_, s, m) for id_, s, m in resultados if id_ in self.chunks]
            if len(conocidos) < len(resultados):
                span.add_event("vectores_desconocidos", {"cantidad": len(resultados) - len(conocidos)})
            _registrar_documentos(
                span,
                [(id_, m.titulo, s, {"motor": "pinecone", "namespace": self.vectorial.namespace}) for id_, s, m in conocidos],
            )
            return [id_ for id_, _, _ in conocidos]

    # --- Búsqueda --------------------------------------------------------------

    async def buscar(self, consulta: str, k: int = config.RAG_TOP_K) -> ResultadoBusqueda:
        with _span("recuperar_documentacion", OpenInferenceSpanKindValues.RETRIEVER, consulta, raiz=True) as span:
            rankings: dict[str, Sequence[str]] = {"lexico": self._ranking_lexico(consulta)}
            modo: ModoBusqueda = self.modo
            aviso: str | None = None

            if self.vectorial is not None:
                try:
                    rankings["vectorial"] = await self._ranking_vectorial(consulta)
                except Exception as error:  # noqa: BLE001 - cualquier falla remota degrada, no rompe
                    log.warning("Búsqueda vectorial caída, se responde solo con BM25: %r", error)
                    span.record_exception(error)
                    modo = "lexico"
                    aviso = (
                        "La búsqueda semántica no respondió; estos resultados son solo por "
                        "coincidencia de palabras. Si no aparece lo que buscás, probá con el "
                        "término literal (código de error, nombre de función)."
                    )

            pesos = self.pesos if modo == "hibrido" else {"lexico": 1.0}
            fusion = fusionar_rrf(rankings, pesos)[:k]
            resultados = [
                FragmentoRecuperado(
                    posicion=i,
                    id=id_,
                    fuente=self.chunks[id_].metadatos.fuente,
                    seccion=self.chunks[id_].metadatos.titulo,
                    categoria=self.chunks[id_].metadatos.categoria,
                    puntaje=round(puntaje, 6),
                    rango_lexico=posiciones.get("lexico"),
                    rango_vectorial=posiciones.get("vectorial"),
                    texto=self.chunks[id_].metadatos.text[:MAX_CARACTERES_FRAGMENTO],
                )
                for i, (id_, puntaje, posiciones) in enumerate(fusion, start=1)
            ]

            _registrar_documentos(
                span,
                [
                    (
                        r.id,
                        r.texto,
                        r.puntaje,
                        {
                            "fuente": r.fuente,
                            "seccion": r.seccion,
                            "categoria": r.categoria,
                            "rango_lexico": r.rango_lexico,
                            "rango_vectorial": r.rango_vectorial,
                        },
                    )
                    for r in resultados
                ],
            )
            span.set_attribute(
                SpanAttributes.METADATA,
                json.dumps({"modo": modo, "pesos": pesos, "k": k, "candidatos": self.candidatos}),
            )
            if aviso:
                span.set_status(Status(StatusCode.OK, "degradado a léxico"))
            return ResultadoBusqueda(consulta=consulta, modo=modo, resultados=resultados, aviso=aviso)
