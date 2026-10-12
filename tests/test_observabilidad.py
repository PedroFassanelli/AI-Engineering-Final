"""Trazabilidad: qué spans genera una corrida y cómo quedan colgados.

Se instrumenta LangChain contra un exportador **en memoria** (en vez del de
Phoenix) y se inspeccionan los spans: que la traza tenga una raíz por turno,
que lleve `session.id = conversacion_id`, que los spans del RAG cuelguen de la
herramienta que lo llamó y que traigan los documentos recuperados.
"""

from __future__ import annotations

import json

import fakeredis
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.graph import construir_grafo
from app.jobs import AlmacenJobs, EstadoJob
from app.tools.research_tools import buscar_documentacion
from app.worker import correr_job
from tests.test_grafo import modelos
from tests.test_hitl import supervisor_feliz

_proveedor = TracerProvider()
_exportador = InMemorySpanExporter()
_proveedor.add_span_processor(SimpleSpanProcessor(_exportador))


@pytest.fixture(scope="module", autouse=True)
def instrumentado():
    # El provider global se puede fijar una sola vez por proceso; este módulo
    # es el único que lo hace.
    trace.set_tracer_provider(_proveedor)
    instrumentador = LangChainInstrumentor()
    instrumentador.instrument(tracer_provider=_proveedor)
    yield
    instrumentador.uninstrument()


@pytest.fixture
def spans():
    _exportador.clear()
    return _exportador


def _por_nombre(lista, nombre):
    return [s for s in lista if s.name == nombre]


def _padre(lista, span):
    return next((s for s in lista if span.parent and s.context.span_id == span.parent.span_id), None)


async def test_el_span_del_rag_cuelga_de_la_herramienta_y_trae_los_documentos(spans):
    await buscar_documentacion.ainvoke({"consulta": "E4090", "k": 2})
    terminados = spans.get_finished_spans()

    [recuperar] = _por_nombre(terminados, "recuperar_documentacion")
    assert _padre(terminados, recuperar).name == "buscar_documentacion"
    atributos = dict(recuperar.attributes)
    assert atributos["openinference.span.kind"] == "RETRIEVER"
    assert atributos["input.value"] == "E4090"
    assert atributos["retrieval.documents.0.document.id"].startswith("0")
    assert "E4090" in atributos["retrieval.documents.0.document.content"]
    metadatos = json.loads(atributos["retrieval.documents.0.document.metadata"])
    assert {"fuente", "seccion", "rango_lexico", "rango_vectorial"} <= set(metadatos)

    [lexica] = _por_nombre(terminados, "busqueda_lexica")
    assert _padre(terminados, lexica) is recuperar


async def test_cada_turno_es_una_traza_con_la_conversacion_como_sesion(spans):
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    jobs = AlmacenJobs(redis, prefijo="obs")
    job = await jobs.crear("¿Qué código falla más en Bazar Pampa y qué porcentaje representa?")
    grafo = construir_grafo(modelos(supervisor_feliz()), checkpointer=InMemorySaver())

    assert await correr_job(grafo, jobs, job["job_id"]) == EstadoJob.DONE
    terminados = spans.get_finished_spans()

    [raiz] = _por_nombre(terminados, "tarea_nimbus")
    assert raiz.parent is None
    atributos = dict(raiz.attributes)
    assert atributos["openinference.span.kind"] == "AGENT"
    assert atributos["session.id"] == job["conversacion_id"]
    metadata = json.loads(atributos["metadata"])
    assert metadata["job_id"] == job["job_id"] and metadata["turno"] == 1

    # Todos los spans de la corrida son de la misma traza, con la misma sesión.
    assert {s.context.trace_id for s in terminados} == {raiz.context.trace_id}
    nombres = {s.name for s in terminados}
    assert {"supervisor", "investigador", "analista", "sintetizador", "recuperar_documentacion"} <= nombres
    # La búsqueda del investigador también cuelga de su herramienta, adentro del grafo.
    for recuperar in _por_nombre(terminados, "recuperar_documentacion"):
        assert _padre(terminados, recuperar).name == "buscar_documentacion"
    await redis.aclose()
