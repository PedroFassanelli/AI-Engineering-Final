"""Observabilidad: trazas de cada ejecución hacia Arize Phoenix (OpenInference).

Dos capas que se complementan:

1. **Auto-instrumentación de LangChain** (`LangChainInstrumentor`). Cada nodo del
   grafo, cada subgrafo ReAct, cada herramienta y cada llamada al LLM se vuelve
   un span, con los tokens de entrada y salida y el nombre del modelo. Con eso
   Phoenix calcula solo el **costo** de cada llamada y de cada traza.
2. **Spans propios** con el decorador `@trazar`. Envuelven lo que LangChain no
   ve: la tarea del worker que corre el grafo (la raíz de cada traza), la
   reanudación después de la aprobación humana y la escritura en la base.

Todas las trazas de una misma **conversación** llevan `session.id =
conversacion_id` (`contexto_de_job`): en la vista *Sessions* de Phoenix cada
conversación es una sesión, con un turno por traza (y dos trazas para un turno
con aprobación humana, antes y después de la pausa). El `job_id`, el turno y la
fase van en `metadata`, filtrables desde el buscador.

La tercera capa es la del RAG (`app/rag/hibrido.py`): spans `RETRIEVER` y
`EMBEDDING` con los documentos recuperados y de qué mitad del híbrido vino cada uno.

Si `TRAZAS_ACTIVAS=0` (los tests) o Phoenix no está instalado, todo esto es un
no-op: `@trazar` usa el tracer global de OpenTelemetry, que sin un provider
configurado no exporta nada.
"""

from __future__ import annotations

import functools
import json
import logging
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterator, ParamSpec, TypeVar

from openinference.instrumentation import get_attributes_from_context, using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app import config

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

NOMBRE_TRACER = "nimbus-intelligence"

#: El `TracerProvider` de Phoenix, una vez inicializado. Uno por proceso.
_proveedor: Any = None


def iniciar_observabilidad(servicio: str) -> bool:
    """Registra el exportador de Phoenix e instrumenta LangChain.

    Idempotente: el worker y los tests pueden llamarla más de una vez. Devuelve
    `True` si quedó activa.
    """
    global _proveedor
    if _proveedor is not None:
        return True
    if not config.TRAZAS_ACTIVAS:
        log.info("Observabilidad desactivada (TRAZAS_ACTIVAS=0).")
        return False

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    _proveedor = register(
        project_name=config.PHOENIX_PROJECT_NAME,
        endpoint=config.PHOENIX_COLLECTOR_ENDPOINT,
        # Batch: exportar span por span bloquearía el event loop en cada nodo.
        batch=True,
        set_global_tracer_provider=True,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=_proveedor)
    log.info(
        "Trazas de %s → Phoenix %s (proyecto %r).",
        servicio,
        config.PHOENIX_COLLECTOR_ENDPOINT,
        config.PHOENIX_PROJECT_NAME,
    )
    return True


def cerrar_observabilidad() -> None:
    """Vacía el buffer de spans antes de que termine el proceso.

    Sin esto, las últimas trazas del batch se pierden al apagar el worker.
    """
    global _proveedor
    if _proveedor is None:
        return
    try:
        _proveedor.force_flush()
        _proveedor.shutdown()
    finally:
        _proveedor = None


@contextmanager
def contexto_de_job(job_id: str, *, conversacion_id: str | None = None, **metadata: Any) -> Iterator[None]:
    """Todo span creado adentro queda asociado al job y a su conversación.

    `session_id` (la conversación) agrupa los turnos en la vista *Sessions*;
    `metadata` se puede filtrar desde el buscador de Phoenix (p. ej.
    `metadata['fase'] == 'reanudacion'` o `metadata['turno'] == 2`).
    """
    sesion = conversacion_id or job_id
    with using_attributes(
        session_id=sesion,
        metadata={"job_id": job_id, "conversacion_id": sesion, **metadata},
        tags=["nimbus", str(metadata.get("fase", "ejecucion"))],
    ):
        yield


def contexto_padre() -> Any:
    """El contexto OTel del span de LangChain que está corriendo, o `None`.

    El instrumentador de LangChain crea sus spans desde callbacks y **no** los
    deja como span "actual" de OpenTelemetry. Sin esto, un span propio creado
    dentro de una herramienta (la búsqueda del RAG) o de un nodo (la escritura
    en la base) colgaría de la raíz de la traza, no de la herramienta o el nodo
    que lo originó. `get_current_span` resuelve el span de LangChain en curso.
    """
    try:
        from openinference.instrumentation.langchain import get_current_span

        span = get_current_span()
    except Exception:  # noqa: BLE001 - sin instrumentación, no hay padre que buscar
        return None
    return trace.set_span_in_context(span) if span is not None else None


def _como_texto(valor: Any) -> str:
    if isinstance(valor, str):
        return valor
    try:
        return json.dumps(valor, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(valor)


def trazar(
    nombre: str | None = None,
    *,
    tipo: OpenInferenceSpanKindValues = OpenInferenceSpanKindValues.CHAIN,
    entrada: Callable[..., Any] | None = None,
    salida: Callable[[Any], Any] | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decora una corutina para que cada llamada sea un span de OpenInference.

    - `tipo`: el *span kind* que muestra Phoenix (CHAIN, TOOL, AGENT...).
    - `entrada` / `salida`: qué registrar como `input.value` / `output.value`.
      Por defecto, los argumentos y el valor de retorno; se pueden recortar
      para no mandar a Phoenix objetos enormes o que no son datos (el `ctx` de
      Arq, por ejemplo).

    Una excepción se registra en el span (con estado ERROR) y se vuelve a
    lanzar: el decorador observa, no decide.
    """

    def decorador(funcion: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        nombre_span = nombre or funcion.__name__

        @functools.wraps(funcion)
        async def envuelta(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = trace.get_tracer(NOMBRE_TRACER)
            with tracer.start_as_current_span(
                nombre_span,
                context=contexto_padre(),
                attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: tipo.value},
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                datos_entrada = entrada(*args, **kwargs) if entrada else {"args": args, "kwargs": kwargs}
                # `using_attributes` solo lo leen los instrumentadores de OpenInference:
                # a los spans propios hay que copiarles sesión, metadata y tags a mano.
                span.set_attributes(dict(get_attributes_from_context()))
                span.set_attribute(SpanAttributes.INPUT_VALUE, _como_texto(datos_entrada))
                try:
                    resultado = await funcion(*args, **kwargs)
                except BaseException as error:  # CancelledError incluido: también es un final
                    span.record_exception(error)
                    span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {error}"))
                    raise
                datos_salida = salida(resultado) if salida else resultado
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, _como_texto(datos_salida))
                span.set_status(Status(StatusCode.OK))
                return resultado

        return envuelta

    return decorador
