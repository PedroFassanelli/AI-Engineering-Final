"""Human-in-the-loop: la pausa obligatoria antes de cualquier efecto secundario.

El grafo de la Pre-entrega 6 solo leía (documentación y pagos). Acá se suma la
primera acción que **escribe**: programar el reintento de pagos fallidos, que
en un sistema real dispara nuevos cobros sobre la tarjeta de un cliente. Eso es
lo que el sistema identifica como tarea crítica.

    sintetizador ──► planificador ──┬──► END                  (solo consulta)
                                    │
                                    └──► aprobacion_humana ──┬──► ejecutar_accion ──► END
                                         (interrupt: el grafo │
                                          queda pausado en    └──► END   (rechazada)
                                          Redis hasta el
                                          POST /approve)

Cómo funciona la pausa:

1. `aprobacion_humana` llama a `interrupt(...)` con todo lo que la persona
   necesita para decidir (qué pagos, cuánto, por qué es crítica).
2. LangGraph guarda el checkpoint en Redis (`AsyncRedisSaver`) y la corrida
   termina. El worker ve el `__interrupt__`, pasa el job a
   `AWAITING_APPROVAL` y **libera el proceso**: no hay ningún hilo esperando.
3. `POST /tasks/{id}/approve` encola la reanudación. El worker retoma el grafo
   con `Command(resume=<decisión>)` desde el checkpoint, en el mismo
   `thread_id` (= la conversación). Puede ser otro proceso, o el mismo después
   de un reinicio: el estado no vive en memoria.

Defensa en profundidad: `ejecutar_accion` vuelve a verificar que exista una
aprobación positiva antes de escribir, aunque el ruteo ya lo garantice.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Final, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt
from openinference.semconv.trace import OpenInferenceSpanKindValues

from app import config
from app.fuentes import db
from app.observability import trazar
from app.state import AccionPropuesta, Aprobacion, EstadoOrquestador, ResultadoAccion

#: Acciones con efectos secundarios y qué efecto tienen. Toda acción de esta
#: tabla es crítica por definición: no hay forma de ejecutarla sin aprobación.
ACCIONES_CON_EFECTOS: Final[dict[str, str]] = {
    "programar_reintentos": (
        "genera nuevos intentos de cobro sobre los medios de pago de los clientes "
        "(escribe en reintentos_programados)"
    ),
}


#: La tabla de códigos de la documentación (la misma que lee el investigador).
DOC_ERRORES: Final[str] = "03-manejo-de-errores.md"
_FILA_CODIGO = re.compile(r"^\|\s*`(E\d{4})`\s*\|[^|]*\|\s*(sí|si|no)\s*\|", re.MULTILINE)


@lru_cache(maxsize=1)
def codigos_reintentables() -> dict[str, bool]:
    """`código -> reintentable`, leído de la tabla de la documentación.

    Es un dato que la persona que aprueba tiene que ver: reintentar un código
    que la documentación marca como no reintentable no lo convierte en
    aprobado, solo genera otro rechazo (y otro cargo de procesamiento).
    """
    ruta = config.DIR_DOCS / DOC_ERRORES
    if not ruta.exists():
        return {}
    return {
        codigo: valor != "no"
        for codigo, valor in _FILA_CODIGO.findall(ruta.read_text(encoding="utf-8"))
    }


class AccionNoAprobada(RuntimeError):
    """Se intentó ejecutar una acción sin una aprobación positiva registrada."""


# --- Datos que respaldan una acción -----------------------------------------


def pagos_consultados(estado: EstadoOrquestador) -> dict[int, dict[str, Any]]:
    """Los pagos que el investigador **realmente** trajo de la base, por comercio.

    Es la única fuente válida para una acción: lo que no pasó por
    `consultar_pagos` no se puede reintentar, lo diga quien lo diga.
    """
    nombres: dict[int, str] = {}
    pagos: dict[int, dict[str, dict[str, Any]]] = {}
    for contribucion in estado.get("contribuciones", []):
        if contribucion["agente"] != "investigador":
            continue
        for evidencia in contribucion["evidencia"]:
            resultado = evidencia["resultado"]
            if not isinstance(resultado, dict):
                continue
            if evidencia["herramienta"] == "buscar_comercio":
                for comercio in resultado.get("comercios", []):
                    nombres[int(comercio["comercio_id"])] = str(comercio["nombre"])
            elif evidencia["herramienta"] == "consultar_pagos" and "pagos" in resultado:
                destino = pagos.setdefault(int(resultado["comercio_id"]), {})
                for pago in resultado["pagos"]:
                    destino[str(pago["pago_id"])] = pago
    return {
        comercio_id: {
            "nombre": nombres.get(comercio_id, f"comercio {comercio_id}"),
            "pagos": list(por_id.values()),
        }
        for comercio_id, por_id in pagos.items()
    }


def evaluar_criticidad(accion: AccionPropuesta) -> list[str]:
    """Por qué una acción es crítica. Lista vacía = no requiere aprobación.

    Las razones se le muestran a quien aprueba: no alcanza con decir "es
    crítica", hay que decir por qué.
    """
    razones: list[str] = []
    if accion["tipo"] in ACCIONES_CON_EFECTOS:
        razones.append(f"Efecto secundario: {ACCIONES_CON_EFECTOS[accion['tipo']]}.")
    if accion["monto_total_centavos"] >= config.UMBRAL_MONTO_ALTO_CENTAVOS:
        razones.append(
            f"Monto alto: {formatear_monto(accion['monto_total_centavos'], accion['moneda'])} "
            f"(umbral {formatear_monto(config.UMBRAL_MONTO_ALTO_CENTAVOS, accion['moneda'])})."
        )
    if len(accion["pago_ids"]) >= config.UMBRAL_PAGOS_MASIVO:
        razones.append(f"Acción masiva: {len(accion['pago_ids'])} pagos de una vez.")
    no_reintentables = sorted(
        {c for c in accion.get("codigos_error", []) if codigos_reintentables().get(c) is False}
    )
    if no_reintentables:
        razones.append(
            f"ADVERTENCIA: {', '.join(no_reintentables)} figura como NO reintentable en "
            f"{DOC_ERRORES}. El reintento probablemente vuelva a fallar."
        )
    return razones


def es_critica(accion: AccionPropuesta | None) -> bool:
    return accion is not None and bool(accion["criticidad"])


def validar_propuesta(
    tipo: str, pago_ids: list[str], motivo: str, estado: EstadoOrquestador
) -> AccionPropuesta | None:
    """Convierte la propuesta del modelo en una acción concreta, o en nada.

    El modelo eligió *qué* pagos; acá se decide cuáles de esos son válidos:
    tienen que haber sido consultados, estar fallidos y ser todos del mismo
    comercio (una aprobación = un comercio). El monto lo suma el código.
    """
    if tipo not in ACCIONES_CON_EFECTOS:
        return None

    consultados = pagos_consultados(estado)
    indice = {
        p["pago_id"]: (comercio_id, p)
        for comercio_id, datos in consultados.items()
        for p in datos["pagos"]
    }

    validos: list[dict[str, Any]] = []
    descartados: list[str] = []
    comercio_id: int | None = None
    for pago_id in dict.fromkeys(pago_ids):  # sin duplicados, en orden
        if pago_id not in indice:
            descartados.append(f"{pago_id}: no figura en los pagos consultados.")
            continue
        su_comercio, pago = indice[pago_id]
        if pago.get("estado") != "fallido":
            descartados.append(f"{pago_id}: está {pago.get('estado')}, no fallido.")
            continue
        if comercio_id is not None and su_comercio != comercio_id:
            descartados.append(f"{pago_id}: es de otro comercio ({su_comercio}).")
            continue
        comercio_id = su_comercio
        validos.append(pago)

    if comercio_id is None:
        return None

    accion = AccionPropuesta(
        tipo=tipo,  # type: ignore[typeddict-item]
        comercio_id=comercio_id,
        comercio=consultados[comercio_id]["nombre"],
        pago_ids=[p["pago_id"] for p in validos],
        monto_total_centavos=sum(int(p["monto_centavos"]) for p in validos),
        moneda=str(validos[0].get("moneda", "")),
        codigos_error=sorted({str(p["codigo_error"]) for p in validos if p.get("codigo_error")}),
        motivo=motivo,
        criticidad=[],
        descartados=descartados,
    )
    accion["criticidad"] = evaluar_criticidad(accion)
    return accion


def formatear_monto(centavos: int, moneda: str) -> str:
    entero, resto = divmod(centavos, 100)
    return f"{moneda} {entero:,}.{resto:02d}".replace(",", "_").replace(".", ",").replace("_", ".")


# --- Nodos ------------------------------------------------------------------


def solicitud_de_aprobacion(accion: AccionPropuesta) -> dict[str, Any]:
    """Lo que viaja en el `interrupt` y le muestra la API a quien aprueba."""
    return {
        "tipo": accion["tipo"],
        "resumen": (
            f"Programar el reintento de {len(accion['pago_ids'])} pago(s) fallido(s) de "
            f"{accion['comercio']} por {formatear_monto(accion['monto_total_centavos'], accion['moneda'])}."
        ),
        "comercio_id": accion["comercio_id"],
        "comercio": accion["comercio"],
        "pago_ids": accion["pago_ids"],
        "monto_total_centavos": accion["monto_total_centavos"],
        "moneda": accion["moneda"],
        "codigos_error": accion["codigos_error"],
        "motivo": accion["motivo"],
        "criticidad": accion["criticidad"],
        "descartados": accion["descartados"],
        "como_responder": "POST /tasks/{job_id}/approve  {\"aprobado\": true|false, \"revisor\": \"...\", \"comentario\": \"...\"}",
    }


def normalizar_respuesta(respuesta: Any) -> Aprobacion:
    """La respuesta humana, validada. Cualquier cosa que no sea un `aprobado: true`
    explícito cuenta como rechazo: el silencio no autoriza cobros."""
    if not isinstance(respuesta, dict):
        respuesta = {}
    return Aprobacion(
        aprobado=respuesta.get("aprobado") is True,
        revisor=str(respuesta.get("revisor") or "desconocido"),
        comentario=respuesta.get("comentario"),
        decidido_en=str(respuesta.get("decidido_en") or _ahora()),
    )


async def aprobacion_humana(estado: EstadoOrquestador) -> dict[str, Any]:
    """Pausa el grafo hasta recibir la decisión de una persona.

    Ojo con la semántica de `interrupt`: al reanudar, el nodo se ejecuta **de
    nuevo desde el principio** y `interrupt` devuelve la respuesta en vez de
    pausar. Por eso acá no hay nada con efectos antes del `interrupt`.
    """
    accion = estado["accion_propuesta"]
    assert accion is not None, "aprobacion_humana sin acción propuesta: error de ruteo"

    aprobacion = normalizar_respuesta(interrupt(solicitud_de_aprobacion(accion)))

    actualizacion: dict[str, Any] = {"aprobacion": aprobacion}
    if not aprobacion["aprobado"]:
        detalle = (
            f"Acción rechazada por {aprobacion['revisor']}"
            + (f": {aprobacion['comentario']}" if aprobacion["comentario"] else "")
            + ". No se programó ningún reintento."
        )
        actualizacion["resultado_accion"] = ResultadoAccion(ejecutada=False, detalle=detalle, pago_ids=[])
        actualizacion["respuesta_final"] = _con_nota(estado, detalle)
        # Queda en la conversación: en el turno siguiente, "¿al final se
        # reintentaron?" tiene respuesta en el historial.
        actualizacion["messages"] = [AIMessage(content=detalle, name="sistema")]
    return actualizacion


@trazar(
    "programar_reintentos",
    tipo=OpenInferenceSpanKindValues.TOOL,
    entrada=lambda accion, aprobacion, job_id: {
        "job_id": job_id,
        "pago_ids": accion["pago_ids"],
        "aprobado_por": aprobacion["revisor"],
    },
)
async def _programar_reintentos(accion: AccionPropuesta, aprobacion: Aprobacion, job_id: str) -> list[str]:
    """La única escritura del sistema, en su propio span (tipo TOOL)."""
    return await db.programar_reintentos(
        accion["pago_ids"],
        job_id=job_id,
        aprobado_por=aprobacion["revisor"],
        comentario=aprobacion["comentario"],
        creado_en=_ahora(),
    )


async def ejecutar_accion(estado: EstadoOrquestador, config: RunnableConfig) -> dict[str, Any]:
    """Ejecuta la acción aprobada. Solo se llega acá con `aprobado=True`."""
    accion = estado.get("accion_propuesta")
    aprobacion = estado.get("aprobacion")
    if accion is None or aprobacion is None or not aprobacion["aprobado"]:
        raise AccionNoAprobada("ejecutar_accion sin una aprobación positiva registrada.")

    # La idempotencia es por turno (job), no por conversación: el mismo pago
    # puede reintentarse en un turno posterior si alguien lo vuelve a aprobar.
    configurable = config.get("configurable", {})
    job_id = str(configurable.get("job_id") or configurable.get("thread_id") or "sin-job")
    nuevos = await _programar_reintentos(accion, aprobacion, job_id)

    ya_estaban = [p for p in accion["pago_ids"] if p not in nuevos]
    detalle = (
        f"Acción ejecutada (aprobada por {aprobacion['revisor']}): se programó el reintento de "
        f"{', '.join(accion['pago_ids'])} por "
        f"{formatear_monto(accion['monto_total_centavos'], accion['moneda'])}."
    )
    if ya_estaban:
        detalle += f" Ya estaban programados por este mismo job: {', '.join(ya_estaban)}."
    return {
        "resultado_accion": ResultadoAccion(ejecutada=True, detalle=detalle, pago_ids=accion["pago_ids"]),
        "respuesta_final": _con_nota(estado, detalle),
        "messages": [AIMessage(content=detalle, name="sistema")],
    }


# --- Aristas ----------------------------------------------------------------


def ruta_tras_planificador(estado: EstadoOrquestador) -> Literal["aprobacion_humana", "__end__"]:
    """Las acciones críticas pasan por la aprobación; las consultas terminan."""
    return "aprobacion_humana" if es_critica(estado.get("accion_propuesta")) else END


def ruta_tras_aprobacion(estado: EstadoOrquestador) -> Literal["ejecutar_accion", "__end__"]:
    aprobacion = estado.get("aprobacion")
    return "ejecutar_accion" if aprobacion and aprobacion["aprobado"] else END


# --- Utilidades -------------------------------------------------------------


def _con_nota(estado: EstadoOrquestador, nota: str) -> str:
    base = estado.get("respuesta_final", "").rstrip()
    return f"{base}\n\n{nota}" if base else nota


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
