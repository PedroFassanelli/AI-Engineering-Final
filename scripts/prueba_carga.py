"""Prueba de carga: 5 peticiones concurrentes contra la API.

    python scripts/prueba_carga.py                 # 5 consultas en paralelo
    python scripts/prueba_carga.py --aprobar       # y aprueba sola la acción crítica
    python scripts/prueba_carga.py --url http://otra-api:8000 --rondas 2

Qué mide (del lado del cliente):

- **Latencia del POST**: cuánto tarda la API en devolver el `job_id`. Tiene
  que ser de milisegundos aunque el grafo tarde un minuto; si crece con la
  carga, algo está bloqueando el event loop.
- **Latencia de punta a punta**: desde el POST hasta que el job llega a un
  estado final (o a AWAITING_APPROVAL). Para los jobs con aprobación se
  informa aparte la fase posterior a la aprobación, porque la espera humana
  no es latencia del sistema.

El costo y la latencia p95 **por traza** no salen de acá: los calcula Phoenix
a partir de las trazas que manda el worker (ver README → Observabilidad). Cada
consulta abre una conversación nueva; el script guarda su `conversacion_id`,
que en Phoenix es el `session.id`, para poder cruzar una cosa con la otra.

El resultado completo queda en `resultados/carga-<fecha>.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

RAIZ = Path(__file__).resolve().parent.parent

#: Cinco consultas distintas, todas del dominio. La última pide una acción con
#: efectos: es la que dispara la pausa de aprobación humana.
CONSULTAS: list[str] = [
    "El comercio Bazar Pampa viene teniendo pagos fallidos. ¿Cuál es el código de error que más "
    "se repite, qué significa según la documentación y cómo se resuelve? Y cuantificalo: qué "
    "porcentaje de sus pagos y qué porcentaje del monto total representan los fallidos por ese código.",
    "¿Qué significa el error E4021 que tuvo Café Tortoni Centro y cómo se resuelve? ¿Qué porcentaje "
    "del monto total de sus pagos representa el pago fallido?",
    "Librería Ávila tuvo un pago fallido. ¿Qué indica su código de error según la documentación y "
    "qué porcentaje de sus pagos representa?",
    "Bazar Pampa tuvo un pago fallido por E4290. ¿Qué significa ese código, es reintentable y cómo "
    "debería manejarlo el comercio?",
    "Viajes Aconcagua tuvo pagos fallidos por el código E5030. Explicame qué significa según la "
    "documentación y programá el reintento de esos pagos fallidos.",
]

FINALES = {"DONE", "FAILED"}


def percentil(valores: list[float], p: float) -> float:
    """Percentil con interpolación lineal (el mismo método que `numpy.percentile`)."""
    ordenados = sorted(valores)
    if not ordenados:
        return float("nan")
    k = (len(ordenados) - 1) * p / 100
    piso = int(k)
    techo = min(piso + 1, len(ordenados) - 1)
    return ordenados[piso] + (ordenados[techo] - ordenados[piso]) * (k - piso)


async def esperar(cliente: httpx.AsyncClient, job_id: str, hasta: set[str], intervalo: float) -> dict[str, Any]:
    while True:
        job = (await cliente.get(f"/tasks/{job_id}")).raise_for_status().json()
        if job["estado"] in hasta:
            return job
        await asyncio.sleep(intervalo)


async def una_peticion(
    cliente: httpx.AsyncClient, indice: int, pregunta: str, *, aprobar: bool, intervalo: float
) -> dict[str, Any]:
    t0 = time.perf_counter()
    respuesta = (await cliente.post("/tasks", json={"pregunta": pregunta})).raise_for_status()
    t_post = time.perf_counter() - t0
    job_id = respuesta.json()["job_id"]
    conversacion_id = respuesta.json()["conversacion_id"]
    print(f"  [{indice}] encolado {job_id}  ({t_post * 1000:.0f} ms)")

    job = await esperar(cliente, job_id, FINALES | {"AWAITING_APPROVAL"}, intervalo)
    t_primera_fase = time.perf_counter() - t0
    registro: dict[str, Any] = {
        "indice": indice,
        "job_id": job_id,
        "conversacion_id": conversacion_id,
        "pregunta": pregunta,
        "latencia_post_s": round(t_post, 4),
        "latencia_hasta_estado_s": round(t_primera_fase, 3),
        "estado_tras_primera_fase": job["estado"],
    }

    if job["estado"] == "AWAITING_APPROVAL":
        print(f"  [{indice}] ⏸  pide aprobación: {job['aprobacion_pendiente']['resumen']}")
        if aprobar:
            t1 = time.perf_counter()
            (await cliente.post(
                f"/tasks/{job_id}/approve",
                json={"aprobado": True, "revisor": "prueba-de-carga", "comentario": "aprobación automática del script"},
            )).raise_for_status()
            job = await esperar(cliente, job_id, FINALES, intervalo)
            registro["latencia_reanudacion_s"] = round(time.perf_counter() - t1, 3)

    registro.update(
        estado_final=job["estado"],
        duracion_worker_s=job.get("duracion_segundos"),
        duracion_reanudacion_worker_s=job.get("duracion_reanudacion_segundos"),
        nodos=job.get("nodos", []),
        error=job.get("error"),
        respuesta=job.get("respuesta"),
        resultado_accion=job.get("resultado_accion"),
    )
    simbolo = {"DONE": "✔", "FAILED": "✘", "AWAITING_APPROVAL": "⏸"}.get(job["estado"], "?")
    print(f"  [{indice}] {simbolo} {job['estado']} en {t_primera_fase:.1f} s")
    return registro


async def correr(url: str, concurrencia: int, rondas: int, aprobar: bool, intervalo: float) -> dict[str, Any]:
    consultas = [CONSULTAS[i % len(CONSULTAS)] for i in range(concurrencia)]
    registros: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=url, timeout=30) as cliente:
        (await cliente.get("/health")).raise_for_status()
        for ronda in range(1, rondas + 1):
            print(f"\nRonda {ronda}: {concurrencia} peticiones concurrentes contra {url}")
            t0 = time.perf_counter()
            resultados = await asyncio.gather(
                *(una_peticion(cliente, i + 1, q, aprobar=aprobar, intervalo=intervalo) for i, q in enumerate(consultas))
            )
            for r in resultados:
                r["ronda"] = ronda
            registros.extend(resultados)
            print(f"Ronda {ronda} completa en {time.perf_counter() - t0:.1f} s")

    posts = [r["latencia_post_s"] for r in registros]
    e2e = [r["latencia_hasta_estado_s"] for r in registros]
    resumen = {
        "url": url,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "peticiones": len(registros),
        "concurrencia": concurrencia,
        "estados_finales": {e: sum(r["estado_final"] == e for r in registros) for e in {r["estado_final"] for r in registros}},
        "post_ms": {"p50": round(percentil(posts, 50) * 1000, 1), "p95": round(percentil(posts, 95) * 1000, 1), "max": round(max(posts) * 1000, 1)},
        "punta_a_punta_s": {"p50": round(percentil(e2e, 50), 2), "p95": round(percentil(e2e, 95), 2), "max": round(max(e2e), 2), "media": round(statistics.mean(e2e), 2)},
    }
    return {"resumen": resumen, "peticiones": registros}


def imprimir_resumen(datos: dict[str, Any]) -> None:
    r = datos["resumen"]
    print("\n" + "=" * 78)
    print(f"{'#':>2}  {'job_id':<32}  {'estado':<17} {'e2e (s)':>8} {'worker (s)':>10}  nodos")
    for p in datos["peticiones"]:
        print(
            f"{p['indice']:>2}  {p['job_id']:<32}  {p['estado_final']:<17} "
            f"{p['latencia_hasta_estado_s']:>8.1f} {p['duracion_worker_s'] or 0:>10.1f}  {len(p['nodos'])}"
        )
    print("-" * 78)
    print(f"POST (encolar):  p50 {r['post_ms']['p50']} ms · p95 {r['post_ms']['p95']} ms · máx {r['post_ms']['max']} ms")
    print(f"Punta a punta:   p50 {r['punta_a_punta_s']['p50']} s · p95 {r['punta_a_punta_s']['p95']} s · máx {r['punta_a_punta_s']['max']} s")
    print(f"Estados finales: {r['estados_finales']}")


def main() -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrencia", type=int, default=5)
    parser.add_argument("--rondas", type=int, default=1)
    parser.add_argument("--aprobar", action="store_true", help="Aprueba automáticamente las acciones críticas.")
    parser.add_argument("--intervalo", type=float, default=1.0, help="Segundos entre consultas de polling.")
    args = parser.parse_args()

    datos = asyncio.run(correr(args.url, args.concurrencia, args.rondas, args.aprobar, args.intervalo))
    imprimir_resumen(datos)

    destino = RAIZ / "resultados" / f"carga-{datetime.now():%Y%m%d-%H%M%S}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultado completo: {destino.relative_to(RAIZ)}")
    return 0 if all(p["estado_final"] != "FAILED" for p in datos["peticiones"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
