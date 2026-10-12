"""Pruebas del sistema completo contra la API real (Docker), con trazas en Phoenix.

    python scripts/pruebas_sistema.py                  # todo, con reinicio del stack en el medio
    python scripts/pruebas_sistema.py --sin-reinicio   # sin reiniciar contenedores

Seis escenarios, cada uno una conversación (= una sesión en Phoenix):

  1. rag_documentacion      Solo documentación: el investigador usa el RAG híbrido.
  2. multipaso              RAG + base de pagos + cálculos: investigador → analista.
  3. conversacion           Dos turnos; el segundo ("¿y es reintentable?") depende del primero.
  4. hitl_aprobado          Pide reintentar pagos: pausa, se aprueba, se ejecuta.
  5. hitl_rechazado         Pide reintentar un código NO reintentable: pausa con advertencia, se rechaza.
  6. comercio_inexistente   Un comercio que no existe: la herramienta devuelve error + sugerencia.

Y el punto que pide la consigna sobre persistencia: entre la fase 1 y la 2 se
**reinician Redis, la API y el worker** (`docker compose restart`). Después del
reinicio:

- el turno 2 de la conversación se responde con el contexto del turno 1, que
  solo existe en el checkpoint de LangGraph guardado en Redis;
- las dos acciones que quedaron en AWAITING_APPROVAL se aprueban/rechazan y el
  grafo se reanuda desde el `interrupt`, en un worker que no es el que pausó.

Cada escenario tiene chequeos **estructurales** (estado final, nodos que
pasaron, fuente citada, acción ejecutada o no): no se compara el texto del LLM
palabra por palabra. El resultado queda en `resultados/pruebas-sistema-<fecha>.json`,
con el `conversacion_id` de cada escenario para buscarlo en Phoenix (Sessions).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx

RAIZ = Path(__file__).resolve().parent.parent
FINALES = {"DONE", "FAILED"}
PAUSA = "AWAITING_APPROVAL"

Chequeo = tuple[str, Callable[[dict[str, Any]], bool]]


def _nodos(job: dict[str, Any]) -> list[str]:
    return job.get("nodos") or []


def _respuesta(job: dict[str, Any]) -> str:
    return job.get("respuesta") or ""


@dataclass
class Escenario:
    nombre: str
    pregunta: str
    chequeos: list[Chequeo]
    #: Si se espera una pausa de aprobación: qué decidir después del reinicio.
    decision: dict[str, Any] | None = None
    chequeos_pausa: list[Chequeo] = field(default_factory=list)
    #: Segundo turno de la misma conversación (después del reinicio).
    turno_2: str | None = None
    chequeos_turno_2: list[Chequeo] = field(default_factory=list)


ESCENARIOS: list[Escenario] = [
    Escenario(
        "rag_documentacion",
        "Estoy integrando nimbus-sdk. ¿Cómo pruebo la integración sin que se mueva plata de verdad, "
        "y qué tengo que revisar antes de salir a producción?",
        [
            ("DONE", lambda j: j["estado"] == "DONE"),
            ("pasó por el investigador", lambda j: "investigador" in _nodos(j)),
            ("no hizo falta el analista", lambda j: "analista" not in _nodos(j)),
            ("cita la guía de inicio", lambda j: "01-guia-de-inicio.md" in _respuesta(j)),
        ],
    ),
    Escenario(
        "multipaso",
        "El comercio Bazar Pampa viene teniendo pagos fallidos. ¿Cuál es el código de error que más "
        "se repite, qué significa según la documentación y cómo se resuelve? Y cuantificalo: qué "
        "porcentaje de sus pagos y qué porcentaje del monto total representan los fallidos por ese código.",
        [
            ("DONE", lambda j: j["estado"] == "DONE"),
            ("investigador y analista", lambda j: {"investigador", "analista"} <= set(_nodos(j))),
            ("identifica E4090", lambda j: "E4090" in _respuesta(j)),
            ("cita el catálogo de errores", lambda j: "03-manejo-de-errores.md" in _respuesta(j)),
            ("da el porcentaje de pagos (42,86 %)", lambda j: "42,86" in _respuesta(j) or "42.86" in _respuesta(j)),
        ],
    ),
    Escenario(
        "conversacion",
        "¿Qué código de error tuvo el pago fallido de Café Tortoni Centro y qué significa?",
        [
            ("DONE", lambda j: j["estado"] == "DONE"),
            ("identifica E4021", lambda j: "E4021" in _respuesta(j)),
        ],
        turno_2="¿Y ese error es reintentable? ¿Qué le recomendarías al comercio?",
        chequeos_turno_2=[
            ("DONE", lambda j: j["estado"] == "DONE"),
            ("es el turno 2", lambda j: j.get("turno") == 2),
            ("resolvió 'ese error' = E4021", lambda j: "E4021" in _respuesta(j)),
            ("dice que no es reintentable", lambda j: "no es reintentable" in _respuesta(j).lower()
             or "no reintentable" in _respuesta(j).lower()),
        ],
    ),
    Escenario(
        "hitl_aprobado",
        "Viajes Aconcagua tuvo pagos fallidos por el código E5030. Explicame qué significa según la "
        "documentación y programá el reintento de esos pagos fallidos.",
        [("pausa para aprobación", lambda j: j["estado"] == PAUSA)],
        decision={"aprobado": True, "revisor": "operaciones", "comentario": "E5030 es reintentable"},
        chequeos_pausa=[
            ("propone pay_10502 y pay_10503", lambda j: sorted(j["aprobacion_pendiente"]["pago_ids"]) == ["pay_10502", "pay_10503"]),
            ("explica por qué es crítica", lambda j: any("Efecto secundario" in c for c in j["aprobacion_pendiente"]["criticidad"])),
        ],
        chequeos_turno_2=[],
    ),
    Escenario(
        "hitl_rechazado",
        "Bazar Pampa tiene pagos fallidos por E4090. Reintentá esos pagos fallidos.",
        [("pausa para aprobación", lambda j: j["estado"] == PAUSA)],
        decision={"aprobado": False, "revisor": "operaciones", "comentario": "E4090 no es reintentable"},
        chequeos_pausa=[
            ("advierte que E4090 no es reintentable",
             lambda j: any("ADVERTENCIA" in c and "E4090" in c for c in j["aprobacion_pendiente"]["criticidad"])),
            ("propone los 3 pagos E4090",
             lambda j: sorted(j["aprobacion_pendiente"]["pago_ids"]) == ["pay_10402", "pay_10403", "pay_10405"]),
            ("si la respuesta lista pagos, lista los 3",
             lambda j: "pay_" not in _respuesta(j) or all(p in _respuesta(j) for p in ("pay_10402", "pay_10403", "pay_10405"))),
        ],
    ),
    Escenario(
        "comercio_inexistente",
        "¿Qué pagos fallidos tuvo el comercio Ferretería Los Andes y por qué fallaron?",
        [
            ("DONE (no FAILED)", lambda j: j["estado"] == "DONE"),
            ("no inventa pagos", lambda j: "pay_" not in _respuesta(j)),
        ],
    ),
]


# --- Cliente ---------------------------------------------------------------------


async def esperar(cliente: httpx.AsyncClient, job_id: str, hasta: set[str], limite: float = 300) -> dict[str, Any]:
    fin = time.monotonic() + limite
    while time.monotonic() < fin:
        job = (await cliente.get(f"/tasks/{job_id}")).raise_for_status().json()
        if job["estado"] in hasta:
            return job
        await asyncio.sleep(1.5)
    raise TimeoutError(f"{job_id} no llegó a {hasta} en {limite} s")


async def turno(cliente: httpx.AsyncClient, pregunta: str, conversacion_id: str | None = None) -> dict[str, Any]:
    cuerpo: dict[str, Any] = {"pregunta": pregunta}
    if conversacion_id:
        cuerpo["conversacion_id"] = conversacion_id
    t0 = time.perf_counter()
    encolado = (await cliente.post("/tasks", json=cuerpo)).raise_for_status().json()
    job = await esperar(cliente, encolado["job_id"], FINALES | {PAUSA})
    job["_e2e_s"] = round(time.perf_counter() - t0, 2)
    return job


def evaluar(chequeos: list[Chequeo], job: dict[str, Any]) -> list[dict[str, Any]]:
    salida = []
    for nombre, condicion in chequeos:
        try:
            ok = bool(condicion(job))
        except Exception:  # noqa: BLE001 - un chequeo que explota es un chequeo que falla
            ok = False
        salida.append({"chequeo": nombre, "ok": ok})
    return salida


def resumen_job(job: dict[str, Any]) -> dict[str, Any]:
    claves = (
        "job_id", "conversacion_id", "turno", "estado", "pregunta", "nodos", "respuesta",
        "aprobacion_pendiente", "aprobacion", "resultado_accion", "error",
        "duracion_segundos", "duracion_reanudacion_segundos", "_e2e_s",
    )
    return {k: job.get(k) for k in claves}


# --- Reinicio ----------------------------------------------------------------------


async def reiniciar_stack(cliente: httpx.AsyncClient) -> float:
    """`docker compose restart redis api worker` y espera a que el worker vuelva con el RAG."""
    print("\n↻ Reiniciando redis, api y worker (docker compose restart)...")
    await asyncio.sleep(2)  # AOF: fsync cada segundo
    t0 = time.monotonic()
    proceso = await asyncio.create_subprocess_exec(
        "docker", "compose", "restart", "redis", "api", "worker", cwd=str(RAIZ),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    if await proceso.wait() != 0:
        raise RuntimeError("docker compose restart falló")
    # El worker vuelve a marcar el modo del RAG al arrancar; mientras tanto, la
    # API puede no responder o responder `sin worker` (la clave quedó de antes,
    # así que se espera a que el worker haya registrado su arranque en los logs).
    while time.monotonic() - t0 < 180:
        try:
            salud = (await cliente.get("/health")).json()
            logs = subprocess.run(
                ["docker", "compose", "logs", "--since", f"{int(time.monotonic() - t0) + 5}s", "worker"],
                cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", errors="replace",
            ).stdout
            if salud.get("rag") not in (None, "sin worker") and "Worker listo" in logs:
                segundos = time.monotonic() - t0
                print(f"✓ Stack de vuelta en {segundos:.0f} s: {salud}")
                return segundos
        except httpx.HTTPError:
            pass
        await asyncio.sleep(2)
    raise TimeoutError("El stack no volvió en 3 minutos")


# --- Corrida ---------------------------------------------------------------------------


async def correr(url: str, reiniciar: bool, phoenix: str = "http://localhost:6006") -> dict[str, Any]:
    registros: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient(base_url=url, timeout=30) as cliente:
        salud = (await cliente.get("/health")).raise_for_status().json()
        print(f"API {url}: {salud}")
        # Sin Phoenix arriba, las pruebas corren igual pero sus trazas se pierden.
        try:
            (await cliente.get(f"{phoenix}/healthz")).raise_for_status()
        except httpx.HTTPError as error:
            raise SystemExit(f"Phoenix ({phoenix}) no responde: {error}. Esperá a que arranque.") from error

        # Validación de entrada (Pydantic): no llega a encolarse nada.
        invalida = await cliente.post("/tasks", json={"pregunta": "?"})
        id_invalido = await cliente.post("/tasks", json={"pregunta": "Hola, ¿cómo va?", "conversacion_id": "x"})
        validacion = {
            "pregunta_corta": invalida.status_code,
            "conversacion_id_invalido": id_invalido.status_code,
        }
        print(f"Validación de entrada: {validacion} (se esperan 422)")

        # --- Fase 1: primeros turnos, en paralelo -------------------------------
        print(f"\nFase 1: {len(ESCENARIOS)} conversaciones en paralelo")
        t0 = time.perf_counter()
        primeros = await asyncio.gather(*(turno(cliente, e.pregunta) for e in ESCENARIOS))
        for escenario, job in zip(ESCENARIOS, primeros, strict=True):
            chequeos = evaluar(escenario.chequeos, job)
            if job["estado"] == PAUSA:
                chequeos += evaluar(escenario.chequeos_pausa, job)
            registros[escenario.nombre] = {
                "conversacion_id": job["conversacion_id"],
                "turnos": [resumen_job(job)],
                "chequeos": chequeos,
            }
            print(f"  {escenario.nombre:<22} {job['estado']:<18} {job['_e2e_s']:>6.1f} s  {job['conversacion_id']}")
        print(f"Fase 1 en {time.perf_counter() - t0:.1f} s")

        reinicio_s = await reiniciar_stack(cliente) if reiniciar else None

        # La conversación sigue en el checkpoint después del reinicio.
        conv = registros["conversacion"]["conversacion_id"]
        tras_reinicio = (await cliente.get(f"/conversations/{conv}")).raise_for_status().json()
        registros["conversacion"]["chequeos"].append({
            "chequeo": "el turno 1 sigue en el checkpoint tras el reinicio",
            "ok": [m["rol"] for m in tras_reinicio["mensajes"]] == ["usuario", "asistente"],
        })

        # --- Fase 2: segundos turnos y decisiones humanas, en paralelo ----------
        print("\nFase 2: turno 2 y aprobaciones (después del reinicio)" if reiniciar else "\nFase 2")

        async def segunda_fase(escenario: Escenario) -> None:
            registro = registros[escenario.nombre]
            primero = registro["turnos"][0]
            if escenario.decision is not None and primero["estado"] == PAUSA:
                (await cliente.post(f"/tasks/{primero['job_id']}/approve", json=escenario.decision)).raise_for_status()
                job = await esperar(cliente, primero["job_id"], FINALES)
                registro["turnos"][0] = resumen_job(job)
                aprobado = escenario.decision["aprobado"]
                registro["chequeos"] += evaluar(
                    [
                        ("DONE tras la decisión", lambda j: j["estado"] == "DONE"),
                        ("se ejecutó" if aprobado else "NO se ejecutó",
                         lambda j: bool(j["resultado_accion"]) and j["resultado_accion"]["ejecutada"] is aprobado),
                        ("reanudó en el mismo hilo", lambda j: j["nodos"][-1] in ("ejecutar_accion", "aprobacion_humana")),
                    ],
                    job,
                )
                print(f"  {escenario.nombre:<22} {'aprobado' if aprobado else 'rechazado'} → {job['estado']}")
            if escenario.turno_2:
                job = await turno(cliente, escenario.turno_2, registro["conversacion_id"])
                registro["turnos"].append(resumen_job(job))
                registro["chequeos"] += evaluar(escenario.chequeos_turno_2, job)
                print(f"  {escenario.nombre:<22} turno 2 → {job['estado']} en {job['_e2e_s']:.1f} s")

        await asyncio.gather(*(segunda_fase(e) for e in ESCENARIOS))

        conversacion = (await cliente.get(f"/conversations/{conv}")).raise_for_status().json()

    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "url": url,
        "reinicio_entre_fases": reiniciar,
        "reinicio_segundos": round(reinicio_s, 1) if reinicio_s else None,
        "validacion_entrada": validacion,
        "escenarios": registros,
        "conversacion_completa": conversacion,
    }


def imprimir(datos: dict[str, Any]) -> bool:
    print("\n" + "=" * 92)
    todo_ok = all(v == 422 for v in datos["validacion_entrada"].values())
    for nombre, registro in datos["escenarios"].items():
        ok = sum(c["ok"] for c in registro["chequeos"])
        total = len(registro["chequeos"])
        todo_ok &= ok == total
        print(f"{'✔' if ok == total else '✘'} {nombre:<22} {ok}/{total} chequeos   sesión Phoenix: {registro['conversacion_id']}")
        for c in registro["chequeos"]:
            if not c["ok"]:
                print(f"      ✘ {c['chequeo']}")
    print("=" * 92)
    return todo_ok


def main() -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--sin-reinicio", action="store_true", help="No reinicia los contenedores entre fases.")
    args = parser.parse_args()

    datos = asyncio.run(correr(args.url, reiniciar=not args.sin_reinicio))
    todo_ok = imprimir(datos)

    destino = RAIZ / "resultados" / f"pruebas-sistema-{datetime.now():%Y%m%d-%H%M%S}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultado completo: {destino.relative_to(RAIZ)}")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
