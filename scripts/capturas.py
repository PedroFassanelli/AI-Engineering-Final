"""Capturas del dashboard de Phoenix para `screenshots/`.

    python scripts/capturas.py                   # todas
    python scripts/capturas.py --solo rag hitl   # algunas

Usa Playwright con el Edge (o Chrome) que ya está instalado: no hace falta
`playwright install`. Las capturas son del Phoenix real que levanta el
docker compose, con las trazas de `scripts/pruebas_sistema.py`.

Cada captura navega a una vista, espera a que carguen los datos y, si hace
falta, interactúa (abrir una traza, elegir un span, pasar el mouse sobre el
gráfico de latencia para que aparezca el tooltip con el p95).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from playwright.async_api import Page, async_playwright

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "screenshots"
PHOENIX = "http://localhost:6006"
PROYECTO = os.getenv("PHOENIX_PROJECT_NAME", "nimbus-intelligence")
ANCHO, ALTO = 1920, 1080


def id_proyecto() -> str:
    """El id GraphQL del proyecto (`UHJvamVjdDo...` = base64 de `Project:<n>`)."""
    r = httpx.post(f"{PHOENIX}/graphql", json={"query": "{ projects { edges { node { id name } } } }"}).json()
    for borde in r["data"]["projects"]["edges"]:
        if borde["node"]["name"] == PROYECTO:
            return borde["node"]["id"]
    raise SystemExit(f"No existe el proyecto {PROYECTO!r} en Phoenix.")


async def esperar_datos(pagina: Page, segundos: float = 3.0) -> None:
    await pagina.wait_for_load_state("networkidle")
    await pagina.wait_for_timeout(int(segundos * 1000))


async def ocultar_columnas(pagina: Page, ids: list[str]) -> None:
    """Destilda columnas de la tabla para que entren latencia, tokens y costo."""
    await pagina.get_by_role("button", name="Columns").click()
    await pagina.wait_for_timeout(500)
    for columna in ids:
        casilla = pagina.locator(f'li[data-column-id="{columna}"] input[type=checkbox]')
        if await casilla.count() and await casilla.first.is_checked():
            await casilla.first.click(force=True)
    await pagina.keyboard.press("Escape")
    await pagina.wait_for_timeout(500)


async def captura(pagina: Page, nombre: str, **kwargs) -> None:
    ruta = DESTINO / f"{nombre}.png"
    await pagina.screenshot(path=str(ruta), **kwargs)
    print("  ✔", ruta.relative_to(RAIZ))


def ultima_corrida() -> dict:
    archivos = sorted((RAIZ / "resultados").glob("pruebas-sistema-*.json"))
    if not archivos:
        raise SystemExit("Primero corré scripts/pruebas_sistema.py")
    return json.loads(archivos[-1].read_text(encoding="utf-8"))


async def abrir_traza(pagina: Page, proyecto: str, rango: str, texto: str, span: str | None = None) -> None:
    """Abre la traza cuya fila contiene `texto` y, si se pide, elige un span del árbol.

    El panel de detalle solo se abre con un click en la fila (navegar directo a
    `/traces/<id>` muestra la tabla sola). Se filtra por filas y elementos
    **visibles**: las celdas truncadas duplican el texto en nodos ocultos.
    """
    await pagina.goto(f"{PHOENIX}/projects/{proyecto}/traces?timeRangeKey={rango}")
    await esperar_datos(pagina)
    await pagina.locator("tbody tr").filter(has_text=texto).first.locator("td").nth(3).click()
    await esperar_datos(pagina, 4)
    if span:
        await pagina.get_by_text(span, exact=True).locator("visible=true").first.click()
        await esperar_datos(pagina, 3)


# --- Vistas -------------------------------------------------------------------


async def trazas(pagina: Page, proyecto: str, rango: str) -> None:
    """Las trazas de las pruebas, con latencia, tokens y costo de cada una."""
    await pagina.goto(f"{PHOENIX}/projects/{proyecto}/traces?timeRangeKey={rango}")
    await esperar_datos(pagina)
    await ocultar_columnas(pagina, ["error", "userId", "annotations", "traceAnnotations", "startTime"])
    await captura(pagina, "01-trazas-latencia-tokens-costo")


async def multipaso(pagina: Page, proyecto: str, rango: str) -> None:
    """La traza del caso multipaso: supervisor → investigador → analista → síntesis."""
    await abrir_traza(pagina, proyecto, rango, "El comercio Bazar Pampa viene")
    await captura(pagina, "02-traza-multipaso-arbol-de-spans")


async def rag(pagina: Page, proyecto: str, rango: str) -> None:
    """El span RETRIEVER del RAG híbrido, con los documentos recuperados."""
    await abrir_traza(pagina, proyecto, rango, "El comercio Bazar Pampa viene", span="recuperar_documentacion")
    await captura(pagina, "03-rag-span-retriever-documentos")
    atributos = pagina.get_by_role("tab", name="Attributes")
    if await atributos.count():
        await atributos.first.click()
        await pagina.wait_for_timeout(1000)
        await captura(pagina, "04-rag-span-retriever-atributos")


async def sesiones(pagina: Page, proyecto: str, rango: str) -> None:
    """Una sesión por conversación (session.id = conversacion_id)."""
    await pagina.goto(f"{PHOENIX}/projects/{proyecto}/sessions?timeRangeKey={rango}")
    await esperar_datos(pagina)
    await captura(pagina, "05-sesiones-por-conversacion")


async def conversacion(pagina: Page, proyecto: str, rango: str) -> None:
    """La conversación de dos turnos: el segundo, después del reinicio del stack."""
    conversacion_id = ultima_corrida()["escenarios"]["conversacion"]["conversacion_id"]
    await pagina.goto(f"{PHOENIX}/projects/{proyecto}/sessions?timeRangeKey={rango}")
    await esperar_datos(pagina)
    await pagina.locator("tbody tr").filter(has_text=conversacion_id[:12]).first.locator("td").nth(1).click()
    await esperar_datos(pagina, 5)
    await captura(pagina, "06-conversacion-dos-turnos")


async def hitl(pagina: Page, proyecto: str, rango: str) -> None:
    """La reanudación tras la aprobación humana, con el span TOOL que escribe en la base."""
    await abrir_traza(pagina, proyecto, rango, 'El código de error `E5030`', span="programar_reintentos")
    await captura(pagina, "07-hitl-reanudacion-aprobada")


async def tarjeta(pagina: Page, titulo: str):
    """La tarjeta del dashboard que tiene ese título (el gráfico y su leyenda)."""
    return pagina.get_by_text(titulo, exact=True).locator(
        "xpath=ancestor::*[.//*[contains(@class,'recharts-wrapper')]][1]"
    )


async def pasar_mouse_por_el_dato(pagina: Page, grafico) -> None:
    """Hover sobre el primer dato del gráfico para que Recharts muestre el tooltip."""
    dato = grafico.locator(".recharts-line-dot, .recharts-bar-rectangle path, .recharts-rectangle").first
    caja = await dato.bounding_box()
    if caja is None:
        return
    await pagina.mouse.move(caja["x"] + caja["width"] / 2, caja["y"] + caja["height"] / 2)
    await pagina.wait_for_timeout(800)


async def metricas(pagina: Page, proyecto: str, rango: str) -> None:
    """Pestaña Metrics: latencia por percentil (p95) y costo por modelo."""
    await pagina.goto(f"{PHOENIX}/projects/{proyecto}/metrics?timeRangeKey={rango}")
    await esperar_datos(pagina, 5)
    latencia = await tarjeta(pagina, "Trace latency")
    await pasar_mouse_por_el_dato(pagina, latencia)
    await captura(pagina, "08-metricas-latencia-p95")

    await pagina.mouse.move(5, 5)
    await pagina.wait_for_timeout(500)
    modelos = await tarjeta(pagina, "Top models by cost")
    await pasar_mouse_por_el_dato(pagina, modelos)
    await captura(pagina, "09-metricas-costo-por-modelo")


VISTAS = {
    "trazas": trazas,
    "multipaso": multipaso,
    "rag": rag,
    "sesiones": sesiones,
    "conversacion": conversacion,
    "hitl": hitl,
    "metricas": metricas,
}


async def main(solo: list[str], rango: str, navegador: str) -> None:
    DESTINO.mkdir(exist_ok=True)
    proyecto = id_proyecto()
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel=navegador, headless=True)
        contexto = await browser.new_context(viewport={"width": ANCHO, "height": ALTO}, color_scheme="dark")
        pagina = await contexto.new_page()
        for nombre in solo or VISTAS:
            print(nombre)
            try:
                await VISTAS[nombre](pagina, proyecto, rango)
            except Exception as error:  # noqa: BLE001 - una vista rota no frena las demás
                print(f"  ✘ {nombre}: {type(error).__name__}: {error}")
        await browser.close()


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solo", nargs="*", choices=list(VISTAS), default=[])
    parser.add_argument("--rango", default="1h", help="timeRangeKey de Phoenix: 15m, 1h, 12h, 1d, 7d...")
    parser.add_argument("--navegador", default="msedge", help="msedge o chrome")
    args = parser.parse_args()
    asyncio.run(main(args.solo, args.rango, args.navegador))
