"""Utilidades de línea de comandos.

    python -m app.cli sembrar              # (re)crea la base de comercios y pagos
    python -m app.cli diagrama [--png]     # Mermaid del grafo → docs/grafo.mmd (y .png)
    python -m app.cli reintentos           # lo que escribió la acción aprobada
    python -m app.cli fragmentos           # cómo queda el chunking (sin red)
    python -m app.cli indexar              # sincroniza Pinecone con el corpus (lo hace el worker al arrancar)
    python -m app.cli buscar "E4090"       # una búsqueda híbrida, con de qué mitad vino cada fragmento
    python -m app.cli evaluar-rag [--k 5]  # léxico vs. vectorial vs. híbrido sobre el golden set

`indexar`, `buscar` y `evaluar-rag` necesitan Redis (el manifiesto del índice
vive ahí: `docker compose up -d redis`) y, en modo híbrido, las claves de
Pinecone y OpenAI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app import config
from app.fuentes import db

DIR_DOCS = config.BASE_DIR.parent / "docs"


@asynccontextmanager
async def _recuperador() -> AsyncIterator:
    from redis.asyncio import Redis

    from app.rag import abrir_recuperador

    redis = Redis.from_url(config.REDIS_URL, decode_responses=True)
    try:
        async with abrir_recuperador(redis) as recuperador:
            yield recuperador
    finally:
        await redis.aclose()


async def _buscar(consulta: str, k: int) -> None:
    async with _recuperador() as recuperador:
        resultado = await recuperador.buscar(consulta, k=k)
    print(f"modo: {resultado.modo}" + (f"  (aviso: {resultado.aviso})" if resultado.aviso else ""))
    for r in resultado.resultados:
        print(
            f"{r.posicion}. {r.id:<34} rrf={r.puntaje:.4f}  bm25#{r.rango_lexico or '-'}  "
            f"pinecone#{r.rango_vectorial or '-'}\n   {r.seccion}"
        )


async def _evaluar(k: int, con_detalle: bool) -> None:
    from app.rag.evaluacion import cargar_golden_set, detalle, evaluar, formatear, validar_contra_corpus, variantes

    casos = await asyncio.to_thread(cargar_golden_set)
    async with _recuperador() as base:
        validar_contra_corpus(casos, base.fuentes)
        print(f"Golden set: {len(casos)} preguntas · {len(base.chunks)} fragmentos · k={k}\n")
        reportes = [await evaluar(r, casos, k, nombre) for nombre, r in variantes(base).items()]
    print(formatear(reportes))
    if con_detalle:
        for reporte in reportes:
            print(detalle(reporte))


def main(argv: list[str] | None = None) -> int:
    config.configurar_consola()
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("sembrar", help="(Re)crea la base de negocio con los datos fijos.")
    diagrama = sub.add_parser("diagrama", help="Exporta el diagrama del grafo.")
    diagrama.add_argument("--png", action="store_true", help="Además, el PNG (usa mermaid.ink).")
    sub.add_parser("reintentos", help="Lista los reintentos programados por acciones aprobadas.")
    sub.add_parser("fragmentos", help="Muestra el chunking del corpus (sin red).")
    sub.add_parser("indexar", help="Sincroniza el índice de Pinecone con el corpus.")
    buscar = sub.add_parser("buscar", help="Una búsqueda en la documentación.")
    buscar.add_argument("consulta")
    buscar.add_argument("--k", type=int, default=config.RAG_TOP_K)
    evaluar = sub.add_parser("evaluar-rag", help="Recall@k y MRR contra el golden set.")
    evaluar.add_argument("--k", type=int, default=config.RAG_TOP_K)
    evaluar.add_argument("--detalle", action="store_true")
    args = parser.parse_args(argv)

    if args.comando == "sembrar":
        print(asyncio.run(db.sembrar()), "→", config.DB_NEGOCIO)
    elif args.comando == "diagrama":
        from app.graph import construir_grafo_para_dibujar

        grafo = construir_grafo_para_dibujar().get_graph()
        DIR_DOCS.mkdir(exist_ok=True)
        destino = DIR_DOCS / "grafo.mmd"
        destino.write_text(grafo.draw_mermaid(), encoding="utf-8")
        print(destino)
        if args.png:
            png = DIR_DOCS / "grafo.png"
            png.write_bytes(grafo.draw_mermaid_png())
            print(png)
    elif args.comando == "reintentos":
        print(json.dumps(asyncio.run(db.listar_reintentos()), ensure_ascii=False, indent=2))
    elif args.comando == "fragmentos":
        from app.rag.corpus import chunks_por_defecto, hash_corpus

        chunks = chunks_por_defecto()
        tokens = [c.metadatos.tokens for c in chunks]
        print(
            f"{len(chunks)} fragmentos · tokens min {min(tokens)} / prom {sum(tokens) // len(tokens)} / "
            f"máx {max(tokens)} · hash {hash_corpus(chunks)[:16]}…\n"
        )
        for c in chunks:
            print(f"  {c.id:<36} {c.metadatos.tokens:>4} tok  [{c.metadatos.categoria}]  {c.metadatos.seccion}")
    elif args.comando == "indexar":
        async def indexar() -> None:
            async with _recuperador() as recuperador:
                print(f"RAG {recuperador.modo}: {len(recuperador.chunks)} fragmentos listos.")

        asyncio.run(indexar())
    elif args.comando == "buscar":
        asyncio.run(_buscar(args.consulta, args.k))
    elif args.comando == "evaluar-rag":
        asyncio.run(_evaluar(args.k, args.detalle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
