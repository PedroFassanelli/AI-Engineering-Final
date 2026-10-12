"""Evaluación de la recuperación contra el golden set.

Mide las tres variantes sobre las mismas preguntas: solo léxico (BM25), solo
vectorial (Pinecone) y el híbrido. Medir solo el híbrido responde "¿qué tan
bueno es?"; compararlo con sus dos mitades responde "¿el híbrido sirve para
algo?", que es la pregunta que importa.

Métricas, a nivel **documento** (lo que el agente cita es el archivo):

- **Recall@k**: de los documentos relevantes, cuántos aparecen en el top-k.
- **Hit@k**: si el documento principal aparece en el top-k.
- **MRR**: 1 / posición del primer fragmento relevante. Distingue traer la
  respuesta primera de traerla tercera, que Recall@k no ve.

Se reportan también por `tipo` de consulta (léxica, semántica, mixta): ahí se
ve qué mitad aporta qué.

    python -m app.cli evaluar-rag            # k = RAG_TOP_K
    python -m app.cli evaluar-rag --k 5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field, field_validator

from app import config
from app.rag.hibrido import RecuperadorHibrido

TipoConsulta = Literal["lexica", "semantica", "mixta"]


class CasoGolden(BaseModel):
    pregunta: str = Field(min_length=10)
    #: El primero es el documento principal.
    relevantes: list[str] = Field(min_length=1)
    tipo: TipoConsulta
    nota: str = ""

    @field_validator("relevantes")
    @classmethod
    def _son_archivos(cls, valor: list[str]) -> list[str]:
        for archivo in valor:
            if not archivo.endswith(".md"):
                raise ValueError(f"{archivo!r} no es un nombre de archivo del corpus")
        return valor


class ResultadoCaso(BaseModel):
    pregunta: str
    tipo: TipoConsulta
    recuperados: list[str]
    recall: float = Field(ge=0, le=1)
    hit: bool
    reciprocal_rank: float = Field(ge=0, le=1)


class Reporte(BaseModel):
    variante: str
    k: int
    casos: list[ResultadoCaso]

    def _promedio(self, valores: Sequence[float]) -> float:
        return sum(valores) / len(valores) if valores else 0.0

    @property
    def recall(self) -> float:
        return self._promedio([c.recall for c in self.casos])

    @property
    def hit_rate(self) -> float:
        return self._promedio([float(c.hit) for c in self.casos])

    @property
    def mrr(self) -> float:
        return self._promedio([c.reciprocal_rank for c in self.casos])

    def mrr_por_tipo(self) -> dict[str, float]:
        tipos = sorted({c.tipo for c in self.casos})
        return {t: self._promedio([c.reciprocal_rank for c in self.casos if c.tipo == t]) for t in tipos}


def cargar_golden_set(ruta: Path | None = None) -> list[CasoGolden]:
    crudo = json.loads((ruta or config.GOLDEN_SET).read_text(encoding="utf-8"))
    return [CasoGolden.model_validate(c) for c in crudo["casos"]]


def validar_contra_corpus(casos: Sequence[CasoGolden], fuentes: set[str]) -> None:
    """Un typo en el golden set da Recall 0 para siempre, igual que un recuperador roto."""
    faltantes = sorted({f for c in casos for f in c.relevantes} - fuentes)
    if faltantes:
        raise ValueError(f"El golden set nombra documentos que no están en el corpus: {faltantes}")


def evaluar_caso(caso: CasoGolden, recuperados: Sequence[str]) -> ResultadoCaso:
    relevantes = set(caso.relevantes)
    presentes = relevantes & set(recuperados)
    primero = next((i for i, f in enumerate(recuperados, start=1) if f in relevantes), None)
    return ResultadoCaso(
        pregunta=caso.pregunta,
        tipo=caso.tipo,
        recuperados=list(recuperados),
        recall=len(presentes) / len(relevantes),
        hit=caso.relevantes[0] in recuperados,
        reciprocal_rank=0.0 if primero is None else 1.0 / primero,
    )


async def evaluar(
    recuperador: RecuperadorHibrido, casos: Sequence[CasoGolden], k: int, variante: str
) -> Reporte:
    resultados = []
    for caso in casos:
        busqueda = await recuperador.buscar(caso.pregunta, k=k)
        resultados.append(evaluar_caso(caso, [r.fuente for r in busqueda.resultados]))
    return Reporte(variante=variante, k=k, casos=resultados)


def variantes(base: RecuperadorHibrido) -> dict[str, RecuperadorHibrido]:
    """Léxico, vectorial e híbrido sobre los mismos fragmentos (y el mismo índice)."""
    chunks = list(base.chunks.values())
    salida = {"lexico (BM25)": RecuperadorHibrido(chunks)}
    if base.vectorial is not None:
        salida["vectorial (Pinecone)"] = RecuperadorHibrido(
            chunks, base.vectorial, peso_lexico=0.0, peso_vectorial=1.0
        )
        salida[f"hibrido {base.pesos['lexico']:.1f}/{base.pesos['vectorial']:.1f}"] = base
    return salida


def formatear(reportes: Sequence[Reporte]) -> str:
    if not reportes:
        return "(sin reportes)"
    k = reportes[0].k
    tipos = sorted({c.tipo for r in reportes for c in r.casos})
    encabezado = f"{'variante':<22} {'Recall@' + str(k):>9} {'Hit@' + str(k):>7} {'MRR':>6}  " + "  ".join(
        f"{'MRR ' + t:>13}" for t in tipos
    )
    lineas = [encabezado, "-" * len(encabezado)]
    for r in reportes:
        por_tipo = r.mrr_por_tipo()
        lineas.append(
            f"{r.variante:<22} {r.recall:>9.1%} {r.hit_rate:>7.1%} {r.mrr:>6.3f}  "
            + "  ".join(f"{por_tipo.get(t, 0):>13.3f}" for t in tipos)
        )
    return "\n".join(lineas)


def detalle(reporte: Reporte) -> str:
    lineas = [f"\nDetalle — {reporte.variante}"]
    for c in reporte.casos:
        marca = "OK " if c.hit else "MISS"
        lineas.append(f"  [{marca}] ({c.tipo:<9}) RR={c.reciprocal_rank:.2f}  {c.pregunta}")
        lineas.append(f"         {', '.join(c.recuperados)}")
    return "\n".join(lineas)
