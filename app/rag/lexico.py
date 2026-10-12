"""La mitad léxica del híbrido: BM25 en memoria sobre los mismos fragmentos.

BM25 no entiende nada, pero si la consulta dice `E4090` y el fragmento dice
`E4090`, lo encuentra primero, siempre. Es justo lo que la búsqueda vectorial
hace mal: los códigos de error, los nombres de función y los encabezados HTTP
son tokens rarísimos para un modelo de embeddings, y su vector es casi ruido.

Corre en memoria y en microsegundos (el corpus son unas decenas de
fragmentos), así que llamarlo desde una corutina no bloquea el event loop de
forma apreciable.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final, Sequence

from rank_bm25 import BM25Okapi

from app.rag.esquemas import Chunk

_TOKEN = re.compile(r"[a-z0-9_]+")

#: Stopwords del castellano, ya sin tildes (el filtro va después de normalizar).
#: Es la lista de la Pre-entrega 4, que salió de medir: sin ella, "¿qué hace el
#: método capturar_pago y qué parámetros recibe?" rankeaba por el andamiaje
#: gramatical de la pregunta y no por la función que nombra.
STOPWORDS: Final[frozenset[str]] = frozenset(
    """
    a al algo ante antes aqui asi aunque cada como con contra cual cuales cuando
    cuanta cuantas cuanto cuantos de del desde donde dos e el ella ellas ello
    ellos en entre era eran es esa esas ese eso esos esta estan estas este esto
    estos fue fueron ha hace hacen hacer hasta hay la las le les lo los mas me
    mi mucho muy no nos o para pero por porque puede pueden que quien se segun
    ser si sin sobre solo son su sus tambien tener tiene tienen todo todos un
    una uno unos y ya
    """.split()
)


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sin acentos, sin puntuación pegada y sin stopwords.

    `E4090`, `capturar_pago` y `X-Nimbus-Signature` sobreviven como tokens (el
    guion bajo se conserva).
    """
    plano = unicodedata.normalize("NFKD", texto.lower())
    sin_tildes = "".join(c for c in plano if not unicodedata.combining(c))
    return [t for t in _TOKEN.findall(sin_tildes) if t not in STOPWORDS]


class IndiceLexico:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("No se puede armar un índice BM25 sin fragmentos.")
        self.chunks = list(chunks)
        # Se indexa título + texto: el título de la sección ("E4090 · Pago
        # duplicado") suele tener justo el literal que se busca.
        self._bm25 = BM25Okapi([tokenizar(c.texto_para_embedding) for c in self.chunks])

    def buscar(self, consulta: str, k: int) -> list[tuple[Chunk, float]]:
        """Los `k` fragmentos con mayor puntaje, descartando los de puntaje 0."""
        puntajes = self._bm25.get_scores(tokenizar(consulta))
        orden = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)
        return [(self.chunks[i], float(puntajes[i])) for i in orden[:k] if puntajes[i] > 0]
