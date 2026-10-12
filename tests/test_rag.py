"""El RAG sin red: chunking, BM25, fusión RRF, degradación y sincronización del índice.

La mitad vectorial se reemplaza por `IndiceFalso`, que cumple la interfaz de
`IndiceVectorial` (embeber, buscar, contar, reemplazar) con rankings fijos.
Redis es `fakeredis`. Lo que se prueba es la lógica de este repo, no Pinecone.
"""

from __future__ import annotations

import fakeredis
import pytest
from pydantic import ValidationError

from app import config
from app.rag.corpus import (
    chunks_por_defecto,
    construir_chunks,
    contar_tokens,
    dividir_en_secciones,
    hash_corpus,
)
from app.rag.esquemas import Chunk, MetadatosChunk
from app.rag.evaluacion import cargar_golden_set, evaluar, evaluar_caso, validar_contra_corpus
from app.rag.hibrido import RecuperadorHibrido, fusionar_rrf
from app.rag.lexico import tokenizar
from app.rag.vectorial import ErrorDeIndice, asegurar_indice, leer_manifiesto, manifiesto_esperado, sincronizar

# --- Chunking -----------------------------------------------------------------


def test_dividir_ignora_encabezados_dentro_de_bloques_de_codigo():
    markdown = "# Doc\n\n## Real\ntexto\n```python\n# Antes (2.x)\nx = 1\n```\n## Otra\nmás"
    secciones = dividir_en_secciones("doc.md", markdown)
    assert [(s.documento, s.seccion) for s in secciones] == [("Doc", "Real"), ("Doc", "Otra")]
    assert "# Antes (2.x)" in secciones[0].texto


def test_el_corpus_son_los_cinco_documentos_partidos_por_seccion():
    chunks = chunks_por_defecto()
    assert {c.metadatos.fuente for c in chunks} == {
        "01-guia-de-inicio.md",
        "02-referencia-api.md",
        "03-manejo-de-errores.md",
        "04-webhooks-y-conciliacion.md",
        "05-changelog-y-migracion.md",
    }
    # Ningún fragmento pasa el techo de tokens, y los ids son únicos y deterministas.
    assert max(c.metadatos.tokens for c in chunks) <= config.CHUNK_MAX_TOKENS
    assert len({c.id for c in chunks}) == len(chunks)
    catalogo = next(c for c in chunks if c.metadatos.seccion == "Catálogo de códigos")
    assert catalogo.id.startswith("03-manejo-de-errores.md::")
    assert catalogo.metadatos.categoria == "manejo de errores"
    assert "E5030" in catalogo.metadatos.text


def test_una_seccion_larga_se_parte_por_tokens_conservando_su_titulo(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 60)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_TOKENS", 10)
    parrafos = "\n\n".join(f"Párrafo {i}: " + "palabra " * 40 for i in range(4))
    chunks = construir_chunks({"99-largo.md": f"# Largo\n\n## Sección única\n{parrafos}"})
    assert len(chunks) > 1
    assert all(c.metadatos.seccion == "Sección única" for c in chunks)
    assert [c.metadatos.parte for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.metadatos.total_partes == len(chunks) for c in chunks)
    assert all(contar_tokens(c.metadatos.text) <= 60 for c in chunks)


def test_el_hash_cambia_si_cambia_el_texto_o_el_chunking():
    base = construir_chunks({"a.md": "# A\n\n## Uno\nhola\n\n## Dos\nchau"})
    assert hash_corpus(base) == hash_corpus(list(reversed(base)))  # estable ante el orden
    editado = construir_chunks({"a.md": "# A\n\n## Uno\nhola!\n\n## Dos\nchau"})
    assert hash_corpus(base) != hash_corpus(editado)


def test_los_metadatos_se_validan_antes_de_indexar():
    with pytest.raises(ValidationError):
        MetadatosChunk(
            text="x", fuente="a.md", documento="A", seccion="S", categoria="a",
            chunk_id=3, total_chunks=2, tokens=1,
        )
    with pytest.raises(ValidationError):
        Chunk(id="sin formato", metadatos=chunks_por_defecto()[0].metadatos)


def test_tokenizar_conserva_codigos_y_quita_acentos_y_stopwords():
    assert tokenizar("¿Qué significa el código E4090 en capturar_pago()?") == [
        "significa", "codigo", "e4090", "capturar_pago"
    ]


# --- Fusión ---------------------------------------------------------------------


def test_rrf_premia_lo_que_aparece_en_los_dos_rankings():
    fusion = fusionar_rrf(
        {"lexico": ["a", "b", "c"], "vectorial": ["c", "d", "a"]}, {"lexico": 0.3, "vectorial": 0.7}
    )
    orden = [id_ for id_, _, _ in fusion]
    assert orden[0] == "c"  # 1.º vectorial (peso alto) y 3.º léxico
    assert dict((i, p) for i, _, p in fusion)["a"] == {"lexico": 1, "vectorial": 3}
    assert set(orden) == {"a", "b", "c", "d"}


def test_rrf_ignora_un_ranking_con_peso_cero():
    fusion = fusionar_rrf({"lexico": ["a"], "vectorial": ["b"]}, {"lexico": 0.0, "vectorial": 1.0})
    assert [id_ for id_, _, _ in fusion] == ["b"]


# --- Híbrido con un índice vectorial falso -----------------------------------------


class IndiceFalso:
    """Devuelve siempre el mismo ranking; anota lo que se le pidió."""

    namespace = "test"

    def __init__(self, ranking: list[str], *, falla: bool = False) -> None:
        self.ranking = ranking
        self.falla = falla
        self.vectores: dict[str, MetadatosChunk] = {}
        self.reemplazos = 0

    async def embeber_consulta(self, texto: str) -> list[float]:
        if self.falla:
            raise ConnectionError("Pinecone no responde")
        return [0.0] * 4

    async def buscar_vector(self, vector, k):
        por_id = {c.id: c.metadatos for c in chunks_por_defecto()}
        return [(id_, 0.9 - i / 100, por_id.get(id_) or por_id[next(iter(por_id))]) for i, id_ in enumerate(self.ranking[:k])]

    async def contar(self) -> int:
        return len(self.vectores)

    async def reemplazar(self, chunks) -> None:
        self.reemplazos += 1
        self.vectores = {c.id: c.metadatos for c in chunks}


def _id_de(seccion: str) -> str:
    return next(c.id for c in chunks_por_defecto() if c.metadatos.seccion == seccion)


async def test_el_hibrido_trae_lo_semantico_que_bm25_no_encuentra():
    sandbox = _id_de("Entorno de pruebas")
    recuperador = RecuperadorHibrido(chunks_por_defecto(), IndiceFalso([sandbox]))  # type: ignore[arg-type]
    resultado = await recuperador.buscar("¿Cómo pruebo sin mover plata de verdad?", k=3)
    assert resultado.modo == "hibrido" and resultado.aviso is None
    primero = resultado.resultados[0]
    assert primero.id == sandbox and primero.rango_vectorial == 1


async def test_si_pinecone_falla_degrada_a_bm25_y_lo_avisa():
    recuperador = RecuperadorHibrido(chunks_por_defecto(), IndiceFalso([], falla=True))  # type: ignore[arg-type]
    resultado = await recuperador.buscar("E4090", k=2)
    assert resultado.modo == "lexico"
    assert "semántica no respondió" in resultado.aviso
    assert resultado.resultados and all(r.rango_vectorial is None for r in resultado.resultados)


async def test_un_vector_que_no_esta_en_el_corpus_local_no_entra_al_ranking():
    recuperador = RecuperadorHibrido(
        chunks_por_defecto(), IndiceFalso(["99-viejo.md::0", _id_de("Catálogo de códigos")])  # type: ignore[arg-type]
    )
    resultado = await recuperador.buscar("códigos de error", k=5)
    assert "99-viejo.md::0" not in {r.id for r in resultado.resultados}


# --- Sincronización del índice -------------------------------------------------------


async def test_sincronizar_indexa_una_vez_y_despues_no_hace_nada():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    indice = IndiceFalso([])
    chunks = chunks_por_defecto()

    manifiesto, reindexo = await sincronizar(indice, chunks, redis, prefijo="t")  # type: ignore[arg-type]
    assert reindexo and indice.reemplazos == 1
    assert manifiesto.total_chunks == len(chunks)
    assert (await leer_manifiesto(redis, "t")).corpus_hash == manifiesto.corpus_hash

    _, reindexo = await sincronizar(indice, chunks, redis, prefijo="t")  # type: ignore[arg-type]
    assert not reindexo and indice.reemplazos == 1


async def test_sincronizar_reindexa_si_cambia_el_corpus_o_falta_un_vector():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    indice = IndiceFalso([])
    chunks = list(chunks_por_defecto())
    await sincronizar(indice, chunks, redis, prefijo="t")  # type: ignore[arg-type]

    await sincronizar(indice, chunks[:-1], redis, prefijo="t")  # type: ignore[arg-type]
    assert indice.reemplazos == 2

    indice.vectores.popitem()  # alguien borró un vector a mano
    _, reindexo = await sincronizar(indice, chunks[:-1], redis, prefijo="t")  # type: ignore[arg-type]
    assert reindexo and indice.reemplazos == 3


def test_el_manifiesto_identifica_modelo_indice_y_namespace():
    manifiesto = manifiesto_esperado(chunks_por_defecto())
    assert manifiesto.fingerprint == f"openai:{config.EMBEDDING_MODEL}:{config.dimension_embeddings()}"
    assert manifiesto.index_name == config.INDEX_NAME
    assert manifiesto.namespace == config.PINECONE_NAMESPACE


async def test_asegurar_indice_se_planta_ante_otra_dimension():
    class Descripcion:
        dimension = 384

    class PineconeFalso:
        creado = False

        async def has_index(self, nombre):
            return True

        async def create_index(self, **kwargs):
            self.creado = True

        async def describe_index(self, nombre):
            return Descripcion()

    with pytest.raises(ErrorDeIndice, match="no se redimensiona"):
        await asegurar_indice(PineconeFalso())


async def test_asegurar_indice_crea_el_indice_serverless_si_no_existe():
    llamadas = {}

    class Descripcion:
        dimension = config.dimension_embeddings()

    class PineconeFalso:
        async def has_index(self, nombre):
            return False

        async def create_index(self, **kwargs):
            llamadas.update(kwargs)

        async def describe_index(self, nombre):
            return Descripcion()

    await asegurar_indice(PineconeFalso())
    assert llamadas["name"] == config.INDEX_NAME
    assert llamadas["dimension"] == config.dimension_embeddings()
    assert llamadas["metric"] == "cosine"


# --- Evaluación -------------------------------------------------------------------------


def test_el_golden_set_nombra_documentos_que_existen():
    casos = cargar_golden_set()
    assert len(casos) >= 10
    validar_contra_corpus(casos, {c.metadatos.fuente for c in chunks_por_defecto()})


def test_metricas_de_un_caso():
    caso = cargar_golden_set()[0]
    resultado = evaluar_caso(caso, ["05-changelog-y-migracion.md", caso.relevantes[0]])
    assert resultado.hit and resultado.reciprocal_rank == 0.5


async def test_bm25_resuelve_las_consultas_lexicas_del_golden_set():
    """La mitad léxica sola: las consultas con literales tienen que salir bien."""
    casos = [c for c in cargar_golden_set() if c.tipo == "lexica"]
    reporte = await evaluar(RecuperadorHibrido(chunks_por_defecto()), casos, k=3, variante="lexico")
    assert reporte.hit_rate == 1.0
