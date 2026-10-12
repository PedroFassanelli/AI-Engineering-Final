"""Base de negocio simulada de Nimbus Pagos (la misma de la Pre-entrega 5).

Dos tablas — `comercios` y `pagos` — sobre SQLite, leídas con `aiosqlite` para
que las herramientas sean `async` de punta a punta.

En la Pre-entrega 7 se suma una tercera, `reintentos_programados`: la única
escritura de todo el sistema. Es el efecto secundario que queda detrás del nodo
de aprobación humana (ver `app/hitl.py`).

Los datos son **literales fijos**: ni `random` ni `datetime.now()`. Una base
sembrada dos veces produce exactamente las mismas filas, que es lo que permite
que las trazas guardadas en el repositorio sigan siendo reproducibles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import aiosqlite

from app import config

ESQUEMA: Final[str] = """
DROP TABLE IF EXISTS reintentos_programados;
DROP TABLE IF EXISTS pagos;
DROP TABLE IF EXISTS comercios;

CREATE TABLE comercios (
    comercio_id INTEGER PRIMARY KEY,
    nombre      TEXT    NOT NULL,
    plan        TEXT    NOT NULL,
    pais        TEXT    NOT NULL,
    alta        TEXT    NOT NULL
);

CREATE TABLE pagos (
    pago_id         TEXT    PRIMARY KEY,
    comercio_id     INTEGER NOT NULL REFERENCES comercios(comercio_id),
    monto_centavos  INTEGER NOT NULL,
    moneda          TEXT    NOT NULL,
    estado          TEXT    NOT NULL,
    codigo_error    TEXT,
    metodo          TEXT    NOT NULL,
    creado_en       TEXT    NOT NULL
);

CREATE INDEX idx_pagos_comercio ON pagos(comercio_id, estado);

-- Un pago se programa una sola vez por job: reanudar dos veces el mismo job
-- (o un reintento de la cola) no duplica filas.
CREATE TABLE reintentos_programados (
    pago_id      TEXT    NOT NULL REFERENCES pagos(pago_id),
    job_id       TEXT    NOT NULL,
    aprobado_por TEXT    NOT NULL,
    comentario   TEXT,
    creado_en    TEXT    NOT NULL,
    PRIMARY KEY (pago_id, job_id)
);
"""

COMERCIOS: Final[tuple[tuple[Any, ...], ...]] = (
    (101, "Librería Ávila", "starter", "AR", "2024-03-11"),
    (102, "Café Tortoni Centro", "growth", "AR", "2023-11-02"),
    (103, "Café Tortoni Norte", "growth", "AR", "2025-01-20"),
    (104, "Bazar Pampa", "scale", "AR", "2022-07-15"),
    (105, "Viajes Aconcagua", "growth", "CL", "2024-09-30"),
)

#: pago_id, comercio_id, monto_centavos, moneda, estado, codigo_error, metodo, creado_en
PAGOS: Final[tuple[tuple[Any, ...], ...]] = (
    # --- 101 Librería Ávila -------------------------------------------------
    ("pay_10101", 101, 185000, "ARS", "aprobado", None, "tarjeta", "2026-09-12T09:41:00"),
    ("pay_10102", 101, 240000, "ARS", "aprobado", None, "tarjeta", "2026-09-13T15:08:00"),
    ("pay_10103", 101, 96000, "ARS", "fallido", "E4012", "tarjeta", "2026-09-16T18:55:00"),
    ("pay_10104", 101, 310000, "ARS", "reembolsado", None, "tarjeta", "2026-09-18T10:02:00"),
    # --- 102 Café Tortoni Centro --------------------------------------------
    ("pay_10201", 102, 620000, "ARS", "aprobado", None, "tarjeta", "2026-09-15T12:30:00"),
    ("pay_10202", 102, 510000, "ARS", "aprobado", None, "transferencia", "2026-09-17T19:14:00"),
    ("pay_10203", 102, 320000, "ARS", "fallido", "E4021", "tarjeta", "2026-09-19T08:47:00"),
    # --- 103 Café Tortoni Norte ---------------------------------------------
    ("pay_10301", 103, 145000, "ARS", "aprobado", None, "tarjeta", "2026-09-14T17:22:00"),
    ("pay_10302", 103, 288000, "ARS", "aprobado", None, "tarjeta", "2026-09-18T13:05:00"),
    # --- 104 Bazar Pampa ----------------------------------------------------
    # El caso de la demo: cuatro fallas, tres del mismo código.
    ("pay_10401", 104, 1250000, "ARS", "aprobado", None, "tarjeta", "2026-09-14T10:12:00"),
    ("pay_10402", 104, 890000, "ARS", "fallido", "E4090", "tarjeta", "2026-09-15T09:03:00"),
    ("pay_10403", 104, 890000, "ARS", "fallido", "E4090", "tarjeta", "2026-09-15T09:04:00"),
    ("pay_10404", 104, 430000, "ARS", "aprobado", None, "transferencia", "2026-09-16T11:40:00"),
    ("pay_10405", 104, 2150000, "ARS", "fallido", "E4090", "tarjeta", "2026-09-17T16:22:00"),
    ("pay_10406", 104, 760000, "ARS", "fallido", "E4290", "tarjeta", "2026-09-18T12:00:00"),
    ("pay_10407", 104, 310000, "ARS", "pendiente", None, "transferencia", "2026-09-19T08:31:00"),
    # --- 105 Viajes Aconcagua -----------------------------------------------
    ("pay_10501", 105, 4200000, "CLP", "aprobado", None, "tarjeta", "2026-09-11T20:19:00"),
    ("pay_10502", 105, 3150000, "CLP", "fallido", "E5030", "tarjeta", "2026-09-16T21:44:00"),
    ("pay_10503", 105, 3150000, "CLP", "fallido", "E5030", "tarjeta", "2026-09-16T21:51:00"),
)


async def sembrar(ruta: Path | None = None) -> dict[str, int]:
    """Crea el esquema y carga los datos. Idempotente: borra y reescribe."""
    destino = ruta or config.DB_NEGOCIO
    destino.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(destino) as conexion:
        await conexion.executescript(ESQUEMA)
        await conexion.executemany(
            "INSERT INTO comercios VALUES (?, ?, ?, ?, ?)", COMERCIOS
        )
        await conexion.executemany(
            "INSERT INTO pagos VALUES (?, ?, ?, ?, ?, ?, ?, ?)", PAGOS
        )
        await conexion.commit()

    return {"comercios": len(COMERCIOS), "pagos": len(PAGOS)}


class BaseNoSembrada(RuntimeError):
    """La base de negocio no existe todavía."""


async def _conectar(ruta: Path | None = None) -> aiosqlite.Connection:
    destino = ruta or config.DB_NEGOCIO
    if not destino.exists():
        raise BaseNoSembrada(
            f"No existe {destino.name}. Creala con:  python -m app.cli sembrar"
        )
    conexion = await aiosqlite.connect(destino)
    conexion.row_factory = aiosqlite.Row
    return conexion


async def buscar_comercios_por_nombre(
    fragmento: str, ruta: Path | None = None
) -> list[dict[str, Any]]:
    """Comercios cuyo nombre contiene `fragmento`, sin distinguir acentos."""
    conexion = await _conectar(ruta)
    try:
        cursor = await conexion.execute("SELECT * FROM comercios ORDER BY comercio_id")
        filas = [dict(f) for f in await cursor.fetchall()]
    finally:
        await conexion.close()

    objetivo = normalizar(fragmento)
    return [f for f in filas if objetivo in normalizar(f["nombre"])]


async def listar_pagos(comercio_id: int, ruta: Path | None = None) -> list[dict[str, Any]]:
    """**Todos** los pagos del comercio, en orden cronológico.

    Sin `LIMIT` a propósito: el analista calcula porcentajes sobre esto, y un
    total calculado sobre un listado recortado es un número que parece correcto
    y no lo es.
    """
    conexion = await _conectar(ruta)
    try:
        cursor = await conexion.execute(
            "SELECT * FROM pagos WHERE comercio_id = ? ORDER BY creado_en",
            (comercio_id,),
        )
        return [dict(f) for f in await cursor.fetchall()]
    finally:
        await conexion.close()


async def programar_reintentos(
    pago_ids: list[str],
    *,
    job_id: str,
    aprobado_por: str,
    comentario: str | None,
    creado_en: str,
    ruta: Path | None = None,
) -> list[str]:
    """Registra el reintento de cada pago. Devuelve los que quedaron **nuevos**.

    `INSERT OR IGNORE` sobre la clave `(pago_id, job_id)` la hace idempotente:
    si el nodo se reejecuta (un reintento de la cola, una reanudación doble),
    no se programa dos veces el mismo cobro.
    """
    conexion = await _conectar(ruta)
    try:
        nuevos: list[str] = []
        for pago_id in pago_ids:
            cursor = await conexion.execute(
                "INSERT OR IGNORE INTO reintentos_programados VALUES (?, ?, ?, ?, ?)",
                (pago_id, job_id, aprobado_por, comentario, creado_en),
            )
            if cursor.rowcount:
                nuevos.append(pago_id)
        await conexion.commit()
        return nuevos
    finally:
        await conexion.close()


async def listar_reintentos(job_id: str | None = None, ruta: Path | None = None) -> list[dict[str, Any]]:
    conexion = await _conectar(ruta)
    try:
        if job_id is None:
            cursor = await conexion.execute("SELECT * FROM reintentos_programados ORDER BY creado_en")
        else:
            cursor = await conexion.execute(
                "SELECT * FROM reintentos_programados WHERE job_id = ? ORDER BY pago_id", (job_id,)
            )
        return [dict(f) for f in await cursor.fetchall()]
    finally:
        await conexion.close()


_ACENTOS = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y sin espacios en los bordes."""
    return texto.translate(_ACENTOS).lower().strip()
