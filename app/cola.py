"""La cola de trabajos: cómo la API le pasa una tarea al worker.

La API no conoce Arq: depende de este protocolo. En producción es `ColaArq`; en
los tests, un doble que anota lo encolado y deja que el test corra el worker
cuando quiere.
"""

from __future__ import annotations

from typing import Any, Protocol

from arq.connections import ArqRedis


class Cola(Protocol):
    async def encolar_ejecucion(self, job_id: str) -> None: ...

    async def encolar_reanudacion(self, job_id: str, decision: dict[str, Any]) -> None: ...


class ColaArq:
    def __init__(self, pool: ArqRedis) -> None:
        self.pool = pool

    async def encolar_ejecucion(self, job_id: str) -> None:
        # `_job_id` determinista: si por un reintento HTTP se encolara dos
        # veces el mismo job, Arq descarta el duplicado.
        await self.pool.enqueue_job("ejecutar_tarea", job_id, _job_id=f"ejecutar:{job_id}")

    async def encolar_reanudacion(self, job_id: str, decision: dict[str, Any]) -> None:
        await self.pool.enqueue_job(
            "reanudar_tarea", job_id, decision, _job_id=f"reanudar:{job_id}"
        )
