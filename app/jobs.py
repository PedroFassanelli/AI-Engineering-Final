"""Estado de los trabajos y de las conversaciones en Redis.

Cada job (= un turno de una conversación) es un hash `nimbus:job:<id>` más una
lista `nimbus:job:<id>:nodos` con el recorrido del grafo. Cada conversación es
un hash `nimbus:conv:<id>` más la lista de sus jobs `nimbus:conv:<id>:jobs`.
El ciclo de vida de un job:

    PENDING ──► RUNNING ──┬──► DONE
       ▲                  ├──► FAILED      (excepción, timeout, worker caído)
       │                  └──► AWAITING_APPROVAL
       │                              │
       └──────── POST /approve ───────┘

Dos reglas sostienen todo:

1. **Las transiciones son atómicas** (`transicionar`, con `WATCH`/`MULTI`).
   Solo se pasa de un estado a otro si el estado actual es el esperado. Dos
   aprobaciones simultáneas del mismo job: una gana, la otra recibe 409. Un
   reintento de la cola sobre un job que ya corrió: no hace nada.
2. **Ningún job queda colgado.** Si el grafo lanza una excepción, el worker lo
   pasa a FAILED (ver `app/worker.py`). Si el worker muere sin poder hacerlo,
   el job deja de recibir latidos, y la próxima lectura lo pasa a FAILED
   (`obtener`). El cliente que hace polling siempre termina viendo un estado
   final.

Y una tercera, por las conversaciones:

3. **Un turno a la vez por conversación.** Dos turnos corriendo sobre el mismo
   `thread_id` se pisarían los checkpoints. `crear` registra el job activo de
   la conversación en la misma transacción que crea el job, y rechaza un turno
   nuevo (`ConversacionOcupada` → 409) mientras el anterior no haya terminado.
   Un turno en `AWAITING_APPROVAL` también ocupa la conversación: hasta que
   alguien decida, el hilo está pausado en el `interrupt`.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app import config


class EstadoJob(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


TERMINALES: frozenset[EstadoJob] = frozenset({EstadoJob.DONE, EstadoJob.FAILED})

#: Campos que se guardan como JSON dentro del hash.
CAMPOS_JSON: frozenset[str] = frozenset(
    {"aprobacion_pendiente", "aprobacion", "resultado_accion", "error", "task_completed"}
)

#: Campos numéricos (epoch, segundos).
CAMPOS_NUMERICOS: frozenset[str] = frozenset(
    {"latido", "duracion_segundos", "duracion_reanudacion_segundos"}
)

#: Campos enteros.
CAMPOS_ENTEROS: frozenset[str] = frozenset({"turno"})


class JobNoEncontrado(LookupError):
    pass


class ConversacionNoEncontrada(LookupError):
    pass


class ConversacionOcupada(RuntimeError):
    """La conversación tiene un turno sin terminar."""

    def __init__(self, conversacion_id: str, job_id: str, estado: str) -> None:
        super().__init__(f"La conversación {conversacion_id} tiene el turno {job_id} en {estado}.")
        self.conversacion_id = conversacion_id
        self.job_id = job_id
        self.estado = estado


def ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AlmacenJobs:
    def __init__(
        self,
        redis: Redis,
        *,
        prefijo: str | None = None,
        ttl_segundos: int | None = None,
        sin_latido_segundos: int | None = None,
        ttl_conversacion_segundos: int | None = None,
    ) -> None:
        # `decode_responses=True` es obligatorio: el resto del módulo trabaja con str.
        self.redis = redis
        self.prefijo = prefijo or config.PREFIJO_REDIS
        self.ttl = ttl_segundos or config.TTL_JOB_SEGUNDOS
        self.sin_latido = sin_latido_segundos or config.SIN_LATIDO_SEGUNDOS
        self.ttl_conversacion = ttl_conversacion_segundos or config.TTL_CONVERSACION_SEGUNDOS

    # --- Claves -----------------------------------------------------------

    def clave(self, job_id: str) -> str:
        return f"{self.prefijo}:job:{job_id}"

    def clave_nodos(self, job_id: str) -> str:
        return f"{self.clave(job_id)}:nodos"

    @property
    def clave_indice(self) -> str:
        return f"{self.prefijo}:jobs"

    def clave_conversacion(self, conversacion_id: str) -> str:
        return f"{self.prefijo}:conv:{conversacion_id}"

    def clave_jobs_conversacion(self, conversacion_id: str) -> str:
        return f"{self.clave_conversacion(conversacion_id)}:jobs"

    # --- Serialización ----------------------------------------------------

    @staticmethod
    def _a_redis(campos: dict[str, Any]) -> dict[str, str]:
        salida: dict[str, str] = {}
        for nombre, valor in campos.items():
            if nombre in CAMPOS_JSON:
                salida[nombre] = json.dumps(valor, ensure_ascii=False, default=str)
            elif isinstance(valor, EstadoJob):
                salida[nombre] = valor.value
            else:
                salida[nombre] = "" if valor is None else str(valor)
        return salida

    @staticmethod
    def _desde_redis(crudo: dict[str, str]) -> dict[str, Any]:
        job: dict[str, Any] = {}
        for nombre, valor in crudo.items():
            if nombre in CAMPOS_JSON:
                job[nombre] = json.loads(valor) if valor else None
            elif nombre in CAMPOS_NUMERICOS:
                job[nombre] = float(valor) if valor else None
            elif nombre in CAMPOS_ENTEROS:
                job[nombre] = int(valor) if valor else None
            else:
                job[nombre] = valor or None
        return job

    # --- Operaciones ------------------------------------------------------

    async def crear(self, pregunta: str, conversacion_id: str | None = None) -> dict[str, Any]:
        """Crea el job de un turno. Sin `conversacion_id`, abre una conversación nueva.

        Atómico con `WATCH` sobre la conversación: entre "el turno anterior ya
        terminó" y "este es el turno activo" no puede colarse otro `POST`.
        """
        nueva = conversacion_id is None
        conversacion_id = conversacion_id or uuid.uuid4().hex
        clave_conv = self.clave_conversacion(conversacion_id)
        job_id = uuid.uuid4().hex

        if not nueva:
            # Si el turno activo murió sin avisar, leerlo lo pasa a FAILED (y
            # libera la conversación) antes de decidir.
            activo = await self.redis.hget(clave_conv, "job_activo")
            if activo:
                await self.obtener(activo)

        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(clave_conv)
                    conversacion = await pipe.hgetall(clave_conv)
                    if not nueva and not conversacion:
                        await pipe.unwatch()
                        raise ConversacionNoEncontrada(conversacion_id)
                    activo = conversacion.get("job_activo") or ""
                    if activo:
                        estado_activo = await pipe.hget(self.clave(activo), "estado")
                        if estado_activo and estado_activo not in TERMINALES:
                            await pipe.unwatch()
                            raise ConversacionOcupada(conversacion_id, activo, estado_activo)

                    turno = int(conversacion.get("turnos") or 0) + 1
                    momento = ahora_iso()
                    campos = {
                        "job_id": job_id,
                        "conversacion_id": conversacion_id,
                        "turno": turno,
                        "estado": EstadoJob.PENDING,
                        "pregunta": pregunta,
                        "creado_en": momento,
                        "actualizado_en": momento,
                        "latido": time.time(),
                    }
                    conv = {"turnos": turno, "job_activo": job_id, "actualizada_en": momento}
                    if nueva:
                        conv |= {"conversacion_id": conversacion_id, "creada_en": momento}

                    pipe.multi()
                    pipe.hset(self.clave(job_id), mapping=self._a_redis(campos))
                    pipe.expire(self.clave(job_id), self.ttl)
                    pipe.zadd(self.clave_indice, {job_id: time.time()})
                    pipe.hset(clave_conv, mapping=self._a_redis(conv))
                    pipe.rpush(self.clave_jobs_conversacion(conversacion_id), job_id)
                    pipe.expire(clave_conv, self.ttl_conversacion)
                    pipe.expire(self.clave_jobs_conversacion(conversacion_id), self.ttl_conversacion)
                    await pipe.execute()
                    break
                except WatchError:
                    continue
        return await self.obtener(job_id)  # type: ignore[return-value]

    async def obtener_conversacion(self, conversacion_id: str) -> dict[str, Any] | None:
        crudo = await self.redis.hgetall(self.clave_conversacion(conversacion_id))
        if not crudo:
            return None
        conversacion = self._desde_redis(crudo)
        conversacion["turnos"] = int(crudo.get("turnos") or 0)
        conversacion["job_ids"] = await self.redis.lrange(
            self.clave_jobs_conversacion(conversacion_id), 0, -1
        )
        return conversacion

    async def _liberar_conversacion(self, job_id: str) -> None:
        """Quita el job terminado como turno activo de su conversación.

        Es limpieza, no la garantía: aunque esto no llegara a correr, `crear`
        mira el estado del job activo y un job terminado no bloquea.
        """
        conversacion_id = await self.redis.hget(self.clave(job_id), "conversacion_id")
        if not conversacion_id:
            return
        clave_conv = self.clave_conversacion(conversacion_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(clave_conv)
                    if await pipe.hget(clave_conv, "job_activo") != job_id:
                        await pipe.unwatch()
                        return
                    pipe.multi()
                    pipe.hset(clave_conv, mapping={"job_activo": "", "actualizada_en": ahora_iso()})
                    pipe.expire(clave_conv, self.ttl_conversacion)
                    await pipe.execute()
                    return
                except WatchError:
                    continue

    async def obtener(self, job_id: str) -> dict[str, Any] | None:
        crudo = await self.redis.hgetall(self.clave(job_id))
        if not crudo:
            return None
        job = self._desde_redis(crudo)

        if job["estado"] == EstadoJob.RUNNING and self._sin_latido(job):
            # El worker murió (o se colgó) sin poder marcar el fallo. Se marca
            # acá, al leer, para que el cliente no haga polling para siempre.
            await self.fallar(
                job_id,
                "WorkerSinLatido",
                f"El job no recibió señales del worker en {self.sin_latido} s.",
                desde=(EstadoJob.RUNNING,),
            )
            return await self.obtener(job_id)

        job["nodos"] = await self.redis.lrange(self.clave_nodos(job_id), 0, -1)
        return job

    def _sin_latido(self, job: dict[str, Any]) -> bool:
        latido = job.get("latido") or 0.0
        return time.time() - latido > self.sin_latido

    async def transicionar(
        self,
        job_id: str,
        desde: Iterable[EstadoJob],
        hacia: EstadoJob,
        **campos: Any,
    ) -> bool:
        """Cambia el estado solo si el actual está en `desde`. Atómico.

        Devuelve `False` si el estado actual no era el esperado (otro proceso
        ya lo movió). Lanza `JobNoEncontrado` si el job no existe.
        """
        permitidos = {EstadoJob(e).value for e in desde}
        clave = self.clave(job_id)
        momento = ahora_iso()
        cambios = {**campos, "estado": hacia, "actualizado_en": momento, "latido": time.time()}
        if hacia in TERMINALES:
            cambios.setdefault("terminado_en", momento)

        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(clave)
                    actual = await pipe.hget(clave, "estado")
                    if actual is None:
                        await pipe.unwatch()
                        raise JobNoEncontrado(job_id)
                    if actual not in permitidos:
                        await pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.hset(clave, mapping=self._a_redis(cambios))
                    pipe.expire(clave, self.ttl)
                    pipe.expire(self.clave_nodos(job_id), self.ttl)
                    await pipe.execute()
                    break
                except WatchError:
                    # Alguien escribió el job entre el WATCH y el EXEC: se
                    # vuelve a leer y se decide de nuevo.
                    continue
        if hacia in TERMINALES:
            await self._liberar_conversacion(job_id)
        return True

    async def latido(self, job_id: str, nodo: str | None = None) -> None:
        """El worker avisa que sigue vivo (y, si corresponde, qué nodo terminó)."""
        clave = self.clave(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            campos: dict[str, Any] = {"latido": time.time(), "actualizado_en": ahora_iso()}
            if nodo:
                campos["nodo_actual"] = nodo
                pipe.rpush(self.clave_nodos(job_id), nodo)
                pipe.expire(self.clave_nodos(job_id), self.ttl)
            pipe.hset(clave, mapping=self._a_redis(campos))
            await pipe.execute()

    async def fallar(
        self,
        job_id: str,
        tipo: str,
        mensaje: str,
        *,
        desde: Iterable[EstadoJob] = (EstadoJob.PENDING, EstadoJob.RUNNING),
    ) -> bool:
        return await self.transicionar(
            job_id, desde, EstadoJob.FAILED, error={"tipo": tipo, "mensaje": mensaje[:2000]}
        )

    async def listar(self, limite: int = 20, estado: EstadoJob | None = None) -> list[dict[str, Any]]:
        """Los jobs más recientes. Limpia del índice los que ya expiraron."""
        ids = await self.redis.zrevrange(self.clave_indice, 0, max(limite * 5, 100) - 1)
        jobs: list[dict[str, Any]] = []
        for job_id in ids:
            job = await self.obtener(job_id)
            if job is None:
                await self.redis.zrem(self.clave_indice, job_id)
                continue
            if estado is None or job["estado"] == estado:
                jobs.append(job)
            if len(jobs) >= limite:
                break
        return jobs
