# Referencia de API de nimbus-sdk

Referencia de los métodos públicos de `NimbusClient`. Todos devuelven modelos
Pydantic; ninguno devuelve diccionarios crudos.

## `autorizar_pago()`

```python
def autorizar_pago(
    monto: int,
    moneda: str,
    tarjeta_token: str,
    idempotency_key: str,
    cuotas: int = 1,
    metadata: dict[str, str] | None = None,
) -> Autorizacion
```

Reserva fondos sin cobrarlos. La reserva vence a las **168 horas** (7 días); si
no se captura antes, se libera sola y el `id` deja de ser capturable.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `monto` | `int` | Importe en centavos. Debe ser mayor a 0. |
| `moneda` | `str` | Código ISO 4217 de tres letras: `ARS`, `USD`, `BRL`. |
| `tarjeta_token` | `str` | Token generado en el front con la clave pública. |
| `idempotency_key` | `str` | Identificador único de intento, máximo 64 caracteres. |
| `cuotas` | `int` | Entre 1 y 12. Solo aplica a `ARS`. |
| `metadata` | `dict` | Hasta 20 claves, valores de hasta 500 caracteres. |

El objeto `Autorizacion` que devuelve expone `id`, `estado`, `monto`,
`moneda`, `creada_en`, `vence_en` y `request_id`. El `estado` puede ser
`autorizada`, `rechazada` o `en_revision`; este último aparece cuando el emisor
pide validación adicional y se resuelve por webhook, no por polling.

## `capturar_pago()`

```python
def capturar_pago(autorizacion_id: str, monto: int | None = None) -> Recibo
```

Cobra una autorización previa. Si se omite `monto`, captura el total autorizado.
Se admite **captura parcial**: un `monto` menor al autorizado cobra esa parte y
libera el resto. Una autorización solo se puede capturar una vez.

La captura parcial es útil cuando el pedido se despacha incompleto. Lo que no se
puede es capturar dos veces la misma autorización para ir cobrando de a partes:
el primer `capturar_pago()` cierra la reserva y libera el saldo no cobrado. Si el
negocio necesita cobros escalonados, hay que autorizar por separado cada envío.

## `reembolsar()`

```python
def reembolsar(recibo_id: str, monto: int | None = None, motivo: str = "") -> Reembolso
```

Devuelve fondos de un pago ya capturado. Admite reembolsos parciales múltiples
mientras la suma no supere el monto capturado. El plazo máximo para reembolsar es
de **180 días** desde la captura.

El reembolso no es instantáneo para el tarjetahabiente: Nimbus lo acredita al
emisor el mismo día, pero el emisor puede tardar entre 2 y 10 días hábiles en
reflejarlo en el resumen. El `estado` del reembolso pasa por `pendiente`,
`acreditado` y eventualmente `fallido`, y cada transición emite un webhook.

El campo `motivo` es libre y aparece en el back-office y en el archivo de
liquidación. Conviene usar un vocabulario corto y consistente — `producto
defectuoso`, `arrepentimiento`, `error de cobro` — porque después se agrupa.

## `consultar_operacion()`

```python
def consultar_operacion(operacion_id: str) -> Autorizacion | Recibo | Reembolso
```

Devuelve el estado actual de cualquier operación a partir de su identificador. El
prefijo del `id` determina el tipo: `aut_` para autorizaciones, `rec_` para
recibos y `ref_` para reembolsos.

Este método está pensado para consultas puntuales y reconciliación manual, no
para enterarse de cambios de estado. Para eso están los webhooks: hacer polling
sobre `consultar_operacion()` consume cuota de rate limit y agrega latencia.

## `listar_operaciones()`

```python
def listar_operaciones(
    desde: date,
    hasta: date,
    estado: str | None = None,
    limite: int = 100,
    cursor: str | None = None,
) -> PaginaOperaciones
```

Devuelve operaciones paginadas por cursor. El `limite` máximo es 100. El rango
entre `desde` y `hasta` no puede superar los 31 días.

La paginación es por cursor y no por número de página a propósito: con offsets,
una operación nueva insertada mientras se recorre desplaza el resto y hace que
una fila aparezca dos veces o ninguna.

```python
cursor = None
while True:
    pagina = cliente.listar_operaciones(desde=d1, hasta=d2, cursor=cursor)
    for operacion in pagina.items:
        procesar(operacion)
    if not pagina.cursor_siguiente:
        break
    cursor = pagina.cursor_siguiente
```

## `descargar_liquidacion()`

```python
def descargar_liquidacion(fecha: date) -> Liquidacion
```

Devuelve el detalle de lo liquidado para una fecha. Está disponible a partir de
las 06:00 hora de Buenos Aires del día hábil siguiente. Pedir una fecha sin
liquidación publicada devuelve una liquidación vacía, no un error.

## Idempotencia

`idempotency_key` es obligatoria en toda operación que mueva dinero. Nimbus
guarda la respuesta asociada a esa clave durante **24 horas**: repetir la misma
solicitud con la misma clave devuelve la respuesta original en vez de crear una
operación nueva.

Reutilizar una `idempotency_key` con un cuerpo distinto devuelve el error
`E4090 - Conflicto de idempotencia`.

La clave tiene que derivarse de algo estable del lado del comercio — el
identificador del pedido, o el del pedido más el número de intento — y no de un
UUID generado en cada click. Si el comprador aprieta "pagar" dos veces y cada
click genera una clave nueva, la idempotencia no protege de nada y se cobran dos
autorizaciones.

Pasadas las 24 horas la clave se libera y una solicitud con la misma clave crea
una operación nueva. Para conciliaciones más largas hay que apoyarse en el
`id` de la operación, no en la clave.

## Límites de los campos de metadata

`metadata` acepta hasta 20 claves por operación, con nombres de hasta 40
caracteres y valores de hasta 500. Los valores se guardan como texto: un número
enviado como `int` vuelve como `str`. No hay que guardar ahí datos personales del
comprador — la metadata aparece en exportaciones y en el back-office, que tiene
un control de acceso más amplio que la base de datos del comercio.
