# Manejo de errores en nimbus-sdk

Todas las excepciones del SDK heredan de `NimbusError` y exponen los atributos
`codigo`, `mensaje`, `reintentable` y `request_id`.

## Jerarquía de excepciones

```
NimbusError
├── ConfiguracionInvalida      (falta la API key, ambiente desconocido)
├── SolicitudInvalida          (4xx del lado del comercio)
│   ├── MontoInvalido
│   ├── TokenExpirado
│   └── ConflictoIdempotencia
├── PagoRechazado              (el emisor rechazó la operación)
├── LimiteDeTasa               (429)
└── ErrorDeServicio            (5xx, reintentable)
```

La jerarquía está pensada para que se pueda capturar con la granularidad que haga
falta: `except NimbusError` atrapa todo, `except SolicitudInvalida` solo los
errores atribuibles al código del comercio, y `except PagoRechazado` solo lo que
hay que mostrarle al comprador.

## Catálogo de códigos

| Código | Excepción | Reintentable | Significado |
|---|---|---|---|
| `E4001` | `MontoInvalido` | no | El monto no es un entero positivo en centavos. |
| `E4012` | `TokenExpirado` | no | El `tarjeta_token` venció; los tokens duran 15 minutos. |
| `E4013` | `SolicitudInvalida` | no | La moneda no es un código ISO 4217 válido. |
| `E4021` | `PagoRechazado` | no | Fondos insuficientes. |
| `E4022` | `PagoRechazado` | no | Tarjeta reportada como robada. Sin reintento. |
| `E4023` | `PagoRechazado` | no | El emisor no autorizó, sin detalle. |
| `E4031` | `SolicitudInvalida` | no | La autorización ya fue capturada. |
| `E4032` | `SolicitudInvalida` | no | La autorización venció (168 horas). |
| `E4090` | `ConflictoIdempotencia` | no | Misma clave, cuerpo distinto. |
| `E4290` | `LimiteDeTasa` | sí | Se superaron las 300 solicitudes por minuto. |
| `E5001` | `ErrorDeServicio` | sí | Falla interna del autorizador. |
| `E5030` | `ErrorDeServicio` | sí | El adquirente no respondió a tiempo. |

La columna **reintentable** es la que importa en el código: el SDK la respeta
automáticamente, y si se implementa un reintento propio hay que respetarla
también. Reintentar un `E4022` no lo convierte en aprobado; solo suma latencia y
alimenta las métricas de fraude del emisor.

## Cómo tratar un rechazo

`PagoRechazado` no es un error de programación: es una respuesta legítima del
emisor. Hay que capturarla y mostrar un mensaje al comprador, no registrarla como
excepción no controlada.

```python
from nimbus.errores import PagoRechazado, LimiteDeTasa

try:
    autorizacion = cliente.autorizar_pago(...)
except PagoRechazado as exc:
    mostrar_al_comprador(exc.mensaje)
except LimiteDeTasa as exc:
    encolar_para_reintento(exc.reintentar_en)
```

Un detalle que suele sorprender: Nimbus **no** informa el motivo real del rechazo
más allá de lo que da el emisor. `E4021` dice fondos insuficientes porque el
emisor lo dijo; `E4023` significa que el emisor rechazó sin explicar. Inventarle
al comprador un motivo que la API no dio genera reclamos de soporte.

Tampoco conviene mostrar el `codigo` crudo en la interfaz. Sirve en los logs, no
en la pantalla del comprador.

## Límite de tasa

El límite por defecto es de **300 solicitudes por minuto por comercio**. Al
superarlo, la API devuelve 429 y el SDK lanza `LimiteDeTasa` con el atributo
`reintentar_en`, en segundos. El reintento automático del SDK **respeta ese
valor** en lugar de aplicar su propio backoff.

El límite se cuenta sobre todas las llamadas, incluidas las de solo lectura como
`consultar_operacion()` y `listar_operaciones()`. La causa más común de llegar al
tope no son los pagos sino el polling: un proceso que consulta el estado de cada
operación pendiente cada pocos segundos consume la cuota entera sin hacer un solo
cobro. Para eso están los webhooks.

Los comercios del plan Enterprise tienen un límite negociado que se configura por
separado. Si la aplicación necesita ráfagas por encima de 300 por minuto —una
migración, una carga masiva— conviene pedirlo con anticipación en vez de
absorberlo con reintentos.

## Errores de servicio

`ErrorDeServicio` agrupa los 5xx y siempre es reintentable. `E5030` en
particular significa que el adquirente no contestó dentro de la ventana: la
operación puede haberse procesado igual del otro lado.

Por eso el reintento **tiene que llevar la misma `idempotency_key`**. Con la
clave repetida, si la operación se había procesado, Nimbus devuelve la original;
sin ella, se cobra dos veces. Es el escenario que convierte un incidente de
infraestructura en un descuadre contable.

## Trazabilidad

Cada respuesta incluye un `request_id` con formato `req_<26 caracteres>`. Al
abrir un ticket de soporte hay que incluirlo: sin el `request_id` no es posible
ubicar la operación en los registros del autorizador.

El `request_id` está en las respuestas exitosas y también en las excepciones, en
`exc.request_id`. Guardarlo junto al pedido, y no solo escribirlo en un log que
rota a los siete días, es lo que después permite responder un contracargo.

```python
try:
    recibo = cliente.capturar_pago(autorizacion.id)
except NimbusError as exc:
    registrar_fallo(pedido.id, exc.codigo, exc.request_id)
    raise
```
