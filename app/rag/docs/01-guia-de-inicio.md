# Guía de inicio de nimbus-sdk

`nimbus-sdk` es la librería cliente oficial en Python para la plataforma de pagos
Nimbus. Cubre autorización, captura, reembolsos y consulta de conciliación.

## Instalación

```bash
pip install nimbus-sdk
```

Requiere Python 3.10 o superior. La librería no trae dependencias nativas: solo
`httpx` y `pydantic`. No necesita compilador ni ruedas específicas de plataforma,
así que instala igual en Linux, macOS y Windows, y dentro de imágenes `slim`.

## Autenticación

El SDK se autentica con un par de claves: una clave pública (`pk_live_...`) para
identificar al comercio y una clave secreta (`sk_live_...`) que firma cada
solicitud. La clave secreta nunca debe viajar al navegador.

```python
from nimbus import NimbusClient

cliente = NimbusClient(
    api_key="sk_live_9f2c...",
    merchant_id="mrc_00412",
    timeout=30.0,
)
```

Si no se pasa `api_key`, el constructor lee la variable de entorno
`NIMBUS_API_KEY`. Si tampoco existe, lanza `ConfiguracionInvalida` antes de hacer
cualquier llamada de red. Es deliberado que falle en el constructor y no en la
primera operación: un error de configuración tiene que aparecer al levantar el
proceso, no cuando entra el primer pago de producción.

Las claves se rotan desde el back-office. Una clave revocada deja de funcionar de
inmediato, sin período de gracia, así que conviene desplegar la clave nueva antes
de revocar la vieja.

## Primer cobro

El flujo mínimo son dos llamadas: autorizar y capturar.

```python
autorizacion = cliente.autorizar_pago(
    monto=150000,             # en centavos, siempre entero
    moneda="ARS",
    tarjeta_token="tok_2h8d...",
    idempotency_key="pedido-8831",
)

if autorizacion.estado == "autorizada":
    recibo = cliente.capturar_pago(autorizacion.id)
```

`monto` se expresa **en centavos y como entero**. Pasar `1500.50` lanza
`MontoInvalido`: la librería no redondea flotantes por decisión de diseño, porque
el redondeo silencioso es la causa más frecuente de descuadres de conciliación.

Separar la autorización de la captura permite reservar los fondos cuando el
comprador confirma y cobrarlos recién cuando el pedido sale del depósito. Si el
negocio cobra siempre en el mismo instante, igual hay que hacer las dos llamadas:
el atajo que las combinaba se eliminó en la versión 3.0.

## Entorno de pruebas

Con `ambiente="sandbox"` el SDK apunta a `https://sandbox.api.nimbus.io` y acepta
las tarjetas de prueba documentadas en la referencia. En sandbox ninguna
operación mueve dinero real y los reembolsos se liquidan de forma inmediata.

```python
cliente = NimbusClient(api_key="sk_test_...", ambiente="sandbox")
```

Las tarjetas de prueba permiten forzar cada resultado: una termina siempre en
aprobación, otra en fondos insuficientes, otra en tarjeta robada y otra en
timeout del adquirente. Conviene tener un test de integración por cada una, no
solo por el camino feliz.

Los identificadores de sandbox y los de producción **no son intercambiables**.
Un `aut_` generado en sandbox devuelve 404 contra la API de producción.

## Tiempos de espera y reintentos

El valor por defecto de `timeout` es 30 segundos. El SDK reintenta
automáticamente hasta 3 veces ante errores de red y respuestas 5xx, con backoff
exponencial y jitter. Los errores 4xx **no se reintentan**: son determinísticos y
reintentar solo suma latencia.

La política se puede ajustar al construir el cliente:

```python
from nimbus import NimbusClient, PoliticaDeReintentos

cliente = NimbusClient(
    api_key="sk_live_...",
    reintentos=PoliticaDeReintentos(
        intentos_maximos=5,
        espera_inicial=0.5,
        espera_maxima=8.0,
        jitter=True,
    ),
)
```

El jitter no es decorativo: sin él, todas las instancias que fallaron al mismo
tiempo reintentan al mismo tiempo y le pegan al adquirente en una sola ráfaga,
justo cuando se está recuperando.

Cualquier operación que mueva dinero necesita `idempotency_key`, y por eso el
reintento automático es seguro: si el primer intento llegó a destino y solo se
perdió la respuesta, el reintento devuelve la operación original en vez de
duplicar el cobro.

## Cliente asincrónico

Para aplicaciones sobre `asyncio` existe `AsyncNimbusClient`, con la misma
superficie de métodos pero esperables:

```python
from nimbus import AsyncNimbusClient

async with AsyncNimbusClient(api_key="sk_live_...") as cliente:
    autorizacion = await cliente.autorizar_pago(
        monto=150000,
        moneda="ARS",
        tarjeta_token=token,
        idempotency_key=f"pedido-{pedido.id}",
    )
```

Usarlo como contexto asincrónico es lo recomendado: al salir cierra el pool de
conexiones. Sin el `async with` hay que llamar a `await cliente.cerrar()` a mano;
olvidarse filtra sockets en procesos de larga duración.

El cliente es seguro para compartir entre corrutinas y conviene instanciarlo una
sola vez por proceso. Crear uno por request desperdicia el pool y multiplica los
handshakes TLS.

## Registro y trazabilidad

El SDK emite logs por el logger `nimbus`, sin configurar handlers propios. Cada
línea incluye el `request_id` de la respuesta, que es lo que pide el soporte de
Nimbus para rastrear una operación.

```python
import logging

logging.getLogger("nimbus").setLevel(logging.INFO)
```

En nivel `DEBUG` se registran los cuerpos de las solicitudes con los datos
sensibles enmascarados: del token de tarjeta solo quedan los últimos cuatro
caracteres y la clave secreta nunca se escribe. Aun así, `DEBUG` en producción es
mala idea por volumen.

## Antes de salir a producción

- Cambiar `ambiente` a `production` y las claves `sk_test_` por `sk_live_`.
- Confirmar que todos los montos se calculan en enteros desde la base de datos.
- Definir cómo se generan las `idempotency_key` y que sean estables ante un
  reintento del usuario: el identificador del pedido sirve, un UUID nuevo por
  click no.
- Configurar el endpoint de webhooks y verificar la firma.
- Guardar el `request_id` de cada operación junto al pedido.
