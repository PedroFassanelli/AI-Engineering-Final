# Webhooks y conciliación en nimbus-sdk

Los webhooks son la única forma confiable de enterarse de eventos que ocurren
fuera del ciclo de la solicitud: contracargos, liquidaciones y reembolsos
iniciados por el emisor.

## Eventos disponibles

| Evento | Cuándo se emite |
|---|---|
| `pago.autorizado` | La autorización fue aprobada por el emisor. |
| `pago.capturado` | Se cobró una autorización. |
| `pago.reembolsado` | Se acreditó un reembolso total o parcial. |
| `pago.vencido` | Una autorización se liberó sin capturarse. |
| `contracargo.abierto` | El tarjetahabiente desconoció el consumo. |
| `contracargo.resuelto` | Se cerró la disputa, a favor del comercio o del comprador. |
| `liquidacion.disponible` | El resumen diario de liquidación está listo. |

La suscripción se configura por endpoint en el back-office. Un mismo comercio
puede tener varios endpoints, cada uno con su propio subconjunto de eventos y su
propio secreto de firma.

## Verificación de firma

Cada webhook llega con el encabezado `X-Nimbus-Signature`, que contiene un HMAC
SHA-256 del cuerpo crudo usando el `webhook_secret` del comercio como clave.

```python
from nimbus.webhooks import verificar_firma, FirmaInvalida

@app.post("/webhooks/nimbus")
async def recibir(request):
    cuerpo = await request.body()          # bytes crudos, sin parsear
    try:
        evento = verificar_firma(
            cuerpo=cuerpo,
            firma=request.headers["X-Nimbus-Signature"],
            secreto=os.environ["NIMBUS_WEBHOOK_SECRET"],
            tolerancia=300,
        )
    except FirmaInvalida:
        return Response(status_code=400)
    procesar(evento)
    return Response(status_code=200)
```

La firma se calcula sobre el **cuerpo crudo en bytes**. Serializar el JSON ya
parseado y firmar eso falla siempre: basta con que cambie el orden de las claves
o el espaciado para que el HMAC no coincida. Es el error de integración más
reportado al soporte de Nimbus.

En frameworks que parsean el cuerpo automáticamente hay que pedir el acceso al
original: `await request.body()` en FastAPI o Starlette, `request.get_data()` en
Flask, `request.body` en Django. Leer el cuerpo después de que el framework lo
consumió puede devolver vacío, así que conviene verificar antes de cualquier otra
cosa.

El parámetro `tolerancia` (300 segundos por defecto) descarta eventos cuyo
timestamp se aleje más de ese margen del reloj local, para cortar ataques de
reenvío. Si el servidor tiene el reloj corrido, empiezan a rechazarse webhooks
legítimos: vale la pena tener NTP andando antes de culpar a la firma.

El `webhook_secret` se rota desde el back-office y se admiten dos secretos
válidos en simultáneo durante 24 horas, para poder desplegar el nuevo sin perder
eventos. `verificar_firma()` acepta una lista en `secreto` para cubrir ese
período.

## Reintentos de entrega

Si el endpoint no responde `2xx` en menos de 10 segundos, Nimbus reintenta la
entrega con backoff exponencial durante **72 horas**: a los 30 segundos, 5
minutos, 30 minutos, 2 horas, y luego cada 6 horas. Después de 72 horas el evento
se descarta y queda visible en el back-office.

De ahí se desprende una regla concreta: el endpoint tiene que responder rápido y
procesar después. Validar la firma, encolar el evento y devolver 200 lleva
milisegundos; hacer el trabajo pesado dentro del request lleva a que Nimbus
reintente mientras el primer intento todavía está corriendo.

Los webhooks pueden llegar **duplicados y fuera de orden**. El campo `evento_id`
es único y estable: hay que deduplicar por ese valor y no asumir que
`pago.autorizado` llega antes que `pago.capturado`.

La forma práctica de deduplicar es una tabla con `evento_id` como clave primaria
y un insert que ignore el conflicto. Si el insert no agregó nada, el evento ya se
procesó y se devuelve 200 sin hacer nada más.

## Contracargos

`contracargo.abierto` llega cuando el tarjetahabiente desconoce el consumo. El
monto se retiene de la próxima liquidación en el mismo momento, antes de que la
disputa se resuelva.

El comercio tiene **21 días corridos** para presentar evidencia: comprobante de
entrega, registro de sesión, comunicación con el comprador. Pasado el plazo, la
disputa se cierra a favor del tarjetahabiente sin revisión.

`contracargo.resuelto` trae el campo `resultado`, que vale `ganado` o `perdido`.
Si es `ganado`, el monto retenido vuelve en la liquidación siguiente.

## Conciliación diaria

El archivo de liquidación se publica todos los días a las 06:00 hora de Buenos
Aires y cubre las operaciones del día hábil anterior.

```python
liquidacion = cliente.descargar_liquidacion(fecha=date(2026, 3, 11))
for linea in liquidacion.lineas:
    print(linea.operacion_id, linea.monto_bruto, linea.comision, linea.monto_neto)
```

`monto_neto` es `monto_bruto` menos `comision` menos `impuestos`. Conciliar
contra `monto_bruto` es un error frecuente: lo que efectivamente se acredita en
la cuenta bancaria del comercio es `monto_neto`.

La liquidación de un día incluye líneas negativas por los reembolsos acreditados
y por los contracargos abiertos en esa fecha, aunque la operación original sea de
semanas atrás. Por eso el total de una liquidación no coincide con la suma de las
ventas del día, y buscar esa coincidencia es la causa más común de investigar un
descuadre que no existe.

Los fondos se acreditan según el plan contratado: 48 horas hábiles en el plan
estándar, 24 en el plan Enterprise. La fecha comprometida viene en el campo
`acreditacion_estimada` de la liquidación.
