# Changelog y guía de migración de nimbus-sdk

La librería sigue versionado semántico. Solo las versiones mayores rompen
compatibilidad.

## 3.1.0 — 2026-05-06

- Nuevo evento `pago.vencido`, que avisa cuando una autorización se libera sin
  haberse capturado.
- `listar_operaciones()` acepta el filtro `moneda`.
- `verificar_firma()` acepta una lista de secretos, para poder rotar el
  `webhook_secret` sin perder eventos durante la ventana de 24 horas.

## 3.0.0 — 2026-02-18

Cambios que rompen compatibilidad:

- Los montos pasan a ser **enteros en centavos** en toda la API pública. Los
  parámetros `monto` que aceptaban `float` ahora lanzan `MontoInvalido`.
- `idempotency_key` deja de ser opcional en `autorizar_pago()`, `capturar_pago()`
  y `reembolsar()`.
- Se eliminó `NimbusClient.pagar()`, el atajo que autorizaba y capturaba en un
  solo paso. Hay que llamar a `autorizar_pago()` y `capturar_pago()` por
  separado.
- El cliente asincrónico pasa de `NimbusAsyncClient` a `AsyncNimbusClient`, por
  coherencia con el resto del ecosistema.
- `listar_operaciones()` pasa de paginación por offset a paginación por cursor.
  El parámetro `pagina` desaparece y lo reemplaza `cursor`.

Novedades:

- Soporte de captura parcial en `capturar_pago()`.
- Nuevo método `descargar_liquidacion()`.
- Los reintentos ahora respetan el valor de `reintentar_en` que devuelve la API
  en los errores 429.
- La política de reintentos se configura con `PoliticaDeReintentos` en vez de
  parámetros sueltos en el constructor.

## 2.4.0 — 2025-11-04

- Se agregó el evento `contracargo.resuelto`.
- `listar_operaciones()` acepta el filtro `estado`.
- Corrección: `verificar_firma()` devolvía `True` cuando el encabezado
  `X-Nimbus-Signature` venía vacío. Es una corrección de seguridad; se recomienda
  actualizar de inmediato.

## 2.3.1 — 2025-09-22

- Corrección: el pool de conexiones no se cerraba al usar el cliente como
  contexto asincrónico, y filtraba sockets en procesos de larga duración.

## Migrar de 2.x a 3.0

El paso obligatorio es revisar cómo se calculan los montos. Si el sistema los
maneja como `Decimal` o `float` en unidades de moneda, hay que convertirlos a
centavos **antes** de llamar al SDK.

```python
# Antes (2.x)
cliente.pagar(monto=1500.50, moneda="ARS", tarjeta_token=token)

# Después (3.0)
centavos = int((Decimal("1500.50") * 100).to_integral_value())
autorizacion = cliente.autorizar_pago(
    monto=centavos,
    moneda="ARS",
    tarjeta_token=token,
    idempotency_key=f"pedido-{pedido.id}",
)
cliente.capturar_pago(autorizacion.id)
```

No hay que usar `round()` sobre un `float` para llegar a los centavos: el binario
flotante ya perdió precisión antes del redondeo. `Decimal` desde el origen, o
enteros desde la base de datos.

### Las claves de idempotencia

Al volverse obligatoria, la migración obliga a decidir de dónde sale cada clave.
La regla es que tiene que ser estable ante un reintento del mismo intento de
pago: si el comprador aprieta "pagar" dos veces, las dos llamadas deben llevar la
misma clave. Derivarla del identificador del pedido funciona; generar un UUID por
click no protege de nada.

### La paginación

El cambio de offset a cursor obliga a reescribir los recorridos:

```python
# Antes (2.x)
pagina = 1
while True:
    resultado = cliente.listar_operaciones(desde=d1, hasta=d2, pagina=pagina)
    if not resultado.items:
        break
    procesar(resultado.items)
    pagina += 1

# Después (3.0)
cursor = None
while True:
    resultado = cliente.listar_operaciones(desde=d1, hasta=d2, cursor=cursor)
    procesar(resultado.items)
    if not resultado.cursor_siguiente:
        break
    cursor = resultado.cursor_siguiente
```

### Orden sugerido de la migración

1. Actualizar la librería y correr la suite de tests contra sandbox.
2. Convertir los montos a enteros, empezando por la capa que lee de la base.
3. Definir el origen de las `idempotency_key`.
4. Reemplazar cada `pagar()` por el par autorizar/capturar.
5. Renombrar `NimbusAsyncClient` y reescribir los recorridos paginados.
6. Recién ahí apuntar a producción.

## Política de soporte

Cada versión mayor recibe correcciones de seguridad durante **12 meses** desde la
publicación de la siguiente mayor. La rama 2.x recibe correcciones hasta el
**18 de febrero de 2027**.

Las versiones menores no reciben soporte por separado: el soporte es sobre la
última menor de cada mayor. Estar en 3.0.0 cuando salió 3.1.0 significa que un
arreglo se entrega como 3.1.1, no como 3.0.1.
