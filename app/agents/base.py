"""Lo que tienen en común los dos especialistas.

Cada especialista es un **subgrafo ReAct** (`create_react_agent`) envuelto en
un nodo del grafo principal. El envoltorio hace tres cosas que el subgrafo
solo no hace:

1. **Aislar el contexto.** El subgrafo no recibe `state["messages"]`. Recibe
   un único `HumanMessage` armado a medida con la instrucción del supervisor y
   el contexto que ese rol necesita. Nada del historial de otros agentes, nada
   de la metadata del orquestador.
2. **Registrar la contribución.** Del historial interno del subgrafo se extrae
   la evidencia (cada llamada a herramienta con su resultado) y el texto final,
   y se guardan como una `Contribucion` con nombre de agente y ronda.
3. **Chequear y depurar el aporte.** Antes de devolverlo al supervisor se
   corren chequeos deterministas. Los bloqueantes van a `problemas` y deciden
   el criterio `respaldo` de la rúbrica: el supervisor no tiene que "darse
   cuenta" de que el analista inventó un número, se lo dice un chequeo. La
   depuración (no bloqueante) saca del texto lo que el agente escribió fuera de
   su rol y lo registra en `avisos`.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from app import config
from app.state import Contribucion, Especialista, EstadoOrquestador, Evidencia, rondas_de

with warnings.catch_warnings():
    # En LangGraph 1.x `create_react_agent` está marcado para migrar a
    # `langchain.agents.create_agent`. Sigue funcionando igual, y es la API que
    # nombra la consigna.
    warnings.simplefilter("ignore", DeprecationWarning)
    from langgraph.prebuilt import create_react_agent

#: Lo que responde `create_react_agent` cuando se queda sin pasos.
SIN_PASOS_REACT = "Sorry, need more steps to process this request."

#: Firma de un chequeo: recibe la contribución armada y el estado con el que
#: se armó la entrada del agente, y devuelve la lista de problemas.
Chequeo = Callable[[Contribucion, EstadoOrquestador], list[str]]

#: Firma de una depuración: devuelve el contenido corregido y los avisos que
#: explican qué se cambió.
Depuracion = Callable[[Contribucion, EstadoOrquestador], tuple[str, list[str]]]


@dataclass(frozen=True)
class Especificacion:
    """Todo lo que distingue a un especialista de otro."""

    nombre: Especialista
    prompt_sistema: str
    herramientas: Sequence[BaseTool]
    #: Arma el mensaje de entrada a partir del estado. Es donde se decide qué
    #: contexto ve el agente — y, sobre todo, cuál no.
    armar_entrada: Callable[[EstadoOrquestador], str]
    #: Chequeos bloqueantes: lo que encuentran va a `problemas`.
    chequeos: Sequence[Chequeo] = ()
    #: Depuración no bloqueante: saca del contenido lo que el agente escribió
    #: fuera de su rol, y deja constancia en `avisos`.
    depurar: Depuracion | None = None


def _contenido_como_dato(contenido: Any) -> Any:
    """El `content` de un `ToolMessage` vuelto a estructura cuando se puede."""
    if isinstance(contenido, str):
        try:
            return json.loads(contenido)
        except json.JSONDecodeError:
            return contenido
    return contenido


def extraer_evidencia(mensajes: Sequence[Any]) -> list[Evidencia]:
    """Empareja cada `tool_call` con el `ToolMessage` que le respondió."""
    llamadas: dict[str, dict[str, Any]] = {}
    for mensaje in mensajes:
        if isinstance(mensaje, AIMessage):
            for llamada in mensaje.tool_calls:
                llamadas[llamada["id"]] = llamada
    evidencia: list[Evidencia] = []
    for mensaje in mensajes:
        if isinstance(mensaje, ToolMessage) and mensaje.tool_call_id in llamadas:
            llamada = llamadas[mensaje.tool_call_id]
            evidencia.append(
                Evidencia(
                    herramienta=llamada["name"],
                    argumentos=llamada["args"],
                    resultado=_contenido_como_dato(mensaje.content),
                )
            )
    return evidencia


def texto_final(mensajes: Sequence[Any]) -> str:
    """El último texto que escribió el modelo del especialista."""
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, AIMessage) and not mensaje.tool_calls:
            contenido = mensaje.content
            if isinstance(contenido, list):  # bloques de contenido (Anthropic)
                contenido = "".join(
                    b.get("text", "") for b in contenido if isinstance(b, dict)
                )
            return str(contenido).strip()
    return ""


# --- Chequeos comunes -------------------------------------------------------


def chequeo_uso_herramientas(contribucion: Contribucion, estado: EstadoOrquestador) -> list[str]:
    """Un especialista que no llamó a ninguna herramienta respondió de memoria."""
    if not contribucion["evidencia"]:
        return ["No usó ninguna herramienta: el aporte no tiene respaldo."]
    return []


def chequeo_errores_sin_resolver(contribucion: Contribucion, estado: EstadoOrquestador) -> list[str]:
    """Una herramienta devolvió `error` y ninguna llamada posterior a la misma
    herramienta salió bien. Un error seguido de un reintento exitoso no es un
    problema: es el ciclo de recuperación funcionando."""
    problemas: list[str] = []
    ultima_por_herramienta: dict[str, Evidencia] = {}
    for evidencia in contribucion["evidencia"]:
        ultima_por_herramienta[evidencia["herramienta"]] = evidencia
    for nombre, evidencia in ultima_por_herramienta.items():
        resultado = evidencia["resultado"]
        if isinstance(resultado, dict) and "error" in resultado:
            problemas.append(f"`{nombre}` terminó en error: {resultado['error']}")
        elif isinstance(resultado, str) and resultado.lower().startswith("error"):
            problemas.append(f"`{nombre}` terminó en error: {resultado[:160]}")
    return problemas


def chequeo_respuesta_vacia(contribucion: Contribucion, estado: EstadoOrquestador) -> list[str]:
    if not contribucion["contenido"]:
        return ["El agente no escribió ninguna conclusión."]
    return []


CHEQUEOS_COMUNES: tuple[Chequeo, ...] = (
    chequeo_respuesta_vacia,
    chequeo_uso_herramientas,
    chequeo_errores_sin_resolver,
)


# --- Nodo -------------------------------------------------------------------


def crear_nodo_especialista(
    especificacion: Especificacion, modelo: BaseChatModel
) -> Callable[[EstadoOrquestador], Any]:
    """Construye el nodo del grafo principal para un especialista.

    El subgrafo ReAct se compila **una vez**, acá, y se reutiliza en cada
    intervención. Cada intervención arranca con un historial nuevo: el
    especialista no recuerda su ronda anterior salvo por lo que el supervisor
    le ponga en la instrucción de corrección.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        subgrafo = create_react_agent(
            modelo,
            tools=list(especificacion.herramientas),
            prompt=especificacion.prompt_sistema,
            name=especificacion.nombre,
        )

    async def nodo(estado: EstadoOrquestador) -> dict[str, Any]:
        entrada = especificacion.armar_entrada(estado)
        ronda = rondas_de(estado.get("contribuciones", []), especificacion.nombre) + 1
        problemas_extra: list[str] = []

        try:
            resultado = await subgrafo.ainvoke(
                {"messages": [HumanMessage(content=entrada)]},
                config={"recursion_limit": config.LIMITE_RECURSION_ESPECIALISTA},
            )
            mensajes = resultado["messages"]
        except GraphRecursionError:
            # El especialista entró en un bucle de herramientas. No se propaga:
            # se registra como problema y el supervisor decide qué hacer.
            mensajes = []
            problemas_extra.append(
                f"Se cortó al llegar a {config.LIMITE_RECURSION_ESPECIALISTA} "
                "super-pasos sin terminar (posible bucle de herramientas)."
            )

        contenido = texto_final(mensajes)
        if contenido == SIN_PASOS_REACT:
            # `create_react_agent` se adelanta al `recursion_limit`: cuando le
            # quedan pocos pasos, en vez de lanzar la excepción responde con
            # este texto fijo. No es una conclusión del agente.
            contenido = ""
            problemas_extra.append(
                f"Se quedó sin pasos ({config.LIMITE_RECURSION_ESPECIALISTA}) sin "
                "terminar (posible bucle de herramientas)."
            )

        contribucion = Contribucion(
            agente=especificacion.nombre,
            ronda=ronda,
            instruccion=estado.get("instruccion_actual", ""),
            contenido=contenido,
            evidencia=extraer_evidencia(mensajes),
            problemas=[],
            avisos=[],
        )
        problemas = problemas_extra + [
            p
            for chequeo in (*CHEQUEOS_COMUNES, *especificacion.chequeos)
            for p in chequeo(contribucion, estado)
        ]
        contribucion["problemas"] = list(dict.fromkeys(problemas))  # sin duplicados
        if especificacion.depurar is not None:
            contribucion["contenido"], contribucion["avisos"] = especificacion.depurar(
                contribucion, estado
            )

        return {
            "contribuciones": [contribucion],
            # Lo único que va a `messages` es la conclusión, con el nombre del
            # agente. Es el registro legible de la conversación, no el canal
            # por el que trabajan los agentes.
            "messages": [
                AIMessage(
                    content=contribucion["contenido"] or "(sin conclusión)",
                    name=especificacion.nombre,
                )
            ],
        }

    nodo.__name__ = f"nodo_{especificacion.nombre}"
    return nodo


# --- Utilidades para los chequeos de cada rol -------------------------------

_NUMERO = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\w])")


def lecturas_por_token(texto: str) -> list[tuple[str, set[float]]]:
    """Cada número del texto, tal como aparece, con todas sus lecturas posibles.

    `6.680.000`, `6,680,000` y `6680000` tienen que ser el mismo número, y
    `58,83` y `58.83` también. Un token ambiguo como `1.250` tiene dos lecturas
    (1,25 y 1250) y no hay forma de saber cuál quiso decir el modelo; por eso
    se devuelven las dos, y quien chequea respaldo acepta el token si
    **cualquiera** de sus lecturas está respaldada.
    """
    resultado: list[tuple[str, set[float]]] = []
    for token in _NUMERO.findall(texto):
        candidatos = {token.replace(".", "").replace(",", "."), token.replace(",", "")}
        if token.count(".") + token.count(",") == 1:
            candidatos.add(token.replace(",", "."))
        valores: set[float] = set()
        for candidato in candidatos:
            try:
                valores.add(float(candidato))
            except ValueError:
                continue
        if valores:
            resultado.append((token, valores))
    return resultado


def numeros_en(texto: str) -> set[float]:
    """Todas las lecturas de todos los números de un texto, en un solo conjunto."""
    return set().union(*(lecturas for _, lecturas in lecturas_por_token(texto)), set())


def numeros_en_estructura(dato: Any) -> set[float]:
    """Todos los números que aparecen en un resultado de herramienta."""
    if isinstance(dato, bool):
        return set()
    if isinstance(dato, (int, float)):
        return {float(dato)}
    if isinstance(dato, str):
        return numeros_en(dato)
    if isinstance(dato, dict):
        return set().union(*(numeros_en_estructura(v) for v in dato.values()), set())
    if isinstance(dato, (list, tuple)):
        return set().union(*(numeros_en_estructura(v) for v in dato), set())
    return set()


#: Números que no necesitan respaldo: conteos chicos y el 100 de "× 100".
CONSTANTES_TRIVIALES: frozenset[float] = frozenset({*map(float, range(11)), 100.0})


def respaldado(valor: float, respaldos: set[float]) -> bool:
    """`valor` coincide con algún respaldo, admitiendo redondeo al informar.

    58.8 respalda a 58.83; 39300 respalda a 3930000 centavos (se prueba cada
    respaldo también dividido por 100, porque pasar de centavos a unidades es
    la única conversión que hacen los agentes).
    """
    if valor in CONSTANTES_TRIVIALES:
        return True
    for respaldo in respaldos:
        for candidato in (respaldo, respaldo / 100):
            if abs(valor - candidato) <= max(0.051, abs(candidato) * 0.001):
                return True
    return False


def cifras_sin_respaldo(texto: str, respaldos: set[float]) -> list[str]:
    """Los números de `texto` (tal como están escritos) sin ningún respaldo.

    Un token ambiguo (`1.250`) se acepta si **cualquiera** de sus lecturas
    está respaldada: acusar un número correcto manda a corregir algo que
    estaba bien, que es peor que dejar pasar una ambigüedad.
    """
    return list(
        dict.fromkeys(
            token
            for token, lecturas in lecturas_por_token(texto)
            if not any(respaldado(valor, respaldos) for valor in lecturas)
        )
    )


def formatear_cifras(valores: Sequence[str | float], maximo: int = 6) -> str:
    """Lista legible para un mensaje de problema: enteros sin `.0` ni `e+06`."""
    def uno(v: str | float) -> str:
        if isinstance(v, str):
            return v
        return f"{v:.0f}" if float(v).is_integer() else f"{v:g}"

    return ", ".join(uno(v) for v in list(valores)[:maximo])


# --- Escala de los montos ---------------------------------------------------

_MONEDAS = r"(?:ARS|CLP|USD|BRL|MXN)"
_CIFRA = r"\d[\d.,]*\d|\d"
_MONTO_CON_MONEDA = re.compile(
    rf"{_MONEDAS}\s*\$?\s*(?P<despues>{_CIFRA})|(?P<antes>{_CIFRA})\s*{_MONEDAS}\b"
)


def montos_en_centavos(dato: Any) -> set[float]:
    """Los valores de todos los campos `monto_centavos` de una estructura."""
    if isinstance(dato, dict):
        propios = {
            float(v) for k, v in dato.items()
            if k == "monto_centavos" and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        return propios.union(*(montos_en_centavos(v) for v in dato.values()))
    if isinstance(dato, (list, tuple)):
        return set().union(*(montos_en_centavos(v) for v in dato), set())
    return set()


def montos_mal_escalados(texto: str, centavos: set[float]) -> list[str]:
    """Cifras escritas junto a una moneda que en realidad son centavos.

    "6,300,000 CLP" cuando el dato es `monto_centavos: 6300000` es un monto
    cien veces más grande que el real. Es el error de escala más fácil de
    cometer con estos datos y el que ningún chequeo de "¿el número existe?"
    atrapa, porque el número existe: lo que está mal es la unidad.
    """
    malos: list[str] = []
    for coincidencia in _MONTO_CON_MONEDA.finditer(texto):
        token = coincidencia.group("despues") or coincidencia.group("antes")
        lecturas = set().union(*(l for _, l in lecturas_por_token(token)), set())
        es_centavos = any(v >= 100 and v in centavos for v in lecturas)
        es_unidades = any(abs(v - c / 100) < 0.01 for v in lecturas for c in centavos)
        if es_centavos and not es_unidades:
            malos.append(token)
    return list(dict.fromkeys(malos))


def tachar_cifras(texto: str, tokens: Sequence[str]) -> str:
    """Reemplaza cada aparición exacta de esos números por una marca visible."""
    a_tachar = set(tokens)
    return _NUMERO.sub(
        lambda m: "[cifra descartada]" if m.group(0) in a_tachar else m.group(0), texto
    )
