"""Los nodos del orquestador: dos especialistas, el supervisor y el sintetizador."""

from app.agents.analyst_agent import crear_analista
from app.agents.research_agent import crear_investigador
from app.agents.supervisor import crear_supervisor, enrutar
from app.agents.synthesizer import crear_sintetizador

__all__ = [
    "crear_analista",
    "crear_investigador",
    "crear_sintetizador",
    "crear_supervisor",
    "enrutar",
]
