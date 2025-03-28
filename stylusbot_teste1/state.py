from typing import Annotated, Dict, List, Optional

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AtendimentoState(TypedDict):
    messages: Annotated[List[dict], add_messages]  # Histórico de mensagens
    session_id: Optional[str]  # ID do atendimento
    user_id: Optional[str]  # ID do usuário (caso esteja logado)
    intent: Optional[str]  # Intenção detectada do usuário
    status: Optional[str]  # Status do atendimento (ex: "em andamento")
    requested_info: Optional[Dict]
    timestamp_start: Optional[str]  # Hora de início do atendimento
    timestamp_last_message: Optional[str]  # Hora da última mensagem
    interaction_count: Optional[int]  # Número de interações
    is_escalated: Optional[bool]  # Se o atendimento foi passado para um humano
    search_results: Optional[List[dict]]  # Última busca de imóveis