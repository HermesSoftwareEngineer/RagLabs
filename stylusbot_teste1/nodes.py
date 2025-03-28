from vector_store import llm
from state import AtendimentoState

def responder(state: AtendimentoState) -> AtendimentoState:
    llm_response = llm.invoke(state['messages'])
    print(state)
    return {'messages': llm_response}