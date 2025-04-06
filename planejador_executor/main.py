from typing_extensions import TypedDict
from typing import List, Annotated
import operator

# Definindo o estado do grafo
class StatePlan(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[tuple], operator.add]
    response: str

