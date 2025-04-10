from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator
from langgraph.graph.message import add_messages
from langchain_core.utils.function_calling import convert_to_openai_function
from typing_extensions import TypedDict
from typing import Annotated, List, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Union
from pydantic import BaseModel, Field
from typing import Literal
from utils import act_class_to_dict_schema

llm = ChatVertexAI(model_name="gemini-1.5-flash")

# Definindo o estado

class StatePlan(TypedDict):
    messages: Annotated[List[str], add_messages]
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]

# Estrutura de resposta do LLM para a etapa de planejamento

class Response(BaseModel):
    """Responder ao usuário"""
    type: Literal["ResponderUsuário"] = Field("ResponderUsuário",
        description="Type of responder"
    )
    response: str = Field(
        description="Responder ao usuário logo"
    )

class Plan(BaseModel):
    type: Literal["plan"] = Field("plan",
        description="Type of plan"
    )
    steps: List[str] = Field(
        description="Diferentes etapas a seguir, devem estar em ordem de classificação"
    )

class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description="Ação a ser executada. Você pode responder ou replanejar.",
        discriminator="type"
    )

# dict_schema_plan = act_class_to_dict_schema(Plan)
# print(f"dict_schema_plan: {dict_schema_plan}")

dict_schema_act = act_class_to_dict_schema(Act)
print(f"dict_schema_act: {dict_schema_act}")

# Nó de planejamento

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é o Stylus Bot, um atendente imobiliário. Crie um plano simples passo a passo para responder o usuário. \
            Este plano deve envolver tarefas individuais que, se executadas corretamente, produzirão a resposta correta. Não adicione nenhuma etapa supérflua. \
            O resultado da etapa final deve ser a resposta final. Certifique-se de que cada etapa tenha todas as informações necessárias - não pule etapas \
            Responda sempre com o TYPE: plan"
        ),
        (
            "placeholder",
            "{messages}"
        ),
    ]
)

planner = planner_prompt | llm.with_structured_output(Plan)

# Etapa de replanejamento

replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é o Stylus Bot, um atendente imobiliário. Crie um plano simples passo a passo para responder o usuário. As vezes o plano é simples, as vezes precisa de mais de um passo. SEJA CLARO NOS PASSOS.
            Este plano deve envolver tarefas individuais que, se executadas corretamente, produzirão a resposta correta. Não adicione etapas supérfluas. 
            O resultado da etapa final deve ser a resposta final. Certifique-se de que cada etapa tenha todas as informações necessárias - não pule etapas.

            Seu objetivo era este:
            {input}

            Seu plano original era este:
            {plan}

            Você atualmente fez as seguintes etapas:
            {past_steps}

            Atualize seu plano adequadamente. Se não forem necessárias mais etapas e você puder retornar ao usuário, responda com isso. Caso contrário, preencha o plano. Adicione apenas etapas ao plano que ainda PRECISAM ser feitas. Não retorne etapas feitas anteriormente como parte do plano. Se quiser responder logo ao usuário.
            
            REPITO, SEJA CLARO NOS PASSOS"""
        ),
        (
            "placeholder",
            "{messages}"
        )
    ]
)

replanner = replanner_prompt | llm.with_structured_output(dict_schema_act)

# elaborando as funções dos nós

def execute_step(state: StatePlan):
    print("ETAPA DE EXECUÇÃO DO PLANO")
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""
        Para o seguinte plano: {plan_str}\n\n
        Você é responsável pela tarefa {1}: {task}.
    """
    agent_response = llm.invoke(task_formatted)

    return {
        "past_steps": [(task, agent_response.content)]
    }

def plan_step(state: StatePlan):
    print("ETAPA DE PLANEJAMENTO")
    plan = planner.invoke(
        {
            "messages": [
                {"role": "user", "content": state["messages"][-1].content}
            ]
        }
    )
    print(f"plan: {plan}")

    return {
        "plan": plan.steps
    }

def replan_steps(state: StatePlan):
    print("ETAPA DE REPLANEJAMENTO")
    print(f"STATE: {state}")
    output = replanner.invoke(
        {
            "input": state["messages"][-1].content,
            "past_steps": state["past_steps"],
            "plan": state["plan"],
            "messages": state["messages"]
        }
    )
    print(f"OUTPUT: {output}")
    if isinstance(output["action"], Response):
        return {"response": output["action"].response, "messages": output["action"].response}
    else:
        return {"plan": output["action"].steps}
    
def should_end(state: StatePlan):
    "ETAPA DE CONFERIR SE RESPONDE OU ITERA"
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"
    
# Construindo o gráfico

graph_builder = StateGraph(StatePlan)

# Definindo os nós
graph_builder.add_node("planner", plan_step)
graph_builder.add_node("agent", execute_step)
graph_builder.add_node("replan", replan_steps)

# Definindo as pontes
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "agent")
graph_builder.add_edge("agent", "replan")
graph_builder.add_conditional_edges("replan", should_end, ["agent", END])

# Compilando o grafo
app = graph_builder.compile()