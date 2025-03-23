from main import graph
from langgraph.types import Command

human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})

config = {"configurable": {"thread_id": "1"}}
def stream_graph_update(user_input: str):
    response = graph.invoke({'messages': user_input}, config, human_command)
    print("Assistent: ", response['messages'][-1].content)

while True:
    user_input = input("User: ")
    if user_input.lower() in ['quit', 'sair', 'q']:
        break
    stream_graph_update(user_input)