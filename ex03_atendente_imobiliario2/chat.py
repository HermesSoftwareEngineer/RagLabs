from app import graph

config = {"configurable": {"thread_id": "1"}}
def stream_graph_update(user_input: str):
    response = graph.invoke({'messages': user_input}, config)
    print("Assistent: ", response['messages'][-1].content)

while True:
    user_input = input("User: ")
    if user_input.lower() in ['quit', 'sair', 'q']:
        break
    stream_graph_update(user_input)