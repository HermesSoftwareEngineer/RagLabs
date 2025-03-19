from main import graph

def stream_graph_update(user_input: str):
    response = graph.invoke({'messages': user_input})
    print("Assistent: ", response['messages'][-1].content)

while True:
    user_input = input("User: ")
    if user_input.lower() in ['quit', 'sair', 'q']:
        break
    stream_graph_update(user_input)