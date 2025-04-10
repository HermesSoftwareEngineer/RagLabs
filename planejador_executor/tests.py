from main import StatePlan, replanner, app
import asyncio

config = {"recursion_limit": 50}
inputs = {"messages": "Olá! Tudo bem?"}

async def main():
    async for event in app.astream(inputs, config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

asyncio.run(main())

# state: StatePlan = {
#     "input": "Desenvolver um aplicativo de lista de tarefas.",
#     "plan": [
#         "1. Definir os requisitos do app",
#         "2. Criar wireframes",
#         "3. Configurar o ambiente de desenvolvimento",
#         "4. Desenvolver funcionalidades básicas"
#     ],
#     "past_steps": [
#         ("1. Definir os requisitos do app", "Feito com sucesso"),
#         ("2. Criar wireframes", "Concluído com base nos requisitos")
#     ],
#     "response": ""
# }

# # Chamada do replanner
# output = replanner.invoke({
#     "input": state["input"],
#     "plan": "\n".join(state["plan"]),  # pois o prompt espera um texto com o plano
#     "past_steps": "\n".join([f"{s[0]} - {s[1]}" for s in state["past_steps"]]),
#     "messages": [],
#     "content": []
# })

# print(output)