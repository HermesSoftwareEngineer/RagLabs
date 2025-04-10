from pydantic import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from typing import Union, Literal, Dict, Any
from utils import act_class_to_dict_schema

class ResponderEngraçado(BaseModel):
    type: Literal['ResponderEngraçado'] = Field('ResponderEngraçado', description='Type of responder')
    response: str = Field(
        description="Responder de forma engraçada!"
    )

class ResponderNormal(BaseModel):
    type: Literal['ResponderNormal'] = Field('ResponderNormal', description='Type of responder')
    response: str = Field(
        description="Responder normal"
    )

class Act(BaseModel):
    action: Union[ResponderEngraçado, ResponderNormal] = Field(
        description="Escolher qual a melhor forma para responder",
        discriminator='type' # Add a discriminator field
    )

# # Manually define the schema
# dict_schema = {
#     "name": "Act",
#     "description": "Escolher qual a melhor forma para responder",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "action": {
#                 "type": "object",
#                 "properties": {
#                     "type": {"type": "string", "enum": ["ResponderEngraçado", "ResponderNormal"]},
#                     "response": {"type": "string", "description": "Responder de forma engraçada ou normal"}
#                 },
#                 "required": ["type", "response"]
#             }
#         },
#         "required": ["action"]
#     }
# }

dict_schema = act_class_to_dict_schema(Act)

llm = ChatVertexAI(model_name="gemini-1.5-flash")
agent = llm.with_structured_output(dict_schema)

response = agent.invoke("Olá! Eu sou o Hermes. Comprimente de maneira")
print(response)