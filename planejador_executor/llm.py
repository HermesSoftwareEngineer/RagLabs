from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI

# Iniciando variáveis de ambiente
load_dotenv()

# Iniciando LLM
llm = ChatVertexAI(model_name='gemini-1.5-flash')