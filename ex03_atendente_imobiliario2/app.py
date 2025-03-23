from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader, UnstructuredExcelLoader
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage

# Carregando as variáveis de ambiente
load_dotenv()

# Iniciando o LLM
llm = ChatVertexAI(model_name='gemini-1.5-flash')

# Iniciando os embeddings
embeddings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
vector_store_instrucoes = InMemoryVectorStore(embeddings)
vector_store_imoveis = InMemoryVectorStore(embeddings)

# Carregando os documentos
loader_instrucoes = TextLoader(r'C:\Users\Asus\PROJETOS_DEV\RagLabs\ex03_atendente_imobiliario2\instrucoes.txt', encoding="utf-8")
docs_instrucoes = loader_instrucoes.load()

loader_imoveis = UnstructuredExcelLoader(r'C:\Users\Asus\PROJETOS_DEV\RagLabs\ex03_atendente_imobiliario2\dados_imoveis.xlsx', encoding="utf-8")
docs_imoveis = loader_imoveis.load()

# Dividindo os documentos
text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
split_instrucoes = text_splitter1.split_documents(docs_instrucoes)
split_imoveis = text_splitter2.split_documents(docs_imoveis)

# Adiciona os documentos aos vector_stores
vector_store_imoveis.add_documents(split_imoveis)
vector_store_instrucoes.add_documents(split_instrucoes)


# ----------------------------------------------------------------------------
# PREPARANDO AS FERRAMENTAS
# ----------------------------------------------------------------------------

@tool
def buscar_imoveis (query: str) -> str:
    """Buscar imóveis disponíveis para locação/aluguel"""
    docs = vector_store_imoveis.similarity_search(query)
    response = "\n\n".join([doc.page_content for doc in docs])
    return response

@tool
def ler_instrucoes (query: str) -> str:
    """Instruções sobre processos administrativos da Imobiliária Stylus"""
    docs = vector_store_instrucoes.similarity_search(query)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [buscar_imoveis, ler_instrucoes]
tools_node = ToolNode(tools)

# ----------------------------------------------------------------------------
# PREPARANDO O PROMPT_TEMPLATE
# ----------------------------------------------------------------------------

prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            "Você é um atendente da Imobiliária Stylus chamado 'StylusBot'. Responda da melhor maneira possível. Utilize as ferramentas sempre que precisar. Se não souber a resposta, basta dizer que não sabe."
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)

# ----------------------------------------------------------------------------
# PREPARANDO OS NÓS
# ----------------------------------------------------------------------------

def chatbot(state: MessagesState):
    """Nó inicial para consultar ou responder"""
    prompt = prompt_template.invoke(state['messages'])
    llm_with_tools = llm.bind_tools(tools, tool_choice="any")
    response = llm_with_tools.invoke(prompt)
   
    return {'messages': response}

def gerar_resposta(state: MessagesState):
    """Gerando respostas com base nas ferramentas"""
    list_messages_tools = [m for m in reversed(state['messages']) if m.type == 'tool']
    docs_messages_tools = '\n\n'.join(m.content for m in reversed(list_messages_tools))

    system_message = SystemMessage(
        'Você é um assistente imobiliário. Sempre formate as respostas de imóveis da seguinte maneira:\n\n'
        '**🏠 Imóvel encontrado!**\n\n'
        '* 📍 **Endereço:** [endereço do imóvel]\n'
        '* 🏡 **Descrição:** [descrição curta do imóvel]\n'
        '* 💰 **Valor:** [valor do aluguel + encargos]\n'
        '* 🔗 **Links de anúncio:** [links disponíveis]\n\n'
        'Se não encontrar imóveis, apenas diga "No momento, não temos imóveis disponíveis com essas características."'
        '\n\n' + docs_messages_tools
    )

    conversation_messages = [
        m for m in state['messages']
        if m.type in ('system', 'human')
        or (m.type == 'ia' and not m.tool_calls)
    ]
    
    prompt = [system_message] + conversation_messages

    return {'messages': llm.invoke(prompt)}

# ----------------------------------------------------------------------------
# PREPARANDO O FLUXO
# ----------------------------------------------------------------------------

graph_builder = StateGraph(MessagesState)

# Adicionando os nós
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_node('tools', tools_node)
graph_builder.add_node('gerar_resposta', gerar_resposta)

# Adicionando as pontes
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_conditional_edges('chatbot', tools_condition, {'tools': 'tools', END: END})
graph_builder.add_edge('tools', 'gerar_resposta')
graph_builder.add_edge('gerar_resposta', END)

# Compilando e preparando o fluxo
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)