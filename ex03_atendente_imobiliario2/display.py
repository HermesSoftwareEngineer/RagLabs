from PIL import Image
from app import graph
from io import BytesIO

# VISUALIZANDO O GRÁFICO
try:
    image_path = graph.get_graph().draw_mermaid_png() # Obter os dados binários da imagem
    img = Image.open(BytesIO(image_path)) # Abrir a imagem com pillow
    img.show() # Mostrar a imagem
except Exception as e:
    print(f"Erro ao exibir a imagem: {e}")
    pass