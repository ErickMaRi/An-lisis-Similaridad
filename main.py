# Importación de bibliotecas
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import networkx as nx

# Nombre del archivo de caché
cache_file = "papers_cache.json"

def load_cache():
    """
    Carga datos del caché desde el archivo JSON si existe.

    Returns:
        dict: Datos cargados desde el caché o un diccionario vacío si el archivo no existe.
    """
    if os.path.exists(cache_file):
        with open(cache_file, "r") as file:
            return json.load(file)
    return {}

def save_to_cache(paper_id, data):
    """
    Guarda datos relevantes de un artículo en el caché.

    Args:
        paper_id (str): Identificador único del artículo.
        data (dict): Datos del artículo a guardar en el caché.
    """
    # Extraer solo la información relevante
    relevant_data = {
        'title': data.get('title'),
        'authors': [author['name'] for author in data.get('authors', [])],
        'references': [ref['paperId'] for ref in data.get('references', [])]
    }
    cache = load_cache()
    cache[paper_id] = relevant_data
    with open(cache_file, "w") as file:
        json.dump(cache, file)

def get_paper_data_from_cache(paper_id):
    """
    Obtiene datos de un artículo desde el caché si está disponible.

    Args:
        paper_id (str): Identificador único del artículo.

    Returns:
        dict: Datos del artículo desde el caché o None si no está en el caché.
    """
    cache = load_cache()
    return cache.get(paper_id)

def get_semantic_scholar_data(paper_ids, debug=False):
    """
    Obtiene datos de artículos académicos desde la API de Semantic Scholar.

    Args:
        paper_ids (list): Lista de identificadores únicos de artículos académicos.

    Returns:
        list: Lista de datos de artículos académicos.
    """
    base_url = "https://api.semanticscholar.org/v1/paper"
    headers = {'User-Agent': 'MiAplicacionDeInvestigacion/1.0 (erick.marinrojas@ucr.ac.cr)'}
    papers = []
    iter = 0
    for paper_id in paper_ids:
        iter += 1
        cached_data = get_paper_data_from_cache(paper_id)
        if cached_data:
            if debug:
                print(f"#{iter}-Usando datos en caché para el artículo {paper_id}")
            papers.append(cached_data)
        else:
            if debug:
                print(f"#{iter}-Obteniendo datos para el artículo {paper_id}")
            response = requests.get(f"{base_url}/{paper_id}", headers=headers)
            if response.status_code == 200:
                paper_data = response.json()
                save_to_cache(paper_id, paper_data)
                papers.append(paper_data)
            else:
                if debug:
                    print(f"#{iter}-Error al obtener el artículo {paper_id}: {response.text}")
    return papers

def search_semantic_scholar(query, max_results=5):
    """
    Realiza una búsqueda de artículos en Semantic Scholar basada en una consulta.

    Args:
        query (str): Consulta de búsqueda.
        max_results (int): Número máximo de resultados a obtener.

    Returns:
        list: Lista de identificadores únicos de artículos encontrados.
    """
    base_search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {'User-Agent': 'MiAplicacionDeInvestigacion/1.0 (erick.marinrojas@ucr.ac.cr)'}
    params = {
        "query": query,
        "limit": max_results,
        "fields": "paperId,title,authors,references"
    }
    response = requests.get(base_search_url, headers=headers, params=params)
    if response.status_code == 200:
        search_results = response.json()
        paper_ids = [paper['paperId'] for paper in search_results.get('data', [])]
        return paper_ids
    else:
        print(f"Error en la búsqueda: {response.text}")
        return []

def procesar_datos_semantic_scholar(papers):
    """
    Procesa datos de artículos académicos obtenidos de Semantic Scholar.

    Args:
        papers (list): Lista de datos de artículos académicos.

    Returns:
        list: Lista de registros estructurados de artículos.
    """
    registros = []
    for paper in papers:
        registro = {
            'title': paper.get('title'),
            'id': paper.get('paperId'),
            'authors': [author['name'] if isinstance(author, dict) else author for author in paper.get('authors', [])],
            'references': [ref['paperId'] if isinstance(ref, dict) else ref for ref in paper.get('references', [])]
        }
        registros.append(registro)
    return registros

def calculate_similarity(df, debug=False):
    """
    Calcula la similitud entre artículos académicos basada en referencias compartidas.

    Args:
        df (DataFrame): DataFrame con registros de artículos académicos.
        debug (bool): Indica si se debe mostrar información de depuración.

    Returns:
        array: Matriz de similitud normalizada.
    """
    num_papers = len(df)
    similarity_matrix = np.zeros((num_papers, num_papers))
    for i in range(num_papers):
        for j in range(num_papers):
            if i != j:
                shared_refs_i = set(df.loc[i, 'references'])
                shared_refs_j = set(df.loc[j, 'references'])
                shared_refs = shared_refs_i.intersection(shared_refs_j)
                if debug:
                    print(f'Referencias compartidas entre el artículo {i+1} y el artículo {j+1}: {len(shared_refs)}')
                similarity_matrix[i][j] = len(shared_refs)
    norm_sim_matrix = cosine_similarity(similarity_matrix)
    return norm_sim_matrix

def plot_interactive_graph(similarity_matrix):
    """
    Genera un gráfico interactivo de matriz de similitud.

    Args:
        similarity_matrix (array): Matriz de similitud.

    """
    fig = go.Figure(data=go.Heatmap(z=similarity_matrix, colorscale='Viridis'))
    fig.update_layout(title='Gráfico Interactivo de Matriz de Similitud')
    fig.show()

def threshold_similarity_matrix(similarity_matrix, threshold):
    """
    Convierte una matriz de similitud en una matriz binaria basada en un umbral.

    Args:
        similarity_matrix (array): Matriz de similitud.
        threshold (float): Umbral para la binarización.

    Returns:
        array: Matriz binaria.
    """
    binary_matrix = np.zeros_like(similarity_matrix)
    binary_matrix[similarity_matrix > threshold] = 1
    return binary_matrix

def plot_binary_matrix_networkx(binary, red_count=60, blue_count=60, green_count=60):
    """
    Visualiza una matriz binaria como un gráfico de red utilizando NetworkX.

    Args:
        binary (array): Matriz binaria.
        red_count (int): Cantidad de nodos rojos.
        blue_count (int): Cantidad de nodos azules.
        green_count (int): Cantidad de nodos verdes.
    """
    threshold = 0.5
    binary[binary < threshold] = 0
    binary[binary >= threshold] = 1
    G = nx.Graph(binary)
    pos = nx.spring_layout(G)

    # Define los colores de los nodos según la consulta
    node_colors = (['red'] * red_count) + (['blue'] * blue_count) + (['green'] * green_count)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, cmap=plt.get_cmap('jet'))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

def generate_color_dict(red_count, blue_count, green_count):
    """
    Genera un diccionario de colores para los nodos en el gráfico de NetworkX.

    Args:
        red_count (int): Cantidad de nodos rojos.
        blue_count (int): Cantidad de nodos azules.
        green_count (int): Cantidad de nodos verdes.

    Returns:
        dict: Diccionario de colores para los nodos.
    """
    color_dict = {}
    for _ in range(red_count):
        color_dict['Forward-Forward Algorithm'] = 'red'
    for _ in range(blue_count):
        color_dict['machine learning'] = 'blue'
    for _ in range(green_count):
        color_dict['nanowire networks'] = 'green'
    return color_dict

def process_queries(queries: dict, thresh=0.1, debug=False):
    """
    Procesa consultas, obtiene datos de artículos, calcula la similitud y visualiza los resultados.

    Args:
        queries (dict): Diccionario de consultas con la cantidad de resultados deseados.
        thresh (float): Umbral para la binarización de la matriz de similitud.
    """
    paper_ids = []
    for query, count in queries.items():
        paper_ids += search_semantic_scholar(query, max_results=count)

    papers = get_semantic_scholar_data(paper_ids, debug)
    records = procesar_datos_semantic_scholar(papers)
    df = pd.DataFrame(records)
    similarity_matrix = calculate_similarity(df)
    if debug:
        print(similarity_matrix)
    binary = threshold_similarity_matrix(similarity_matrix, thresh)

    np.fill_diagonal(binary, 0)
    if debug:
        print(binary)
    plot_binary_matrix_networkx(binary)

# Ejemplo de consultas
queries = {
    "Forward-Forward Algorithm": 60,
    "gaussian splatting": 60,
    "unsupervised learning": 60
}

# Procesar las consultas
process_queries(queries, debug=True)
