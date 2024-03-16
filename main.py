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
import matplotlib.cm as cm
import plotly.graph_objs as go
import networkx as nx
import random
from scipy.stats import entropy

used_ids = set()  # This set will keep track of all IDs to prevent collisions.
cache_file = "papers_cache.json" # Nombre del archivo de caché

def generate_unique_id(existing_ids, real_id_length):
    while True:
        # Generate a long integer ID by concatenating random digits to double the size of a real ID.
        fake_id = ''.join(str(random.randint(0, 9)) for _ in range(real_id_length * 2))
        # Ensure the fake ID hasn't already been used.
        if fake_id not in existing_ids:
            return fake_id

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
    global used_ids
    real_id_length = 40  
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
                # Only save to cache if a paperId is present
                if 'paperId' in paper_data and paper_data['paperId'] not in used_ids:
                    used_ids.add(paper_data['paperId'])
                    save_to_cache(paper_id, paper_data)
                    papers.append(paper_data)
                else:
                    # Generate a unique fake paperId that is double the size of a real ID
                    fake_paper_id = generate_unique_id(used_ids, real_id_length)
                    used_ids.add(fake_paper_id)  # Add the fake ID to the used IDs set
                    paper_data['paperId'] = fake_paper_id  # Assign the fake ID
                    save_to_cache(fake_paper_id, paper_data)
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
    headers = {'User-Agent': 'elpepepipe'}
    params = {
        "query": query,
        "limit": max_results,
        "fields": "paperId"
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

def evaluate_network(G):
    """
    Evalúa la calidad de la red enfocándose en la diversidad de la dimensionalidad aparente local de los nodos.

    Args:
        G (Graph): Grafo de NetworkX.

    Returns:
        float: Puntaje de calidad de la red.
    """
    avg_clustering = nx.average_clustering(G)
    degrees = np.array([degree for _, degree in G.degree()])
    degree_entropy = entropy(degrees + 1)  # Agregamos 1 para evitar el logaritmo de 0
    degree_variation_coefficient = np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else 0
    network_density = nx.density(G)

    # Definir pesos para cada factor (estos valores pueden ser ajustados)
    weight_clustering = 0
    weight_entropy = 0  # Peso para la entropía de la distribución de grados
    weight_variation_coefficient = -1.2/1.2803997096318764  # Peso para el coeficiente de variación de grados
    weight_density = -0.4/0.03837781181972103

    # Calcular el puntaje final, incorporando las nuevas métricas
    score = -1/(weight_clustering * avg_clustering +
             weight_entropy * degree_entropy +
             weight_variation_coefficient * degree_variation_coefficient +
             weight_density * network_density)
    #print(f'score: {score} = {1 * avg_clustering, 1 * degree_entropy, weight_variation_coefficient * degree_variation_coefficient, weight_density * network_density}')
    return score

def find_optimal_threshold(similarity_matrix, debug=False, min_threshold=0.001, max_threshold=0.5, step=0.001):
    """
    Encuentra el umbral óptimo para la matriz de similitud.

    Args:
        similarity_matrix (array): Matriz de similitud.
        min_threshold (float): Umbral mínimo a considerar.
        max_threshold (float): Umbral máximo a considerar.
        step (float): Paso entre umbrales consecutivos.

    Returns:
        float: Umbral óptimo encontrado.
    """
    best_threshold = min_threshold
    best_score = -1

    for threshold in np.arange(min_threshold, max_threshold, step):
        binary_matrix = np.zeros_like(similarity_matrix)
        binary_matrix[similarity_matrix > threshold] = 1
        G = nx.from_numpy_array(binary_matrix)

        score = evaluate_network(G)

        if score > best_score:
            best_score = score
            best_threshold = threshold
    if debug:
        print(f'Best threshold is: {best_threshold}, with a score of: {best_score}.')
    return best_threshold

def threshold_similarity_matrix(similarity_matrix, threshold='auto', debug=False):
    """
    Convierte una matriz de similitud en una matriz binaria basada en un umbral, densidad, o de forma autónoma.

    Args:
        similarity_matrix (array): Matriz de similitud.
        threshold (float, str): Umbral para la binarización, 'densidad', o 'auto'.
        density_percentage (int): Porcentaje de conexiones más fuertes a mantener si se utiliza 'densidad'.

    Returns:
        array: Matriz binaria.
    """
    if threshold == 'auto':
        threshold = find_optimal_threshold(similarity_matrix, debug=debug)
    
    # Crear la matriz binaria
    binary_matrix = np.zeros_like(similarity_matrix)
    binary_matrix[similarity_matrix > threshold] = 1
    return binary_matrix

def plot_binary_matrix_networkx(binary, df, id_to_query_map):
    # Generar colores
    unique_queries = list(set(id_to_query_map.values()))
    n_queries = len(unique_queries)
    colors = cm.rainbow(np.linspace(0, 1, n_queries))

    # Crear un mapeo de consultas a colores
    query_to_color = {query: colors[i] for i, query in enumerate(unique_queries)}
    default_color = 'grey'  # Color por defecto para IDs no mapeados

    # Asignar colores a los nodos basados en su consulta de origen
    node_colors = []
    for idx in range(len(df)):
        paper_id = df.loc[idx, 'id']
        query = id_to_query_map.get(paper_id)
        if query:
            node_colors.append(query_to_color[query])
        else:
            node_colors.append(default_color)  # Usar color por defecto

    # Crear y visualizar la red
    G = nx.Graph(binary)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    labels = {i: df.loc[i, 'title'] for i in range(len(df))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
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

def plot_interactive_networkx(similarity_matrix, df, id_to_query_map, paper_ids):
    """
    Visualiza una matriz de similitud como un gráfico de red interactivo utilizando Plotly.
    
    Args:
        binary (array): Matriz binaria.
        df (DataFrame): DataFrame con registros de artículos académicos.
        id_to_query_map (dict): Diccionario mapeando los IDs de los artículos a sus respectivas consultas.
    """
    # Convertir la matriz de similitud en un grafo y obtener la posición de los nodos
    G = nx.from_numpy_array(similarity_matrix)
    
    # Identificar los componentes conectados y filtrar los nodos no conectados
    connected_components = list(nx.connected_components(G))
    # Podemos decidir qué componentes mantener (por ejemplo, aquellos con más de 1 nodo)
    nodes_to_keep = {node for component in connected_components for node in component if len(component) > 1}
    # Creamos un subgrafo solo con los nodos conectados
    G = G.subgraph(nodes_to_keep).copy()
    pos = nx.spring_layout(G)  # Puede necesitar reajuste dado que se han eliminado nodos

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y, node_text, node_symbols, node_colors_by_query, node_colors_by_connectivity = [], [], [], [], [], []
    shapes = ['triangle-up', 'square', 'cross', 'diamond', 'circle', 'star', 'hexagon', 'pentagon']
    unique_queries = set(id_to_query_map.values())
    colors = cm.rainbow(np.linspace(0, 1, len(unique_queries)))
    query_shapes = {query: shapes[i % len(shapes)] for i, query in enumerate(unique_queries)}
    query_to_color = {query: colors[i] for i, query in enumerate(unique_queries)}

    # Color y forma por defecto
    default_color = 'rgba(150,150,150,0.5)'
    default_shape = 'circle'
    #print(df)
    real_id_length = 17  # Assuming real paperIds are 17 digits long
    for node in G.nodes():
        paper_id = paper_ids[int(node)]
        if paper_id is None:
            # If there's a None value, generate a fake ID and replace it
            fake_paper_id = generate_unique_id(used_ids, real_id_length)
            used_ids.add(fake_paper_id)
            df.at[node, 'id'] = fake_paper_id  # Update the DataFrame with the fake ID
            paper_id = fake_paper_id  # Use the new fake ID
        query = id_to_query_map.get(paper_id, None)
        #print(f'node is: {node}, and paper id is: {paper_id}.')
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        node_text.append(df.loc[node, 'title'])
        node_symbols.append(query_shapes.get(query, default_shape))
        node_colors_by_query.append(
            'rgba({}, {}, {}, 0.8)'.format(
                *map(lambda x: int(x*255), query_to_color.get(query, (0.6, 0.6, 0.6)))
            ) if query is not None else default_color
        )
        node_colors_by_connectivity.append(len(list(G.adj[node])))
    
    node_trace_by_query = go.Scatter(
        x=node_x, y=node_y, text=node_text, mode='markers', hoverinfo='text',
        marker=dict(
            size=10,
            color=node_colors_by_query,
            symbol=node_symbols
        )
    )
    
    node_trace_by_connectivity = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(
            size=10,
            color=node_colors_by_connectivity, 
            colorscale='Viridis', 
            showscale=True,
            symbol=node_symbols
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace_by_query, node_trace_by_connectivity], layout=go.Layout(
        title='Red de artículos interactiva',
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))

    for query, shape in query_shapes.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker_symbol=shape,
            marker=dict(
                size=10, 
                color='rgba({}, {}, {}, 0.8)'.format(*map(lambda x: int(x*255), query_to_color[query]))
            ),
            name=query
        ))

    fig.show()


def process_queries(queries: dict, thresh="auto", debug=False):
    '''
    Procesa los queries, correlaciona los papers recogidos, los transforma
    en una matriz binaria basado en el threshold, luego esa matriz de conexiones
    se usa para dibujar una figura interactiva en el buscador para:
        1- El grafo de la matriz binaria
        2- La matriz binaria
        3- El grafo de la matriz de similitud
    
    '''
    paper_ids_set = set()
    paper_ids_ordered = []
    id_to_query_map = {}  # Nuevo diccionario para mapear IDs a consultas

    for query, count in queries.items():
        current_paper_ids = search_semantic_scholar(query, max_results=count)
        for paper_id in current_paper_ids:
            if paper_id not in paper_ids_set:
                paper_ids_set.add(paper_id)
                paper_ids_ordered.append(paper_id)
                id_to_query_map[paper_id] = query  # Mapear el ID al query

    #print(f'id to query map: {id_to_query_map}')

    papers = get_semantic_scholar_data(paper_ids_ordered, debug)
    records = procesar_datos_semantic_scholar(papers)
    df = pd.DataFrame(records)
    similarity_matrix = calculate_similarity(df)
    np.fill_diagonal(similarity_matrix, 0)
    if debug:
        print(df)

    binary = threshold_similarity_matrix(similarity_matrix, thresh, debug)
    np.fill_diagonal(binary, 0)
    if debug:
        print(binary)
    plot_interactive_networkx(binary, df, id_to_query_map, paper_ids_ordered)  # Pasar el mapeo a la función
    plot_interactive_graph(binary)
    plot_interactive_networkx(similarity_matrix, df, id_to_query_map, paper_ids_ordered)
    plot_interactive_graph(similarity_matrix)

# Ejemplo de consultas
queries = { #Máximo 45 resultados por query!!!!
    "CONDA architechture":45, "LSTM architecture": 45}

# Procesar las consultas
process_queries(queries, thresh="auto", debug=True)
