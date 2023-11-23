# Análisis Similaridad
<image src="newplot.png" alt="Descripción de la imagen">

Este proyecto permite realizar un análisis de similitud entre artículos académicos utilizando la API de Semantic Scholar. Puede buscar artículos basados en consultas y calcular la similitud entre ellos en función de las referencias compartidas. También proporciona visualizaciones interactivas de los resultados.

## Requisitos

Asegúrese de tener las siguientes bibliotecas de Python instaladas antes de ejecutar el código:

- requests
- pandas
- numpy
- scikit-learn
- json
- os
- matplotlib
- plotly
- networkx

Puede instalar estas bibliotecas utilizando pip:

```
pip install requests
pip install pandas
pip install numpy
pip install scikit-learn
pip install json
pip install os
pip install matplotlib
pip install plotly
pip install networkx
```

## Uso

1. Clone este repositorio o descargue los archivos del proyecto.
2. Ejecute el archivo `main.py` para realizar un análisis de similitud entre artículos académicos.
3. Cambie los datos en el diccionario query para buscar n papers por query en Semantic Scholar
4. Los papers que tienen referencias comunes tienen valores altos en la matriz de similitud ([cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html))
5. Luego usando un threshold elegido arbitrariamente se produce una matriz con unos y ceros, que define el grafo a plotear usando [plotly](https://plotly.com/python/).
6. El gráfico interactivo se dibuja como un nuevo tab en su buscador por defecto y permite ver:
   - El trace cero para apreciar las líneas que conectan los nodos solamente.
   - El trace 1 para apreciar los nodos coloreados y con forma basada en su query de origen.
   - El trace 2 para apreciar los nodos coloreados por conectividad y con formas basadas en el query.
  
## Funciones Principales

- load_cache(): Carga datos del caché desde el archivo JSON si existe.
- save_to_cache(): Guarda datos relevantes de un artículo en el caché.
- get_paper_data_from_cache(): Obtiene datos de un artículo desde el caché si está disponible.
- get_semantic_scholar_data(): Obtiene datos de artículos académicos desde la API de Semantic Scholar.
- search_semantic_scholar(): Realiza una búsqueda de artículos en Semantic Scholar basada en una consulta.
- procesar_datos_semantic_scholar(): Procesa datos de artículos académicos obtenidos de Semantic Scholar.
- calculate_similarity(): Calcula la similitud entre artículos académicos basada en referencias compartidas.
- plot_interactive_graph(): Genera un gráfico interactivo de matriz de similitud.
- threshold_similarity_matrix(): Convierte una matriz de similitud en una matriz binaria basada en un umbral.
- plot_binary_matrix_networkx(): Visualiza una matriz binaria como un gráfico de NetworkX.
- plot_interactive_networkx(): Visualiza una matriz binaria como un gráfico de red interactivo utilizando Plotly.
- process_queries(): Procesa las consultas, calcula la matriz de similitud y grafica los gráficos interactivos y los mapas de calor.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulte el archivo [LICENSE](LICENSE.txt) para obtener más detalles.

---

**Nota:** Asegúrese de seguir las políticas de uso de la API de Semantic Scholar al realizar consultas, el sistema de cache evita solicitar dos veces los datos para un mismo paper, pero la función que produce una lista de paper ids basado en el query produce una solicitud con el API siempre.

```
