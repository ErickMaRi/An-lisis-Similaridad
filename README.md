# Análisis Similaridad

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
pip install requests pandas numpy scikit-learn matplotlib plotly networkx
```

## Uso

1. Clone este repositorio o descargue los archivos del proyecto.
2. Ejecute el archivo `main.py` para realizar un análisis de similitud entre artículos académicos.
3. Siga las instrucciones proporcionadas en la consola para ingresar consultas y configurar el análisis.

## Funciones Principales

- `search_semantic_scholar(query, max_results=5)`: Realiza una búsqueda de artículos en Semantic Scholar.
- `calculate_similarity(df, debug=False)`: Calcula la similitud coseno entre artículos basada en referencias compartidas.
- `plot_interactive_graph(similarity_matrix)`: Genera un gráfico interactivo de matriz de similitud.
- `plot_binary_matrix_networkx(binary, red_count=60, blue_count=60, green_count=60)`: Visualiza la matriz binaria como un gráfico de red utilizando NetworkX.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulte el archivo [LICENSE](LICENSE.txt) para obtener más detalles.

---

**Nota:** Asegúrese de seguir las políticas de uso de la API de Semantic Scholar al realizar consultas.

```

**Licencia MIT**

El texto en inglés anterior es el README en formato Markdown para el proyecto. La Licencia MIT es una licencia de software de código abierto que permite a los usuarios utilizar, modificar y distribuir el código con ciertas restricciones y sin garantías. Es una licencia ampliamente utilizada en la comunidad de desarrollo de software de código abierto.

Es importante mencionar que la licencia se aplica al código fuente del proyecto, y los usuarios deben respetar sus términos y condiciones al utilizarlo. El README también proporciona instrucciones sobre cómo utilizar el proyecto y menciona las bibliotecas requeridas.
