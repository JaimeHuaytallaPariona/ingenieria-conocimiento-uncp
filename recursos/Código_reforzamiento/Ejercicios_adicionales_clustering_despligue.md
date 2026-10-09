# Ejercicios Propuestos — Clustering y Despliegue ML
## Material de Refuerzo — Ingeniería del Conocimiento (ISO56B)
### Msc. Jaime Antonio Huaytalla Pariona — UNCP, 2026-II

---

## Nivel 1: Básico (Comprensión de conceptos)

### Ejercicio 1.1 — Exploración visual con datos sintéticos
**Dataset:** `make_blobs`, `make_moons`, `make_circles` de sklearn

**Tareas:**
1. Generar 3 datasets sintéticos: blobs (4 centros), moons (noise=0.15), circles (noise=0.08)
2. Visualizar cada dataset con scatter plot, coloreando por la etiqueta real
3. Estandarizar cada dataset con StandardScaler
4. Aplicar K-Means (K=2 para moons/circles, K=4 para blobs) y visualizar los clusters
5. Responder: ¿En cuál(es) dataset(s) K-Means funciona bien? ¿En cuáles falla? ¿Por qué?

**Entregable:** 6 gráficos (3 reales + 3 con K-Means) y explicación escrita.

---

### Ejercicio 1.2 — K-Means manual paso a paso
**Dataset:** Crear 12 puntos 2D manualmente:
```python
X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],
              [8,2],[10,2],[9,3],[3,7],[4,8],[3,5]])
```

**Tareas:**
1. Visualizar los 12 puntos con scatter plot
2. Inicializar 2 centroides en posiciones de tu elección
3. **Manualmente** (sin sklearn): calcular distancias, asignar clusters, recalcular centroides para 3 iteraciones
4. Comparar tu resultado manual con `KMeans(n_clusters=2)` de sklearn
5. Responder: ¿Cambiaron los clusters si hubieras elegido otros centroides iniciales?

**Entregable:** Pasos manuales documentados, comparación con sklearn.

---

### Ejercicio 1.3 — Importancia de la estandarización
**Dataset:** Crear datos con escalas muy diferentes:
```python
np.random.seed(42)
ingresos = np.random.normal(50000, 15000, 200)  # Soles
edad = np.random.normal(35, 10, 200)  # Años
X = np.column_stack([ingresos, edad])
```

**Tareas:**
1. Aplicar K-Means (K=3) SIN estandarizar y visualizar
2. Aplicar K-Means (K=3) CON StandardScaler y visualizar
3. Mostrar ambos resultados lado a lado
4. Calcular el Silhouette Score para ambos casos
5. Responder: ¿Por qué los resultados son tan diferentes? ¿Cuál es más razonable?

**Entregable:** Gráficos comparativos y explicación.

---

## Nivel 2: Intermedio (Aplicación de algoritmos)

### Ejercicio 2.1 — Selección de K con Iris
**Dataset:** `sklearn.datasets.load_iris()` (sin usar las etiquetas)

**Tareas:**
1. Estandarizar los datos
2. Aplicar el Método del Codo (K=2 a 10)
3. Calcular el Silhouette Score para K=2 a 10
4. Graficar ambas métricas
5. Seleccionar K óptimo y aplicar K-Means
6. Comparar con las etiquetas reales: calcular ARI y NMI
7. Visualizar clusters con PCA (2D) y colorear: una vez por cluster, una vez por etiqueta real
8. Responder: ¿El clustering encontró las 3 especies de Iris?

**Entregable:** Gráficos, métricas y análisis comparativo.

---

### Ejercicio 2.2 — DBSCAN: efecto de eps y min_samples
**Dataset:** `make_moons(n_samples=500, noise=0.1)`

**Tareas:**
1. Crear el gráfico de K-distancias para estimar eps
2. Probar DBSCAN con eps = [0.05, 0.1, 0.2, 0.3, 0.5] y min_samples=5
3. Para cada configuración: reportar número de clusters, puntos de ruido y Silhouette Score
4. Crear una cuadrícula de gráficos (5 columnas) mostrando los clusters
5. Repetir con eps=0.2 y min_samples = [2, 5, 10, 15, 20]
6. Responder: ¿Cómo afectan eps y min_samples al resultado? ¿Cuál es la mejor combinación?

**Entregable:** Cuadrículas de gráficos, tabla de resultados y análisis.

---

### Ejercicio 2.3 — Clustering Jerárquico: comparación de linkages
**Dataset:** `sklearn.datasets.load_wine()` (sin etiquetas)

**Tareas:**
1. Estandarizar los datos
2. Crear dendrogramas con 4 métodos de linkage: single, complete, average, ward
3. Aplicar clustering aglomerativo con K=3 para cada linkage
4. Calcular Silhouette Score y ARI para cada uno
5. Visualizar con PCA (2D) los 4 resultados
6. Responder: ¿Qué linkage produce los mejores clusters? ¿Por qué ward es el predeterminado?

**Entregable:** 4 dendrogramas, tabla comparativa, gráficos PCA.

---

### Ejercicio 2.4 — Serialización y carga de modelos
**Dataset:** `sklearn.datasets.load_wine()` (usando etiquetas para clasificación)

**Tareas:**
1. Entrenar un pipeline completo: StandardScaler → RandomForestClassifier
2. Evaluar con cross-validation y reportar accuracy
3. Guardar el pipeline con `joblib.dump`
4. En una celda **nueva** (simular nuevo script): cargar con `joblib.load`
5. Verificar que las predicciones del modelo cargado sean idénticas al original
6. Guardar también con `pickle` y comparar tamaños de archivo
7. Responder: ¿Por qué es importante guardar el pipeline completo y no solo el modelo?

**Entregable:** Notebook con demostración de serialización/deserialización.

---

## Nivel 3: Avanzado (Comparación y despliegue)

### Ejercicio 3.1 — Comparación justa de algoritmos de clustering
**Datasets:** Iris, Wine, Digits (solo con 3 clases)

**Tareas:**
1. Para cada dataset: estandarizar y aplicar K-Means, DBSCAN, Jerárquico (ward), GMM
2. Evaluar con 3 métricas internas: Silhouette, Calinski-Harabasz, Davies-Bouldin
3. Evaluar con 2 métricas externas: ARI, NMI
4. Crear una tabla de resultados (filas=dataset×algoritmo, columnas=métricas)
5. Crear heatmaps por métrica
6. Responder:
   - ¿Existe un algoritmo que sea el mejor en todos los datasets?
   - ¿Las métricas internas y externas coinciden en el ranking?
   - ¿Qué algoritmo recomendarías como "primera opción"?

**Entregable:** Tablas, heatmaps y argumentación escrita.

---

### Ejercicio 3.2 — API REST con Flask
**Dataset:** Modelo entrenado en Wine

**Tareas:**
1. Entrenar y guardar un pipeline (StandardScaler + RandomForest) con Wine
2. Crear un archivo `app.py` con Flask que:
   - Endpoint `/predict` (POST): recibe JSON con features, retorna clase y probabilidades
   - Endpoint `/health` (GET): retorna status del servicio
   - Manejo de errores para datos malformados
3. Probar la API localmente (ejecutar y hacer requests)
4. Documentar los endpoints (qué recibe, qué retorna)
5. **Bonus:** Crear un script cliente (`test_api.py`) que envíe 10 muestras de prueba

**Entregable:** Archivos `app.py`, `test_api.py` y documentación de la API.

---

### Ejercicio 3.3 — Interfaz interactiva con Gradio
**Dataset:** Modelo entrenado en California Housing (regresión) o Wine (clasificación)

**Tareas:**
1. Entrenar y guardar un pipeline
2. Crear una interfaz Gradio con:
   - Inputs numéricos para cada feature (con valores por defecto)
   - Output: predicción (clase o valor numérico)
3. Lanzar la interfaz y probar con diferentes valores
4. Publicar con `share=True` para obtener un enlace público temporal
5. Capturar screenshots de la interfaz funcionando

**Entregable:** Notebook con código Gradio funcional y screenshots.

---

## Nivel 4: Desafío (Integración y pensamiento crítico)

### Ejercicio 4.1 — Segmentación de clientes: proyecto mini completo
**Dataset:** Mall Customers (descargar de Kaggle) o generar datos sintéticos similares:
```python
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=200, n_features=4, centers=5, cluster_std=1.5, random_state=42)
df = pd.DataFrame(X, columns=['Edad', 'Ingreso_Anual', 'Frecuencia_Compra', 'Gasto_Promedio'])
df['Edad'] = np.abs(df['Edad']) * 5 + 20
df['Ingreso_Anual'] = np.abs(df['Ingreso_Anual']) * 10000 + 30000
df['Frecuencia_Compra'] = np.abs(df['Frecuencia_Compra']) * 3 + 1
df['Gasto_Promedio'] = np.abs(df['Gasto_Promedio']) * 50 + 20
```

**Tareas:**
1. **Exploración:** Distribuciones, correlaciones, outliers
2. **Preprocesamiento:** Estandarizar
3. **Selección de K:** Codo + Silueta
4. **Clustering:** Aplicar K-Means con K óptimo
5. **Perfilado:** Caracterizar cada cluster (medias por feature, gráficos radar)
6. **Nombrar clusters:** Asignar nombres descriptivos (ej: "Clientes Premium", "Compradores Ocasionales")
7. **Serialización:** Guardar el pipeline
8. **Despliegue:** Crear una función que reciba datos de un nuevo cliente y lo asigne a un segmento
9. **(Bonus):** Crear interfaz Gradio para la segmentación

**Entregable:** Notebook completo tipo proyecto, desde exploración hasta despliegue funcional.

---

### Ejercicio 4.2 — Compresión de imágenes con K-Means
**Dataset:** Una imagen a color (usar cualquier imagen disponible)

**Tareas:**
1. Cargar una imagen con matplotlib (`plt.imread`)
2. Reshapear a (n_pixels, 3) — cada fila es un píxel RGB
3. Aplicar K-Means con K = 2, 4, 8, 16, 32, 64, 128, 256
4. Reconstruir la imagen reemplazando cada píxel por su centroide
5. Visualizar las 8 versiones junto a la original
6. Calcular la razón de compresión para cada K
7. Graficar: calidad visual vs razón de compresión
8. Responder: ¿Cuál es el K mínimo que produce una imagen aceptable?

**Pista:**
```python
from sklearn.cluster import KMeans
img = plt.imread('imagen.jpg') / 255.0
pixels = img.reshape(-1, 3)
km = KMeans(n_clusters=K, random_state=42)
labels = km.fit_predict(pixels)
img_compressed = km.cluster_centers_[labels].reshape(img.shape)
```

**Entregable:** Mosaico de imágenes comprimidas con diferentes K.

---

*Total: 12 ejercicios graduados (3 básicos + 4 intermedios + 3 avanzados + 2 desafíos)*

*Material de refuerzo — Ingeniería del Conocimiento (ISO56B) — UNCP*
