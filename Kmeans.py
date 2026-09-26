import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def getData():
    # Cargamos el dataset de e-commerce
    df = pd.read_csv("data/ecommerce_customers.csv")
    
    # Seleccionamos automáticamente las primeras 3 columnas numéricas para evitar errores de nombres
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    feature_cols = numeric_df.columns[:3].tolist()
    
    # Si por alguna razón tuviera menos de 3, tomamos las que haya
    if len(feature_cols) < 2:
        feature_cols = numeric_df.columns.tolist()
        
    return df[feature_cols].to_dict(orient="records"), df, feature_cols

def implementClustering():
    data, df, feature_cols = getData()
    
    # Extraemos los valores usando dinámicamente las columnas detectadas
    x = [[row[col] for col in feature_cols] for row in data]
    
    # Escalado de los datos
    scaler = StandardScaler()
    Xscaled = scaler.fit_transform(x)

    # Configuración del modelo K-Means
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(Xscaled)
    
    # Construcción de los resultados
    result = []
    for i, row_data in enumerate(data):
        row = row_data.copy()
        row["cluster"] = int(labels[i])
        result.append(row)

    # Resumen de conteo por clúster
    summaryClusters = {}
    for label in labels:
        label = int(label)
        summaryClusters[label] = summaryClusters.get(label, 0) + 1

    # Centroides en la escala original
    centroids_scaled = model.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled).tolist()

    # Cálculo del coeficiente de silueta
    score = silhouette_score(Xscaled, labels)

    return {
        "results": result,
        "summary": summaryClusters,
        "centroids": centroids,
        "silhouette_score": score,
        "feature_cols": feature_cols
    }