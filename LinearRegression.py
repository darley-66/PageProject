import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from sklearn.linear_model import LinearRegression

# Cargar dataset
df = pd.read_csv("data/Student_Performance.csv")

# Asegurar tipos numéricos y limpiar
df["study_hours"] = pd.to_numeric(df["study_hours"], errors="coerce")
df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")
df = df.dropna(subset=["study_hours", "overall_score"])

# Variables
X = df[["study_hours"]]
y = df["overall_score"]

# Entrenar modelo
model = LinearRegression()
model.fit(X, y)

def CalculateGrade(hours: float) -> float:
    return float(model.predict([[hours]])[0])

def GeneratePlot(hours: float = None) -> str:
    plt.figure(figsize=(6,4))
    plt.scatter(X, y, color="blue", alpha=0.5, label="Datos reales")
    plt.plot(X, model.predict(X), color="red", label="Regresión lineal")

    if hours is not None:
        predicted = CalculateGrade(hours)
        plt.scatter([hours], [predicted], color="green", s=100, marker="x", label="Predicción")

    plt.xlabel("Study Hours")
    plt.ylabel("Overall Score")
    plt.title("Linear Regression - Study Hours vs Overall Score")
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()
    return f"data:image/png;base64,{plot_url}"
