import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

def train_model():
    data = pd.read_csv("data/Student_Performance.csv")
    X = data["study_hours"].values.reshape(-1, 1)
    y = data["overall_score"].values

    model = LinearRegression()
    model.fit(X, y)

    return model

model = train_model()

def CalculateGrade(hours):
    prediction = model.predict([[hours]])[0]
    return round(prediction, 2)

def GeneratePlot(hours=None):
    data = pd.read_csv("data/Student_Performance.csv")

    X = data["study_hours"].values.reshape(-1, 1)
    y = data["overall_score"].values

    y_pred = model.predict(X)

    plt.figure(figsize=(7, 5))
    plt.scatter(X, y, label="Datos reales")
    plt.plot(X, y_pred, label="Regresión lineal")

    if hours is not None:
        prediction = model.predict([[hours]])[0]
        plt.scatter([hours], [prediction], s=100, label="Predicción")

    plt.xlabel("Study Hours")
    plt.ylabel("Overall Score")
    plt.title("Linear Regression - Student Performance")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode("utf-8")