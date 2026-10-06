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


def GeneratePlot(hours: float = None) -> str:
    data = pd.read_csv("data/Student_Performance.csv")

    X = data["study_hours"].values.reshape(-1, 1)
    y = data["overall_score"].values

    plt.figure(figsize=(6, 4))
    plt.scatter(
        X,
        y,
        color="blue",
        alpha=0.5,
        label="Observed Data"
    )

    plt.plot(
        X,
        model.predict(X),
        color="red",
        label="Linear Regression"
    )

    if hours is not None:
        predicted = CalculateGrade(hours)

        plt.scatter(
            [hours],
            [predicted],
            color="green",
            s=100,
            marker="x",
            label="Prediction"
        )

    plt.xlabel("Study Hours")
    plt.ylabel("Overall Score")
    plt.title("Linear Regression - Student Performance")
    plt.legend()

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight"
    )

    plt.close()

    buf.seek(0)

    return base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")