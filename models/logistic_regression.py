import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def implementLogisticRegression():
    data = pd.read_csv("data/Student_Performance.csv")

    data["risk"] = (data["overall_score"] < 60).astype(int)

    X = data[
        [
            "study_hours",
            "attendance_percentage",
            "age"
        ]
    ]

    y = data["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions).tolist()
    coefs = model.coef_[0]

    return {
        "model": model,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": matrix,
            # Claves que la plantilla ya usa; credit_amount se alimenta con
            # attendance_percentage porque el modelo no tiene esa variable.
            "intercept": float(model.intercept_[0]),
            "coefficients": {
                "duration_months": float(coefs[0]),
                "credit_amount": float(coefs[1]),
                "age": float(coefs[2])
            }
        }
    }


logistic_assets = implementLogisticRegression()


def PredictRisk(duration, amount, age):
    model = logistic_assets["model"]

    study_hours = float(duration)
    attendance = 80.0

    input_data = [[
        study_hours,
        attendance,
        age
    ]]

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return {
        "default_risk": int(prediction),
        "default_probability": float(probability),
        "approved": bool(prediction == 0),
        "metrics": logistic_assets["metrics"]
    }