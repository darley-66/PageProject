import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def getData():
    # Dataset German Credit (numeric)
    df = pd.read_csv('german.data-numeric', sep=r'\s+', header=None)

    # Seleccionamos variables relevantes
    X = df[[1, 4, 12]].copy()
    X.columns = ['duration_months', 'credit_amount', 'age']

    # Variable objetivo: 1 = default, 0 = no default
    y = df.iloc[:, -1].apply(lambda x: 1 if x == 2 else 0)
    return X, y

def implementLogisticRegression():
    X, y = getData()

    # Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar modelo
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = model.predict(X_test_scaled)

    # Métricas
    metrics = {
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "coefficients": dict(zip(X.columns, model.coef_[0].tolist())),
        "intercept": round(float(model.intercept_[0]), 4)
    }

    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "metrics": metrics
    }

# Entrenamos una sola vez
_model_assets = implementLogisticRegression()

def PredictRisk(duration_months, credit_amount, age):
    features = [[duration_months, credit_amount, age]]
    scaled_features = _model_assets["scaler"].transform(features)

    prediction = int(_model_assets["model"].predict(scaled_features)[0])
    probabilities = _model_assets["model"].predict_proba(scaled_features)[0]
    default_probability = float(probabilities[1])

    return {
        "approved": prediction == 0,
        "default_risk": "HIGH" if prediction == 1 else "LOW",
        "default_probability": round(default_probability, 4),
        "metrics": _model_assets["metrics"]
    }
