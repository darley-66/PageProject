import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def getData():
    df = pd.read_csv('german.data-numeric', sep=r'\s+', header=None)
    X = df[[1, 4, 12]].copy()
    X.columns = ['duration_months', 'credit_amount', 'age']
    y = df.iloc[:, -1].apply(lambda x: 1 if x == 2 else 0)
    return X, y

def implementLogisticRegression():
    X, y = getData()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    coefficients = dict(zip(X.columns, model.coef_[0].tolist()))
    intercept = float(model.intercept_[0])

    return {
        "model": model,
        "scaler": scaler,
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "coefficients": coefficients,
            "intercept": round(intercept, 4)
        }
    }

# Entrenamos los recursos una sola vez para reutilizarlos en las predicciones
_model_assets = implementLogisticRegression()

def PredictRisk(duration_months, credit_amount, age):
    features = [[duration_months, credit_amount, age]]
    scaled_features = _model_assets["scaler"].transform(features)
    
    prediction = int(_model_assets["model"].predict(scaled_features)[0])
    probabilities = _model_assets["model"].predict_proba(scaled_features)[0]
    default_probability = float(probabilities[1])

    return {
        "approved": default_probability < 0.5,
        "default_risk": "HIGH" if prediction == 1 else "LOW",
        "default_probability": round(default_probability, 4),
        "metrics": _model_assets["metrics"]
    }