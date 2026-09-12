from flask import Flask, render_template, request, jsonify
from LinearRegression import CalculateGrade, GeneratePlot
from LogisticRegression import PredictRisk, implementLogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64

# Librerías para SVM y métricas
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

app = Flask(__name__)

# ============================
# Logistic Regression (assets)
# ============================
logistic_assets = implementLogisticRegression()

# ============================
# SVM: Entrenamiento inicial
# ============================
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)

# Calcular métricas del modelo SVM
y_pred = model_svm.predict(X_test)
metrics_svm = {
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

# ============================
# Función para graficar dataset
# ============================
def plot_dataset():
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x=df['mean radius'], y=df['mean texture'],
        hue=df['target'], palette="coolwarm"
    )
    plt.title("Breast Cancer Dataset (Radius vs Texture)")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ============================
# Home & Information
# ============================
@app.route("/")
def home():
    return render_template("homePage.html")

@app.route("/information/")
def information():
    return render_template("index.html")

@app.route("/concepts/")
def concepts():
    return render_template("concepts.html")

# ============================
# Linear Regression
# ============================
@app.route("/conceptsLinear/")
def concepts_linear():
    return render_template("conceptsLinear.html")

@app.route("/LinearRegression/", methods=["GET", "POST"])
def calculate():
    data_lin = pd.read_csv("data/Student_Performance.csv")
    records = len(data_lin)

    calculateResult = None
    plot_url = None
    metrics = None

    # Variables reales del dataset
    X = data_lin["study_hours"].values.reshape(-1, 1)
    y = data_lin["overall_score"].values

    if request.method == "POST":
        try:
            hours = float(request.form.get("hours"))
            calculateResult = CalculateGrade(hours)
            plot_url = GeneratePlot(hours)

            y_pred = [CalculateGrade(h) for h in data_lin["study_hours"]]
            metrics = {
                "r2": r2_score(y, y_pred),
                "mse": mean_squared_error(y, y_pred)
            }
        except (ValueError, TypeError):
            calculateResult = None
            plot_url = GeneratePlot()
    else:
        plot_url = GeneratePlot()
        y_pred = [CalculateGrade(h) for h in data_lin["study_hours"]]
        metrics = {
            "r2": r2_score(y, y_pred),
            "mse": mean_squared_error(y, y_pred)
        }

    return render_template("temLinearRegression.html", 
                           result=calculateResult, 
                           records=records, 
                           plot_url=plot_url,
                           metrics=metrics)

# ============================
# Logistic Regression
# ============================
@app.route("/logistic-concepts/")
def concepts_logistic():
    return render_template("temLogisticConcepts.html")

@app.route("/logistic-application/", methods=["GET", "POST"])
def calculate_logistic():
    result = None
    metrics = logistic_assets["metrics"]

    duration = None
    amount = None
    age = None

    if request.method == "POST":
        try:
            duration = int(request.form.get("duration_months"))
            amount = float(request.form.get("credit_amount"))
            age = int(request.form.get("age"))

            result = PredictRisk(duration, amount, age)
        except (ValueError, TypeError):
            result = None

    return render_template("temLogisticRegression.html",
                           result=result,
                           metrics=metrics,
                           duration=duration,
                           amount=amount,
                           age=age)

@app.route("/api/predict-risk", methods=["POST"])
def api_predict_risk():
    data_req = request.get_json()
    if not data_req:
        return jsonify({"error": "No data provided"}), 400

    duration = int(data_req.get("duration_months", 24))
    amount = float(data_req.get("credit_amount", 3000))
    age = int(data_req.get("age", 30))

    result = PredictRisk(duration, amount, age)

    return jsonify({
        "input_data": data_req,
        "prediction": {
            "default_risk": result["default_risk"],
            "default_probability": result["default_probability"],
            "approved": result["approved"]
        },
        "model_performance": result["metrics"]
    })

# ============================
# Support Vector Machine (SVM)
# ============================
@app.route("/svm-concepts/")
def concepts_svm():
    return render_template("conceptsSVM.html")

@app.route("/SVM/", methods=["GET", "POST"])
def calculate_svm():
    result = None
    plot_url = plot_dataset()

    if request.method == "POST":
        try:
            mean_radius = float(request.form.get("mean_radius"))
            mean_texture = float(request.form.get("mean_texture"))
            mean_perimeter = float(request.form.get("mean_perimeter"))

            input_data = [0]*X.shape[1]
            input_data[0] = mean_radius
            input_data[1] = mean_texture
            input_data[2] = mean_perimeter

            prediction = model_svm.predict([input_data])[0]
            result = "Malignant" if prediction == 0 else "Benign"
        except:
            result = "Invalid input"

    return render_template("temSVMApp.html", 
                           result=result, 
                           plot_url=plot_url,
                           metrics=metrics_svm)

# ============================
# Use Cases
# ============================
@app.route("/example/")
def example():
    return render_template("use_case1.html")

@app.route("/use-case-2/")
def use_case_2():
    return render_template("use_case2.html")

@app.route("/use-case-3/")
def use_case3():
    return render_template("use_case3.html")

@app.route("/use-case-4/")
def use_case4():
    return render_template("use_case4.html")

if __name__ == "__main__":
    app.run(debug=True)
