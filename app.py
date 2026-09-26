from flask import Flask, render_template, request, jsonify
from models.linear_regression import CalculateGrade, GeneratePlot
from models.logistic_regression import PredictRisk, implementLogisticRegression
from flask import Blueprint, render_template
from Kmeans import KMeans

from models.svm_model import (
    get_breast_cancer_svm_metrics,
    plot_breast_cancer_dataset,
    predict_breast_cancer,
)
from models.kmeans_model import (
    get_centroids,
    get_cluster_summary,
    get_generated_images,
    get_variance_comparison,
    get_assignments,
    run_kmeans,
)
import base64
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
from flask import Flask, render_template, request
from SGDRegression import train, GRID, START, GOAL, ACTION_NAMES
# Librerías para SVM y métricas
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# ============================
# Logistic Regression (assets)
# ============================
logistic_assets = implementLogisticRegression()

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
    data_lin = pd.read_csv(BASE_DIR / "data" / "Student_Performance.csv")
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
    plot_url = plot_breast_cancer_dataset()

    if request.method == "POST":
        try:
            mean_radius = float(request.form.get("mean_radius"))
            mean_texture = float(request.form.get("mean_texture"))
            mean_perimeter = float(request.form.get("mean_perimeter"))

            result = predict_breast_cancer(
                mean_radius,
                mean_texture,
                mean_perimeter,
            )
        except:
            result = "Invalid input"

    return render_template("temSVMApp.html", 
                           result=result, 
                           plot_url=plot_url,
                           metrics=get_breast_cancer_svm_metrics())

# ============================
# K-Means
# ============================
@app.route("/kmeans-concepts/")
def concepts_kmeans():
    return render_template("conceptsKMeans.html")


@app.route("/kmeans-manual/")
def kmeans_manual():
    try:
        get_centroids()
    except RuntimeError:
        run_kmeans()

    dataset = pd.read_csv(BASE_DIR / "data" / "ecommerce_customers.csv")
    images = {
        name: base64.b64encode(Path(path).read_bytes()).decode("ascii")
        for name, path in get_generated_images().items()
    }
    assignments_by_iteration = get_assignments()["iterations"]
    assignment_downloads = {
        iteration: "data:text/csv;base64," + base64.b64encode(
            assignments.to_csv(index=False).encode("utf-8")
        ).decode("ascii")
        for iteration, assignments in assignments_by_iteration.items()
    }

    centroids = get_centroids()
    return render_template(
        "temKMeans.html",
        dataset_name="ecommerce_customers.csv",
        dataset_rows=len(dataset),
        dataset_preview=dataset[
            ["customer_id", "purchase_frequency", "monthly_spending"]
        ].head(5).to_dict("records"),
        initial_centroids=centroids[
            centroids["stage"] == "Initial"
        ].to_dict("records"),
        images=images,
        variance_rows=get_variance_comparison().to_dict("records"),
        summary_rows=get_cluster_summary().to_dict("records"),
        assignment_downloads=assignment_downloads,
    )

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
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    
app = Flask(__name__)
@app.route('/SGDRegression', methods=['GET', 'POST'])
def reinforcement():
    result = None
    if request.method == 'POST':
        result = train(episodes=1000)
# Pass the result and grid settings to the template.
    return render_template(
        'SGDRegression.html',
        result=result,
        grid=GRID,
        start=START,
        goal=GOAL,
        actions=ACTION_NAMES,
    )
if __name__ == '__main__':
    app.run()

@app.route("/kmeans-app/")
def kmeans_app():
    from Kmeans import implementClustering
    result_data = implementClustering()
    
    return render_template(
        "kmeans.html", 
        results=result_data["results"],
        summary=result_data["summary"],
        centroids=result_data["centroids"],
        score=result_data.get("silhouette_score")
    )