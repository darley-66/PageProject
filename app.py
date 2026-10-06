from flask import Flask, render_template, request, jsonify, abort, send_file
from models.linear_regression import CalculateGrade, GeneratePlot
from models.logistic_regression import PredictRisk, implementLogisticRegression
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
from SGDRegression import train, GRID, START, GOAL, ACTION_NAMES
import base64
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

logistic_assets = implementLogisticRegression()


@app.route("/")
def home():
    return render_template("homePage.html")


@app.route("/information/")
def information():
    return render_template("index.html")


@app.route("/concepts/")
def concepts():
    return render_template("concepts.html")


@app.route("/conceptsLinear/")
def concepts_linear():
    return render_template("conceptsLinear.html")


@app.route("/LinearRegression/", methods=["GET", "POST"])
def calculate():
    data_lin = pd.read_csv(BASE_DIR / "data" / "Student_Performance.csv")
    records = len(data_lin)

    calculateResult = None
    plot_url = GeneratePlot()

    X = data_lin["study_hours"].values.reshape(-1, 1)
    y = data_lin["overall_score"].values

    y_pred = CalculateGrade(data_lin["study_hours"].iloc[0])

    metrics = {
        "r2": r2_score(y, [y_pred] * len(y)),
        "mse": mean_squared_error(y, [y_pred] * len(y))
    }

    if request.method == "POST":
        try:
            hours = float(request.form.get("hours"))
            calculateResult = CalculateGrade(hours)
            plot_url = GeneratePlot(hours)
        except (ValueError, TypeError):
            calculateResult = None

    return render_template(
        "temLinearRegression.html",
        result=calculateResult,
        records=records,
        plot_url=plot_url,
        metrics=metrics
    )


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

    return render_template(
        "temLogisticRegression.html",
        result=result,
        metrics=metrics,
        duration=duration,
        amount=amount,
        age=age
    )


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

    return render_template(
        "temSVMApp.html",
        result=result,
        plot_url=plot_url,
        metrics=get_breast_cancer_svm_metrics()
    )


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


@app.route("/SGDRegression", methods=["GET", "POST"])
def reinforcement():
    result = None

    if request.method == "POST":
        result = train(episodes=1000)

    return render_template(
        "SGDRegression.html",
        result=result,
        grid=GRID,
        start=START,
        goal=GOAL,
        actions=ACTION_NAMES,
    )


@app.route("/kmeans-app/")
def kmeans_app():
    from Kmeans import implementClustering

    result_data = implementClustering()

    return render_template(
        "kmeans.html",
        results=result_data["results"],
        summary=result_data["summary"],
        centroids=result_data["centroids"],
        score=result_data.get("silhouette_score"),
        feature_cols=result_data["feature_cols"],
        plot_path=result_data["plot_path"],
    )


@app.route("/kmeans-plot/<path:relative_path>")
def kmeans_plot_file(relative_path):
    if relative_path != "outputs/kmeans_clusters.png":
        abort(404)

    return send_file(
        BASE_DIR / relative_path,
        mimetype="image/png"
    )


if __name__ == "__main__":
    app.run(debug=True)