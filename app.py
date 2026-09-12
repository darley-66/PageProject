from flask import Flask, render_template, request, jsonify
from LinearRegression import CalculateGrade, GeneratePlot
from svm import PredictSpam, GeneratePlot as GeneratePlotLogistic
from LogisticRegression import PredictRisk, implementLogisticRegression
import pandas as pd

app = Flask(__name__)

# Recursos del modelo de regresión logística para renderizado inicial
logistic_assets = implementLogisticRegression()

# Home & Information
@app.route("/")
def home():
    return render_template("homePage.html")

@app.route("/information/")
def information():
    return render_template("index.html")

@app.route("/concepts/")
def concepts():
    return render_template("concepts.html")

# Linear Regression
@app.route("/conceptsLinear/")
def concepts_linear():
    return render_template("conceptsLinear.html")

@app.route("/LinearRegression/", methods=["GET", "POST"])
def calculate():
    data = pd.read_csv("data/Student_Performance.csv")
    records = len(data)

    calculateResult = None
    plot_url = None
    if request.method == "POST":
        try:
            hours = float(request.form.get("hours"))
            calculateResult = CalculateGrade(hours)
            plot_url = GeneratePlot(hours)
        except (ValueError, TypeError):
            calculateResult = None
            plot_url = GeneratePlot()
    else:
        plot_url = GeneratePlot()

    return render_template("temLinearRegression.html", 
                           result=calculateResult, 
                           records=records, 
                           plot_url=plot_url)

# Logistic Regression
@app.route("/logistic-concepts/")
def concepts_logistic():
    return render_template("temLogisticConcepts.html")

@app.route("/logistic-application/", methods=["GET", "POST"])
def calculate_logistic():
    return render_template("temLogisticRegression.html", metrics=logistic_assets["metrics"])

# Endpoint para peticiones AJAX fetch() en temLogisticRegression.html
@app.route("/api/predict-risk", methods=["POST"])
def api_predict_risk():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    duration = int(data.get("duration_months", 24))
    amount = float(data.get("credit_amount", 3000))
    age = int(data.get("age", 30))

    result = PredictRisk(duration, amount, age)

    return jsonify({
        "input_data": data,
        "prediction": {
            "default_risk": result["default_risk"],
            "default_probability": result["default_probability"],
            "approved": result["approved"]
        },
        "model_performance": result["metrics"]
    })

# Support Vector Machine (SVM)
@app.route("/svm-concepts/")
def concepts_svm():
    return render_template("conceptsSVM.html")

@app.route("/SVM/", methods=["GET", "POST"])
def calculate_svm():
    data = pd.read_csv("data/spam.csv", encoding="latin-1")
    records = len(data)

    calculateResult = None
    plot_url = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email_text", "")
        if email_text.strip():
            calculateResult = PredictSpam(email_text)
            plot_url = GeneratePlotLogistic(calculateResult)
        else:
            plot_url = GeneratePlotLogistic()
    else:
        plot_url = GeneratePlotLogistic()

    return render_template("temSVMApp.html", 
                           result=calculateResult, 
                           plot_url=plot_url, 
                           email_text=email_text,
                           records=records)

# Use Cases
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