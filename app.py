from flask import Flask, render_template, request
from LinearRegression import CalculateGrade

app = Flask(__name__)

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

@app.route("/LinearRegression/", methods=["GET", "POST"])
def calculate():
    calculateResult = None
    if request.method == "POST":
        try:
            hours = float(request.form.get("hours"))
            calculateResult = CalculateGrade(hours)
        except (ValueError, TypeError):
            calculateResult = None
            
    return render_template("temLinearRegression.html", result=calculateResult)

if __name__ == "__main__":
    app.run(debug=True)
