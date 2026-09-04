from flask import Flask, render_template, request
from LinearRegression import CalculateGrade
import requests

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

@app.route("/example/")
def example():
    return render_template("example.html")

@app.route("/LinearRegression/", methods=["GET", "POST"])
def calculate():
    calculateResult = None
    hours = requests
    calculateResult = CalculateGrade(5)
    return render_template("temLinearRegression.html", result=calculateResult)

if __name__ == "__main__":
    app.run(debug=True)