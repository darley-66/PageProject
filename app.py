from flask import Flask, render_template, request
# import LinearRegression
from LinearRegression import CalculateGrade
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("homePage.html")

@app.route("/pagina/")
def pagina():
    return render_template("index.html")

@app.route("/LinearRegression/", methods =["GET","POST"])
def calculate():
    calculateResult = None
    hours = requests
    calculateResult = CalculateGrade(5)
    return render_template("temLinearRegression.html", result = calculateResult)

@app.route("/example/")
def example():
    return render_template("example.html")