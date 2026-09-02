from flask import Flask, render_template, request
import LinearRegression
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return  "Hello World"

@app.route("/pagina/")
def pagina():
    return render_template("index.html")

@app.route("/LinearRegression/", methods =["GET","POST"])
def calculate():
    calculateResult = None
    hours = requests
    calculateResult = LinearRegression.calculateGrade(5)
    return render_template("temLinearRegression.html", result = calculateResult)
