import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

data = {
    "Study Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Final Grade": [2.0, 2.5, 3.0, 3.5, 3.7, 4.0, 4.2, 4.5]
}

df = pd.DataFrame(data)

x = df[["Study Hours"]]
y = df["Final Grade"]

model = LinearRegression()
model.fit(x, y)

def CalculateGrade(hours):
    return model.predict(pd.DataFrame([[hours]], columns=["Study Hours"]))[0]

def GeneratePlot():
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, color="blue", label="Datos reales")
    plt.plot(x, model.predict(x), color="red", label="Regresión lineal")
    plt.xlabel("Study Hours")
    plt.ylabel("Final Grade")
    plt.title("Linear Regression Example")
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()
    print("Longitud del string base64:", len(plot_url))
    return plot_url
