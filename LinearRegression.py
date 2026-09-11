import pandas as pd 
import matplotlib.pyplot as plt
import io 
import base64
from sklearn.linear_model import LinearRegression



data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)

x= df[["Study Hours"]]
y= df["Final Grade"]

model =LinearRegression()
model.fit(x,y)

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
    print("Longitud del string base64:", len(plot_url))
    plt.close()
    return plot_url
