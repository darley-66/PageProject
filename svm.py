import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Cargar dataset de spam
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convertir etiquetas a binario
df['binary_label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']
y = df['binary_label']

# Vectorización de texto
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Entrenar modelo SVM
model = SVC(kernel='linear')
model.fit(X_vec, y)

def PredictSpam(email_text):
    text_vec = vectorizer.transform([email_text])
    return model.predict(text_vec)[0]

def GeneratePlot(prediction=None):
    plt.figure(figsize=(6, 4))
    counts = df['binary_label'].value_counts()
    categories = ['Not Spam (0)', 'Spam (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.bar(categories, values, color=['blue', 'red'])
    plt.xlabel("Class")
    plt.ylabel("Number of messages")
    plt.title("Spam vs Non-Spam Distribution (SVC)")

    # Marcar la predicción del usuario
    if prediction is not None:
        if prediction == 1:
            plt.text(1, values[1] + 50, "← Predicción: SPAM",
                     color="green", fontsize=12, ha="center", fontweight="bold")
        else:
            plt.text(0, values[0] + 50, "← Predicción: HAM",
                     color="green", fontsize=12, ha="center", fontweight="bold")

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()
    return f"data:image/png;base64,{plot_url}"

if __name__ == "__main__":
    sample_text = "WINNER!! You have won a $1000 gift card! Call now to claim."
    pred = PredictSpam(sample_text)
    print("Predicción para el correo:", pred)
    grafica_base64 = GeneratePlot(pred)
    print("Longitud del plot_url:", len(grafica_base64))
