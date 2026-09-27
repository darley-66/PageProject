import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Cargar dataset de spam
data_path = Path(__file__).resolve().parent.parent / "data" / "spam.csv"
df = pd.read_csv(data_path, encoding="latin-1")
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


def _train_breast_cancer_svm():
    data = load_breast_cancer()
    features = data.data
    targets = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }
    return data, model, metrics


_breast_cancer_data, _breast_cancer_model, _breast_cancer_metrics = (
    _train_breast_cancer_svm()
)


def get_breast_cancer_svm_metrics():
    return _breast_cancer_metrics


def predict_breast_cancer(mean_radius, mean_texture, mean_perimeter):
    input_data = [0] * _breast_cancer_data.data.shape[1]
    input_data[0] = mean_radius
    input_data[1] = mean_texture
    input_data[2] = mean_perimeter

    prediction = _breast_cancer_model.predict([input_data])[0]
    return "Malignant" if prediction == 0 else "Benign"


def plot_breast_cancer_dataset():
    data = _breast_cancer_data
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x=df["mean radius"],
        y=df["mean texture"],
        hue=df["target"],
        palette="coolwarm",
    )
    plt.title("Breast Cancer Dataset (Radius vs Texture)")

    image = io.BytesIO()
    plt.savefig(image, format="png")
    image.seek(0)
    plot_url = base64.b64encode(image.getvalue()).decode("utf-8")
    plt.close()
    return plot_url
