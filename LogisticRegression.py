import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['binary_label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']
y = df['binary_label']

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

model = SVC(kernel='linear')
model.fit(X_vec, y)

def PredictSpam(email_text):
    text_vec = vectorizer.transform([email_text])
    return model.predict(text_vec)[0]

def GeneratePlot():
    plt.figure(figsize=(6, 4))
    
    counts = df['binary_label'].value_counts()
    categories = ['Not Spam (0)', 'Spam (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]
    
    plt.bar(categories, values, color=['blue', 'red'])
    plt.xlabel("Class")
    plt.ylabel("number of messages")
    plt.title("Spam vs non-spam distributio (SVC)")
    
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    print("the length of the base64 srting:", len(plot_url))
    plt.close()
    return plot_url

if __name__ == "__main__":
    sample_text = "WINNER!! You have won a $1000 gift card! Call now to claim."
    print("Predicción para el correo:", PredictSpam(sample_text))
    
    grafica_base64 = GeneratePlot()