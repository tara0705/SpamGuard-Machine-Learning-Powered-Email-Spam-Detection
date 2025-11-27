import os
import re
import string
import joblib
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from wordcloud import WordCloud

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#                    1. LOAD DATASET

data_path = "spam.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Put the dataset in the repo or update data_path.")

df = pd.read_csv(data_path, encoding='latin-1')

# Your CSV columns are: Category, Message
if 'Category' in df.columns and 'Message' in df.columns:
    df = df[['Category', 'Message']].rename(columns={'Category': 'label', 'Message': 'text'})
else:
    raise KeyError("CSV file does not contain the expected 'Category' and 'Message' columns.")

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Dataset shape:", df.shape)
print(df.head())
print(df['label'].value_counts())


#                  2. EXPLORATORY ANALYSIS
df['text_len'] = df['text'].apply(lambda x: len(str(x)))
print(df['text_len'].describe())

plt.figure(figsize=(5, 3))
df['label'].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.show()


#               3. TEXT PREPROCESSING FUNCTIONS

contractions = {
    "don't": "do not", "can't": "cannot", "i'm": "i am",
    "it's": "it is", "you're": "you are"
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def expand_contractions(text):
    for c, full in contractions.items():
        text = re.sub(c, full, text)
    return text


def clean_text(text):
    text = text.lower()
    text = expand_contractions(text)

    # Remove URLs, emails, HTML tags
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<.*?>", "", text)

    # Keep alphabets only
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]

    return " ".join(tokens)


df["clean_text"] = df["text"].apply(clean_text)

#               4. WORDCLOUD VISUALIZATION

spam_words = " ".join(df[df['label'] == 1]['clean_text'])
ham_words = " ".join(df[df['label'] == 0]['clean_text'])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=600, height=300).generate(spam_words))
plt.title("Spam WordCloud")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=600, height=300).generate(ham_words))
plt.title("Ham WordCloud")
plt.axis("off")
plt.show()

#                5. TOP TOKENS (CountVectorizer)

cv = CountVectorizer(max_features=50)
Xc = cv.fit_transform(df['clean_text'])

tokens = pd.DataFrame({
    "token": cv.get_feature_names_out(),
    "count": np.array(Xc.sum(axis=0)).ravel()
}).sort_values("count", ascending=False)

tokens.head(20).plot.bar(x="token", y="count", figsize=(10, 4))
plt.title("Top Tokens")
plt.show()


#                        6. SPLIT DATA
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#              7. MODEL PIPELINES (Default)
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=2)

pipelines = {
    "NaiveBayes": Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())]),
    "LogisticRegression": Pipeline([("tfidf", tfidf), ("clf", LogisticRegression(max_iter=1000))]),
    "SVM": Pipeline([("tfidf", tfidf), ("clf", LinearSVC(max_iter=10000))]),
    "RandomForest": Pipeline([("tfidf", tfidf), ("clf", RandomForestClassifier(n_estimators=200))]),
    "XGBoost": Pipeline([("tfidf", tfidf), ("clf", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])
}

#                   8. MODEL EVALUATION HELPER
results = []


def evaluate(name, model):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n================ {name} =================")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

    return y_pred

#                 9. TRAIN & EVALUATE ALL MODELS
trained_models = {}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    evaluate(name, pipe)
    trained_models[name] = pipe

#                        10. RESULTS TABLE
res_df = pd.DataFrame(results).sort_values("f1", ascending=False)
print(res_df)

res_df.plot.bar(x="model", y="f1", figsize=(10, 4))
plt.title("Model F1 Comparison")
plt.xticks(rotation=45)
plt.show()
#              11. PICK BEST MODEL BY F1 SCORE
best_model_name = res_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)
#                  12. CONFUSION MATRIX PLOT

cm = confusion_matrix(y_test, best_model.predict(X_test))

plt.figure(figsize=(4, 3))
plt.imshow(cm)
plt.title(f"Confusion Matrix: {best_model_name}")
plt.colorbar()
plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])

for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, v, ha="center", va="center")

plt.show()

#                    13. SAVE BEST MODEL
os.makedirs("saved_models", exist_ok=True)
joblib.dump(best_model, "saved_models/best_spam_model.pkl")
print("Model saved → saved_models/best_spam_model.pkl")

#                  14. PREDICTION FUNCTION
def predict_messages(model, msgs):
    cleaned = [clean_text(m) for m in msgs]
    preds = model.predict(cleaned)
    probs = model.predict_proba(cleaned)[:, 1] if hasattr(model, "predict_proba") else [None] * len(preds)
    return list(zip(msgs, preds, probs))


# Example
sample = [
    "Congratulations! You've won a free vacation",
    "Can we talk later?"
]
print(predict_messages(best_model, sample))
