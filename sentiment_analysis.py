"""
Sentiment Analysis on Amazon Fine Food Reviews

Pehli baar setup (terminal mein):
    pip install kagglehub nltk textblob vaderSentiment wordcloud scikit-learn seaborn matplotlib pandas

Run karna:
    python sentiment_analysis.py

Dataset: Reviews.csv isi file ke folder mein rakho (Kaggle se manually download),
warna code kagglehub se download karne ki koshish karega.
Charts screen par aayenge aur 'charts' folder mein PNG ke roop mein save bhi honge.
"""

import os
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_STATE = 42
SAMPLE_SIZE = 50000
LABELS = ["negative", "neutral", "positive"]
COLORS = ["#d9534f", "#f0ad4e", "#5cb85c"]

sns.set_style("whitegrid")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def save_and_show(name):
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------
# 1. NLTK data
# ---------------------------------------------------------------
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ---------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------
local_csv = os.path.join(BASE_DIR, "Reviews.csv")
if os.path.exists(local_csv):
    csv_path = local_csv
elif os.path.exists("Reviews.csv"):
    csv_path = "Reviews.csv"
else:
    print("Reviews.csv nahi mili, kagglehub se download kar rahe hain...")
    import kagglehub
    folder = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
    csv_path = os.path.join(folder, "Reviews.csv")

df = pd.read_csv(csv_path)
print("Full dataset shape:", df.shape)

# ---------------------------------------------------------------
# 3. Cleaning + labels
# ---------------------------------------------------------------
df = df[["Score", "Summary", "Text"]].dropna()
df = df.drop_duplicates(subset="Text")
df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)


def rating_to_label(score):
    if score >= 4:
        return "positive"
    if score == 3:
        return "neutral"
    return "negative"


df["true_label"] = df["Score"].apply(rating_to_label)
print("\nLabel distribution:")
print(df["true_label"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "never"}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


print("\nPreprocessing chal rahi hai (1-3 minute lag sakte hain)...")
df["clean_text"] = df["Text"].apply(clean_text)

# ---------------------------------------------------------------
# 5. Lexicon-based sentiment (VADER + TextBlob)
# ---------------------------------------------------------------
vader = SentimentIntensityAnalyzer()


def vader_label_from_score(c):
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"


def textblob_label_from_score(p):
    if p > 0.05:
        return "positive"
    if p < -0.05:
        return "negative"
    return "neutral"


print("VADER aur TextBlob chal rahe hain...")
df["vader_compound"] = df["Text"].apply(lambda t: vader.polarity_scores(t)["compound"])
df["vader_label"] = df["vader_compound"].apply(vader_label_from_score)
df["textblob_polarity"] = df["Text"].apply(lambda t: TextBlob(t).sentiment.polarity)
df["textblob_label"] = df["textblob_polarity"].apply(textblob_label_from_score)

# ---------------------------------------------------------------
# 6. Lexicon vs actual ratings
# ---------------------------------------------------------------
for name, col in [("VADER", "vader_label"), ("TextBlob", "textblob_label")]:
    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(df["true_label"], df[col]), 4))
    print(classification_report(df["true_label"], df[col], labels=LABELS, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (name, col) in zip(axes, [("VADER", "vader_label"), ("TextBlob", "textblob_label")]):
    cm = confusion_matrix(df["true_label"], df[col], labels=LABELS)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_title(f"{name}: Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (from rating)")
save_and_show("1_lexicon_confusion_matrices.png")

# ---------------------------------------------------------------
# 7. ML model: TF-IDF + Logistic Regression
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["true_label"],
    test_size=0.2, random_state=RANDOM_STATE, stratify=df["true_label"]
)

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(Xtr, y_train)
pred = model.predict(Xte)

print("\n===== TF-IDF + Logistic Regression =====")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, labels=LABELS))

cm = confusion_matrix(y_test, pred, labels=LABELS)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Logistic Regression: Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
save_and_show("2_ml_confusion_matrix.png")

# ---------------------------------------------------------------
# 8. Comparison (same test set)
# ---------------------------------------------------------------
test_idx = X_test.index
comparison = pd.DataFrame({
    "Method": ["VADER", "TextBlob", "TF-IDF + Logistic Regression"],
    "Accuracy": [
        accuracy_score(df.loc[test_idx, "true_label"], df.loc[test_idx, "vader_label"]),
        accuracy_score(df.loc[test_idx, "true_label"], df.loc[test_idx, "textblob_label"]),
        accuracy_score(y_test, pred),
    ],
}).round(4)
print("\n===== Model Comparison =====")
print(comparison.to_string(index=False))

plt.figure(figsize=(7, 4))
sns.barplot(data=comparison, x="Method", y="Accuracy", palette="viridis")
plt.ylim(0, 1)
plt.title("Accuracy Comparison (same test set)")
plt.xticks(rotation=10)
save_and_show("3_accuracy_comparison.png")

# ---------------------------------------------------------------
# 9. Visualizations
# ---------------------------------------------------------------
# Sentiment distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, col) in zip(axes, [("Actual (from rating)", "true_label"),
                                    ("VADER", "vader_label"),
                                    ("TextBlob", "textblob_label")]):
    sns.countplot(data=df, x=col, order=LABELS, palette=COLORS, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
plt.suptitle("Sentiment Distribution", y=1.02, fontsize=14)
save_and_show("4_sentiment_distribution.png")

# VADER score by star rating
plt.figure(figsize=(8, 4.5))
sns.boxplot(data=df, x="Score", y="vader_compound", palette="RdYlGn")
plt.title("VADER Compound Score by Star Rating")
plt.xlabel("Star rating")
plt.ylabel("VADER compound")
save_and_show("5_vader_by_rating.png")

# Word clouds
n_wc = 5000
pos_pool = df.loc[df["true_label"] == "positive", "clean_text"]
neg_pool = df.loc[df["true_label"] == "negative", "clean_text"]
pos_text = " ".join(pos_pool.sample(min(n_wc, len(pos_pool)), random_state=1))
neg_text = " ".join(neg_pool.sample(min(n_wc, len(neg_pool)), random_state=1))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, text, title, cmap in [(axes[0], pos_text, "Positive Reviews", "Greens"),
                              (axes[1], neg_text, "Negative Reviews", "Reds")]:
    wc = WordCloud(width=800, height=500, background_color="white",
                   colormap=cmap, max_words=100).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
save_and_show("6_wordclouds.png")

# Top indicator words
feature_names = np.array(tfidf.get_feature_names_out())
classes = list(model.classes_)
top_pos = feature_names[np.argsort(model.coef_[classes.index("positive")])[-15:]][::-1]
top_neg = feature_names[np.argsort(model.coef_[classes.index("negative")])[-15:]][::-1]
print("\nTop positive indicator words:", ", ".join(top_pos))
print("Top negative indicator words:", ", ".join(top_neg))

# Review length vs sentiment
df["word_count"] = df["Text"].str.split().str.len()
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="true_label", y="word_count", order=LABELS,
            palette=COLORS, showfliers=False)
plt.title("Review Length by Sentiment")
plt.xlabel("")
plt.ylabel("Words per review")
save_and_show("7_review_length.png")
print("\nMedian review length (words):")
print(df.groupby("true_label")["word_count"].median())

# ---------------------------------------------------------------
# 10. Try custom text
# ---------------------------------------------------------------
def predict_sentiment(text):
    c = vader.polarity_scores(text)["compound"]
    ml = model.predict(tfidf.transform([clean_text(text)]))[0]
    return {"text": text, "vader": vader_label_from_score(c), "ml_model": ml}


samples = [
    "Absolutely loved this coffee, will buy again!",
    "Terrible taste and the package arrived damaged.",
    "It is okay, nothing special.",
    "Not good at all, waste of money.",
]
print("\n===== Custom text predictions =====")
print(pd.DataFrame([predict_sentiment(s) for s in samples]).to_string(index=False))

# ---------------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------------
df[["Score", "true_label", "vader_compound", "vader_label",
    "textblob_polarity", "textblob_label"]].to_csv(
    os.path.join(BASE_DIR, "sentiment_results.csv"), index=False)
print("\nDone! Charts 'charts' folder mein aur results 'sentiment_results.csv' mein save ho gaye.")