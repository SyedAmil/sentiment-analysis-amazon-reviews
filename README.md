# Sentiment Analysis on Amazon Reviews

Classifies Amazon food reviews as **positive, negative or neutral** using NLP techniques and sentiment lexicons, and derives insights useful for marketing and product development.

## Dataset
[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (Kaggle). A random sample of 50,000 reviews is used.
Star ratings are mapped to labels: 4-5 = positive, 3 = neutral, 1-2 = negative.

## Approach
1. Data cleaning and sampling
2. Text preprocessing: lowercasing, HTML/URL/punctuation removal, stopword removal (negations kept), lemmatization
3. Lexicon-based sentiment with **VADER** and **TextBlob**
4. Comparison of lexicon labels against actual star ratings
5. Supervised model: **TF-IDF + Logistic Regression**
6. Visualizations: sentiment distribution, confusion matrices, word clouds, top indicator words, review length
7. Business insights

## Results
| Method | Accuracy |
|---|---|
| VADER | _fill after running_ |
| TextBlob | _fill after running_ |
| TF-IDF + Logistic Regression | _fill after running_ |

## How to run
```bash
pip install kagglehub nltk textblob vaderSentiment wordcloud scikit-learn seaborn matplotlib pandas
python sentiment_analysis.py
```
Download `Reviews.csv` from Kaggle and place it next to the script (the script tries `kagglehub` if the file is missing). Charts are saved in the `charts/` folder and predictions in `sentiment_results.csv`.

The full dataset is not included in this repo because of its size.

## Tech stack
Python, pandas, NLTK, VADER, TextBlob, scikit-learn, matplotlib, seaborn, WordCloud
