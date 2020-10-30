import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

stop_words=STOP_WORDS


nlp=spacy.load('en_core_web_lg')

df=pd.read_csv('bbc-text.csv')

df.sort_values('text',inplace=True, ascending=False)

duplicated_articles_series = df.duplicated('text', keep = False)

df = df[~duplicated_articles_series]

def lemmatizer(text):        
    sent = []
    doc = nlp(text,disable=['parser','ner'])
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)


news_articles=df.copy()
news_articles['text']=  df.apply(lambda x: lemmatizer(x['text']), axis=1)


from sklearn.model_selection import train_test_split

X = news_articles['text']
y = news_articles['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

text_clf_lsvc= Pipeline([('tfidf', TfidfVectorizer(stop_words=stop_words)),
                     ('clf', LinearSVC()),
])
text_clf_lsvc.fit(X_train, y_train)

news='Cristiano ronaldo'

tfidfvectorizer=TfidfVectorizer(stop_words=stop_words,min_df=0)
tfidftext=tfidfvectorizer.fit_transform(news_articles['text'])
tfidfnews=tfidfvectorizer.transform([news])

def tfidf_based_model(tfidfnews, num_similar_items):
    couple_dist = pairwise_distances(tfidftext,tfidfnews)
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    return pd.DataFrame({'Recommended':df['text'][indices].values})

tfidf_based_model(tfidfnews,5)

predictions=text_clf_lsvc.predict([news])[0]

predictions

from sklearn.metrics import classification_report

print(classification_report(y_test,text_clf_lsvc.predict(X_test)))

