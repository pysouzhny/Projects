from nltk.stem.snowball import SnowballStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob, Word
from datetime import datetime as dt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer
import pandas as pd 
import os



os.chdir(r"C:\Users\Dominique Njinkeu\Documents\GitHub\Projects\notebook")
df=pd.read_csv("tweets_sample.csv")
df["Label"]=df.Support.map({"Neutral":0,"Panthers":1,"Broncos":2})
#determine feature and label variables
X=df.text
y=df.Label
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2,stratify=y)

def split_train_test(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2,stratify=y)
	return X_train, X_test, y_train, y_test




def test_classifier(vect,clf):
    X_train_dtm=vect.fit_transform(X_train)
    print "Features: ",X_train_dtm.shape[1]
    clf.fit(X_train_dtm,y_train)
    X_test_dtm=vect.transform(X_test)
    y_pred_class=clf.predict(X_test_dtm)
    print "Accuracy: ",metrics.accuracy_score(y_test,y_pred_class)

def confusion_ROC(vect,clf):
    X_train_dtm=vect.fit_transform(X_train)
    print "Features: ",X_train_dtm.shape[1]
    X_test_dtm=vect.transform(X_test)
    clf.fit(X_train_dtm,y_train)
    y_pred_class=clf.predict(X_test_dtm)
    print "confusion matrix ",metrics.confusion_matrix(y_test, y_pred_class)

def split_into_lemmas(text):
    stemmer = SnowballStemmer('english')
    text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]

def split_into_stem(text):
    text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [stemmer.stem(word) for word in words]
