from nltk.stem.snowball import SnowballStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from textblob import TextBlob, Word
from datetime import datetime as dt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer
import pandas as pd 
import os

#**********classifiers************
#multinomialNB classifier
from sklearn.naive_bayes import MultinomialNB
clfNB=MultinomialNB()
#knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15)
#linear SVM
from sklearn import svm
linear_svc=svm.SVC(kernel='linear')
#RBF SVM
rbf_svm=svm.SVC(kernel='rbf',C=470)
#Logistic Regression
from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
#*********************************



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

def test_score(vect,clf):
    X_train_dtm=vect.fit_transform(X_train)
    
    clf.fit(X_train_dtm,y_train)
    X_test_dtm=vect.transform(X_test)
    y_pred_class=clf.predict(X_test_dtm)
    
    return metrics.accuracy_score(y_test,y_pred_class)

def find_best_nb_of_features(v=None,c=None,clf=None):
    scores=[]
    if v=="vect":
        for ci in range(300,c,2):
            vect=CountVectorizer(max_features=ci,ngram_range=(1,2),stop_words='english')
            scores.append([test_score(vect,clf),ci])
        scores=[score for score in scores if score==max(scores)]
        print "maximum attainable score and features: ",scores
    elif v=="tfidf":
        for ci in range(1,c,2):
            vect=TfidfVectorizer(stop_words="english",max_features=ci,ngram_range=(1,3))
            scores.append([test_score(vect,clf),ci])
        scores=[score for score in scores if score==max(scores)]
        print "maximum attainable score and features: ",scores
    else:
        print "you chose the wrong option"
    return scores

        




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
    print "confusion matrix "
    print ""
    print metrics.confusion_matrix(y_test, y_pred_class)

def split_into_lemmas(text):

    text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]

def split_into_stem(text):
    stemmer = SnowballStemmer('english')
    text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [stemmer.stem(word) for word in words]
