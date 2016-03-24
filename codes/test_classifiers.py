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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from pprint import pprint
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from time import time

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
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y)

def classifier(vect,clf):
	scores=[]
	X_train_dtm=vect.fit_transform(X_train)
	print "Features: ",X_train_dtm.shape[1]
	clf.fit(X_train_dtm,y_train)
	X_test_dtm=vect.transform(X_test)
	y_pred_class=clf.predict(X_test_dtm)
	print "classification_report"
	print classification_report(y_test,y_pred_class)
	accuracy=accuracy_score(y_test,y_pred_class)
	f1=f1_score(y_test,y_pred_class,average=None)
	precision=precision_score(y_test,y_pred_class,average=None)
	recall=recall_score(y_test,y_pred_class,average=None)
	print "accuracy_score",accuracy
	for prec,rec,f in zip(precision,recall,f1):
		scores.append(prec)
		scores.append(rec)
		scores.append(f)
		
	scores.append(accuracy)
	return scores
def grid(pipeline,parameters):
	cv = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,cv=5,scoring='accuracy')
	print "Performing grid search..."
	print "pipeline:",[name for name, _ in pipeline.steps]
	print "parameters:"
	pprint(parameters)
	t0 = time()
	cv.fit(X_train, y_train)
	print "done in %0.3fs" % (time() - t0)
	print""
	print "Best score: %0.3f" % cv.best_score_
	print "Best parameters set for tfidf:"
	best_parameters = cv.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print "\t%s: %r" % (param_name, best_parameters[param_name])




