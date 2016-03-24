# fix the cross-validation process using Pipeline
from sklearn.pipeline import make_pipeline
#from __future__ import division # ensure that all division is float division
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, NuSVC, SVC
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
#import all the necessary modules
import os, sys, re
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
sns.set_style("whitegrid")

os.chdir(r"C:\Users\Dominique Njinkeu\Documents\GitHub\Projects\notebook")
df=pd.read_csv("tweets_sample.csv")
df.Support.value_counts()
# map the label values for scikit-learn
df["Label"]=df.Support.map({"Neutral":0,"Panthers":1,"Broncos":2})
#determine feature and label variables
X=df.text
y=df.Label
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

#multinomialNB classifier
from sklearn.naive_bayes import MultinomialNB
clfNB=MultinomialNB()
#knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
#linear SVM
from sklearn import svm
linear_svc=LinearSVC()
#Logistic Regression
from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)

from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from time import time
os.chdir(r"C:\Users\Dominique Njinkeu\Documents\GitHub\Projects\codes")
import ParseOutText

from sklearn.pipeline import Pipeline

def tokens(x):
	return x.split(' ')

pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english',ngram_range=(1,2),max_df=.5)),
    #('tfidf', TfidfVectorizer(stop_words='english',max_df=.5,ngram_range=(1,2),norm="l2",use_idf=False)),
    
    ('clf', linear_model.LogisticRegression(C=1000,penalty="l2")),
    ]
)

parameters= {
    #"vect__analyzer":(ParseOutText.split_into_lemmas,ParseOutText.split_into_stem),
   
    #'vect__ngram_range':((1,1),(1,2),(1,3)),
    #"vect__max_df":(0.5, 0.75, 1.0),
    #"vect__min_df":(.4,.3,.2,.1),
    "vect__max_features":range(9000,10000,25),
    #"clf__n_neighbors":range(1,30),
    #"clf__alpha":[a*0.1 for a in range(0,11)]
    
}


"""parameters = {
	#"tfidf__analyzer":(ParseOutText.split_into_lemmas,ParseOutText.split_into_stem),
	
    #'tfidf__use_idf': (True, False),
    #'tfidf__ngram_range':((1,1),(1,2),(1,3)),
   #"tfidf__max_df":(0.5, 0.75, 1.0),
    #'tfidf__norm': ('l1', 'l2'),
    "tfidf__max_features":range(1,6000),
    #"tfidf__min_df":(.4,.3,.2,.1)
    #"clf__n_neighbors":range(1,30)
    #"clf__alpha":[a*.1 for a in range(11)]
    #'bow__analyzer': (ParseOutText.split_into_lemmas, ParseOutText.split_into_stem),
}
"""
###parameters for knn
"""parameters={
	#"clf__weights":['uniform', 'distance'],
	#"clf__algorithm": ['ball_tree', 'kd_tree'],
	#"clf__leaf_size":range(10,100,5),
	#"clf__solver":["newton-cg", "lbfgs"],
	#"clf__p":[1,2],
	"clf__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
	"clf__penalty":['l1',"l2"]
	#"clf__n_neighbors":range(1,30)


}"""
from pprint import pprint

if __name__ == "__main__":
	cv = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,cv=10,scoring='accuracy')
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



