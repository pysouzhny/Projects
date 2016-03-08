from nltk.stem.snowball import SnowballStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob, Word

def parseOutText(text_string):
	stemmer=SnowballStemmer('english')
	words = ""
	if len(text_string) > 1:
	    ### remove punctuation
	    #text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

	    
	    w=TextBlob(text_string)
	    word=[stemmer.stem(word) for word in w.words]
	    for w in word:
	        words+=w+" "

	    

	    ### split the text string into individual words, stem each word,
	    #words=[stemmer.stem(word) for word in text_string]
	    ### and append the stemmed word to words (make sure there's a single
	    ### space between each stemmed word)
	    




	return str(words)


def stemmed_word(text):
	#text = unicode(text, 'utf-8').lower()
	text=str(text).encode('ascii',"ignore").decode("ascii")
	w=TextBlob(text)
	
	stemmed_words=[stemmer.stem(word) for word in w.words]
	return stemmed_words


def split_into_lemmas(text):
	#text = unicode(text, 'utf-8').lower()
	text=str(text).encode('ascii',"ignore").decode("ascii")
	words = TextBlob(text).words
	lemmatized_words=[word.lemmatize() for word in words]
	return lemmatized_words
    

 	