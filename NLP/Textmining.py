# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:52:21 2022

@author: Dell
"""


sentence="we are learning TextMining from 360DGTMG"
# To check whether TextMining word is there in sentence
"TextMining" in sentence
#if we want to know postion of learning
sentence.index('learning')
#if we want to split the sentence and ckeck the index of "TextMining"
sentence.split().index('TextMining')
# here it is going to show index of word
sentence.split()[2]
# if i want to reverse the characters of learning
sentence.split()[2][::-1]
words=sentence.split()
first_word=words[0]
last_word=words[len(words)-1]
concat_word=first_word+last_word
print(concat_word)
# if want to print the words with even length
[words for i in range(len(words)) if i%2==0]
#we are learning TextMining from 360DGTMG" ,it starts from -3 i.e.T to end
sentence[-3:]
#if I want sentence string in reverse order
sentence[::-1]
print(" ".join(word[::-1] for word in words))
#it will select each word and reverse and join as a sentence

##Tokenization
nltk.download('punkt')
import nltk
from nltk import word_tokenize
words=word_tokenize("I am reading NLP Fundamentals")
print(words)
nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(words)
# This will display all parts of speech tagging
# stop words from nltk package
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=stopwords.words('English')
print(stop_words)
sentence1="I am learning NLP.It is one of the most popular library in Python"
sentence_words=word_tokenize(sentence1)
print(sentence_words)
# filtering stop words which are there in sentence1
sentence_no_stop='  '.join( [word for word in sentence_words if word not in stop_words])
print(sentence_no_stop)
# To perform normalization and replace appropriate words
sentence2="I visited MY from IND on 14-02-20"
normalized_sentence=sentence2.replace("MY","Malaysia").replace("IND","India").replace("-20","-2020")
print(normalized_sentence)
# Auto correction in sentence
pip install autocorrect
from autocorrect import Speller
spell=Speller(lang="en")
spell("Natureal")


sentence3=word_tokenize("Ntural Lunguge processin deals with the ert")
print(sentence3)
sentence_corrected=' '.join([spell(word) for word in sentence3])
print(sentence_corrected)

#singularization and plurilization
pip install textblob
from textblob import TextBlob
sentence9=TextBlob('she sells seashells on the seashore')
sentence9.words
sentence9.words[2].singularize()
sentence9.words[5].pluralize()
##language translation from spanish to engilish
# here en:engilsh and es: spanish
from textblob import TextBlob
en_blob=TextBlob(u'muy bien')
en_blob.translate(from_lang='es',to='en')

##custom stopwords removal
from nltk import word_tokenize
sentence9="She sells seashells on the seashore"
custom_stop_word_list=['she','on','the','am','is','not']
' '.join([word for word in word_tokenize(sentence9) if word.lower() not in custom_stop_word_list])
#' '.join([word for word in word_tokenize(sentence9) if word.lower() not in custom_stop_word_list])

#Extrtacting general features from raw text 
#number of words
#Detect presence of wh words
#polarity
#subjectivity
#language identification

import pandas as pd
df=pd.DataFrame([['The vaccine for covid-19 will be announced on 1st August.'],['Do you know how much expectation the world poulaation is having from this research?'],['This risk of virus will end on 31 st July']])
df
df.columns=['text']
df
# to extract words from dataframe,you can write lambda function,which will take txt column
#seperate words from dataframe
from textblob import TextBlob
df['number_of_words']=df['text'].apply(lambda x:len(TextBlob(x).words))
df['number_of_words']

# Detect presence of words wh
wh_words=set(['why','who','which','what','where','when','how'])
df['is_wh_words_present']=df['text'].apply(lambda x:True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']   
#use the apply function to iterate through each row of column text convert the 
#text into text blob oject and extract words from them to check whether any of them is belong
#to the list of 'wh' words that has been declared in dataframe
# here two sets are compared one from text blob and another from the declared list,if it is
#present then it means intersection  is greater than 0 in respective row of the text column
# Here how is present in second row of df

#polarity and subjectivity,which is to extract sentiments +1 means positive polarity and -1 means
#negative ploarity,subjectivity lies between [0,1] subjectivity quantifies amount of personal opinion 
#compared to facts
sentence10='I  like this example'
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10='This is fantastic example and I like very much'
pol=TextBlob(sentence10).sentiment.polarity
pol
# we will check the polarity in given dataframe
df['polarity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']

# to check the subjectivity
sentence11='This was a helpful example but I would prefer another one'
sub=TextBlob(sentence11).sentiment.subjectivity
sub

sentence11='This is my personal opinion that ,it was a helpful example but I would prefer another one'
sub=TextBlob(sentence11).sentiment.subjectivity
sub

# To check the subjectivity
df['subjectivity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']

# Bag of Words
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['At least seven Indian pharma companies are working to develop a vaccine against coronavirus',
'the deadly virus that has already infected more than 14 million globally.',
'Bharat Biotech, Indian Immunologicals, are among the domestic pharma firms working on the coronavirus vaccines in India.'
]

bag_of_words_model = CountVectorizer()
print(bag_of_words_model.fit_transform(corpus).todense()) # bag of words

bag_of_word_df = pd.DataFrame(bag_of_words_model.fit_transform(corpus).todense())
bag_of_word_df.columns = sorted(bag_of_words_model.vocabulary_)
bag_of_word_df.head()
# TFIDF 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model=TfidfVectorizer()
#TfidfVectorizer converts collection of raw documents to matrix of TFIDF features,it is equivaleny
#to Count vectorizer followed by tfidfTransformer
print(tfidf_model.fit_transform(corpus).todense())
# Now you can save them in the dataframe
tfidf_df=pd.DataFrame(tfidf_model.fit_transform(corpus).todense())
# if you will check tfidf dataframe,there are only column numbers no words are there
# to give words to column following code is written
tfidf_df.columns=sorted(tfidf_model.vocabulary_)
# Now you can check each word is coming for each column
tfidf_df.head()
#########################
tfidf_model_small=TfidfVectorizer(max_features=5)
tfidf_df_small=pd.DataFrame(tfidf_model_small.fit_transform(corpus).todense())
tfidf_df_small.columns=sorted(tfidf_model_small.vocabulary_)
tfidf_df_small.head()
#############################
####Feature Engineering(Text similarity)
import nltk
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
lemmatizer=WordNetLemmatizer()
pair1=['Do you have Covid-19','your body temperature will tell you']
pair2=['I travelled to Malaysia','where did you travell']
pair3=['He is a programmer','is he not a programmer']
def extract_text_similarity_jaccard(text1,text2):
    words_text1=[lemmatizer.lemmatize(word.lower())for word in word_tokenize(text1)]
    words_text2=[lemmatizer.lemmatize(word.lower())for word in word_tokenize(text2)]
    numr=len(set(words_text1).intersection(set(words_text2)))
    der=len(set(words_text1).union(set(words_text2)))
    jaccard_sim=numr/der
    return jaccard_sim
extract_text_similarity_jaccard(pair1[0],pair1[1])
extract_text_similarity_jaccard(pair2[0],pair2[1])
extract_text_similarity_jaccard(pair3[0],pair3[1])
tfidf_model=TfidfVectorizer()
# Creating a corpus which will have texts of pair1, pair2 and pair 3 respectively
corpus = [pair1[0], pair1[1], pair2[0], pair2[1], pair3[0], pair3[1]]

tfidf_results = tfidf_model.fit_transform(corpus).todense()
#cosine similarity between texts of pair1
cosine_similarity(tfidf_results[0], tfidf_results[1])

#cosine similarity between texts of pair2
cosine_similarity(tfidf_results[2], tfidf_results[3])

#cosine similarity between texts of pair3
cosine_similarity(tfidf_results[4], tfidf_results[5])
