# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 11:53:45 2022

@author: Dell
"""
import requests
from bs4 import BeautifulSoup as bs
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#Before we will proceed first check the connectivity
link="https://www.amazon.in/Samsung-Inverter-Convertible-Bacteria-AR18BYMZAUR/product-reviews/B09P8LG2RC/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
response=requests.get(link)
response
#if you are getting 503 means connection to Amazon has been lost
#you must get 200 for successfull conectivity
soup=bs(response.content,"html.parser")

#create a empty list to which we will append all the reviews

product_reviews=[]
for i in range(1,3):
    link="https://www.amazon.in/Samsung-Inverter-Convertible-Bacteria-AR18BYMZAUR/product-reviews/B09P8LG2RC/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"+str(i)
    response=requests.get(link)
    soup=bs(response.content,"html.parser")
    ##Now to scrap the contents go to product page and inspect the source code
    ##identify the tag and class name so as to scrap the contents
    reviews=soup.find_all("span",class_="a-size-base review-text review-text-content")
    ip=[]
    for i in range(0,len(reviews)):
        ip.append(reviews[i].get_text())
    product_reviews=product_reviews+ip    

# writng reviews in a text file 
with open("SamsungAC.txt", "w", encoding='utf8') as output:
    output.write(str(product_reviews))
# Removing unwanted symbols incase if exists
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(product_reviews)

import nltk
# from nltk.corpus import stopwords

# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+", " ", ip_rev_string).lower()
# ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)

# words that contained in the reviews
ip_reviews_words = ip_rev_string.split(" ")

ip_reviews_words = ip_reviews_words[1:]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(ip_reviews_words,use_idf=True,ngram_range=(1,1))
X=vectorizer.fit_transform(ip_reviews_words)
with open("C:/360DG/Datasets/stop.txt", "r") as sw:
    stop_words = sw.read()
stop_words = stop_words.split("\n")
ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)
# WordCloud can be performed on the string inputs.
# Corpus level word cloud

wordcloud_ip = WordCloud(background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)
plt.imshow(wordcloud_ip)
# positive words # Choose the path for +ve words stored in system
with open("C:/360DG/Datasets/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)
# negative words Choose path for -ve words stored in system
with open("C:/360DG/Datasets/negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)
####################################
# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)
# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)

# Remove stop words
text_content = [word for word in text_content if word not in stopwords_wc]
# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

# nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)
################
# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])
##################################
# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=stopwords_wc)

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

