# -*- coding: utf-8 -*-
"""
@filename: SentimentAnalytics.py
@author: yashw
"""

# download
#import nltk
#nltk.download('all')
#import nltk
#nltk.download('vader_lexicon')

# imports
import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns

# file-input.py
f = open('./data/tweets.txt','r')
strText = f.read()
f.close()

# print file text
print(strText)

# print object type
print(type(strText))

# split scene_one into sentences: sentences
from nltk.tokenize import sent_tokenize
lstLines = sent_tokenize(strText)

# print file text
print(lstLines)

# print object type
print(type(lstLines))

# make a frequency list of lengths: line_num_words
lstWordLength = [len(vline) for vline in lstLines]
print(lstWordLength)

# print line 1 & chars in line 1
print(lstLines[0])
print(lstWordLength[0])

# plot a histogram of the line lengths
# histogram
plt.figure()
sns.distplot(lstWordLength, bins=7, kde=False, color='b')
plt.show()


# clean

# convert into lowercase
lstLines = [t.lower() for t in lstLines]
print(lstLines)

# remove punctuations
import string
lstLines = [t.translate(str.maketrans('','',string.punctuation)) for t in lstLines]
print(lstLines)


# import
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# classifier
def nltk_sentiment(sentence):
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score

# test
lstOneLine = "Today I am very happy"
print(lstOneLine)
saResults = nltk_sentiment(lstOneLine)
print(saResults)

# test
lstOneLine = "Today is a bad day"
print(lstOneLine)
saResults = nltk_sentiment(lstOneLine)
print(saResults)

# test
lstOneLine = "Today I went to school"
print(lstOneLine)
saResults = nltk_sentiment(lstOneLine)
print(saResults)

# call clasiifier
saResults = [nltk_sentiment(t) for t in lstLines]
print(saResults)

# check
print(lstLines[0])
print(saResults[0])

# find result
def getResult(pos, nue, neg):
    if (pos > nue and pos > neg):
        return ("Positive")
    elif (neg > nue and neg > pos):
        return ("Negative")
    else:
        return('Neutral')

# create dataframe
df = pd.DataFrame(lstLines, columns=['Lines'])
df['Pos']=[t['pos'] for t in saResults]
df['Nee']=[t['neu'] for t in saResults]
df['Neg']=[t['neg'] for t in saResults]
df['Result']= [getResult(t['pos'],t['neu'],t['neg']) for t in saResults]

