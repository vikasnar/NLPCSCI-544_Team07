from __future__ import division, unicode_literals
from textblob import TextBlob as tb
from stop_words import get_stop_words
import sys
import math

reload(sys)
sys.setdefaultencoding('utf8')

it_stop = get_stop_words('italian')

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def get_nonstop_words(text):
    tokens = text.split()
    stopped_tokens = [i for i in tokens if not i in it_stop]
    return stopped_tokens

summary_text = eval(open("summary.txt").read())
input_text = eval(open("input.txt").read())

summarybloblist = []
inputbloblist = []

for comment in summary_text:
    summarybloblist.append(tb(' '.join(get_nonstop_words(comment))))

for comment in input_text:
    inputbloblist.append(tb(' '.join(get_nonstop_words(comment))))

def calculate_tfidf_sum(bloblist):
    scores = {}
    for i,blob in enumerate(bloblist):
        scores.update({word: tfidf(word, blob, bloblist) for word in blob.words})
    tf_idf_sum = 0
    for key in scores:
        tf_idf_sum += scores[key]
    return tf_idf_sum/len(scores)

tf_idf_summary = calculate_tfidf_sum(summarybloblist)
tf_idf_input = calculate_tfidf_sum(inputbloblist)

print "Retention Rate = ", tf_idf_summary/tf_idf_input
