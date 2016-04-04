import numpy as np
import lda
import lda.datasets
import re
import scipy.sparse as sp
import json
import codecs
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

tokenizer = RegexpTokenizer(r'\w+')

it_stop = get_stop_words('italian')


X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
doc_set = []
video = json.load(open('/Users/sridharyadav/Downloads/SenTube/tablets_IT/video__fKtsmt2-00-annotator:Agata.json'))
for comment in video['comments']:
    doc_set.append(re.sub('[^a-zA-Z0-9\s\P{P}\']+',r'',comment.get('text').replace("\n", " ")))

texts = []

# loop through document list
for i in doc_set:

    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in it_stop]

    # add tokens to list
    texts = texts + stopped_tokens


vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0)
matrix = vectorizer.fit_transform(doc_set)
matrix = sp.csr_matrix(matrix, dtype=np.int64, copy=False)
feature_names = vectorizer.get_feature_names()

vocab = tuple(texts)
titles = tuple(doc_set)

model = lda.LDA(n_topics=4, n_iter=500, random_state=1)
model.fit(matrix)
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_
dictionary = defaultdict(list)
file = codecs.open("clustering.txt", "w", "utf-8")
for i in range(len(doc_set)):
    dictionary[int(doc_topic[i].argmax())].append(str(titles[i]))


for key, value in dictionary.iteritems():
    file.write("(top topic: {}) - {}  \n".format(key, value))
file.close()
