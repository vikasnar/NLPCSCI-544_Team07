import numpy as np
import lda
import lda.datasets
import re
import json
import codecs
from collections import defaultdict

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
doc_set = []
video = json.load(open('/Users/sridharyadav/Downloads/SenTube/tablets_IT/video__fKtsmt2-00-annotator:Agata.json'))
for comment in video['comments']:
    doc_set.append(re.sub('[^a-zA-Z0-9\s\P{P}\']+',r'',comment.get('text').replace("\n", " ")))
titles = tuple(doc_set)
X.shape
(395, 4258)
model = lda.LDA(n_topics=4, n_iter=500, random_state=1)
model.fit(X)
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


doc_topic = model.doc_topic_
dict = defaultdict(list)
file = codecs.open("clustering.txt", "w", "utf-8")
for i in range(93):
    dict[int(doc_topic[i].argmax())].append(str(titles[i]))


for key, value in dict.iteritems():
    file.write("(top topic: {}) - {}  \n".format(key, value))
file.close()