import numpy as np
import lda.datasets
import re
import scipy.sparse as sp
import json
import codecs
import networkx as nx
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

ranking_file = codecs.open("topComments.txt", "w", "utf-8")
cluster_file = codecs.open("clustering.txt", "w", "utf-8")
it_stop = get_stop_words('italian')
tokenizer = RegexpTokenizer(r'\w+')
c = CountVectorizer()
# Number of comments to Fetch within cluster
k = 5
# Json file containing the comments
comments_file = 'test.json'


def rankcomments(commentcluster, k):
    bow_matrix = c.fit_transform(commentcluster)
    normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
    similarity_graph = normalized_matrix * normalized_matrix.T
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph, 0.85)
    ranked = sorted(((scores[i], s) for i, s in enumerate(commentcluster)), reverse=True)
    for tC in range(0, k):
        ranking_file.write("Comment {} : {}\n".format(tC, ranked[tC][1]))


def cluster_comments(doc_set, texts):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0)
    matrix = vectorizer.fit_transform(doc_set)
    matrix = sp.csr_matrix(matrix, dtype=np.int64, copy=False)
    # feature_names = vectorizer.get_feature_names()
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
    for i in range(len(doc_set)):
        dictionary[int(doc_topic[i].argmax())].append(str(titles[i]))
    return dictionary


def read_data(comments):
    video = json.load(open(comments))
    comment_set = []
    for comment in video['comments']:
        comment_set.append(re.sub('[^a-zA-Z0-9\s\P{P}\']+', r'', comment.get('text').replace("\n", " ")))
    return comment_set


def clean_data(comments_list):
    comment_texts = []
    # loop through document list
    for i in comments_list:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in it_stop]
        # add tokens to list
        comment_texts = comment_texts + stopped_tokens
    return comment_texts


def main():
    doc_set = read_data(comments_file)
    texts = clean_data(doc_set)
    clusters = cluster_comments(doc_set, texts)
    for key, value in clusters.iteritems():
        cluster_file.write("(top topic: {}) - {}  \n".format(key, value))
        ranking_file.write("Top comments in topic are\n")
        rankcomments(value, k)
    cluster_file.close()
    ranking_file.close()

main()
