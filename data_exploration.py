import pandas as pd
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from nltk.cluster import KMeansClusterer
from nltk import FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords
nltk.download('punkt')

"""
Things to try first:
1) Get vector embeddings for description and transcript. (or title?)
2) Cluster them and see if we can make any sense out of the clusters.
3) Try to find patterns in the transcript talks.  
"""


# TED data
df = pd.read_csv('datasets/ted_main.csv')
df_transcript = pd.read_csv('datasets/transcripts.csv')


"""
Use Doc2Vec to vectorize, visualize the vectors.  Cluster and reduce dimension, then plot clusters.
"""
descriptions = [x for x in df['description']]
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(descriptions)]
model = Doc2Vec(documents, vector_size=8, window=2, min_count=1, workers=4)
tags = list(range(0, len(documents)))
vector_list = model[tags]

# reduce dim of description vectors...
data_embed1 = TSNE(n_components=2, perplexity=50, verbose=2, method='barnes_hut').fit_transform(vector_list)

# plot the reduced vectors
x_axis = data_embed1[:, 0]
y_axis = data_embed1[:, 1]
plt.scatter(x_axis, y_axis, s=5)
plt.show()

# how about now I try to cluster them?
NUM_CLUSTERS = 3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters1 = kclusterer.cluster(vector_list, assign_clusters=True)

# just to see which description is clustered where.
words = list(model.vocab)
for i, word in enumerate(documents):
    print(word.words + ":" + str(assigned_clusters1[i]))

# plot the reduced vectors with colors of clusters
X1 = data_embed1
plt.scatter(X1[:, 0], X1[:, 1], c=assigned_clusters1, s=50, cmap='viridis')


"""
Try a tf-idf approach to vectorizing the descriptions.  Then cluster and plot.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = descriptions
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# encode document
vector = vectorizer.transform(text)
vector_array = vector.toarray()

# cluster the tfidf vectors
NUM_CLUSTERS = 3 #3 might not be the best number here.. just trying something.
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(vector_array, assign_clusters=True)
# reduce dimension of tfidf vectors so I can visualize
data_embed = TSNE(n_components=2, perplexity=50, verbose=2, method='barnes_hut').fit_transform(vector_array)
X = data_embed
plt.scatter(X[:, 0], X[:, 1], c=assigned_clusters, s=50, cmap='viridis')

"""
Trying the titles for clustering and visualizing.
"""
titles = [x for x in df['name']]
documents_title = [TaggedDocument(doc, [i]) for i, doc in enumerate(titles)]
model_title = Doc2Vec(documents_title, vector_size=8, window=2, min_count=1, workers=4)
tags_title = list(range(0, len(documents_title)))
vector_list_title = model[tags_title]
# reduce dim
data_embed_title = TSNE(n_components=2, perplexity=50, verbose=2, method='barnes_hut').fit_transform(vector_list_title)
# cluster & plot
assigned_clusters_title = kclusterer.cluster(vector_list_title, assign_clusters=True)
X_title = data_embed_title
plt.scatter(X_title[:, 0], X_title[:, 1], c=assigned_clusters_title, s=50, cmap='viridis')

"""
Exploring the most frequent n_grams in the descriptions.
"""
transcripts = [x for x in df_transcript['transcript']]


def compute_freq(text_body, ngram_n=6):
    stop_words = set(stopwords.words('english'))
    n_gramfdist = FreqDist()
    for line in text_body:
        if len(line) > 1:
            tokens = line.strip().split(' ')
            # tokens_without_stops = [x.lower() for x in tokens if x.lower() not in stop_words]
            # n_grams = ngrams(tokens_without_stops, 3)
            n_grams = ngrams(tokens, ngram_n)
            n_gramfdist.update(n_grams)

    return n_gramfdist


compute_freq(text_body=transcripts, ngram_n=6)