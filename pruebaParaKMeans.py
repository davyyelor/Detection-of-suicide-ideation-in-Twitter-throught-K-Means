import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

import kMeans

###########################################################################################################################################################################
#########################################################           IMPORTACIONES       ##################################################################################
###########################################################################################################################################################################

train  = pd.read_csv("suicidal_data.csv",sep=",", encoding='cp1252')




###########################################################################################################################################################################
#########################################################           PREPROCESO       ##################################################################################
###########################################################################################################################################################################


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

print('Dataset size:',train.shape)
print('columns are:',train.columns)

length_train = train['tweet'].str.len()

plt.hist(length_train, bins=20, label="train_tweets")
plt.legend()
plt.show()

train['tidy_tweet'] = np.vectorize(remove_pattern)(train['tweet'], "@[\w]*")

train['tidy_tweet'] = train['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))


tokenized_tweet = train['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

train['tidy_tweet'] = tokenized_tweet

import nltk.corpus

stopword = nltk.corpus.stopwords.words('english')
stopword.extend(['fuck', 'shit'])


def remove_stopwords(text):
  text = [word for word in text if word not in stopword]
  return text

# combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: remove_stopwords(x)) # stemming
# combi.head()

train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopword)]))

#visualize all the words our data using the wordcloud plot
all_words = ' '.join([text for text in train['tidy_tweet']])





###########################################################################################################################################################################
#########################################################           MUESTRA DEL CONJUNTO       ##################################################################################
###########################################################################################################################################################################
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


#Non-suicide
normal_words =' '.join([text for text in train['tidy_tweet'][train['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


#Suicide
negative_words = ' '.join([text for text in train['tidy_tweet'][train['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()



###########################################################################################################################################################################
#########################################################           VECTORIZACION       ##################################################################################
###########################################################################################################################################################################
#Bag-of-words
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(train['tidy_tweet'])
bow.shape
# bow[:5]


#Tf-IDF
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import gensim

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(train['tidy_tweet'])




#Word embbeding
tokenized_tweet = train['tidy_tweet'].apply(lambda x: x.split()) # tokenizing

model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            vector_size=200, # desired no. of features/independent variables
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)

model_w2v.train(tokenized_tweet, total_examples= len(train['tidy_tweet']), epochs=20)

print(model_w2v.wv.most_similar(positive="die"))

print(model_w2v.wv.most_similar(positive="suicid"))

model_w2v.wv.get_vector('suicid')



def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v.wv.get_vector(word).reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary

            continue
    if count != 0:
        vec /= count
    return vec


wordvec_arrays = np.zeros((len(tokenized_tweet), 200))

for i in range(len(tokenized_tweet)):
    wordvec_arrays[i, :] = word_vector(tokenized_tweet[i], 200)

wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import TaggedDocument

def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(TaggedDocument(s, ["tweet_" + str(i)]))
    return output

labeled_tweets = add_label(tokenized_tweet) # label all the tweets

labeled_tweets[:6]


model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model
                                  dm_mean=1, # dm = 1 for using mean of the context word vectors
                                  vector_size=200, # no. of desired features
                                  window=5, # width of the context window
                                  negative=7, # if > 0 then negative sampling will be used
                                  min_count=5, # Ignores all words with total frequency lower than 2.
                                  workers=3, # no. of cores
                                  alpha=0.1, # learning rate
                                  seed = 23)
print("conseguid")
model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])

model_d2v.train(labeled_tweets, total_examples= len(train['tidy_tweet']), epochs=15)

docvec_arrays = np.zeros((len(tokenized_tweet), 200))

for i in range(len(train)):
    docvec_arrays[i, :] = model_d2v.dv[i].reshape((1, 200))

docvec_df = pd.DataFrame(docvec_arrays)
docvec_df.shape


print("hecho lo dificil")


###########################################################################################################################################################################
#########################################################           CLUSTERING       ##################################################################################
###########################################################################################################################################################################
#Model buiding
#Logical Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

##################################################################################################################### Bag-of-Words Features
train_bow = bow

# train_bow = bow[:7365,0:1]
# test_bow = bow[7365:,1:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow.toarray(), train['label'],
                                                          random_state=42,
                                                          test_size=0.2)
# xtrain_bow, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
train_bow.shape


lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.2
prediction_int = prediction_int.astype(int)

print("Puntuación F-Score para bow con lr",f1_score(yvalid, prediction_int)) # calculating f1 score


##################################################################################################################### tf-idf Features
# TI-IDF Features
train_tfidf = tfidf_matrix

xtrain_tfidf = train_tfidf[ytrain.index].toarray()
xvalid_tfidf = train_tfidf[yvalid.index].toarray()

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.2
prediction_int = prediction_int.astype(int)

print("Puntuación F-Score para tf-idf con lr", f1_score(yvalid, prediction_int))


##################################################################################################################### Word-Embbedding Features
# Word2vec Features
train_w2v = wordvec_df

xtrain_w2v = train_w2v.iloc[ytrain.index,:]
xvalid_w2v = train_w2v.iloc[yvalid.index,:]

xtrain_w2v.shape

lreg.fit(xtrain_w2v, ytrain)

prediction = lreg.predict_proba(xvalid_w2v)
prediction_int = prediction[:,1] >= 0.2
prediction_int = prediction_int.astype(int)
print("Puntuación F-Score para word-embedding con lr", f1_score(yvalid, prediction_int))

print("Nos metemos al Kmeans duro")
###########################################################################################################################################################################
#########################################################           CLUSTERING       ##################################################################################
###########################################################################################################################################################################
from kMeans import *


###############################################################################KMEANS tf-idf
# Convierte la matriz TF-IDF a un formato adecuado para tu KMeans
tfidf_array = tfidf_matrix.toarray()

# Luego, instanciamos la clase KMeans_Clustering y ajustamos el modelo a tus datos.
# Debes ajustar el número de clusters, el método de inicialización y otros parámetros según tu preferencia.
algoritmo = KMeans_Clustering(n_cluster=2, initialisation_method='random', iter_max=100, p_value=2)
algoritmo.ajustar(instances=tfidf_array)

# Ahora que el modelo K-Means ha sido ajustado, puedes obtener las etiquetas predichas para tus datos.
# Las etiquetas estarán en algoritmo.labels
predicted_labels = algoritmo.labels

# Imprime las etiquetas predichas
print("Etiquetas predichas:", predicted_labels)


####################################################################################KMEANS BOW
# Convierte la matriz TF-IDF a un formato adecuado para tu KMeans
bow_array = bow.toarray()

# Luego, instanciamos la clase KMeans_Clustering y ajustamos el modelo a tus datos.
# Debes ajustar el número de clusters, el método de inicialización y otros parámetros según tu preferencia.
algoritmo = KMeans_Clustering(n_cluster=2, initialisation_method='random', iter_max=100, p_value=2)
algoritmo.ajustar(instances=bow_array)

# Ahora que el modelo K-Means ha sido ajustado, puedes obtener las etiquetas predichas para tus datos.
# Las etiquetas estarán en algoritmo.labels
predicted_labels = algoritmo.labels

# Imprime las etiquetas predichas
print("Etiquetas predichas:", predicted_labels)




################################################################################KMEANS word-embbeding
# Convierte la matriz TF-IDF a un formato adecuado para tu KMeans
wd_array = train_w2v

# Luego, instanciamos la clase KMeans_Clustering y ajustamos el modelo a tus datos.
# Debes ajustar el número de clusters, el método de inicialización y otros parámetros según tu preferencia.
algoritmo = KMeans_Clustering(n_cluster=2, initialisation_method='random', iter_max=100, p_value=2)
algoritmo.ajustar(instances=wd_array)

# Ahora que el modelo K-Means ha sido ajustado, puedes obtener las etiquetas predichas para tus datos.
# Las etiquetas estarán en algoritmo.labels
predicted_labels = algoritmo.labels

# Imprime las etiquetas predichas
print("Etiquetas predichas:", predicted_labels)