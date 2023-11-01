# Autores:
# Descripción:


###########################################################################################################################################################################
#########################################################           IMPORTACIONES       ###############################################################
###########################################################################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, LancasterStemmer
#import emoji
import inflect as inflect
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD

import kMeans
from kMeans import *


###########################################################################################################################################################################
#########################################################           MÉTODOS           ###############################################################
###########################################################################################################################################################################
def barPlotInstanciasPorClase(dfTweetsData):
    # re-arrange:  y:class and x:features
    y_train = dfTweetsData['label'].copy()
    # num samples per class
    value_counts = y_train.value_counts()
    # Ordenar los valores y etiquetas por valor descendente
    sorted_counts = value_counts.sort_index(ascending=True)
    sorted_labels = sorted_counts.index
    max_count = sorted_counts.max()  # Recuento máximo
    # Crear el gráfico de barras
    plt.bar(sorted_labels, sorted_counts)
    # Establecer una altura fija para todas las etiquetas de texto
    label_height = max_count + 10  # Puedes ajustar este valor según tus preferencias
    # Agregar etiquetas de texto con el recuento en la altura fija
    for index, value in enumerate(sorted_counts):
        plt.text(index, label_height, str(value), ha='center', va='bottom')
    # Ajustar el eje x para mostrar cada valor individualmente
    plt.xticks(range(len(sorted_labels)), sorted_labels)
    # Agregar etiquetas en los ejes
    plt.xlabel('Clase')
    plt.ylabel('Número de Instancias')
    # Añadir título
    plt.title('Barplot del conjunto train')
    plt.show()




def analisisDeDato(tweets):  ##david
    #Comprobar si hay valores faltantes, calcular máximo, medio, mínimo, intancias repetidas, número y tipo de atributos y rango de valores.
    #Precondición: Dado el conjunto de datos ya leido y leido en pandas.
    #Postcondición: imprime mensajes avisando de valores faltantes... y en caso de haberlos solucionandolos o borrando o añadiendo con la media...
    print(tweets[(tweets['label']=="no")].head(5))
    print(tweets[(tweets['label']=="si")].head(5))
    print()
    print("Las instancias están repartidas en las dos clases de la siguiente forma:")
    print(tweets['label'].value_counts(), end="\n")
    print()



###################################################################### PREPROCESO #############################################################
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

def remove_stopwords(text,stopword):
  text = [word for word in text if word not in stopword]
  return text



def preproceso(train):
    print('Dataset size:', train.shape)
    print('columns are:', train.columns)
    length_train = train['tweet'].str.len()
    train['tidy_tweet'] = np.vectorize(remove_pattern)(train['tweet'], "@[\w]*")

    train['tidy_tweet'] = train['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

    train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

    tokenized_tweet = train['tidy_tweet'].apply(lambda x: x.split())
    tokenized_tweet.head()

    from nltk.stem.porter import *
    stemmer = PorterStemmer()

    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])  # stemming
    tokenized_tweet.head()

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    train['tidy_tweet'] = tokenized_tweet

    import nltk.corpus

    stopword = nltk.corpus.stopwords.words('english')
    stopword.extend(['fuck', 'shit'])

    # combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: remove_stopwords(x)) # stemming
    # combi.head()

    train['tidy_tweet'] = train['tidy_tweet'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stopword)]))

    # visualize all the words our data using the wordcloud plot
    all_words = ' '.join([text for text in train['tidy_tweet']])
    return all_words, train

##################################################################### VECTORIZACION ###############################################################

def bow (train):
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(train['tidy_tweet'])
    bow.shape
    return processed_features,bow

def tfidf(train):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # TF-IDF feature matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(train['tidy_tweet'])
    return processed_features, tfidf_vectorizer




def wordEmbeddings(train):
    # Word embbeding
    tokenized_tweet = train['tidy_tweet'].apply(lambda x: x.split())  # tokenizing

    model_w2v = gensim.models.Word2Vec(
        tokenized_tweet,
        vector_size=200,  # desired no. of features/independent variables
        window=5,  # context window size
        min_count=2,
        sg=1,  # 1 for skip-gram model
        hs=0,
        negative=10,  # for negative sampling
        workers=2,  # no.of cores
        seed=34)

    model_w2v.train(tokenized_tweet, total_examples=len(train['tidy_tweet']), epochs=20)

    print(model_w2v.wv.most_similar(positive="die"))

    print(model_w2v.wv.most_similar(positive="suicid"))

    model_w2v.wv.get_vector('suicid')

    wordvec_arrays = np.zeros((len(tokenized_tweet), 200))

    for i in range(len(tokenized_tweet)):
        wordvec_arrays[i, :] = word_vector(tokenized_tweet[i], 200)

    wordvec_df = pd.DataFrame(wordvec_arrays)
    wordvec_df.shape

    from tqdm import tqdm
    tqdm.pandas(desc="progress-bar")
    from gensim.models.doc2vec import TaggedDocument

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


def vectorizacion(tweets, opc):   #david
    #Hay que vectorizar el conjunto de datos de tal forma que los mensjes de twitter para que la información de los mensajes se pueda contar y sea lo más representativa posible
    #Precondición: La lista de mensajes ya procesada
    #Postcondición: Una lista con las palabras mas representativas y el numero de apariciones o lo que sea usando tf-idf, bow o embedding.
    if opc == "tf-idf":
        processed_features, tfidf_vectorizer = tfidf(tweets)
        tfidf_vectors = tfidf_vectorizer.transform(processed_features)
        return processed_features, tfidf_vectors
    elif opc == "bow":
        processed_features,bow_vector = bow(tweets)  # Call the bow function and get the result
        return processed_features, bow_vector
    elif opc == "word-embedding":
        wordEmbeddings(tweets)
        # En este caso, no tienes que devolver nada, ya que los embeddings se calcularán pero no se almacenarán en la función.
        return tweets, None
    else:
        raise ValueError("Se ha escogido una opción que no existe")






def redimensionar(X_train):     #albert
    ##comprobar que numero de dimensiones es mejor para esta practica, cual ofrece más información o si se pierde al redimensionar.
    #Precondición: Recibe los mensajes vectorizados
    #Postcondición: devuelve y redimensiona si procediese
    print('Dim originally: ', X_train.shape)
    # Reducir las dimensiones para visualizarlas: PCA
    svd = TruncatedSVD(n_components=5)
    X_train_PCAspace = svd.fit_transform(X_train)
    print('Dim after TruncatedSVD: ', X_train_PCAspace.shape)
    return X_train_PCAspace

def clustering(X,n):    #alberto              #IMPORTANTE: hay que probar con distintos clusters, en vez de dos clusters que seán si o no, por ejemplo 5 que sean, si o si, muy posible, posible, poco posible, imposible
    #Utilizar K-Means sin usar ninguna librería para poder clasificar las instancias actuales y futuras.
    #Precondición: Recibe los tweets vectorizados y redimensionados
    #Postcondición: Devuelve un modelo K-Means con los mejores hiperparámetros y con las instancias calculadas vs label real
    pass

def palabrasRepresentativas(all_words):
    from wordcloud import WordCloud
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

    # Non-suicide
    normal_words = ' '.join([text for text in train['tidy_tweet'][train['label'] == 0]])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

    # Suicide
    negative_words = ' '.join([text for text in train['tidy_tweet'][train['label'] == 1]])
    wordcloud = WordCloud(width=800, height=500,
                          random_state=21, max_font_size=110).generate(negative_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()




def obtenerPuntuaciones():    ###bermu
    #calcular las puntuaciones con diferentes métricas para ver la calidad de nuestro moodelo, si tiene capacidad de mejora o por el contrario ya podría clusterizar todo con un alto grado de confianza.
    #Precondición: Recibe el conjutno ya clasificado
    #Postcondición: Y devuelve las puntuaciones con las métricas más representativas.
    pass
def representacionResultados():        #bermudez
    #además de las puntuaciones, presentar barplots mapas de calor... para poder representar mejor los resultados obtenidos y la bonanza de nuestro modelo
    #Precondición: Recibe el conjutno ya clasificado
    #Postcondición: Y representa los resultados graficamente
    pass

def guardarModelo():          ###bermudez
    #crear un archivo .sav para guardar el mejor modelo y así poder clasificar nuevas instancias
    #Precondición: Recibe el modelo K-Means óptimo
    #Postcondición: Lo guarda en un archivo .sav para el futuro
    pass
def clasificarInstancia():      ##bermudez
    #clasificar una nueva instancia dada
    #Precondición: Recibe un tweet correcto y (no se si) si vectorizado junto al modelo guardado con K-Means.
    #Postcondición: Devuelve un label con el valor de la instancia
    pass










###########################################################################################################################################################################
#########################################################           Inicialización         ###############################################################
###########################################################################################################################################################################
if __name__=="__main__":
    train = pd.read_csv("../suicidal_data.csv", sep=",", encoding='cp1252')

    copyTrain = train

    # Mapear los valores en la columna 'label'
    copyTrain['label'] = copyTrain['label'].map({0: 'no', 1: 'si'})

    # to check out what we are going to be working with
    copyTrain.info()

    barPlotInstanciasPorClase(copyTrain)

    analisisDeDato(copyTrain)

    all_words, train = preproceso(copyTrain)

    opcion = "bow"

    palabrasRepresentativas(all_words)

    processed_features, vector = vectorizacion(train, opcion)

    redimensioned_vect = redimensionar(vector)

    print("Se ha vectorizado con", opcion)
    print(vector)
    print(vector.shape)

    X = vector
    y_real = train['label']



    n= 2
    #y_pred =clustering(X,n)
    #for etiqueta in y_pred:
        #print(etiqueta)

    print(redimensioned_vect)

    algoritmo = kMeans.KMeans_Clustering(n_cluster=n)

    import matplotlib.pyplot as plt

    # Asume que "X" contiene tus vectores de características reducidos a dos dimensiones
    # Asume que "y_real" contiene las etiquetas reales

    # Etiquetas de las clases
    classes = ['si', 'no']

    # Colores para las clases
    colors = ['r', 'b']

    # Crear un gráfico de dispersión
    plt.figure(figsize=(8, 6))
    for i in range(len(classes)):
        class_idx = y_real[y_real == classes[i]].index
        plt.scatter(X[class_idx, 0], redimensioned_vect[class_idx, 1], c=colors[i], label=classes[i])

    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.legend()
    plt.title('Tweets Vectorizados en 2D')

    plt.show()

    algoritmo.ajustar(instances=redimensioned_vect)



    y_pred = algoritmo.labels
    print(y_pred)






