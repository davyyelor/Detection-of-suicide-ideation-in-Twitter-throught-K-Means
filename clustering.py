# Autores:
# Descripción:


###########################################################################################################################################################################
#########################################################           IMPORTACIONES       ###############################################################
###########################################################################################################################################################################
import time
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
from kMeans import *
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from sklearn.cluster import KMeans

import kMeans
from kMeans import *
from kMeans_cuda import *


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
    print(dfTweetsData['label'].value_counts(), end="\n")
    print()



###################################################################### PREPROCESO #############################################################
def remove_non_ascii(word):
    """Se eliminan todas las palabras que no este en formato ascii"""
    new_words = []
    new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Todas las letras se transforman en minuscula"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Se borra la puntuacion de las palabras"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Se convierten los numeros en su representacion con palabras"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Se eliminan las stopwords"""
    stop_words = set(stopwords.words('english'))
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def preprocesado(df_train):
    features = df_train['tweet'].values.tolist()
    labels = df_train['label'].values
    processed_features = []

    for words in range(0, len(features)):
        words = str(features[words])
        #words = emoji.demojize((words), delimiters=("", ""))
        # words = words.split(" ")[1:-1]
        # words = ' '.join([str(elem) for elem in words])
        words = remove_non_ascii(words)
        words = to_lowercase(words)
        words = remove_punctuation(words)
        words = replace_numbers(words)
        words = remove_stopwords(words)
        words = lemmatize_verbs(words)
        words = stem_words(words)
        words = ' '.join([str(elem) for elem in words])
        processed_features.append(words)
    return labels, processed_features







##################################################################### VECTORIZACION ###############################################################

def bow (processed_features):
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=20000, stop_words='english')
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(processed_features)
    bow.shape
    return processed_features,bow

def tfidf(processed_features):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=20000, stop_words='english')
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(processed_features)
    return processed_features, tfidf_vectorizer


def wordEmbeddings(processed_features):
    # Word embbeding
    tokenized_tweet = [text.split() for text in processed_features]  # tokenizing

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

    model_w2v.train(tokenized_tweet, total_examples=len(processed_features), epochs=20)
    we_vectorizado= np.vectorize(processed_features, model=model_w2v, strategy='average')

    # Ahora puedes hacer uso del modelo entrenado para obtener representaciones vectoriales de palabras o realizar tareas de procesamiento de texto con word embeddings.
    # Por ejemplo, para encontrar palabras similares a "die":
    similar_words = model_w2v.wv.most_similar(positive="die")
    print(similar_words)

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






def redimensionar():     #albert
    ##comprobar que numero de dimensiones es mejor para esta practica, cual ofrece más información o si se pierde al redimensionar.
    #Precondición: Recibe los mensajes vectorizados
    #Postcondición: devuelve y redimensiona si procediese
    pass

def clustering(X,n):    #alberto              #IMPORTANTE: hay que probar con distintos clusters, en vez de dos clusters que seán si o no, por ejemplo 5 que sean, si o si, muy posible, posible, poco posible, imposible
    #Utilizar K-Means sin usar ninguna librería para poder clasificar las instancias actuales y futuras.
    #Precondición: Recibe los tweets vectorizados y redimensionados
    #Postcondición: Devuelve un modelo K-Means con los mejores hiperparámetros y con las instancias calculadas vs label real
    modelo = KMeans(n_clusters=n, random_state=42)
    # Entrenar el modelo K-Means
    modelo.fit(X)

    # Obtener las etiquetas predichas por el modelo
    y_pred = modelo.labels_

    return y_pred





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
    dfTweetsData = []
    dfTweetsData = pd.read_csv("suicidal_data.csv",sep=",",encoding='cp1252')

    # Mapear los valores en la columna 'label'
    dfTweetsData['label'] = dfTweetsData['label'].map({0: 'no', 1: 'si'})

    # to check out what we are going to be working with
    #dfTweetsData.info()

    #barPlotInstanciasPorClase(dfTweetsData)

    #analisisDeDato(dfTweetsData)


    labels, tweets = preprocesado(dfTweetsData)

    opcion = "bow"

    processed_features, vector = vectorizacion(tweets, opcion)


    '''print("Se ha vectorizado con", opcion)
    instancia = vector.toarray()[0]
    print(instancia)
    print(type(instancia))'''

    X = vector.toarray()
    y_real = dfTweetsData['label']
    n = 2
    #y_pred =clustering(X,n)
    #for etiqueta in y_pred:
        #print(etiqueta)
    tiempo = time.time()
    kmeans = KMeans_Clustering_CUDA(n_cluster=n, iter_max=100, p_value=6)
    kmeans.ajustar(instances=X)


    y_pred = kmeans.labels

    print(f"Ha tardado {time.time() - tiempo} segundos")





