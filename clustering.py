# Autores:
# Descripción:

###########################################################################################################################################################################
#########################################################           IMPORTACIONES       ###############################################################
###########################################################################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt

###########################################################################################################################################################################
#########################################################           MÉTODOS           ###############################################################
###########################################################################################################################################################################
def barPlotInstanciasPorClase(y_train):
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

    

def preproceso():    ##david
    #Hay que cambiar los mensajes para quitar mayusculas, stop words, lematización....
    #Precondición: Dada la lista de mensajes
    #Postcondición: Procesarlos quitando las stop words, pasando a minúsculas, lematizando, tokenizacion....
    pass

def vectorizacion():   #david
    #Hay que vectorizar el conjunto de datos de tal forma que los mensjes de twitter para que la información de los mensajes se pueda contar y sea lo más representativa posible
    #Precondición: La lista de mensajes ya procesada
    #Postcondición: Una lista con las palabras mas representativas y el numero de apariciones o lo que sea usando tf-idf, bow o embedding.
    pass
def redimensionar():     #albert
    ##comprobar que numero de dimensiones es mejor para esta practica, cual ofrece más información o si se pierde al redimensionar.
    #Precondición: Recibe los mensajes vectorizados
    #Postcondición: devuelve y redimensiona si procediese
    pass

def clustering():    #alberto              #IMPORTANTE: hay que probar con distintos clusters, en vez de dos clusters que seán si o no, por ejemplo 5 que sean, si o si, muy posible, posible, poco posible, imposible
    #Utilizar K-Means sin usar ninguna librería para poder clasificar las instancias actuales y futuras.
    #Precondición: Recibe los tweets vectorizados y redimensionados
    #Postcondición: Devuelve un modelo K-Means con los mejores hiperparámetros y con las instancias calculadas vs label real
    pass
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

    print("Las instancias están repartidas en las dos clases de la siguiente forma:")
    print(dfTweetsData['label'].value_counts(), end="\n")
    print()
    # to check out what we are going to be working with
    #dfTweetsData.info()

    '''
    # re-arrange:  y:class and x:features
    y_train = dfTweetsData['label'].copy()
    dfTweetsData.drop('label', axis=1, inplace=True)
    '''
    analisisDeDato(dfTweetsData)