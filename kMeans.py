import random
from sklearn.cluster import k_means
import numpy as np


class KMeans_Clustering():

    def __init__(self, n_cluster=None, initialisation_method=None, iter_max=100, p_value=1):
        self.n_clusters = n_cluster
        self.method = initialisation_method
        self.iter_max = iter_max
        self.p_value = p_value
        # Devolver una matriz que sea un vector de vectores (n_clusters vectores que contienen las coordenadas de los centros)

    def minkowski_distance(self, vector1, vector2, p_value):

        def p_exp(x, y, p):
            return abs(x - y) ** p

        distance = 0
        for idx in range(len(vector1)):
            distance += p_exp(vector1[idx], vector2[idx], p_value)

        return distance ** (1 / p_value)

    def ajustar(self, instances):  # Itera el algoritmo de KMeans, calculando los centros (La matriz)
        # Instances:
        # [[ valores_palabra ],
        #  [ valores_palabra ]]
        N = len(instances[0])

        # Matriz de centroides, filas son el índice del centroide y columnas los valores para esa componente en cada centroide
        if self.method == None:
            self.centroides = random.sample(instances, self.n_clusters)

        for _ in range(self.iter_max):
            ###################################################
            ### REASIGNACIÓN DE INSTANCIAS CON SU CENTROIDE ###
            ###################################################
            centroides_asignados = {}
            centroides_asignados = {i: [] for i in range(self.n_clusters)}

            for instance_idx in range(len(instances)):
                distanciaMin = float('inf')
                for centroid_idx in range(len(self.centroides)):
                    distancia = self.minkowski_distance(instances[instance_idx], self.centroides[centroid_idx],
                                                        self.p_value)
                    if distancia < distanciaMin:
                        distanciaMin = distancia
                        centroid = centroid_idx
                centroides_asignados[centroid].append(instance_idx)

            ###################################################
            ###         CALCULAR NUEVOS CENTROIDES          ###
            ###################################################

            for numero_cluster in centroides_asignados.keys():
                lista_idx_instancias = centroides_asignados[numero_cluster]
                if lista_idx_instancias != []:
                    lista_instancias = np.array([instances[i] for i in lista_idx_instancias])
                    nuevo_cluster = np.mean(lista_instancias, axis=0)
                    self.centroides[numero_cluster] = nuevo_cluster
                else:
                    self.centroides[numero_cluster] = np.random.rand(N)

        self.labels = []
        for instance_idx in range(len(instances)):
            for centroid_idx in centroides_asignados.keys():
                if instance_idx in centroides_asignados[centroid_idx]:
                    self.labels.append(centroid_idx)

    def calcular_clusters(self):  # Etiqueta las instancias de inicialización
        pass