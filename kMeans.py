import random
from sklearn.cluster import k_means
import numpy as np
from scipy.spatial import distance as sp_distance


class KMeans_Clustering():

    def __init__(self, n_cluster=None, initialisation_method='random', iter_max=100, p_value=1):
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
            p1 = vector1[idx]
            p2 = vector2[idx]
            if p1 != p2:
                distance += p_exp(p1, p2, p_value)

        return distance ** (1 / p_value)

    def ajustar(self, instances):  # Itera el algoritmo de KMeans, calculando los centros (La matriz)
        # Instances:
        # [[ valores_palabra ],
        #  [ valores_palabra ]]
        '''N = len(instances[0])'''
        N = instances.shape[0]
        print("Numero de instancias", N)
        

        # Matriz de centroides, filas son el índice del centroide y columnas los valores para esa componente en cada centroide
        if self.method == 'random':
            self.centroides = random.sample( instances.tolist(), self.n_clusters ) #De la lista de instancias, cogemos N para inicializar los clusters

        centroides_prev = None

        for iter in range(self.iter_max):
            ###################################################
            ### REASIGNACIÓN DE INSTANCIAS CON SU CENTROIDE ###
            ###################################################
            centroides_asignados = {}
            centroides_asignados = {i: [] for i in range(self.n_clusters)}

            for instance_idx in range(N):
                distanciaMin = float('inf')
                for centroid_idx in range(len(self.centroides)):
                    distancia = self.minkowski_distance(instances[instance_idx], self.centroides[centroid_idx], self.p_value)
                    if distancia < distanciaMin:
                        distanciaMin = distancia
                        centroid = centroid_idx
                centroides_asignados[centroid].append(instance_idx)

            ###################################################
            ###         CALCULAR NUEVOS CENTROIDES          ###
            ###################################################

            for numero_cluster in centroides_asignados.keys():
                lista_idx_instancias = centroides_asignados[numero_cluster]
                lista_instancias = [instances[i] for i in lista_idx_instancias]
                nuevo_cluster = np.mean(lista_instancias, axis=0)
                self.centroides[numero_cluster] = nuevo_cluster
            
            if centroides_prev == centroides_asignados:
                print("\n\nConverge")
                break
            else:
                centroides_prev = centroides_asignados

            ###################################################
            ###           PRINT DE LAS ITERACIONES          ###
            ###################################################

            print("\nIteracion", iter)
            for centroid_idx in centroides_asignados.keys():
                num_inst = len(centroides_asignados[centroid_idx])
                print("Número de instancias para cluster ", centroid_idx, "\t", num_inst)
            print("======================================")

        self.labels = []
        for instance_idx in range(N):
            for centroid_idx in centroides_asignados.keys():
                if instance_idx in centroides_asignados[centroid_idx]:
                    self.labels.append(centroid_idx)

    def calcular_clusters(self):  # Etiqueta las instancias de inicialización
        pass