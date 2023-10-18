from sklearn.cluster import k_means
import numpy as np

class KMeans_Clustering():

    def __init__(self, n_cluster=None, initialisation_method=None, iter_max=100, p_value=1):
        self.n_clusters = n_cluster
        self.method = initialisation_method
        self.iter_max = iter_max
        self.p_value = p_value
        #Devolver una matriz que sea un vector de vectores (n_clusters vectores que contienen las coordenadas de los centros)

    def minkowski_distance(self, vector1, vector2, p_value):

        def p_exp(x, y, p):
            return abs(x-y)**p

        distance = 0
        for idx in range(len(vector1)):
            distance += p_exp(vector1[idx], vector2[idx], p_value)
        

        
        return distance**(1/p_value)


    def ajustar(self, instances): #Itera el algoritmo de KMeans, calculando los centros (La matriz)
        #Instances: 
        # [[ valores_palabra ],
        #  [ valores_palabra ]]
        
        centroids_asignados = {}

        if self.method == None:
            self.centroides = np.random.rand(self.n_clusters, instances[1].size())
        
        ###################################################
        ### REASIGNACIÓN DE INSTANCIAS CON SU CENTROIDE ###
        ###################################################

        for instance_idx in range(instances[0].size()):
            distanciaMin = float('inf')
            for centroid_idx in range(self.centroids[0]):
                distancia = self.minkowski_distance(instances[instance_idx], self.centroids[centroid_idx])
                if distancia < distanciaMin:
                    distanciaMin = distancia
                    centroid = centroid_idx
            centroids_asignados[centroid].append(instance_idx)


        ###################################################
        ###         CALCULAR NUEVOS CENTROIDES          ###
        ###################################################

        for vectores in centroids_asignados.values():
            np.array



    def calcular_clusters(self): #Etiqueta las instancias de inicialización
        pass