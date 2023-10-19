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
        
        centroides_asignados = {}

        #Matriz de centroides, filas son el índice del centroide y columnas los valores para esa componente en cada centroide
        if self.method == None:
            self.centroides = np.random.rand(self.n_clusters, instances[0].size())

        for _ in self.iter_max:
            ###################################################
            ### REASIGNACIÓN DE INSTANCIAS CON SU CENTROIDE ###
            ###################################################

            for instance_idx in range(instances.size()):
                distanciaMin = float('inf')
                for centroid_idx in range(self.centroides[0].size()):
                    distancia = self.minkowski_distance(instances[instance_idx], self.centroides[centroid_idx])
                    if distancia < distanciaMin:
                        distanciaMin = distancia
                        centroid = centroid_idx
                centroides_asignados[centroid].append(instance_idx)


            ###################################################
            ###         CALCULAR NUEVOS CENTROIDES          ###
            ###################################################
            
            for numero_cluster in centroides_asignados.keys():
                lista_idx_instancias = centroides_asignados[numero_cluster]
                lista_instancias = np.array(instances[i] for i in lista_idx_instancias)
                nuevo_cluster = np.mean(lista_instancias, axis=0)
                self.centroides[numero_cluster] = nuevo_cluster




    def calcular_clusters(self): #Etiqueta las instancias de inicialización
        pass