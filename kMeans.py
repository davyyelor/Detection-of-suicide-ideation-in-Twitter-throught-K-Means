import random
from sklearn.cluster import k_means
import numpy as np
from scipy.spatial import distance as sp_distance
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

from clasificarNuevaInstancia import *
class KMeans_Clustering():

    def __init__(self, n_cluster=None, initialisation_method='random', iter_max=100, p_value=1, eType='standard'):
        self.n_clusters = n_cluster
        self.method = initialisation_method
        self.iter_max = iter_max
        self.p_value = p_value
        self.type = eType
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

    '''
    def gpu_minkowski_distance(self, vec1, vec2, p_value):
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        mod = SourceModule("""
            __global__ void minkowski_distance(float *a, float *b, float *result, int n, float p) {
                extern __shared__ float sdata[];

                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                float sum = 0;

                // Compute the local sum for this thread
                if (idx < n) {
                    float diff = fabs(a[idx] - b[idx]);
                    sum += powf(diff, p);
                }

                // Load local sum into shared memory and synchronize
                sdata[threadIdx.x] = sum;
                __syncthreads();

                // Do the reduction in shared memory
                for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s) {
                        sdata[threadIdx.x] += sdata[threadIdx.x + s];
                    }
                    __syncthreads();
                }

                // Write the block's result to global memory
                if (threadIdx.x == 0) {
                    atomicAdd(result, sdata[0]);
                }
            }
            """)


        func = mod.get_function("minkowski_distance")
        block_size = (512, 1, 1)
        grid_size = ((len(vec1) + block_size[0] - 1) // block_size[0], 1, 1)

        # Transfer data to GPU
        vec1_gpu = cuda.mem_alloc(vec1.nbytes)
        vec2_gpu = cuda.mem_alloc(vec2.nbytes)
        result_gpu = cuda.mem_alloc(np.float32(0).nbytes)
        cuda.memcpy_htod(vec1_gpu, vec1)
        cuda.memcpy_htod(vec2_gpu, vec2)

        # Launch kernel
        func(vec1_gpu, vec2_gpu, result_gpu, np.int32(len(vec1)), np.float32(p_value),
             block=block_size, grid=grid_size, shared=np.dtype('float32').itemsize * block_size[0])

        # Transfer result back to CPU
        result = np.empty(1, dtype=np.float32)
        cuda.memcpy_dtoh(result, result_gpu)
        return np.power(result[0], 1 / p_value)
    '''

    def ajustar(self, instances):
        self.instancias = instances
        if self.method == 'random':
            print("Algoritmo inicializado con clusters aleatorios")
            self.ajustar_random(instances)
        elif self.method == '2k':
            print("Algoritmo inicializando 2k clusters")
            self.ajustar_2k(instances)
        elif self.method == 'DivisionEspacio':
            print("Algoritmo inicializado por división del Espacio")
            self.ajustarPorDivisionDelEspacio(instances, k=2)


    def ajustar_random(self, instances):  # Itera el algoritmo de KMeans, calculando los centros (La matriz)
        # Instances:
        # [[ valores_palabra ],
        #  [ valores_palabra ]]
        N = instances.shape[0]
        instances = np.array(instances, dtype=np.float32)
        if self.type == 'CUDA':
            print("Calculando mediante GPU con CUDA")
        else:
            print("Calculando mediante CPU")

        # Matriz de centroides, filas son el índice del centroide y columnas los valores para esa componente en cada centroide
        self.centroides = random.sample(instances.tolist(),
                                        self.n_clusters)  # De la lista de instancias, cogemos N para inicializar los clusters
        self.centroides = np.array(self.centroides, dtype=np.float32)

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
                    if self.type == 'CUDA':
                        distancia = self.gpu_minkowski_distance(instances[instance_idx], self.centroides[centroid_idx],
                                                                self.p_value)
                    else:
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
                lista_instancias = [instances[i] for i in lista_idx_instancias]
                nuevo_centroide = np.mean(lista_instancias, axis=0)
                self.centroides[numero_cluster] = nuevo_centroide

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

    def ajustar_2k(self, instances):
        from scipy.spatial.distance import cdist
        clusters = self.n_clusters
        self.n_clusters *= 2

        # Generamos 2k clusters primero
        self.ajustar_random(instances)

        # Seleccionamos los centroides que están más lejos entre si mismos
        selected_points = np.array([self.centroides[np.random.randint(len(self.centroides))]])
        remaining_points = np.delete(self.centroides, 0, axis=0)

        while len(selected_points) < clusters:
            distances = cdist(selected_points, remaining_points, metric='minkowski')
            furthest_point_idx = np.argmax(np.min(distances, axis=0))
            selected_points = np.append(selected_points, [remaining_points[furthest_point_idx]], axis=0)
            remaining_points = np.delete(remaining_points, furthest_point_idx, axis=0)

        # Una vez tenemos los centroides con mayor distancia entre ellos los usamos como centroides
        # iniciales para calcular nuestro K-Means
        self.centroides = selected_points

        N = instances.shape[0]
        instances = np.array(instances, dtype=np.float32)

        centroides_prev = None

        for iter in range(self.iter_max):
            ###################################################
            ### REASIGNACIÓN DE INSTANCIAS CON SU CENTROIDE ###
            ###################################################
            centroides_asignados = {}
            centroides_asignados = {i: [] for i in range(clusters)}

            for instance_idx in range(N):
                distanciaMin = float('inf')
                for centroid_idx in range(len(self.centroides)):
                    if self.type == 'CUDA':
                        distancia = self.gpu_minkowski_distance(instances[instance_idx], self.centroides[centroid_idx],
                                                                self.p_value)
                    else:
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

    def dibujar_dendrograma(self):
        linkage_matrix = linkage(data, method='ward')
        dendrogram(linkage_matrix, orientation="top")
        plt.title("Dendrograma de Clusters")
        plt.xlabel("Tweets")
        plt.ylabel("Distancia")
        plt.show()


    def calcular_clusters(self):  # Etiqueta las instancias de inicialización
        pass

    def guardar_modelo(self, filename):
        # Crea un diccionario con los atributos del modelo que deseas guardar
        model_data = {
            "n_clusters": self.n_clusters,
            "centroides": self.centroides
            # Puedes agregar más atributos si es necesario
        }
        print("Modelo guardado con: ", self.n_clusters, "y con los centroides: ", self.centroides)
        # Utiliza joblib para guardar el modelo en un archivo .sav
        joblib.dump(model_data, filename)

    def cargar_modelo(self, model_path):
        # Carga el modelo guardado utilizando joblib
        model_data = joblib.load(model_path)

        self.n_clusters = model_data["n_clusters"]
        self.centroides = model_data["centroides"]



    def ajustarPorDivisionDelEspacio(instances, k):
        # Encuentra los límites del espacio de características
        #k son el número de subregiones
        min_values = np.min(instances, axis=0)
        max_values = np.max(instances, axis=0)

        # Divide el espacio en k subregiones
        subregions = []
        for i in range(k):
            subregion = []
            for j in range(len(min_values)):
                # Calcula el centro de la subregión en cada dimensión
                subregion_center = min_values[j] + (i + 0.5) * (max_values[j] - min_values[j]) / k
                subregion.append(subregion_center)
            subregions.append(subregion)

        return np.array(subregions)

    def calcular_inercia(self):
        inercia = 0
        for i in range(self.n_clusters):
            instances_in_cluster = self.instancias[np.array(self.labels) == i]
            centroid = self.centroides[i]
            distances = np.sum((instances_in_cluster - centroid) ** 2)
            inercia += distances
        return inercia



if __name__ == '__main__':
    #############################################Test
    nuevos_mensajes = ["This is a test message", "ANother message to send to one of my little friends, I hope I´ll be seeing him soon", "i shouldnt want to kill myself i have so many advantages in life all of my immediate family is alive and still involved in my life i have several friends who i frequently talk to i make more than enough money to live on easily but every night i go home and drink hoping i will forget about how much i just want it all to end every single moment that i am sober i spend thinking that people might actually see me for who i really am a lazy waste of space who lucked into success and wheni amdrunk i worry that people will find out thati am trying to slowly kill myself with alcohol fuck i always thought if i could just be successful i would be happy well i am not"]
    opcion = 'd2v'

    print("Se van a preprocesar y vectorizar las nuevas instancias con: ", opcion)

    nuevoArray = preproceso(nuevos_mensajes, opcion)

    print("vectorizado completado con ", opcion)

    modelo = KMeans_Clustering(n_cluster=2)
    modelo.cargar_modelo('modelo.sav')

    modelo.labels = modelo.ajustar(nuevoArray)

    print(modelo.labels)

    ########################################## Calcular Punto de codo
    inercias = []
    for k in range(1, 4):
        modelo = KMeans_Clustering(n_cluster=k, iter_max=20, p_value=2)
        modelo.ajustar(nuevoArray)
        inercia = modelo.calcular_inercia()
        inercias.append(inercia[0])
        print("La inercia para", k, "clusters es", inercia[0])

    # Plot the inercia
    plt.plot(range(1, 4), inercias, marker='o')
    plt.title('Método del Codo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.show()


