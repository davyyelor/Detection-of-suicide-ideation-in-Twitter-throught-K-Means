import random
from sklearn.cluster import k_means
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.spatial import distance as sp_distance

# Define the CUDA kernel
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

class KMeans_Clustering_CUDA():

    def __init__(self, n_cluster=None, initialisation_method=None, iter_max=100, p_value=1):
        self.n_clusters = n_cluster
        self.method = initialisation_method
        self.iter_max = iter_max
        self.p_value = p_value
        # Devolver una matriz que sea un vector de vectores (n_clusters vectores que contienen las coordenadas de los centros)

    def gpu_minkowski_distance(self, vec1, vec2, p_value):
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
        return np.power(result[0], 1/p_value)

    def ajustar(self, instances):  # Itera el algoritmo de KMeans, calculando los centros (La matriz)
        # Instances:
        # [[ valores_palabra ],
        #  [ valores_palabra ]]
        '''N = len(instances[0])'''
        N = instances.shape[0]
        instances = np.array(instances, dtype=np.float32)
        print("Numero de instancias", N)
        

        # Matriz de centroides, filas son el índice del centroide y columnas los valores para esa componente en cada centroide
        if self.method == None:
            self.centroides = random.sample( instances.tolist(), self.n_clusters ) #De la lista de instancias, cogemos N para inicializar los clusters
            self.centroides = np.array(self.centroides, dtype=np.float32)

        for iter in range(self.iter_max):
            ###################################################
            ### REASIGNACIÓN DE INSTANCIAS CON SU CENTROIDE ###
            ###################################################
            centroides_asignados = {}
            centroides_asignados = {i: [] for i in range(self.n_clusters)}

            for instance_idx in range(N):
                distanciaMin = float('inf')
                for centroid_idx in range(len(self.centroides)):
                    distancia = self.gpu_minkowski_distance(instances[instance_idx], self.centroides[centroid_idx], self.p_value)
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