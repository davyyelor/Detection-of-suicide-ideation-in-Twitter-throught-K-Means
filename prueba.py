import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from kMeans import KMeans_Clustering


x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
instancias = []

for i in range(len(x)):
    instancias.append([x[i], y[i]])

datos = list(zip(x,y))
tiempo1 = time.time_ns()
k = KMeans(n_clusters=2, n_init=2)
k.fit(instancias)
tiempo = time.time_ns() - tiempo1
print("Tiempo en ns sklearn:\t\t\t\t", tiempo)

tiempo1 = time.time_ns()
k2 = KMeans_Clustering(n_cluster=2, iter_max=1000, p_value=1)
k2.ajustar(instancias)
tiempo = time.time_ns() - tiempo1
print("Tiempo en ns kmeans pocho:\t\t\t", tiempo)

print(k.labels_)
print(k2.labels)


plt.scatter(x, y, c=k.labels_)
plt.show()

plt.scatter(x, y, c=k2.labels)
plt.show()