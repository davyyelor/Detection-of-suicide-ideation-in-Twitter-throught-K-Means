class KMeans_Clustering():

    def __init__(self, n_cluster=None, initialisation_method=None):
        pass

    def minkowski_distance(point1, point2, p):
        return np.power(np.sum(np.power(np.abs(point1 - point2), p), axis=0), 1 / p)

    # Función para asignar cada punto al centroide más cercano
    def assign_to_clusters(data, centroids, p):
        num_clusters = centroids.shape[0]
        num_points = data.shape[0]
        labels = np.zeros(num_points, dtype=int)
        distances = np.zeros((num_points, num_clusters))

        for i in range(num_clusters):
            distances[:, i] = minkowski_distance(data.T, centroids[i].T, p)

        labels = np.argmin(distances, axis=1)
        return labels

    # Función principal de K-Means con distancia de Minkowski
    def kmeans_minkowski(data, num_clusters, max_iters=100, p=2):
        num_points, num_features = data.shape

        # Inicialización de los centroides aleatoriamente
        random_indices = np.random.choice(num_points, num_clusters, replace=False)
        centroids = data[random_indices]

        for iter in range(max_iters):
            # Paso 1: Asignar cada punto al centroide más cercano
            labels = assign_to_clusters(data, centroids, p)

            # Paso 2: Actualizar los centroides
            new_centroids = update_centroids(data, labels, num_clusters)

            # Comprobar si los centroides han convergido
            if np.array_equal(centroids, new_centroids):
                break

            centroids = new_centroids

        return labels, centroids