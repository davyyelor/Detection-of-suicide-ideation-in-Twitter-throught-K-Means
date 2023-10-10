


class KMeans_Clustering():

    def __init__(self, n_cluster=None, initialisation_method=None):
        pass

    def minkowski_distance(self, vector1, vector2, p_value):

        distance = 0
        for idx in len(vector1):
            distance += p_root(vector1[idx], vector2[idx], p_value)
        

        def p_root(x, y, p):
            return abs(x-y)**p
        return distance**(1/p_value)
        
eoe = KMeans_Clustering()
x = [0, 3, 4, 5]
y = [7, 6, 3, -1]

distancia = eoe.minkowski_distance(x, y, 3)
print(distancia)