class KMeans_Clustering():

    def __init__(self, n_cluster=None, initialisation_method=None):
        pass

    def minkowski_distance(self, vector1, vector2, p_value):

        def p_exp(x, y, p):
            return abs(x-y)**p

        distance = 0
        for idx in range(len(vector1)):
            distance += p_exp(vector1[idx], vector2[idx], p_value)
        

        
        return distance**(1/p_value)
        
eoe = KMeans_Clustering()
x = [0, 5, 4, 5]
y = [7, 6, 64, -1]

distancia = eoe.minkowski_distance(x, y, -.5)
print(distancia)