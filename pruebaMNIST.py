import pandas as pd
import sys, getopt, random
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score
from Kmeans import KMeans_Clustering

n_clusters = 10
reduction_method = None
dimensions = 2

arguments = sys.argv[1:]
options = "c:r:s:"
long_options = ["Clusters", "Reduction", "Silhouette"]

try:
    arguments, values = getopt.getopt(arguments, options, long_options)
    for arg, val in arguments:
        if arg in ("-c", "--Clusters"):
            n_clusters = int(val)
        if arg in ("-r", "--Reduction"):
            reduction_method = val
except getopt.error as err:
    print(str(err))


#Dataset handling
X_train = pd.read_csv("data/mnist_train.csv")

y_train = X_train['label'].copy()
labels = X_train['label'].copy()
X_train.drop('label', axis=1, inplace=True)

#Reduction method pre clustering
if reduction_method != None:
    if reduction_method == "TSNE" or reduction_method == "tsne":
        print("Reducción mediante tSNE previo al clustering")
        from sklearn.manifold import TSNE
        tsne = TSNE().fit_transform(X_train)

        kmeans = KMeans(n_clusters)
        kmeans.fit(tsne)
        kmeansLabels = kmeans.predict(tsne)

        samples = 300
        sc = plt.scatter(tsne[:samples,0],
                    tsne[:samples,1], 
                    cmap=plt.cm.get_cmap('nipy_spectral', 10),
                    c=kmeansLabels[:samples])

        plt.colorbar()
        # Etiqueta numérica: clase 
        for i in range(samples):
            plt.text(tsne[i,0], tsne[i,1], y_train[i])

        plt.show()
        X_set = tsne

    elif reduction_method == "PCA" or reduction_method == "pca":
        print("Reducción mediante PCA previo al clustering")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dimensions)
        pca.fit(X_train)
        X_train_PCA = pca.transform(X_train)

        kmeans = KMeans(n_clusters, n_init=n_clusters)
        kmeans.fit(X_train_PCA)
        kmeansLabels = kmeans.predict(X_train_PCA)

        samples = 300
        sc = plt.scatter(X_train_PCA[:samples,0],
                    X_train_PCA[:samples,1], 
                    cmap=plt.cm.get_cmap('nipy_spectral', 10),
                    c=kmeansLabels[:samples])

        plt.colorbar()
        # Etiqueta numérica: clase 
        for i in range(samples):
            plt.text(X_train_PCA[i,0], X_train_PCA[i,1], y_train[i])

        plt.show()
        X_set = X_train_PCA
else:
    print("Number of clusters:", n_clusters)
    kmeans = KMeans_Clustering(n_cluster=10)
    X_train = list(X_train)
    print(len(X_train[0]))
    kmeans.ajustar(X_train)
    kmeansLabels = kmeans.labels
    print(kmeansLabels)
    X_set = X_train

    pca = PCA(n_components=dimensions)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)

    samples = 300
    sc = plt.scatter(X_train_PCA[:samples,0],
                X_train_PCA[:samples,1], 
                cmap=plt.cm.get_cmap('nipy_spectral', 10),
                c=kmeansLabels[:samples])

    plt.colorbar()
    # Etiqueta numérica: clase 
    for i in range(samples):
        plt.text(X_train_PCA[i,0], X_train_PCA[i,1], y_train[i])

    plt.show()

    # Si se ejecuta con el parámetro de reducción de dimensiones, se mostrará
    # un scatter plot de las dimensiones reducidas y se pasará a calcular KMeans 
    # tras la reducción del número de dimensiones


### REASIGNACIÓN DE CLUSTERS ###

cluster_to_class = {}
for cluster_id in range(n_clusters):
    cluster_indices = np.where(kmeansLabels == cluster_id)[0]
    true_labels = y_train.iloc[cluster_indices].values
    most_common_label = np.bincount(true_labels).argmax()
    cluster_to_class[cluster_id] = most_common_label

reassigned_labels: np.ndarray = np.vectorize(cluster_to_class.get)(kmeansLabels)


### CALCULAR LA PUNTUACIÓN DE SILUETA ###


silhouette_avg = silhouette_score(X_set, reassigned_labels)
print("Puntuación de Silueta:", silhouette_avg)

### MATRIZ DE CONFUSION ###

cm = confusion_matrix(y_train, reassigned_labels)
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()

 

### 4 IMAGENES ALEATORIAS ###

df = pd.DataFrame({'Cluster': kmeansLabels})
df['Index'] = df.index
selected_indices = []
for cluster_id in range(n_clusters):
    cluster_indices = df[df['Cluster'] == cluster_id]['Index'].tolist()
    selected_indices.extend(random.sample(cluster_indices, 4))

fig, axes = plt.subplots(n_clusters, 4, figsize=(12, 12))
for i, idx in enumerate(selected_indices):
    row = i // 4
    col = i % 4
    image = X_train.iloc[idx].values.reshape(28, 28)  # Assuming MNIST images are 28x28 pixels
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].axis('off')
    axes[row, col].set_title(f'Cluster {kmeansLabels[idx]}')

plt.tight_layout()
plt.show()