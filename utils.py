import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def renombrar_clusters(predicted_labels, train):
# Reasignar etiquetas de clúster basadas en las etiquetas verdaderas más comunes dentro de cada clúster
    cluster_to_class = {}
    for cluster_id in range(2):
        cluster_indices = np.where(predicted_labels == cluster_id)[0]
        print(cluster_indices)
        true_labels = train['label'].iloc[cluster_indices].values
        most_common_label = np.bincount(true_labels).argmax()
        cluster_to_class[cluster_id] = most_common_label

    # Mapear las etiquetas de clúster a etiquetas de clase
    reassigned_labels = np.vectorize(cluster_to_class.get)(predicted_labels)

    # Calculate the confusion matrix with the reassigned labels
    cm = confusion_matrix(predicted_labels, reassigned_labels)
    total_correct = np.trace(cm)  # Suma de valores en la diagonal principal
    total_samples = np.sum(cm)    # Suma de todos los valores en la matriz de confusión
    # Create a heatmap
    plt.xlabel('Clase Real')
    plt.ylabel('Cluster')
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de confusión con las etiquetas ajustadas")
    plt.show()

    # Calculate the confusion matrix with the reassigned labels
    cm = confusion_matrix(predicted_labels, reassigned_labels)
    total_correct = np.trace(cm)  # Suma de valores en la diagonal principal
    total_samples = np.sum(cm)    # Suma de todos los valores en la matriz de confusión

    # Calcular el número total de clasificaciones incorrectas
    total_incorrect = total_samples - total_correct

    # Calcular la tasa de error
    error_rate = total_incorrect / total_samples

    print("Tasa de Error:", error_rate)

def matriz_de_confusion(predicted_labels, train):
    # Visualizar la matriz de confusión con el número de instancias
    # El atributo generado por K-means es int, hay que pasarlos a string
    to_string = lambda x : str(x)
    # Obtener matriz de confusión Class to clustering eval
    cm = confusion_matrix(np.vectorize(to_string)(predicted_labels), np.vectorize(to_string)(train['label']))
    # Mapa de calor a partir de la matriz de confusion sin números
    ax = sns.heatmap(cm, annot=False, cmap="Blues")
    plt.xlabel('Clase Real')
    plt.ylabel('Cluster')
    plt.title('Matriz de Confusión sin relabelización')
    plt.show()