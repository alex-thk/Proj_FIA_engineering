from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import numpy as np


class ObjectiveFunction:
    def __init__(self, data, labels, cluster_labels):
        """
        Initialization of the class ObjectiveFunction
        :param data: complete dataset in numpy array
        :param labels: right labels of class incremento in numpy array 1D
        :param cluster_labels: numpy array 1D with cluster labels
        """
        self.data = data
        self.labels = labels
        self.cluster_labels = cluster_labels
        self.num_clusters = len(set(cluster_labels))

    def purity(self):
        total_elements = len(self.labels)  # calculate the total number of elements
        purity_sum = 0

        for cluster in set(self.cluster_labels):  # loop over each cluster
            cluster_indices = np.where(self.cluster_labels == cluster)[0]  # index of points in this cluster, [0] to get the array of indices of samples in this cluster
            cluster_labels_ = self.labels[cluster_indices]  # true labels of points in this cluster

            most_common_class = max(np.bincount(cluster_labels_))  # find the most common class

            purity_sum += most_common_class  # add the most common class to the sum

        return purity_sum / total_elements  # return the purity

    def silhouette(self):
        print("chiamata metodo silhouette..")
        # Calcola l'indice di silhouette per l'intero insieme di dati
        if len(set(self.cluster_labels)) > 1:
            print("calcolo silhouette entrata nell'if..")
            print("data: ", self.data)
            print("cluster_labels: ", self.cluster_labels)
            # Verifica la lunghezza del dataset
            print(f"Lunghezza del dataset: {len(self.data)}")
            # Verifica la lunghezza del vettore delle etichette
            print(f"Lunghezza delle etichette dei cluster: {len(self.cluster_labels)}")
            return silhouette_score(self.data, self.cluster_labels)
        else:
            return 0  # Se c'è solo un cluster, silhouette non è definito

    def penalty(self):
        # Penalità proporzionale al numero di cluster
        return 0.05 * self.num_clusters

    def compute_final_score(self):
        # Calcola le metriche normalizzate
        print("calcolo purita..)")
        purity_score = self.purity()

        print("calcolo silhouette..")
        silhouette_score_ = self.silhouette()

        # Normalizzazione delle metriche tra 0 e 1
        print("normalizzazione dei risultati..")
        normalized_purity = min(max(purity_score, 0), 1)
        normalized_silhouette = min(max(silhouette_score_, 0), 1)

        # Calcola la media delle metriche normalizzate e sottrai la penalità
        print("calcolo del punteggio finale..")
        final_score_ = (normalized_purity + normalized_silhouette) / 2 - self.penalty()

        return final_score_


"""
if __name__ == "__main__":
    # Creiamo un piccolo dataset sintetico con make_blobs
    # Generate 2D points from 3 clusters
    data, labels = make_blobs(n_samples=30, centers=3, random_state=42)

    # Utilizziamo KMeans per creare dei cluster sui dati
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(data)  # Etichette di cluster assegnate dal modello KMeans

    # Creiamo l'oggetto ObjectiveFunction con i dati sintetici
    obj_function = ObjectiveFunction(data, labels, cluster_labels)

    # Calcoliamo la purezza
    purity = obj_function.purity()
    print(f"Purity: {purity}")

    # Calcoliamo il Silhouette score
    silhouette = obj_function.silhouette()
    print(f"Silhouette Score: {silhouette}")

    # Calcoliamo il punteggio finale
    final_score = obj_function.compute_final_score()
    print(f"Final Score: {final_score}")
"""
