from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np


class ObjectiveFunction:
    """
    Classe che calcola la funzione obiettivo del clustering
    """
    def __init__(self, data, labels, cluster_labels, num_clusters):
        """
        Inizializzazione della classe ObjectiveFunction
        :param data: dataset completo in array numpy
        :param labels: labels classe incremento in array numpy 1D
        :param cluster_labels: labels cluster in array numpy 1D
        """
        self.data = data
        self.labels = labels
        self.cluster_labels = cluster_labels
        self.num_clusters = num_clusters

    def purity(self):
        """
        Metodo che calcola la purità totale del dataset considerando ogni singolo cluster alla volta
        :return: Purity score normalizzato tra 0 e 1
        """
        total_elements = len(self.labels)  # calculate the total number of elements
        purity_sum = 0

        for cluster in set(self.cluster_labels):  # loop su ogni cluster
            cluster_indices = np.where(self.cluster_labels == cluster)[0]  # trovo gli indici dei punti in questo cluster
            cluster_labels_ = self.labels[cluster_indices]  # labels di classe dei punti in questo cluster

            most_common_class = max(np.bincount(cluster_labels_))  # trovo la classe piu comune

            purity_sum += most_common_class  # aggiungo la classe più comune alla somma

        return purity_sum / total_elements  # restituisco la purità

    def silhouette(self):
        """
        Metodo che calcola l'indice di silhouette per l'intero insieme di dati
        :return: Silhouette score normalizzato tra 0 e 1
        """
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
            # Calcola silhouette per ogni singolo punto
            silhouette_values = silhouette_samples(self.data, self.cluster_labels)
            print(f"Silhouette values (before normalization): {silhouette_values}")

            # Normalizza ogni valore di silhouette tra 0 e 1
            silhouette_values_normalized = (silhouette_values + 1) / 2
            print(f"Silhouette values (after normalization): {silhouette_values_normalized}")
            return np.mean(silhouette_values_normalized)  # Calcola la media dei valori di silhouette normalizzati
        else:
            return 0  # Se c'è solo un cluster, silhouette non è definito

    def penalty(self):
        """
        Metodo che calcola la penalità in base al numero di cluster
        :return: Penalità
        """
        return 0.05 * self.num_clusters

    def compute_final_score(self):
        """
        Metodo che calcola il punteggio finale normalizzato
        :return: Punteggio finale normalizzato tra 0 e 1
        """
        # Calcola le metriche normalizzate
        print("calcolo purita..)")
        purity_score = self.purity()

        print("calcolo silhouette..")
        silhouette_score_ = self.silhouette()

        # Normalizzazione delle metriche tra 0 e 1
        print("normalizzazione dei risultati..")
        normalized_purity = min(max(purity_score, 0), 1)
        print("purity normalizzata: ", normalized_purity)
        print("silhouette normalizzata: ", silhouette_score_)
        penalty = self.penalty()
        print("numero di cluster: ", self.num_clusters)
        print("penalità: ", penalty)

        # Calcola la media delle metriche normalizzate e sottrai la penalità
        print("calcolo del punteggio finale..")
        final_score_ = (normalized_purity + silhouette_score_) / 2 - penalty
        print("punteggio finale: ", final_score_)

        return final_score_


# Test del codice commentato

if __name__ == "__main__":
    # Creiamo un piccolo dataset sintetico con make_blobs
    # Generate 2D points from 3 clusters
    data, labels = make_blobs(n_samples=30, centers=3, random_state=42)

    # Utilizziamo KMeans per creare dei cluster sui dati
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(data)  # Etichette di cluster assegnate dal modello KMeans

    # Creiamo l'oggetto ObjectiveFunction con i dati sintetici
    obj_function = ObjectiveFunction(data, labels, cluster_labels, 3)

    # Calcoliamo la purezza
    purity = obj_function.purity()
    print(f"Purity: {purity}")

    # Calcoliamo il Silhouette score
    silhouette = obj_function.silhouette()
    print(f"Silhouette Score: {silhouette}")

    # Calcoliamo il punteggio finale
    final_score = obj_function.compute_final_score()
