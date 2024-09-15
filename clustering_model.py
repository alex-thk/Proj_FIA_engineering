import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes


class ClusteringModel:
    """
    Classe che effettua il clustering dei dati, il kmeans è completamente implementato ma non il kprototypes
    """

    def __init__(self, algorithm, n_clusters):
        """
        Inizializzazione della classe ClusteringModel
        :param algorithm: Algoritmo di clustering da utilizzare (kmeans o kprototypes)
        :param n_clusters: Numero di cluster da creare
        """
        self.algorithm = algorithm  # string that can be 'kmeans' or 'kprototypes'
        self.n_clusters = n_clusters  # number of clusters
        self.model = None  # KMeans or KPrototypes instance

        print(f"il numero di cluster che arrivano a clustering model durante l'istanziamento è: {self.n_clusters}")

        if self.algorithm == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
        elif self.algorithm == 'kprototypes':
            self.model = KPrototypes(n_clusters=n_clusters, init='Huang')  # instance of KPrototypes
        else:
            raise ValueError("Algoritmo non supportato: scegli tra 'kmeans' o 'kprototypes'")

    def fit(self, x, categorical=None):
        """
        Esegue il clustering dei dati.
        :param x: DataFrame che rappresenta i dati da clusterizzare.
        :param categorical: Indici delle colonne categoriche (solo per KPrototypes).
        :return: Dizionario con labels, centroids (e colonne categoriali per KPrototypes).
        """
        results = {}

        if self.algorithm == 'kmeans':
            if not isinstance(x, pd.DataFrame):
                raise ValueError("Per KMeans, X deve essere un DataFrame numerico")

            self.model.fit(x)
            results['labels'] = self.model.labels_
            results['centroids'] = self.model.cluster_centers_

        elif self.algorithm == 'kprototypes':
            if not isinstance(x, pd.DataFrame):
                raise ValueError("Per KPrototypes, X deve essere un DataFrame")
            if categorical is None:
                raise ValueError("Per KPrototypes, devi specificare gli indici delle colonne categoriche")

            self.model.fit(x, categorical=categorical)
            results['labels'] = self.model.labels_
            results['centroids'] = self.model.cluster_centers_
            results['categorical'] = categorical

        else:
            raise ValueError(f"Algoritmo {self.algorithm} non supportato")

        return results

"""
# Test del KMeans protetto da __name__
if __name__ == "__main__":
    # Creiamo dei dati di esempio per testare KMeans
    data = {
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [1.0, 1.5, 3.5, 4.5, 5.5]
    }
    df = pd.DataFrame(data)

    # Istanziare il modello KMeans
    kmeans_model = ClusteringModel(algorithm='kmeans', n_clusters=2)

    # Eseguire il fit sui dati e stampare i cluster risultanti
    labels = kmeans_model.fit(df)
    print("Cluster assegnati:", labels)
"""