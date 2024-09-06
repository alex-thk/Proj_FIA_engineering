import numpy as np
import pandas as pd
from collections import Counter
from hyperparameter_tuner import HyperparameterTuner
from clustering_model import ClusteringModel
from sklearn.datasets import make_blobs


class ClusteringResultAnalyzer:
    def __init__(self, labels, data, centroids=None, y_true=None):
        """
        Inizializza l'analizzatore di clustering.
        :param labels: Etichette dei cluster assegnati dal modello (obbligatorio)
        :param data: Dataset originale (obbligatorio)
        :param centroids: Centroidi dei cluster (opzionale, solo se il modello li fornisce)
        :param y_true: Etichette di classe reali per calcolare la purità (opzionale)
        """
        self.labels = labels
        self.centroids = centroids
        self.data = data
        self.y_true = y_true  # Etichette vere, opzionale

    def cluster_summary(self):
        """
        Crea una tabella riassuntiva che mostra la numerosità e, se disponibile, la purità dei cluster.
        """
        unique_labels, counts = np.unique(self.labels, return_counts=True)

        # Tabella base con numerosità per cluster
        summary_table = pd.DataFrame({
            'Cluster': unique_labels,
            'Numerosità': counts
        })

        if self.y_true is not None:
            # Calcolo della purità per ciascun cluster
            purity_list = []
            for label in unique_labels:
                true_labels_in_cluster = self.y_true[self.labels == label]
                most_common_class_count = Counter(true_labels_in_cluster).most_common(1)[0][1]
                purity = most_common_class_count / len(true_labels_in_cluster)
                purity_list.append(purity)

            # Aggiungi la purità alla tabella
            summary_table['Purità'] = purity_list

        return summary_table

    def display_summary(self):
        """
        Mostra la tabella riassuntiva dei cluster.
        """
        summary_table = self.cluster_summary()

        # Usa Pandas per visualizzare la tabella
        print(summary_table)

    def analyze_centroids(self):
        """
        Analizza i centroidi, mostrando la loro struttura (se presenti).
        """
        if self.centroids is not None:
            print("Centroidi dei cluster:")
            print(pd.DataFrame(self.centroids))
        else:
            print("Questo algoritmo non fornisce centroidi.")

# Test del tuner protetto da __name__
if __name__ == "__main__":

    # Generiamo un dataset artificiale
    data, labels = make_blobs(n_samples=200, centers=4, n_features=2, random_state=42)
    data_df = pd.DataFrame(data, columns=['feature1', 'feature2'])

    # Definizione della griglia di iperparametri per il tuning
    param_grid = {
        'n_clusters': [2, 3, 4, 5],  # numero di cluster che vogliamo testare
    }

    # Istanzia il tuner per KMeans
    tuner = HyperparameterTuner(algorithm='kmeans', param_grid=param_grid, data=data_df, labels=labels)

    # Eseguiamo il Grid Search per trovare la combinazione migliore di iperparametri
    best_params, best_score = tuner.perform_grid_search()

    clustering_model = ClusteringModel(algorithm='kmeans', n_clusters=best_params['n_clusters'])

    results = clustering_model.fit_opt(data_df)

    # Istanzia l'analizzatore dei risultati di clustering
    analyzer = ClusteringResultAnalyzer(
        labels=results['labels'],  # Estrai le etichette dei cluster
        data=results['data'],  # Estrai il dataset originale
        centroids=results['centroids'], # Estrai i centroidi (se presenti)
        y_true=labels  # Etichette vere per calcolare la purità
    )

    # Ora puoi chiamare i metodi dell'analyzer
    analyzer.display_summary()
    analyzer.analyze_centroids()