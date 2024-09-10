import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_blobs
from clustering_model import ClusteringModel
from objective_function import ObjectiveFunction


class HyperparameterTuner:
    def __init__(self, algorithm, param_grid, data, labels):
        """
        Inizializzatore per la classe HyperparameterTuner
        :param algorithm: Nome dell'algoritmo, ad esempio 'kmeans'
        :param param_grid: Dizionario con gli iperparametri da ottimizzare
        :param data: Dataset su cui effettuare il clustering (in formato DataFrame)
        :param labels: Etichette corrette per valutare la funzione obiettivo (in formato numpy array)
        """
        self.algorithm = algorithm
        self.param_grid = param_grid
        self.data = data
        self.labels = labels
        self.grid = list(ParameterGrid(param_grid))  # Crea tutte le combinazioni di iperparametri

    def perform_grid_search(self):
        """
        Esegue il Grid Search sugli iperparametri per trovare la combinazione ottimale.
        Salva labels, centroidi e score per ogni combinazione di parametri testati.
        :return: La migliore combinazione di iperparametri, il relativo punteggio e i risultati del miglior clustering.
        """
        best_score = -np.inf
        best_params = None
        best_centroids = None
        best_labels = None
        all_results = []

        # Itera su tutte le combinazioni nella griglia
        for params in self.grid:
            print(f"Testing parameters: {params}")

            print("istanzio classe clustering model..")

            # Istanzia il modello di clustering con i parametri attuali
            clustering_model = ClusteringModel(algorithm=self.algorithm, n_clusters=params['n_clusters'])

            print("faccio fit..")

            results = clustering_model.fit(self.data)  # fit restituisce labels e centroids

            print("istanzio la classe Objective funtion..")

            # Calcola la funzione obiettivo
            objective_function = ObjectiveFunction(data=self.data, labels=self.labels, cluster_labels=results['labels'])

            print("calcolo il punteggio finale..")

            score = objective_function.compute_final_score()

            print("salvo i risultati..")

            # Salva i risultati per questa combinazione di parametri
            all_results.append({
                'params': params,
                'score': score,
                'labels': results['labels'],
                'centroids': results['centroids']
            })

            # Se il punteggio attuale è il migliore, aggiorna
            if score > best_score:
                best_score = score
                best_params = params
                best_labels = results['labels']
                best_centroids = results['centroids']

            print(f"Score for parameters {params}: {score}")

        # Restituisce i migliori parametri, il punteggio, il miglior clustering e tutti i risultati
        print(f"parametri MIGLIORI che restituisce grid search :{best_params}, {best_score}, {best_labels}, {best_centroids}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_labels': best_labels,
            'best_centroids': best_centroids
        }


'''
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
    best_params, best_score, best_clustering = tuner.perform_grid_search()

    print(best_params), print(best_score), print(best_clustering)
'''
