from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


class ClusteringAnalyzer:
    """
    Classe per l'analisi dei dati tramite clustering KMeans per trovare le feature più significative per il clustering.
    """

    def __init__(self, max_clusters=10):
        """
        Inizializza la classe con il numero massimo di cluster da considerare per l'Elbow Method.
        :param max_clusters: il numero massimo di cluster da considerare per l'Elbow Method.
        """
        self.max_clusters = max_clusters
        self.n_clusters = None

    def determine_optimal_clusters(self, data):
        """
        Determina il numero ottimale di cluster usando l'Elbow Method.
        :param data: i dati su cui eseguire il clustering.
        """
        distortions = []
        K = range(1, self.max_clusters + 1)

        for k in K:
            model = KMeans(n_clusters=k)
            model.fit(data)
            distortions.append(model.inertia_)

        return distortions

    def find_optimal_k_for_quarters(self, grouped_data):
        """
        Determina il numero ottimale di cluster per ogni quadrimestre usando la media delle distorsioni
        :param grouped_data: dizionario con i dati suddivisi per anno e quadrimestre.
        :return: dizionario con il numero ottimale di cluster per ogni quadrimestre.
        """
        quarter_distortions = {}

        # Calcola le distorsioni per ogni gruppo di dati
        for (year, quarter), data in grouped_data.items():
            distortions = self.determine_optimal_clusters(data)
            if quarter not in quarter_distortions:
                quarter_distortions[quarter] = []
            quarter_distortions[quarter].append(distortions)

        # Media delle distorsioni per ogni k, per ogni quadrimestre
        optimal_k_for_quarters = {}
        for quarter, distortions_list in quarter_distortions.items():
            mean_distortions = np.mean(distortions_list, axis=0)

            # Trova il gomito nel grafico delle distorsioni medie:
            # la logica è quella di trovare il punto in cui la distorsione media tra un k e il successivo è minore rispetto a quel k e al precedente
            elbow = next((i for i in range(1, len(mean_distortions) - 1) if
                          mean_distortions[i - 1] - mean_distortions[i] > mean_distortions[i] - mean_distortions[
                              i + 1]), 1)

            optimal_k_for_quarters[quarter] = elbow

            # Plot per l'Elbow Method con le distorsioni medie
            K = range(1, self.max_clusters + 1)
            plt.figure(figsize=(8, 4))
            plt.plot(K, mean_distortions, 'bx-')
            plt.xlabel('Numero di cluster')
            plt.ylabel('Distorsione media')
            plt.title(f'Elbow Method per il quadrimestre {quarter}')
            plt.show()

        return optimal_k_for_quarters

    def evaluate_feature_stability(self, grouped_data, optimal_k_for_quarters):
        """
        Valuta la stabilità delle feature tra anni diversi per lo stesso quadrimestre usando l'Adjusted Rand Index.
        :param grouped_data: dizionario con i dati suddivisi per anno e quadrimestre.
        :param optimal_k_for_quarters: dizionario con il numero ottimale di cluster per ogni quadrimestre.
        :return: dizionario con la stabilità delle feature per ogni quadrimestre.
        """
        feature_stability = {}

        # Per ogni quadrimestre, compara i cluster tra anni diversi
        for quarter in set(k[1] for k in grouped_data.keys()):
            years = sorted(set(k[0] for k in grouped_data.keys() if k[1] == quarter))

            if len(years) > 1:
                for combination in combinations(grouped_data[(years[0], quarter)].columns,
                                                1):  # Considera le feature singolarmente
                    feature_name = combination[0]

                    # Valuta la similarità dei cluster tra gli anni per quella feature
                    stability_scores = []
                    for i in range(1, len(years)):
                        year_prev = years[i - 1]
                        year_curr = years[i]

                        data_prev = grouped_data[(year_prev, quarter)][[feature_name]]
                        data_curr = grouped_data[(year_curr, quarter)][[feature_name]]

                        # Prendi il numero minimo di campioni tra i due anni
                        min_samples = min(len(data_prev), len(data_curr))

                        # Campiona gli stessi campioni da entrambi i set
                        data_prev = data_prev.sample(n=min_samples, random_state=42)
                        data_curr = data_curr.sample(n=min_samples, random_state=42)

                        # Recupera il numero ottimale di cluster per il quadrimestre
                        k_optimal = optimal_k_for_quarters[quarter]

                        # Esegui il clustering per ciascun anno e ottieni le etichette
                        model_prev = KMeans(n_clusters=k_optimal)
                        model_curr = KMeans(n_clusters=k_optimal)

                        model_prev.fit(data_prev)
                        model_curr.fit(data_curr)

                        # Calcola la similarità tra i cluster usando l'Adjusted Rand Index (ARI)
                        ari_score = adjusted_rand_score(model_prev.labels_, model_curr.labels_)
                        stability_scores.append(ari_score)

                    # Salva la media della stabilità per quella feature
                    if quarter not in feature_stability:
                        feature_stability[quarter] = {}
                    feature_stability[quarter][feature_name] = np.mean(stability_scores)

        return feature_stability

    def select_significant_features(self, feature_stability, threshold=0.5):
        """
        Seleziona le feature che hanno una stabilità sopra una certa soglia.
        :param feature_stability: dizionario con la stabilità delle feature.
        :param threshold: soglia per considerare una feature significativa.
        :return: dizionario con le feature significative per ogni quadrimestre.
        """
        significant_features = {}

        for quarter, stability_scores in feature_stability.items():
            significant_features[quarter] = [feature for feature, stability in stability_scores.items() if
                                             stability > threshold]

        return significant_features

    def apply_clustering_on_significant_features(self, grouped_data, optimal_k_for_quarters, significant_features):
        """
        Applica il clustering KMeans sulle feature significative per ciascun gruppo (anno, quadrimestre).
        :param grouped_data: dizionario con i dati suddivisi per anno e quadrimestre.
        :param optimal_k_for_quarters: dizionario con il numero ottimale di cluster per ogni quadrimestre.
        :param significant_features: dizionario con le feature significative per ogni quadrimestre.
        :return: dizionario con i risultati del clustering per ogni gruppo.
        """
        clustering_results = {}

        # Applica il clustering a ciascun gruppo (anno, quadrimestre)
        for (year, quarter), data in grouped_data.items():
            # Filtra solo le feature significative
            if quarter in significant_features:
                data_filtered = data[significant_features[quarter]]

                # Recupera il numero ottimale di cluster per il quadrimestre
                optimal_k = optimal_k_for_quarters[quarter]

                # Inizializza e addestra il modello KMeans
                model = KMeans(n_clusters=optimal_k)
                model.fit(data_filtered)

                # Salva i risultati del clustering (etichette dei cluster)
                clustering_results[(year, quarter)] = model.labels_

        return clustering_results

    def calculate_cluster_increment(self, grouped_data, clustering_results):
        """
        Calcola l'incremento nel numero di campioni che appartengono a cluster simili negli anni successivi
        e aggiunge questo incremento (basato sul numero reale di campioni) come nuova feature al dataset originale.
        Usa l'Adjusted Rand Index (ARI) per misurare la similarità tra i cluster.
        :param grouped_data: dizionario con i dati suddivisi per anno e quadrimestre.
        :param clustering_results: dizionario con i risultati del clustering per ogni gruppo.
        :return: dataframe aggiornato con la nuova feature 'incremento numerico'.
        """
        increments = []
        increment_summary = []  # Per salvare anno, quadrimestre, incremento totale

        # Otteniamo il primo anno disponibile per ciascun quadrimestre
        first_year = min(k[0] for k in clustering_results.keys())

        # Compara i cluster tra anni per lo stesso quadrimestre
        for quarter in set(k[1] for k in clustering_results.keys()):
            years = sorted(set(k[0] for k in clustering_results.keys() if k[1] == quarter))

            # Imposta incremento 0 per l'anno di partenza (2019 nel tuo caso)
            if first_year in years:
                data_first_year = grouped_data[(first_year, quarter)].copy()
                data_first_year['incremento numerico'] = 0  # Assegna incremento 0 per tutti i campioni del primo anno
                increments.append(data_first_year)

                # Aggiungi il primo anno con incremento 0 al riepilogo
                increment_summary.append({'year': first_year, 'quarter': quarter, 'total_increment': 0})

            if len(years) > 1:
                for i in range(1, len(years)):
                    year_prev = years[i - 1]
                    year_curr = years[i]
                    labels_prev = clustering_results[(year_prev, quarter)]
                    labels_curr = clustering_results[(year_curr, quarter)]

                    # Per il confronto usiamo un numero minimo di campioni
                    min_samples = min(len(labels_prev), len(labels_curr))

                    # Campioniamo i dati per avere lo stesso numero di campioni per fare il confronto
                    labels_prev_sampled = labels_prev[:min_samples]
                    labels_curr_sampled = labels_curr[:min_samples]

                    # Creiamo una matrice di similarità basata sull'Adjusted Rand Index (ARI)
                    clusters_prev = np.unique(labels_prev_sampled)
                    clusters_curr = np.unique(labels_curr_sampled)
                    similarity_matrix = np.zeros((len(clusters_prev), len(clusters_curr)))

                    for idx_prev, cluster_prev in enumerate(clusters_prev):
                        for idx_curr, cluster_curr in enumerate(clusters_curr):
                            # Troviamo i campioni che appartengono a ciascun cluster nel campione
                            cluster_prev_samples = (labels_prev_sampled == cluster_prev)
                            cluster_curr_samples = (labels_curr_sampled == cluster_curr)

                            # Calcoliamo l'ARI tra l'intero set di etichette
                            similarity_matrix[idx_prev, idx_curr] = adjusted_rand_score(cluster_prev_samples,
                                                                                        cluster_curr_samples)

                    # Utilizziamo l'algoritmo di assegnazione per trovare la corrispondenza migliore
                    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

                    # Otteniamo i risultati per ogni cluster corrente
                    data_curr = grouped_data[(year_curr, quarter)].copy()
                    data_curr['incremento numerico'] = 0  # Inizializza l'incremento a 0 per tutti i campioni

                    total_increment = 0  # Variabile per sommare gli incrementi per anno/quadrimestre

                    for idx_prev, idx_curr in zip(row_ind, col_ind):
                        # Ottieni il cluster precedente e corrente dalla corrispondenza trovata
                        cluster_prev = clusters_prev[idx_prev]
                        cluster_curr = clusters_curr[idx_curr]

                        # Conta il numero reale di campioni per ciascun cluster
                        size_prev_real = np.sum(labels_prev == cluster_prev)
                        size_curr_real = np.sum(labels_curr == cluster_curr)

                        # Calcoliamo l'incremento reale in termini di numero di campioni
                        increment = size_curr_real - size_prev_real
                        total_increment += increment

                        # Assegna questo incremento solo ai campioni appartenenti al cluster corrente
                        data_curr.loc[labels_curr == cluster_curr, 'incremento numerico'] = increment

                    # Aggiungi i dati con l'incremento alla lista degli incrementi
                    increments.append(data_curr)

                    # Aggiungi l'incremento totale per anno e quadrimestre alla lista di riepilogo
                    increment_summary.append(
                        {'year': year_curr, 'quarter': quarter, 'total_increment': total_increment})

        # Combina tutti i dati con incrementi in un unico dataframe
        final_dataset_with_increments = pd.concat(increments)

        # Crea un dataframe per salvare il riepilogo anno-quadrimestre-incremento
        increment_summary_df = pd.DataFrame(increment_summary)

        # Salva il CSV con anno, quadrimestre e incremento
        increment_summary_df.to_csv('increment_summary.csv', index=False)

        return final_dataset_with_increments

    def categorize_increment(self, increment):
        """
        Funzione per discretizzare i valori di incremento.
        """
        if increment > 5000:
            return 'HIGH'
        elif -5000 <= increment <= 5000:
            if abs(increment) <= 1000:  # Costante, vicino a zero
                return 'CONSTANT'
            else:
                return 'MEDIUM'
        elif increment < -5000:
            return 'LOW'
        else:
            return 'UNKNOWN'

    def add_increment_category(self, final_dataset_with_increments):
        """
        Aggiunge la nuova feature 'increment_category' al dataset finale
        sulla base della discretizzazione dei valori di incremento.
        """
        # Applica la funzione per ogni valore di incremento
        final_dataset_with_increments['incremento'] = final_dataset_with_increments['incremento numerico'].apply(
            self.categorize_increment)

        return final_dataset_with_increments
