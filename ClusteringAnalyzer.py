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

            # Plot per l'Elbow Method con le distorsioni medie
            K = range(1, self.max_clusters + 1)
            plt.figure(figsize=(8, 4))
            plt.plot(K, mean_distortions, 'bx-')
            plt.xlabel('Numero di cluster')
            plt.ylabel('Distorsione media')
            plt.title(f'Elbow Method per il quadrimestre {quarter}')
            plt.show()


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

    def select_significant_features(self, feature_stability, top_n=2):
        """
        Seleziona le prime N feature che hanno la stabilità più alta.
        :param feature_stability: dizionario con la stabilità delle feature.
        :param top_n: numero di feature da selezionare.
        :return: dizionario con le feature significative per ogni quadrimestre.
        """
        significant_features = {}

        for quarter, stability_scores in feature_stability.items():
            print(f"scores: {stability_scores} per il quadrimestre {quarter}")

            # Ordina le feature in base ai punteggi di stabilità, dal più alto al più basso
            sorted_features = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)

            # Prendi solo le prime 'top_n' feature
            significant_features[quarter] = [feature for feature, stability in sorted_features[:top_n]]

        print(f"Feature significative per ciascun quadrimestre: {significant_features}")
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
                # Verifica se data_filtered è vuoto
                if data_filtered is None or len(data_filtered) == 0:
                    raise ValueError("Il dataset filtrato è vuoto.")
                else:
                    print(f"Dimensioni del dataset filtrato: {data_filtered.shape}")
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

        # Otteniamo il primo anno disponibile per ciascun quadrimestre
        first_year = min(k[0] for k in clustering_results.keys())
        print(f"Primo anno disponibile per i quarti: {first_year}")

        # Compara i cluster tra anni per lo stesso quadrimestre
        for quarter in set(k[1] for k in clustering_results.keys()):
            print(f"\nElaborazione del trimestre: {quarter}")
            years = sorted(set(k[0] for k in clustering_results.keys() if k[1] == quarter))
            print(f"Anni disponibili per il trimestre {quarter}: {years}")

            # Imposta incremento 0 per l'anno di partenza (2019 nel tuo caso)
            if first_year in years:
                print(f"Assegno incremento 0 per l'anno: {first_year}, trimestre: {quarter}")
                data_first_year = grouped_data[(first_year, quarter)].copy()
                data_first_year['incremento numerico'] = 0  # Assegna incremento 0 per tutti i campioni del primo anno
                data_first_year['anno'] = first_year  # Aggiungi l'anno come colonna
                increments.append(data_first_year)

            if len(years) > 1:
                for i in range(1, len(years)):
                    year_prev = years[i - 1]  # Anno precedente
                    year_curr = years[i]  # Anno corrente
                    print(f"\nConfronto tra {year_prev} e {year_curr} per il trimestre {quarter}")

                    labels_prev = clustering_results[
                        (year_prev, quarter)]  # Etichette dei cluster per l'anno precedente
                    labels_curr = clustering_results[(year_curr, quarter)]  # Etichette dei cluster per l'anno corrente

                    print(f"Numero di campioni nell'anno precedente ({year_prev}): {len(labels_prev)}")
                    print(f"Numero di campioni nell'anno corrente ({year_curr}): {len(labels_curr)}")

                    # Viene determinato il numero minimo di campioni tra i due anni (min_samples)
                    min_samples = min(len(labels_prev), len(labels_curr))
                    print(f"Numero minimo di campioni usati per il confronto: {min_samples}")

                    # Campionamento per confrontare lo stesso numero di campioni
                    labels_prev_sampled = labels_prev[:min_samples]
                    labels_curr_sampled = labels_curr[:min_samples]

                    # Creiamo una matrice di similarità basata sull'Adjusted Rand Index (ARI)
                    clusters_prev = np.unique(labels_prev_sampled)
                    clusters_curr = np.unique(labels_curr_sampled)
                    similarity_matrix = np.zeros((len(clusters_prev), len(clusters_curr)))

                    print(f"Numero di cluster nell'anno precedente: {len(clusters_prev)}")
                    print(f"Numero di cluster nell'anno corrente: {len(clusters_curr)}")

                    for idx_prev, cluster_prev in enumerate(clusters_prev):
                        for idx_curr, cluster_curr in enumerate(clusters_curr):
                            # Troviamo i campioni che appartengono a ciascun cluster
                            cluster_prev_samples = (labels_prev_sampled == cluster_prev)
                            cluster_curr_samples = (labels_curr_sampled == cluster_curr)

                            # Calcoliamo l'ARI tra l'intero set di etichette
                            similarity_matrix[idx_prev, idx_curr] = adjusted_rand_score(cluster_prev_samples,
                                                                                        cluster_curr_samples)

                    print(f"Matrice di similarità (ARI) tra cluster:\n{similarity_matrix}")

                    # Utilizziamo l'algoritmo di assegnazione per trovare la corrispondenza migliore
                    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
                    print(f"Risultato dell'assegnazione: row_ind = {row_ind}, col_ind = {col_ind}")

                    # Otteniamo i risultati per ogni cluster corrente
                    data_curr = grouped_data[(year_curr, quarter)].copy()
                    data_curr['incremento numerico'] = 0  # Inizializza l'incremento a 0 per tutti i campioni
                    data_curr['anno'] = year_curr  # Aggiungi l'anno come colonna

                    for idx_prev, idx_curr in zip(row_ind, col_ind):
                        print(f"Processando il cluster {idx_prev} dell'anno precedente e {idx_curr} dell'anno corrente")
                        cluster_prev = clusters_prev[idx_prev]
                        cluster_curr = clusters_curr[idx_curr]

                        # Conta il numero reale di campioni per ciascun cluster
                        size_prev_real = np.sum(labels_prev == cluster_prev)
                        size_curr_real = np.sum(labels_curr == cluster_curr)

                        print(f"Numero di campioni nel cluster precedente {cluster_prev}: {size_prev_real}")
                        print(f"Numero di campioni nel cluster corrente {cluster_curr}: {size_curr_real}")

                        # Calcoliamo l'incremento reale in termini di numero di campioni
                        increment = size_curr_real - size_prev_real
                        print(f"Incremento calcolato per il cluster corrente {cluster_curr}: {increment}")

                        # Assegna questo incremento solo ai campioni appartenenti al cluster corrente
                        data_curr.loc[labels_curr == cluster_curr, 'incremento numerico'] = increment

                    # Aggiungi i dati con l'incremento alla lista degli incrementi
                    increments.append(data_curr)

        # Combina tutti i dati con incrementi in un unico dataframe
        final_dataset_with_increments = pd.concat(increments)

        # Filtra i dati per rimuovere quelli del primo anno (ad esempio, 2019)
        final_dataset_with_increments = final_dataset_with_increments[
            final_dataset_with_increments['anno'] != first_year]

        # Rimuovi la colonna 'anno' prima di restituire il dataset
        final_dataset_with_increments = final_dataset_with_increments.drop(columns=['anno'])

        print("Elaborazione completa, dataset finale creato senza i dati del primo anno e senza la colonna anno.")

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
