from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity


class ClusteringAnalyzer:
    """
    Classe per la creazione della variabile incremento numerico e la sua discretizzazione a partire dai risultati del
    clustering KMeans effettuato sulle feature significative per ciascun semestre e anno.
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
        Determina la distorsione per un numero di cluster variabile da 1 a max_clusters.
        :param data: i dati su cui eseguire il clustering.
        :return: una lista di distorsioni per ogni valore di k.
        """
        distortions = []
        K = range(1, self.max_clusters + 1)

        for k in K:
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(data)
            distortions.append(model.inertia_)

        return distortions

    def find_optimal_k_for_semesters(self, grouped_data):
        """
        Trova il grafico che mostra il numero ottimale di cluster per ciascuno dei due semestri.
        :param grouped_data:
        :return:
        """
        semester_distortions = {}

        # Calcola le distorsioni per ogni gruppo di dati
        for (year, semester), data in grouped_data.items():
            distortions = self.determine_optimal_clusters(data)
            if semester not in semester_distortions:
                semester_distortions[semester] = []
            semester_distortions[semester].append(distortions)

        # Media delle distorsioni per ogni k, per ogni semestre
        for semester, distortions_list in semester_distortions.items():
            mean_distortions = np.mean(distortions_list, axis=0)

            # Plot per l'Elbow Method con le distorsioni medie
            K = range(1, self.max_clusters + 1)
            plt.figure(figsize=(8, 4))
            plt.plot(K, mean_distortions, 'bx-')
            plt.xlabel('Numero di cluster')
            plt.ylabel('Distorsione media')
            plt.title(f'Elbow Method per il semestre {semester}')
            plt.show()

    def calculate_jsd_distribution_single_feature(self, data_prev, data_curr, sample_points):
        """
        Calcola la distanza di Jensen-Shannon tra due distribuzioni di una singola feature.
        :param data_prev: Dati del primo anno (una singola feature).
        :param data_curr: Dati del secondo anno (una singola feature).
        :param sample_points: Punti di campionamento nello spazio dei dati.
        :return: Distanza di Jensen-Shannon (JSD) tra le due distribuzioni.
        """
        kde_prev = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_prev)
        kde_curr = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_curr)

        p = np.exp(kde_prev.score_samples(sample_points))
        q = np.exp(kde_curr.score_samples(sample_points))

        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        return jensenshannon(p, q)

    def evaluate_feature_stability_jsd(self, grouped_data):
        """
        Valuta la stabilità delle feature tra anni diversi per lo stesso semestre usando la distanza di Jensen-Shannon (JSD).
        :param grouped_data: dizionario con i dati suddivisi per anno e semestre.
        :return: dizionario con la stabilità delle feature per ogni semestre.
        """
        feature_stability = {}

        for semester in set(k[1] for k in grouped_data.keys()):
            years = sorted(set(k[0] for k in grouped_data.keys() if k[1] == semester))

            if len(years) > 1:
                for combination in combinations(grouped_data[(years[0], semester)].columns, 1):
                    feature_name = combination[0]

                    stability_scores = []
                    for i in range(1, len(years)):
                        year_prev = years[i - 1]
                        year_curr = years[i]

                        data_prev = grouped_data[(year_prev, semester)][[feature_name]].values
                        data_curr = grouped_data[(year_curr, semester)][[feature_name]].values

                        # Definisci i punti di campionamento per la JSD
                        sample_points = np.linspace(np.min(data_prev), np.max(data_prev), 100).reshape(-1, 1)

                        # Calcola la distanza di Jensen-Shannon tra la distribuzione della feature nei due anni
                        jsd_score = self.calculate_jsd_distribution_single_feature(data_prev, data_curr, sample_points)
                        stability_scores.append(jsd_score)

                    # In questo caso, un punteggio più basso indica maggiore stabilità (distribuzioni più simili)
                    if semester not in feature_stability:
                        feature_stability[semester] = {}
                    feature_stability[semester][feature_name] = np.mean(stability_scores)

        return feature_stability

    def select_significant_features(self, feature_stability, top_n=2):
        """
        Seleziona le prime N feature che hanno la stabilità più alta (minima JSD).
        :param feature_stability: dizionario con la stabilità delle feature.
        :param top_n: numero di feature da selezionare.
        :return: dizionario con le feature significative per ogni semestre.
        """
        significant_features = {}

        for semester, stability_scores in feature_stability.items():
            print(f"Stabilità delle feature (JSD): {stability_scores} per il semestre {semester}")

            # Ordina le feature in base ai punteggi di stabilità (minima JSD), dal più basso al più alto
            sorted_features = sorted(stability_scores.items(), key=lambda x: x[1])

            # Prendi solo le prime 'top_n' feature
            significant_features[semester] = [feature for feature, stability in sorted_features[:top_n]]

        print(f"Feature significative per ciascun semestre: {significant_features}")
        return significant_features

    def apply_clustering_on_significant_features(self, grouped_data, optimal_k_for_semesters, significant_features):
        """
        Applica il clustering KMeans sulle feature significative per ciascun gruppo (anno, semestre).
        :param grouped_data: dizionario con i dati suddivisi per anno e semestre.
        :param optimal_k_for_semesters: dizionario con il numero ottimale di cluster per ogni semestre.
        :param significant_features: dizionario con le feature significative per ogni semestre.
        :return: dizionario con i risultati del clustering per ogni gruppo.
        """
        clustering_results = {}

        # Applica il clustering a ciascun gruppo (anno, semestre)
        for (year, semester), data in grouped_data.items():
            # Filtra solo le feature significative
            if semester in significant_features:
                data_filtered = data[significant_features[semester]]

                # Recupera il numero ottimale di cluster per il semestre
                optimal_k = optimal_k_for_semesters[semester]

                # Inizializza e addestra il modello KMeans
                model = KMeans(n_clusters=optimal_k, random_state=42)
                # Verifica se data_filtered è vuoto
                if data_filtered is None or len(data_filtered) == 0:
                    raise ValueError("Il dataset filtrato è vuoto.")
                else:
                    print(f"Dimensioni del dataset filtrato: {data_filtered.shape}")
                    model.fit(data_filtered)

                # Salva i risultati del clustering (etichette dei cluster)
                clustering_results[(year, semester)] = model.labels_

        return clustering_results



    def calculate_jsd(self, kde1, kde2, sample_points):
        """
        Calcola la distanza di Jensen-Shannon tra due distribuzioni KDE.
        :param kde1: Kernel Density Estimation per il cluster 1.
        :param kde2: Kernel Density Estimation per il cluster 2.
        :param sample_points: punti di campionamento nello spazio delle feature.
        :return: distanza di Jensen-Shannon.
        """
        p = np.exp(kde1.score_samples(sample_points))
        q = np.exp(kde2.score_samples(sample_points))
        return jensenshannon(p, q)

    def calculate_jsd_distribution(self, features, labels):
        """
        Calcola la distribuzione per ciascun cluster usando Kernel Density Estimation (KDE).
        :param features: array 2D delle feature.
        :param labels: array delle etichette dei cluster.
        :return: dizionario contenente le distribuzioni (KDE) per ciascun cluster.
        """
        unique_labels = np.unique(labels)
        distributions = {}
        for label in unique_labels:
            cluster_data = features[labels == label]
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(cluster_data)
            distributions[label] = kde
        return distributions

    def calculate_cluster_increment(self, grouped_data, clustering_results, significant_features):
        """
        Calcola l'incremento nel numero di campioni che appartengono a cluster simili negli anni successivi
        e aggiunge questo incremento (basato sul numero reale di campioni) come nuova feature al dataset originale.
        Utilizza la distanza di Jensen-Shannon (JSD) per misurare la similarità tra i cluster.
        :param grouped_data: dizionario con i dati suddivisi per anno e semestre.
        :param clustering_results: dizionario con i risultati del clustering per ogni gruppo.
        :param significant_features: dizionario con le feature significative per ogni semestre.
        :return: dataframe aggiornato con la nuova feature 'incremento numerico'.
        """
        increments = []
        first_year = min(k[0] for k in clustering_results.keys())
        print(f"Primo anno disponibile per i semestri: {first_year}")

        for semester in set(k[1] for k in clustering_results.keys()):
            print(f"\nElaborazione del semestre: {semester}")
            years = sorted(set(k[0] for k in clustering_results.keys() if k[1] == semester))
            print(f"Anni disponibili per il semestre {semester}: {years}")

            # Verifica se ci sono feature significative per il semestre corrente
            if semester not in significant_features:
                print(f"Nessuna feature significativa trovata per il semestre {semester}.")
                continue  # Salta se non ci sono feature significative

            # Filtra le feature significative per il semestre
            selected_features = significant_features[semester]

            if first_year in years:
                print(f"Assegno incremento 0 per l'anno: {first_year}, semestre: {semester}")
                data_first_year = grouped_data[(first_year, semester)].copy()
                data_first_year['incremento numerico'] = 0
                data_first_year['anno'] = first_year
                increments.append(data_first_year)

            if len(years) > 1:
                for i in range(1, len(years)):
                    year_prev = years[i - 1]
                    year_curr = years[i]
                    print(f"\nConfronto tra {year_prev} e {year_curr} per il semestre {semester}")

                    # Etichette dei cluster
                    labels_prev = clustering_results[(year_prev, semester)]
                    labels_curr = clustering_results[(year_curr, semester)]

                    print(f"Numero di campioni nell'anno precedente ({year_prev}): {len(labels_prev)}")
                    print(f"Numero di campioni nell'anno corrente ({year_curr}): {len(labels_curr)}")

                    # Feature corrispondenti per ogni anno
                    features_prev = grouped_data[(year_prev, semester)][selected_features].values
                    features_curr = grouped_data[(year_curr, semester)][selected_features].values

                    # Calcola le distribuzioni KDE per i cluster di ciascun anno
                    kde_prev = self.calculate_jsd_distribution(features_prev, labels_prev)
                    kde_curr = self.calculate_jsd_distribution(features_curr, labels_curr)

                    # Definisci punti di campionamento uniformi nello spazio delle feature
                    sample_points = np.linspace(np.min(features_prev, axis=0), np.max(features_prev, axis=0), 100)

                    # Calcola la matrice di similarità (Jensen-Shannon)
                    clusters_prev = np.unique(labels_prev)
                    clusters_curr = np.unique(labels_curr)
                    jsd_matrix = np.zeros((len(clusters_prev), len(clusters_curr)))

                    print(f"Numero di cluster nell'anno precedente: {len(clusters_prev)}")
                    print(f"Numero di cluster nell'anno corrente: {len(clusters_curr)}")

                    for idx_prev, cluster_prev in enumerate(clusters_prev):
                        for idx_curr, cluster_curr in enumerate(clusters_curr):
                            jsd_matrix[idx_prev, idx_curr] = self.calculate_jsd(kde_prev[cluster_prev],
                                                                                kde_curr[cluster_curr], sample_points)

                    print(f"Matrice di similarità (JSD) tra cluster:\n{jsd_matrix}")

                    # Utilizza np.nan_to_num per sostituire i NaN con 0
                    jsd_matrix = np.nan_to_num(jsd_matrix, nan=1.0)

                    # Utilizza l'algoritmo di assegnazione per trovare la corrispondenza migliore
                    row_ind, col_ind = linear_sum_assignment(jsd_matrix)
                    print(f"Risultato dell'assegnazione: row_ind = {row_ind}, col_ind = {col_ind}")

                    # Ottieni i risultati per ogni cluster corrente
                    data_curr = grouped_data[(year_curr, semester)].copy()
                    data_curr['incremento numerico'] = 0
                    data_curr['anno'] = year_curr

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

        # Filtra i dati per rimuovere quelli del primo anno
        final_dataset_with_increments = final_dataset_with_increments[
            final_dataset_with_increments['anno'] != first_year]

        # Rimuovi la colonna 'anno' prima di restituire il dataset
        final_dataset_with_increments = final_dataset_with_increments.drop(columns=['anno'])

        print("Elaborazione completa, dataset finale creato senza i dati del primo anno e senza la colonna anno.")
        return final_dataset_with_increments

    def categorize_increment(self, increment):
        """
        Funzione per discretizzare i valori di incremento.
        :param increment: valore di incremento numerico.
        :return: categoria di incremento (LOW, MEDIUM, HIGH, CONSTANT).
        """
        if -300 <= increment < 300:  # Incremento costante
            return 'CONSTANT'
        elif increment < -300:  # Incremento negativo
            return 'LOW'
        elif 300 <= increment < 5000:  # Incremento positivo moderato
            return 'MEDIUM'
        elif increment >= 5000:  # Incremento positivo elevato
            return 'HIGH'
        else:
            return 'UNKNOWN'

    def add_increment_category(self, final_dataset_with_increments):
        """
        Aggiunge la nuova feature 'increment_category' al dataset finale
        sulla base della discretizzazione dei valori di incremento.
        :param final_dataset_with_increments: dataset finale con la feature 'incremento numerico'.
        :return: dataset finale con la nuova feature 'incremento'.
        """
        # Applica la funzione per ogni valore di incremento
        final_dataset_with_increments['incremento'] = final_dataset_with_increments['incremento numerico'].apply(
            self.categorize_increment)

        return final_dataset_with_increments
