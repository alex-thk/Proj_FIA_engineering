import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler


class PostProcessing:
    def __init__(self, dataframe_for_visuals, dataframe_for_silhouette, best_params, best_score, best_labels, best_centroids):
        self.dataframe = dataframe_for_visuals
        self.dataframe_for_silhouette = dataframe_for_silhouette
        self.dataframes_by_cluster = {}
        self.stats_by_cluster = {}
        self.best_params = best_params
        self.best_score = best_score
        self.best_labels = best_labels
        self.best_centroids = best_centroids

    def process_clusters(self):

        # Calcola una sola volta i silhouette samples per tutti i punti del dataset
        silhouette_values = silhouette_samples(
            self.dataframe_for_silhouette.drop(columns=['cluster']),  # Rimuovi la colonna 'cluster' per il calcolo delle silhouette
            self.dataframe_for_silhouette['cluster']  # Utilizza la colonna 'cluster' per le etichette
        )

        # Trova tutti i valori unici della colonna 'cluster'
        cluster_values = self.dataframe['cluster'].unique()

        # Per ogni valore di cluster, crea un sotto-dataframe
        for cluster in cluster_values:
            self.dataframes_by_cluster[f'dataframe_cluster_{cluster}'] = self.dataframe[self.dataframe['cluster'] == cluster]

        # Calcola le statistiche per ogni cluster
        for cluster_name, df in self.dataframes_by_cluster.items():
            media_age = df['age'].mean().item()  # Media della colonna 'age'
            media_duration = df['duration_minutes'].mean().item()  # Media della colonna 'duration_minutes'
            conto_sesso_male = df['sesso_male'].sum().item()  # Conteggio dei True nella colonna 'sesso_male'
            conto_sesso_female = df['sesso_female'].sum().item()  # Conteggio dei True nella colonna 'sesso_female'
            conto_nord = df['zona_residenza_Nord'].sum().item()  # Conteggio dei True nella colonna 'Nord'
            conto_centro = df['zona_residenza_Centro'].sum().item()  # Conteggio dei True nella colonna 'Centro'
            conto_sud = df['zona_residenza_Sud'].sum().item()  # Conteggio dei True nella colonna 'Sud'

            numero_record = df.shape[0]

            # Calcola la silhouette media
            if numero_record > 1:  # Silhouette ha senso solo per cluster con più di un record
                mask_cluster = self.dataframe_for_silhouette['cluster'] == df['cluster'].iloc[0]
                silhouette_media = silhouette_values[mask_cluster].mean()
            else:
                silhouette_media = np.nan  # Non calcolabile per cluster con un solo record

            # Calcola la classe più comune rispetto alla variabile 'incremento'
            classe_incremento_comune = df['incremento'].mode()[0]
            mappatura_incremento = {0: 'CONSTANT', 1: 'LOW', 2: 'MEDIUM', 3: 'HIGH'}
            classe_comune = mappatura_incremento[classe_incremento_comune]

            # Calcola la purezza del cluster
            numero_classe_comune = df[df['incremento'] == classe_incremento_comune].shape[0]
            purezza_cluster = numero_classe_comune / numero_record

            # Memorizza le statistiche in un dizionario
            self.stats_by_cluster[cluster_name] = {
                'media_age': media_age,
                'media_duration_minutes': media_duration,
                'conto_sesso_male': conto_sesso_male,
                'conto_sesso_female': conto_sesso_female,
                'conto_nord': conto_nord,
                'conto_centro': conto_centro,
                'conto_sud': conto_sud,
                'numero_record': numero_record,
                'silhouette_media': silhouette_media,
                'classe_comune_incremento': classe_comune,
                'purezza_cluster': purezza_cluster
            }

    def save_stats_to_json(self, file_name='stats_by_cluster.json'):
        # Salva il dizionario stats_by_cluster in un file JSON
        data_to_save = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'stats_by_cluster': self.stats_by_cluster,
            'best_centroids': self.best_centroids.tolist(),  # Converti in lista se è un array numpy
            'best_labels': self.best_labels.tolist(),  # Converti in lista se è un array numpy
        }
        with open(file_name, 'w') as f:
            json.dump(data_to_save, f, indent=4)


# Esegui il test
if __name__ == '__main__':
    # Creazione di un dataframe di esempio
    np.random.seed(42)  # Fissiamo il seed per ripetibilità
    data = {
        'age': np.random.randint(18, 70, size=100),
        'duration_minutes': np.random.randint(10, 120, size=100),
        'sesso_male': np.random.randint(0, 2, size=100),
        'sesso_female': np.random.randint(0, 2, size=100),
        'zona_residenza_Nord': np.random.randint(0, 2, size=100),
        'zona_residenza_Centro': np.random.randint(0, 2, size=100),
        'zona_residenza_Sud': np.random.randint(0, 2, size=100),
        'incremento': np.random.randint(0, 4, size=100),
        'cluster': np.random.randint(0, 4, size=100)
    }

    dataframe_for_visuals = pd.DataFrame(data)

    # Copiamo il dataframe e applichiamo lo scaling solo a certe colonne per silhouette
    dataframe_for_silhouette = dataframe_for_visuals.copy()
    features_to_scale = ['age', 'duration_minutes', 'incremento']
    scaler = StandardScaler()
    dataframe_for_silhouette[features_to_scale] = scaler.fit_transform(dataframe_for_silhouette[features_to_scale])

    # Esegui l'istanza di PostProcessing
    post_processing = PostProcessing(dataframe_for_visuals, dataframe_for_silhouette)
    post_processing.process_clusters()

    # Salva le statistiche in un file JSON
    post_processing.save_stats_to_json('stats_by_cluster.json')