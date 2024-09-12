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

            # Conteggio delle zone di residenza
            # Conteggio per ciascuna regione
            conto_Abruzzo = df['residenza_Abruzzo'].sum().item()
            conto_Basilicata = df['residenza_Basilicata'].sum().item()
            conto_Calabria = df['residenza_Calabria'].sum().item()
            conto_Campania = df['residenza_Campania'].sum().item()
            conto_Emilia_romagna = df['residenza_Emilia romagna'].sum().item()
            conto_Friuli_venezia_giulia = df['residenza_Friuli venezia giulia'].sum().item()
            conto_Lazio = df['residenza_Lazio'].sum().item()
            conto_Liguria = df['residenza_Liguria'].sum().item()
            conto_Lombardia = df['residenza_Lombardia'].sum().item()
            conto_Marche = df['residenza_Marche'].sum().item()
            conto_Molise = df['residenza_Molise'].sum().item()
            conto_Piemonte = df['residenza_Piemonte'].sum().item()
            conto_Puglia = df['residenza_Puglia'].sum().item()
            conto_Sardegna = df['residenza_Sardegna'].sum().item()
            conto_Sicilia = df['residenza_Sicilia'].sum().item()
            conto_Toscana = df['residenza_Toscana'].sum().item()
            conto_Trentino_alto_adige = df['residenza_Trentino alto adige'].sum().item()
            conto_Umbria = df['residenza_Umbria'].sum().item()
            conto_Valle_daosta = df['residenza_Valle daosta'].sum().item()
            conto_Veneto = df['residenza_Veneto'].sum().item()

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
                'conto_Abruzzo': conto_Abruzzo,
                'conto_Basilicata': conto_Basilicata,
                'conto_Calabria': conto_Calabria,
                'conto_Campania': conto_Campania,
                'conto_Emilia_romagna': conto_Emilia_romagna,
                'conto_Friuli_venezia_giulia': conto_Friuli_venezia_giulia,
                'conto_Lazio': conto_Lazio,
                'conto_Liguria': conto_Liguria,
                'conto_Lombardia': conto_Lombardia,
                'conto_Marche': conto_Marche,
                'conto_Molise': conto_Molise,
                'conto_Piemonte': conto_Piemonte,
                'conto_Puglia': conto_Puglia,
                'conto_Sardegna': conto_Sardegna,
                'conto_Sicilia': conto_Sicilia,
                'conto_Toscana': conto_Toscana,
                'conto_Trentino_alto_adige': conto_Trentino_alto_adige,
                'conto_Umbria': conto_Umbria,
                'conto_Valle_daosta': conto_Valle_daosta,
                'conto_Veneto': conto_Veneto,
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
# Esegui il test
if __name__ == '__main__':
    # Creazione di un dataframe di esempio
    np.random.seed(42)  # Fissiamo il seed per ripetibilità
    data = {
        'age': np.random.randint(18, 70, size=100),
        'duration_minutes': np.random.randint(10, 120, size=100),
        'sesso_male': np.random.randint(0, 2, size=100),
        'sesso_female': np.random.randint(0, 2, size=100),
        'residenza_Abruzzo': np.random.randint(0, 2, size=100),
        'residenza_Basilicata': np.random.randint(0, 2, size=100),
        'residenza_Calabria': np.random.randint(0, 2, size=100),
        'residenza_Campania': np.random.randint(0, 2, size=100),
        'residenza_Emilia romagna': np.random.randint(0, 2, size=100),
        'residenza_Friuli venezia giulia': np.random.randint(0, 2, size=100),
        'residenza_Lazio': np.random.randint(0, 2, size=100),
        'residenza_Liguria': np.random.randint(0, 2, size=100),
        'residenza_Lombardia': np.random.randint(0, 2, size=100),
        'residenza_Marche': np.random.randint(0, 2, size=100),
        'residenza_Molise': np.random.randint(0, 2, size=100),
        'residenza_Piemonte': np.random.randint(0, 2, size=100),
        'residenza_Puglia': np.random.randint(0, 2, size=100),
        'residenza_Sardegna': np.random.randint(0, 2, size=100),
        'residenza_Sicilia': np.random.randint(0, 2, size=100),
        'residenza_Toscana': np.random.randint(0, 2, size=100),
        'residenza_Trentino alto adige': np.random.randint(0, 2, size=100),
        'residenza_Umbria': np.random.randint(0, 2, size=100),
        'residenza_Valle daosta': np.random.randint(0, 2, size=100),
        'residenza_Veneto': np.random.randint(0, 2, size=100),
        'incremento': np.random.randint(0, 4, size=100),
        'cluster': np.random.randint(0, 4, size=100)
    }

    dataframe_for_visuals = pd.DataFrame(data)

    # Copiamo il dataframe e applichiamo lo scaling solo a certe colonne per silhouette
    dataframe_for_silhouette = dataframe_for_visuals.copy()
    features_to_scale = ['age', 'duration_minutes', 'incremento']
    scaler = StandardScaler()
    dataframe_for_silhouette[features_to_scale] = scaler.fit_transform(dataframe_for_silhouette[features_to_scale])

    best_params = {'n_clusters': 4}
    best_score = 0.5
    best_labels = np.random.randint(0, 4, size=100)
    best_centroids = np.random.rand(4, 3)

    # Esegui l'istanza di PostProcessing
    post_processing = PostProcessing(dataframe_for_visuals, dataframe_for_silhouette, best_params, best_score, best_labels, best_centroids)
    post_processing.process_clusters()

    # Salva le statistiche in un file JSON
    post_processing.save_stats_to_json('stats_by_cluster.json')

    # Stampa i risultati in formato leggibile
    print(json.dumps(post_processing.stats_by_cluster, indent=4))