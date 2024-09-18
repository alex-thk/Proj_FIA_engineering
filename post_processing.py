import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler


class PostProcessing:
    """
    Classe per il post-processing dei risultati del clustering
    """
    def __init__(self, dataframe_for_visuals, dataframe_for_silhouette, best_params, best_score, best_centroids, best_labels):
        """
        Inizializzazione della la classe PostProcessing
        :param dataframe_for_visuals: DataFrame con le features originali e le colonne 'cluster' e "incremento"
        :param dataframe_for_silhouette: DataFrame con le features scalate e la colonna 'cluster' e "incremento" scalato
        :param best_params: Iperparametri migliori trovati dal tuning
        :param best_score: Punteggio migliore ottenuto dal tuning
        :param best_centroids: Centroidi migliori trovati dal clustering
        :param best_labels: Etichette migliori trovate dal clustering
        """
        self.dataframe = dataframe_for_visuals
        self.dataframe_for_silhouette = dataframe_for_silhouette
        self.dataframes_by_cluster = {}
        self.stats_by_cluster = {}
        self.best_params = best_params
        self.best_score = best_score
        self.best_labels = best_labels
        self.best_centroids = best_centroids

    def process_clusters(self):
        """
        Calcola le statistiche per ciascun cluster e le memorizza in un dizionario che diventa attributo della classe
        """
        # Calcola una sola volta i silhouette samples per tutti i punti del dataset
        silhouette_values = silhouette_samples(
            self.dataframe_for_silhouette.drop(columns=['cluster']),  # Rimuovi la colonna 'cluster' per il calcolo delle silhouette
            self.dataframe_for_silhouette['cluster']  # Utilizza la colonna 'cluster' per le etichette dei cluster
        )

        # Trova tutti i valori unici della colonna 'cluster'
        cluster_values = self.dataframe['cluster'].unique()

        # Per ogni valore di cluster, crea un sotto-dataframe
        for cluster in cluster_values:
            self.dataframes_by_cluster[f'dataframe_cluster_{cluster}'] = self.dataframe[self.dataframe['cluster'] == cluster]

        # Calcola le statistiche per ogni cluster
        for cluster_name, df in self.dataframes_by_cluster.items():
            media_age = df['age'].mean().item()  # Media della colonna 'age'
            #  media_duration = df['duration_minutes'].mean().item()  # Media della colonna 'duration_minutes'
            #  conto_sesso_male = df['sesso_male'].sum().item()  # Conteggio dei True nella colonna 'sesso_male'
            #  conto_sesso_female = df['sesso_female'].sum().item()  # Conteggio dei True nella colonna 'sesso_female'

            # Conteggio delle zone di residenza
            # Conteggio per ciascuna regione
            '''conto_Abruzzo = df['residenza_Abruzzo'].sum().item()
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
            conto_Veneto = df['residenza_Veneto'].sum().item()'''
            conto_ASN= df['codice_tipologia_professionista_sanitario_ASN'].sum().item()
            conto_DIE = df['codice_tipologia_professionista_sanitario_DIE'].sum().item()
            conto_EDP = df['codice_tipologia_professionista_sanitario_EDP'].sum().item()
            conto_FIS = df['codice_tipologia_professionista_sanitario_FIS'].sum().item()
            conto_INF = df['codice_tipologia_professionista_sanitario_INF'].sum().item()
            conto_LPD = df['codice_tipologia_professionista_sanitario_LPD'].sum().item()
            conto_OST = df['codice_tipologia_professionista_sanitario_OST'].sum().item()
            conto_POD = df['codice_tipologia_professionista_sanitario_POD'].sum().item()
            conto_PSI = df['codice_tipologia_professionista_sanitario_PSI'].sum().item()
            conto_TNP = df['codice_tipologia_professionista_sanitario_TNP'].sum().item()
            conto_TRO = df['codice_tipologia_professionista_sanitario_TRO'].sum().item()
            conto_TRP = df['codice_tipologia_professionista_sanitario_TRP'].sum().item()
            conto_sesso_male = df['sesso_male'].sum().item()
            conto_sesso_female = df['sesso_female'].sum().item()

            numero_record = df.shape[0]

            # Calcola la silhouette media
            if numero_record > 1:  # Silhouette ha senso solo per cluster con più di un record
                mask_cluster = self.dataframe_for_silhouette['cluster'] == df['cluster'].iloc[0]
                silhouette_media = silhouette_values[mask_cluster].mean()
            else:
                silhouette_media = np.nan  # Non calcolabile per cluster con un solo record

            # Calcola la classe più comune rispetto alla variabile 'incremento'
            classe_incremento_comune = df['incremento'].mode()[0]
            mappatura_incremento = {0: 'LOW', 1: 'CONSTANT', 2: 'MEDIUM', 3: 'HIGH'}
            classe_comune = mappatura_incremento[classe_incremento_comune]

            # Calcola la purezza del cluster
            numero_classe_comune = df[df['incremento'] == classe_incremento_comune].shape[0]
            purezza_cluster = numero_classe_comune / numero_record

            # Memorizza le statistiche in un dizionario
            self.stats_by_cluster[cluster_name] = {
                'media_age': media_age,
                # 'media_duration_minutes': media_duration,
                'conto_sesso_male': conto_sesso_male,
                'conto_sesso_female': conto_sesso_female,
                'frazione_male': conto_sesso_male / numero_record,
                'frazione_female': conto_sesso_female / numero_record,
                'conto_ASN': conto_ASN,
                'conto_DIE': conto_DIE,
                'conto_EDP': conto_EDP,
                'conto_FIS': conto_FIS,
                'conto_INF': conto_INF,
                'conto_LPD': conto_LPD,
                'conto_OST': conto_OST,
                'conto_POD': conto_POD,
                'conto_PSI': conto_PSI,
                'conto_TNP': conto_TNP,
                'conto_TRO': conto_TRO,
                'conto_TRP': conto_TRP,
                'numero_record': numero_record,
                'silhouette_media': silhouette_media,
                'classe_comune_incremento': classe_comune,
                'purezza_cluster': purezza_cluster,
            }

    def save_stats_to_json(self, file_name='stats_by_cluster.json'):
        """
        Salva le statistiche dei cluster in un file JSON
        """
        data_to_save = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'stats_by_cluster': self.stats_by_cluster,
            'best_centroids': self.best_centroids.tolist(),  # Converti in lista se è un array numpy
        }
        with open(file_name, 'w') as f:
            json.dump(data_to_save, f, indent=4)


# test per classe PostProcessing
"""
if __name__ == '__main__':
    # Creazione di un dataframe di esempio
    np.random.seed(42)  # Fissiamo il seed per ripetibilità
    data = {
        # 'age': np.random.randint(18, 70, size=100),
        # 'duration_minutes': np.random.randint(10, 120, size=100),
        'sesso_male': np.random.randint(0, 2, size=100),
        'sesso_female': np.random.randint(0, 2, size=100),
        'incremento': np.random.randint(0, 4, size=100),
        'cluster': np.random.randint(0, 4, size=100),
        'codice_tipologia_professionista_sanitario_ASN': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_DIE': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_EDP': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_FIS': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_INF': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_LPD': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_OST': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_POD': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_PSI': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_TNP': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_TRO': np.random.randint(0, 2, size=100),
        'codice_tipologia_professionista_sanitario_TRP': np.random.randint(0, 2, size=100),
    }

    dataframe_for_visuals = pd.DataFrame(data)
    print(dataframe_for_visuals.head())

    # Copiamo il dataframe e applichiamo lo scaling solo a certe colonne per silhouette
    dataframe_for_silhouette = dataframe_for_visuals.copy()
    features_to_scale = ['incremento']
    scaler = StandardScaler()
    dataframe_for_silhouette[features_to_scale] = scaler.fit_transform(dataframe_for_silhouette[features_to_scale])

    best_params = {'n_clusters': 4}
    best_score = 0.5
    best_labels = np.random.randint(0, 4, size=100)
    best_centroids = np.random.rand(4, 3)

    # Esegui l'istanza di PostProcessing
    post_processing = PostProcessing(dataframe_for_visuals, dataframe_for_silhouette, best_params, best_score, best_centroids, best_labels)
    post_processing.process_clusters()

    # Salva le statistiche in un file JSON
    post_processing.save_stats_to_json('stats_by_cluster.json')

    # Stampa i risultati in formato leggibile
    print(json.dumps(post_processing.stats_by_cluster, indent=4))
"""