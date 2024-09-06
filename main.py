import pandas as pd

from data_reduction import DataReduction
from data_cleaning import DataCleaning
from data_transformation import DataTransformation
from data_standardization import FeatureScaler
from ClusteringAnalyzer import ClusteringAnalyzer
from hyperparameter_tuner import HyperparameterTuner

if __name__ == '__main__':
    file = 'challenge_campus_biomedico_2024.parquet'
    df = pd.read_parquet(file)

    cleaner = DataCleaning()
    cleaner.calculate_precentage_missing_values_in_df(df)
    df = cleaner.handle_last_column(df)
    df = cleaner.add_relevant_columns(df)
    df = cleaner.impute_missing_values(df)
    cleaner.calculate_precentage_missing_values_in_df(df)
    cleaner.show_head(df)

    reducer = DataReduction()
    # removing all records with missing values in the column 'comune_residenza'
    # a 0.03% of the total records will be removed (not a big deal)
    # ---------------------
    # 'ora_inizio_erogazione' and 'ora_fine_erogazione' columns are not useful anymore
    # since now we have the 'duration' column
    # ---------------------
    # also dropping other columns that are not significant for the analysis
    df = reducer.remove_insignificant_columns(df, ['comune_residenza', 'ora_inizio_erogazione', 'ora_fine_erogazione', 'id_prenotazione', 'id_paziente', 'codice_regione_residenza', 'codice_asl_residenza', 'codice_provincia_residenza',
                        'codice_comune_residenza', 'tipologia_servizio', 'data_contatto', 'codice_regione_erogazione', 'codice_asl_erogazione',
                        'codice_provincia_erogazione', 'struttura_erogazione', 'codice_struttura_erogazione', 'id_professionista_sanitario',
                        'tipologia_professionista_sanitario', 'data_nascita'])
    cleaner.calculate_precentage_missing_values_in_df(df) # should print no missing values

    # per fare 'nord sud centro' toglieremo
    # 'regione_residenza, asl_residenza, provincia_residenza, comune_residenza'
    # print(df['codice_tipologia_professionista_sanitario'].unique())

    # assistenza sanitaria diretta 1
    # infermiere, ostetrica, fisioterapista, podologo

    # assistenza riabilitativa e terapeutica psico-sociale 2
    # psicologo, tecnico riabilitazione psichiatrica, terapista occupazionale, terapista della neuro e psicomotricita dell'eta evolutiva

    # supporto salute e benessere 3
    # dietista, educatore professionale, assistente sanitario, logopedista

    # mapping for codice_tipologia_professionista_sanitario
    mapping1 = {
        'INF': 'Assistenza sanitaria diretta' , 'OST': 'Assistenza sanitaria diretta' , 'FIS': 'Assistenza sanitaria diretta' , 'POD': 'Assistenza sanitaria diretta' ,
        'PSI': 'Assistenza riabilitativa e terapeutica psico-sociale' , 'TRP': 'Assistenza riabilitativa e terapeutica psico-sociale' , 'TRO': 'Assistenza riabilitativa e terapeutica psico-sociale' , 'TNP': 'Assistenza riabilitativa e terapeutica psico-sociale' ,
        'DIE': 'Supporto salute e benessere' , 'EDP': 'Supporto salute e benessere' , 'ASN': 'Supporto salute e benessere' , 'LPD': 'Supporto salute e benessere'
    }

    transformer = DataTransformation()
    df = transformer.transform_col(df, 'codice_tipologia_professionista_sanitario', mapping1, 'group_professionisti')

    # NORD --> 1
    # CENTRO --> 2
    # SUD --> 3

    region_mapping = {
        "Piemonte": 'Nord', "Valle d`aosta": 'Nord', "Valle d'aosta": 'Nord', "Liguria": 'Nord', "Lombardia": 'Nord', "Trentino alto adige": 'Nord', "Veneto": 'Nord', "Friuli venezia giulia": 'Nord', "Emilia romagna": 'Nord', "Prov. auton. bolzano": 'Nord', "Prov. auton. trento": 'Nord',
        "Toscana": 'Centro', "Umbria": 'Centro', "Marche": 'Centro', "Lazio": 'Centro',
        "Abruzzo": 'Sud', "Molise": 'Sud', "Campania": 'Sud', "Puglia": 'Sud', "Basilicata": 'Sud', "Calabria": 'Sud', "Sicilia": 'Sud', "Sardegna": 'Sud'
    }

    df = transformer.transform_col(df,'regione_residenza', region_mapping, 'zona_residenza')
    df = transformer.transform_col(df, 'regione_erogazione', region_mapping, 'zona_erogazione')

    df = reducer.remove_insignificant_columns(df, ['regione_residenza', 'asl_residenza', 'provincia_residenza', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione', 'codice_tipologia_professionista_sanitario', 'codice_tipologia_professionista_sanitario_tuple', 'regione_residenza_tuple', 'regione_erogazione_tuple'])

    # It is very importante to note that where data_disdetta is True (so the patient has cancelled the appointment) the duration must be set to 0
    df = cleaner.handle_cancelled_appointments(df)
    print(df.info())

    # now onto creating dummy variables for group_professionisti, zona_residenza and zona_erogazione
    df = transformer.create_dummies(df,'group_professionisti')
    df = transformer.create_dummies(df, 'zona_residenza')
    df = transformer.create_dummies(df, 'zona_erogazione')
    df = transformer.create_dummies(df, 'sesso')

    # now we have created dummies for the "medical area of the professional", the "residence area of the patient" and the "area where the service was provided"

    # removing other columns insignificant to our analysis
    df = reducer.remove_insignificant_columns(df, ['data_disdetta', 'duration', 'descrizione_attivita', 'codice_descrizione_attivita', 'tipologia_struttura_erogazione', 'codice_tipologia_struttura_erogazione', 'data_erogazione'])

    # now onto standardization
    scaler = FeatureScaler()
    df = scaler.standardize(df, ['age', 'duration_minutes'])

    print(df.info())

    # Caricamento del dataset
    dataset = df

    # Inizializzazione dell'analizzatore di clustering
    analyzer = ClusteringAnalyzer()

    # Suddivisione dei dati per anno e quadrimestre
    grouped_data = dataset.groupby(['year', 'quarter'])
    grouped_data_dict = {name: group.drop(columns=['year', 'quarter']) for name, group in grouped_data}

    # Step 1: Determinazione del numero ottimale di cluster per ogni quadrimestre
    optimal_k_for_quarters = analyzer.find_optimal_k_for_quarters(grouped_data_dict)

    # Step 2: Valutare la stabilità delle feature tra anni diversi per ogni quadrimestre
    feature_stability = analyzer.evaluate_feature_stability(grouped_data_dict, optimal_k_for_quarters)

    # Step 3: Seleziona le feature significative sulla base della stabilità
    significant_features = analyzer.select_significant_features(feature_stability, threshold=0.5)

    # Step 4: Rifai il clustering usando solo le feature significative
    clustering_results_significant = analyzer.apply_clustering_on_significant_features(grouped_data_dict,
                                                                                       optimal_k_for_quarters,
                                                                                       significant_features)

    # Step 5: Calcola l'incremento nei cluster e crea il dataset finale con la nuova feature
    final_dataset_with_increments = analyzer.calculate_cluster_increment(grouped_data_dict,
                                                                         clustering_results_significant)

    # Step 6: Aggiungi la colonna delle categorie di incremento
    final_dataset_with_categories = analyzer.add_increment_category(final_dataset_with_increments)

    # Step 7: Salva il dataset finale con la colonna delle categorie
    final_dataset_with_categories.to_csv('final_dataset_with_categories.csv', index=False)

    # Visualizza il dataset finale
    print("Dataset finale con incremento dei cluster e categorie:")
    print(final_dataset_with_categories)

    # Show the unique values in the 'cluster_increment' feature
    cluster_increment_values = final_dataset_with_increments['incremento numerico'].unique()

    # Display the unique values
    print("Valori unici della feature 'cluster_increment':")
    print(cluster_increment_values)

    # Definizione della griglia di iperparametri per il tuning
    param_grid = {
        'n_clusters': [4, 5, 6, 7, 8],  # numero di cluster che vogliamo testare
    }

    print(final_dataset_with_increments.info())

    # Dizionario per mappare le stringhe ai numeri
    mapping = {
        'CONSTANT': 0,
        'LOW': 1,
        'MEDIUM': 2,
        'HIGH': 3
    }

    # Applicazione del mapping alla colonna 'incremento'
    final_dataset_with_increments['incremento'] = final_dataset_with_increments['incremento'].map(mapping)

    # Estrai la colonna "incremento" da final_dataset_with_increments e crea un array numpy 1D
    labels = final_dataset_with_increments['incremento'].to_numpy()

    print("Istanzio classe HyperparameterTuner..")

    # Istanzia il tuner per KMeans
    tuner = HyperparameterTuner(algorithm='kmeans', param_grid=param_grid, data=final_dataset_with_increments, labels=labels)

    print("Inizio la ricerca dell'ottimo..")
    # Eseguiamo il Grid Search per trovare la combinazione migliore di iperparametri
    best_params, best_score = tuner.perform_grid_search()

    print("Ricerca terminata")

