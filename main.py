import pandas as pdimport numpy as npimport jsonfrom data_reduction import DataReductionfrom data_cleaning import DataCleaningfrom data_transformation import DataTransformationfrom data_standardization import FeatureScalerfrom ClusteringAnalyzer import ClusteringAnalyzerfrom hyperparameter_tuner import HyperparameterTunerfrom outlier_detector import outlier_detectorfrom clustering_result_analyzer import ClusteringResultAnalyzerfrom clustering_model import ClusteringModelfrom post_processing import PostProcessingfrom visualizer import Visualizerif __name__ == '__main__':    file = 'challenge_campus_biomedico_2024.parquet'    df = pd.read_parquet(file)    cleaner = DataCleaning()    cleaner.calculate_precentage_missing_values_in_df(df)    df = cleaner.handle_last_column(df)    df = cleaner.add_relevant_columns(df)    df = cleaner.impute_missing_values(df)    cleaner.calculate_precentage_missing_values_in_df(df)    cleaner.show_head(df)    reducer = DataReduction()    # removing all records with missing values in the column 'comune_residenza'    # a 0.03% of the total records will be removed (not a big deal)    # ---------------------    # 'ora_inizio_erogazione' and 'ora_fine_erogazione' columns are not useful anymore    # since now we have the 'duration' column    # ---------------------    # also dropping other columns that are not significant for the analysis    df = reducer.remove_insignificant_columns(df, ['comune_residenza', 'ora_inizio_erogazione', 'ora_fine_erogazione', 'id_prenotazione', 'id_paziente', 'codice_regione_residenza', 'codice_asl_residenza', 'codice_provincia_residenza',                                                   'codice_comune_residenza', 'tipologia_servizio', 'data_contatto', 'codice_regione_erogazione', 'codice_asl_erogazione',                                                   'codice_provincia_erogazione', 'struttura_erogazione', 'codice_struttura_erogazione', 'id_professionista_sanitario',                                                   'tipologia_professionista_sanitario', 'data_nascita'])    cleaner.calculate_precentage_missing_values_in_df(df)  # should print no missing values    transformer = DataTransformation()    region_mapping = {        # per creare le dummie per ogni regione        "Valle d`aosta": 'Valle daosta',        "Valle d'aosta": 'Valle daosta',        "Prov. auton. bolzano": 'Trentino alto adige',        "Prov. auton. trento": 'Trentino alto adige',        "Piemonte": "Piemonte",        "Lombardia": "Lombardia",        "Veneto": "Veneto",        "Friuli venezia giulia": "Friuli venezia giulia",        "Liguria": "Liguria",        "Emilia romagna": "Emilia romagna",        "Toscana": "Toscana",        "Umbria": "Umbria",        "Marche": "Marche",        "Lazio": "Lazio",        "Abruzzo": "Abruzzo",        "Molise": "Molise",        "Campania": "Campania",        "Puglia": "Puglia",        "Basilicata": "Basilicata",        "Calabria": "Calabria",        "Sicilia": "Sicilia",        "Sardegna": "Sardegna"    }    df = transformer.transform_col(df, 'regione_residenza', region_mapping, 'residenza')    df = reducer.remove_insignificant_columns(df, ['regione_residenza', 'asl_residenza', 'provincia_residenza', 'asl_erogazione', 'provincia_erogazione', 'regione_residenza_tuple'])    # It is very importante to note that where data_disdetta is True (so the patient has cancelled the appointment)    # the duration must be set to 0    df = cleaner.handle_cancelled_appointments(df)    print(df.info())    df = transformer.create_dummies(df, 'codice_tipologia_professionista_sanitario')    df = transformer.create_dummies(df, 'residenza')    df = transformer.create_dummies(df, 'sesso')    # removing other columns insignificant to our analysis    df = reducer.remove_insignificant_columns(df, ['regione_erogazione', 'data_disdetta', 'duration', 'descrizione_attivita', 'codice_descrizione_attivita', 'tipologia_struttura_erogazione', 'codice_tipologia_struttura_erogazione', 'data_erogazione'])    print(df.columns)    detector = outlier_detector()    df = detector.detect_and_drop_outliers(df, 'age', 0, 100)    df = detector.detect_and_drop_outliers(df, 'duration_minutes', 5, 90)    # now onto standardization    scaler = FeatureScaler()    dataframe_for_visuals = df.copy()    df = scaler.standardize_(df, ['age', 'duration_minutes'])    print(df.info())    # Caricamento del dataset    dataset = df    # Inizializzazione dell'analizzatore di clustering    analyzer = ClusteringAnalyzer()    # Suddivisione dei dati per anno e semestre    grouped_data = dataset.groupby(['year', 'semester'])    grouped_data_dict = {name: group.drop(columns=['year', 'semester']) for name, group in grouped_data}    # Step 1: Determinazione del numero ottimale di cluster per ogni semestre    analyzer.find_optimal_k_for_semesters(grouped_data_dict)    # Dopo aver visualizzato i grafici, scegli manualmente il valore di k per ciascun semestre    optimal_k_for_semesters = {        1: 2,  # Valore ottimale per il primo semestre        2: 2  # Valore ottimale per il secondo semestre    }    # Step 2: Valutare la stabilità delle feature tra anni diversi per ogni semestre    feature_stability = analyzer.evaluate_feature_stability(grouped_data_dict, optimal_k_for_semesters)    # Step 3: Seleziona le feature significative sulla base della stabilità    significant_features = analyzer.select_significant_features(feature_stability)    # Step 4: Rifai il clustering usando solo le feature significative    clustering_results_significant = analyzer.apply_clustering_on_significant_features(        grouped_data_dict,        optimal_k_for_semesters,        significant_features    )    # Step 5: Calcola l'incremento nei cluster e crea il dataset finale con la nuova feature    # NOTA: Ora passa anche 'significant_features' alla funzione 'calculate_cluster_increment'    final_dataset_with_increments = analyzer.calculate_cluster_increment(        grouped_data_dict,        clustering_results_significant,        significant_features  # Passaggio delle feature significative    )    # Step 6: Aggiungi la colonna delle categorie di incremento    final_dataset_with_categories = analyzer.add_increment_category(final_dataset_with_increments)    # Step 7: Salva il dataset finale con la colonna delle categorie    final_dataset_with_categories.to_csv('final_dataset_with_categories.csv', index=False)    # Visualizza il dataset finale    print("Dataset finale con incremento dei cluster e categorie:")    print(final_dataset_with_categories)    # Mostra i valori unici della feature 'cluster_increment'    cluster_increment_values = final_dataset_with_increments['incremento numerico'].unique()    # Visualizza i valori unici    print("Valori unici della feature 'cluster_increment':")    print(cluster_increment_values)    dataframe_final = reducer.remove_insignificant_columns(final_dataset_with_increments, ['incremento numerico'])    # dataframe_final = final_dataset_with_increments.copy()    # Definizione della griglia di iperparametri per il tuning    param_grid = {        'n_clusters': [5]  # numero di cluster che vogliamo testare    }    print(final_dataset_with_increments.info())    # Dizionario per mappare le stringhe ai numeri    mapping = {        'CONSTANT': 0,        'LOW': 1,        'MEDIUM': 2,        'HIGH': 3    }    # Applicazione del mapping alla colonna 'incremento'    dataframe_final['incremento'] = dataframe_final['incremento'].map(mapping)    # Estrai la colonna "incremento" da final_dataset_with_increments e crea un array numpy 1D    labels = dataframe_final['incremento'].to_numpy()    print("effettuo feature scaling per la variabile incremento")    dataframe_final = scaler.standardize_(dataframe_final, ['incremento'])    dataframe_for_silhouette = dataframe_final.copy()    # do più peso alla feature incremento    coefficiente = 3    dataframe_final['incremento'] = dataframe_final['incremento'] * coefficiente    """dataframe_final['age'] = dataframe_final['age'] * coefficiente    dataframe_final['sesso_female'] = dataframe_final['sesso_female'] * coefficiente"""    """# do più peso ad età e durata    coefficiente2 = 5    dataframe_final['age'] = dataframe_final['age'] * coefficiente2    dataframe_final['duration_minutes'] = dataframe_final['duration_minutes'] * coefficiente2"""    print("Istanzio classe HyperparameterTuner..")    # Istanzia il tuner per KMeans    tuner = HyperparameterTuner(algorithm='kmeans', param_grid=param_grid, data=dataframe_final, labels=labels)    print("Inizio la ricerca dell'ottimo..")    print("il dataframe finale è: ", dataframe_final)    print(f"i valori di incremento standardizzati sono {dataframe_final['incremento'].unique()}")    # Eseguiamo il Grid Search per trovare la combinazione migliore di iperparametri    results = tuner.perform_grid_search()    print("Grid search completato..")    # Ora estrai i singoli valori dal dizionario `results`    best_params = results['best_params']    best_score = results['best_score']    best_labels = results['best_labels']    best_centroids = results['best_centroids']    print("ora istanzio l'analizzatore..")    # removing records where 'year' = 2019    dataframe_for_visuals = dataframe_for_visuals[dataframe_for_visuals['year'] != 2019]    # adding cluster column to the dataframe and dataframe_for_silhouette    dataframe_for_visuals['cluster'] = best_labels    dataframe_for_silhouette['cluster'] = best_labels    # adding column 'incremento' to the dataframe    dataframe_for_visuals['incremento'] = labels    print(best_labels)    # sending the dataframe to a csv file    dataframe_for_visuals.to_csv('final_dataset_with_clusters.csv', index=False)    print("ora visualizzo i grafici..")    # visualizzazione dei grafici    visualizer = Visualizer()    list_of_cols = ['duration_minutes', 'age', 'incremento', 'cluster']    visualizer.scatter_plots(dataframe_for_visuals, list_of_cols)    # POST PROCESSING    print("creo il file json per le statistiche su ogni cluster..")    post_processor = PostProcessing(dataframe_for_visuals, dataframe_for_silhouette, best_params, best_score, best_centroids)    post_processor.process_clusters()    post_processor.save_stats_to_json('stats_by_cluster.json')