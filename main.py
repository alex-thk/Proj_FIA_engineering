import pandas as pd

from data_reduction import DataReduction
from data_cleaning import DataCleaning
from data_transformation import DataTransformation

if __name__ == '__main__':
    file = 'challenge_campus_biomedico_2024.parquet'
    df = pd.read_parquet(file)

    cleaner = DataCleaning(df)
    cleaner.calculate_precentage_missing_values_in_df()
    cleaner.handle_last_column()
    cleaner.add_relevant_columns()
    cleaner.impute_missing_values()
    cleaner.calculate_precentage_missing_values_in_df()
    cleaner.show_head()

    reducer = DataReduction(df)
    reducer.remove_insignificant_columns()
    cleaner.calculate_precentage_missing_values_in_df() # should print no missing values
    print(df.info())

    # per fare 'nord sud centro' toglieremo
    # 'regione_residenza, asl_residenza, provincia_residenza, comune_residenza'
    print(df['codice_tipologia_professionista_sanitario'].unique())

    # assistenza sanitaria diretta 1
    # infermiere, ostetrica, fisioterapista, podologo

    # assistenza riabilitativa e terapeutica psico-sociale 2
    # psicologo, tecnico riabilitazione psichiatrica, terapista occupazionale, terapista della neuro e psicomotricita dell'eta evolutiva

    # supporto salute e benessere 3
    # dietista, educatore professionale, assistente sanitario, logopedista

    # mapping for codice_tipologia_professionista_sanitario
    mapping1 = {
        'INF': 1, 'OST': 1, 'FIS': 1, 'POD': 1,
        'PSI': 2, 'TRP': 2, 'TRO': 2, 'TNP': 2,
        'DIE': 3, 'EDP': 3, 'ASN': 3, 'LPD': 3
    }

    transformer = DataTransformation(df)
    transformer.transform_col('codice_tipologia_professionista_sanitario', mapping1)

    region_mapping = {
        "Piemonte": 1, "Valle d'Aosta": 1, "Liguria": 1, "Lombardia": 1, "Trentino-Alto Adige": 1, "Veneto": 1, "Friuli venezia giulia": 1, "Emilia Romagna": 1,
        "Toscana": 2, "Umbria": 2, "Marche": 2, "Lazio": 2,
        "Abruzzo": 3, "Molise": 3, "Campania": 3, "Puglia": 3, "Basilicata": 3, "Calabria": 3, "Sicilia": 3, "Sardegna": 3
    }

    transformer.transform_col('regione_residenza', region_mapping)
    transformer.transform_col('regione_erogazione', region_mapping)