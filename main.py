from idlelib.pyparse import trans

import pandas as pd

from data_reduction import DataReduction
from data_cleaning import DataCleaning
from data_transformation import DataTransformation
from data_standardization import FeatureScaler

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
