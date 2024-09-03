from data_reduction import DataReduction
from data_cleaning import DataCleaning
from data_transformation import DataTransformation
from kmodes.kprototypes import KPrototypes

""""
POTENZIALI FEATURES INUTILI 
id_prenotazione 
id_paziente 
codice_regione_residenza 
codice_asl_residenza
codice_provincia_residenza 
codice_comune_residenza
tipologia_servizio  
data_contatto
codice_regione_erogazione
codice_asl_erogazione
codice_provincia_erogazione
struttura_erogazione
codice_struttura_erogazione
codice_tipologia_struttura_erogazione
id_professionista_sanitario
codice_tipologia_professionista_sanitario
"""

if __name__ == '__main__':
    file = 'challenge_campus_biomedico_2024.parquet'
    cleaner = DataCleaning(file)
    cleaner. calculate_precentage_missing_values_in_df()
    cleaner.handle_last_column()
    cleaner.add_relevant_columns()
    cleaner.impute_missing_values()
    cleaner.remove_insignificant_columns()
    cleaner.calculate_precentage_missing_values_in_df()  # should print no missing values
    cleaner.show_head()
    cleaner.show_info()


