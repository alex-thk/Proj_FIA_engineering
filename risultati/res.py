# questo script serve per calcolare alcune statistiche sul dataset
# e per generare i grafici relativi ai risultati ottenuti
import pandas as pd
import json

from data_reduction import DataReduction
from data_cleaning import DataCleaning
from data_transformation import DataTransformation
from data_standardization import FeatureScaler

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
df = reducer.remove_insignificant_columns(df, ['comune_residenza', 'ora_inizio_erogazione', 'ora_fine_erogazione',
                                               'id_prenotazione', 'id_paziente', 'codice_regione_residenza',
                                               'codice_asl_residenza', 'codice_provincia_residenza',
                                               'codice_comune_residenza', 'tipologia_servizio', 'data_contatto',
                                               'codice_regione_erogazione', 'codice_asl_erogazione',
                                               'codice_provincia_erogazione', 'struttura_erogazione',
                                               'codice_struttura_erogazione', 'id_professionista_sanitario',
                                               'tipologia_professionista_sanitario', 'data_nascita',
                                               'asl_residenza', 'provincia_residenza', 'codice_descrizione_attivita',
                                               'asl_erogazione', 'provincia_erogazione', 'tipologia_struttura_erogazione',
                                               'duration' ])

cleaner.calculate_precentage_missing_values_in_df(df)  # should print no missing values

transformer = DataTransformation()
region_mapping = {
    # per creare le dummie per ogni regione
    "Valle d`aosta": 'Valle daosta',
    "Valle d'aosta": 'Valle daosta',
    "Prov. auton. bolzano": 'Trentino alto adige',
    "Prov. auton. trento": 'Trentino alto adige',
    "Piemonte": "Piemonte",
    "Lombardia": "Lombardia",
    "Veneto": "Veneto",
    "Friuli venezia giulia": "Friuli venezia giulia",
    "Liguria": "Liguria",
    "Emilia romagna": "Emilia romagna",
    "Toscana": "Toscana",
    "Umbria": "Umbria",
    "Marche": "Marche",
    "Lazio": "Lazio",
    "Abruzzo": "Abruzzo",
    "Molise": "Molise",
    "Campania": "Campania",
    "Puglia": "Puglia",
    "Basilicata": "Basilicata",
    "Calabria": "Calabria",
    "Sicilia": "Sicilia",
    "Sardegna": "Sardegna"
}

df = transformer.transform_col(df, 'regione_residenza', region_mapping, 'residenza')
df = reducer.remove_insignificant_columns(df, ['regione_residenza', 'regione_residenza_tuple', 'data_disdetta'])

print(df.info())
print(df.columns)

# Initialize a dictionary to store the results
results = {}

# Iterate over each unique year
for year in sorted(df['year'].unique()):
    df_year = df[df['year'] == year]

    # Sum of same 'residenza' value
    residenza_sum = df_year['residenza'].value_counts().to_dict()

    # Sum of how many 'sesso' (male and female)
    sesso_sum = df_year['sesso'].value_counts().to_dict()

    # Sum of 'codice_tipologia_professionista_sanitario'
    codice_tipologia_sum = df_year['codice_tipologia_professionista_sanitario'].value_counts().to_dict()

    # Calculate the average duration in minutes and average age
    duration_avg = df_year['duration_minutes'].mean()  # Assuming 'duration_minutes' is the column for duration
    age_avg = df_year['age'].mean()  # Assuming 'age' is the column for age

    # Store the results in the dictionary
    results[year] = {
        'residenza_sum': residenza_sum,
        'sesso_sum': sesso_sum,
        'codice_tipologia_sum': codice_tipologia_sum,
        'duration_avg': duration_avg,  # Add average duration
        'age_avg': age_avg  # Add average age
    }
# Convert the keys to strings before dumping to JSON
results_str_keys = {str(year): data for year, data in results.items()}

# Write the results to a JSON file
output_file_path = 'yearly_stats.json'
with open(output_file_path, 'w') as output_file:
    json.dump(results_str_keys, output_file, indent=4)

print(f'Yearly statistics written to {output_file_path}')
