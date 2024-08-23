import pandas
import numpy as np
from datetime import datetime
"""
----- DATA CLEANING -----
1) gestire i dati mancanti
2) smoothing dei rumori nei dati 
3) identificare/rimuovere outliers
4) rimuovere dati duplicati 

----- DATA TRANSFORMATION -----
1) normalizzazione e aggregazione dei dati
 
----- DATA REDUCTION -----
1) riduzione dei campioni 
2) rimuovere le colonne ridondanti
"""

# writing a function that takes as input the parquet file and return
# a percentage of missing values for each feature (column)
file = 'challenge_campus_biomedico_2024.parquet'

def data_cleaning(file):
    # GESTIONE ULTIMA COLONNA "data_disdetta"
    df = pandas.read_parquet(file)
    # print(df.head())
    # Substitute "data_disdetta" with boolean values
    df['data_disdetta'] = df['data_disdetta'].notnull()
    print(df.head())
    # now from the dataframe analyze each column and print out the percentage of missing values
    for column in df.columns:
        missing_values = df[column].isnull().sum()
        # NB shape[0] is the number of rows in the column
        total_values = df[column].shape[0]
        percentage = (missing_values/total_values) * 100
        # print(f'Percentage of missing values in {column} is {np.round(percentage, 2)}%')

    # GESTIONE DATI MANCANTI SU "ora_inizio_erogazione" e "ora_fine_erogazione"
    # converting columns to datetime format
    df = df.dropna(subset=['ora_inizio_erogazione', 'ora_fine_erogazione']) # dropping records with None entries in "ora_inizio_erogazione" and "ora_fine_erogazione"
    df['ora_inizio_erogazione'] = pandas.to_datetime(df['ora_inizio_erogazione'])
    df['ora_fine_erogazione'] = pandas.to_datetime(df['ora_fine_erogazione'])
    df['duration'] = (df['ora_fine_erogazione'] - df['ora_inizio_erogazione'])  # in days

    print(type(df['ora_inizio_erogazione'][0]))
    print(type(df['duration'][0]))
    print(df.head())

if __name__ == '__main__':
    data_cleaning(file)