import pandas as pd
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
    df = pd.read_parquet(file)
    # print(df.head())
    # Substitute "data_disdetta" with boolean values
    df['data_disdetta'] = df['data_disdetta'].notnull()

    # adding relevant colmuns to the dataframe
    # converting columns to datetime format making sure to handle different time zones
    df['ora_inizio_erogazione'] = pd.to_datetime(df['ora_inizio_erogazione'], utc=True, errors='coerce')
    df['ora_fine_erogazione'] = pd.to_datetime(df['ora_fine_erogazione'], utc=True, errors='coerce')

    df['duration'] = (df['ora_fine_erogazione'] - df['ora_inizio_erogazione'])
    df['quarter'] = df['ora_inizio_erogazione'].dt.quarter
    df['year'] = df['ora_inizio_erogazione'].dt.year
    print(df[['ora_inizio_erogazione', 'duration', 'quarter', 'year']].head())

    # print(df.head())
    # now from the dataframe analyze each column and print out the percentage of missing values
    for column in df.columns:
        missing_values = df[column].isnull().sum()
        # NB shape[0] is the number of rows in the column
        total_values = df[column].shape[0]
        percentage = (missing_values / total_values) * 100
        print(f'Percentage of missing values in {column} is {np.round(percentage, 2)}%')


if __name__ == '__main__':
    data_cleaning(file)