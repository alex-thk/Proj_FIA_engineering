import pandas as pd
import numpy as np
import scipy.stats as stats
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


def calculate_precentage_missing_values_in_df(dataframe: pd.DataFrame) -> None:
    for column in dataframe.columns:
        missing_values = dataframe[column].isnull().sum()
        # NB shape[0] is the number of rows in the column
        total_values = dataframe[column].shape[0]
        percentage = (missing_values / total_values) * 100
        if percentage > 0:
            print(f'Percentage of missing values in {column} is {np.round(percentage, 2)}%')


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
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], utc=True, errors='coerce')

    df['duration'] = (df['ora_fine_erogazione'] - df['ora_inizio_erogazione'])
    df['quarter'] = df['data_erogazione'].dt.quarter
    df['year'] = df['data_erogazione'].dt.year

    """" Imputa i valori mancanti nella colonna "duration" con la media dei valori presenti
    mean_duration = df['duration'].mean()
    df['duration'].fillna(mean_duration, inplace=True)
    """""

    # mean and std of standard distribution of the duration column
    vector_distrib = df['duration'].dropna()
    mu, std = stats.norm.fit(vector_distrib.dt.total_seconds())
    # print(f'Mean: {mu}, Std: {std}')
    # creating a vector of random values with the same mean and std of the duration column
    # the size of the vector is equal to the number of missing values in the duration column
    random_durations = np.random.normal(loc=mu, scale=std, size=df['duration'].isnull().sum())
    # replacing the missing values with the random values
    df.loc[df['duration'].isnull(), 'duration'] = random_durations
    print(df[['duration', 'quarter', 'year']].head())

    # print(df.head())
    # now from the dataframe analyze each column and print out the percentage of missing values
    calculate_precentage_missing_values_in_df(df)
    print('------------------------------------')

    # 'ora_inizio_erogazione' and 'ora_fine_erogazione' columns are not useful anymore
    # since now we have the 'duration' column
    df.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)  # inplace=True to modify the original dataframe

    # removing all records with missing values in the column 'comune_residenza'
    # a 0.03% of the total records will be removed (not a big deal)
    df.dropna(subset=['comune_residenza'], inplace=True)

    calculate_precentage_missing_values_in_df(df)
    print('------------------------------------')

    # 'codice_provincia_erogazione' has around 6% missing values, 'provincia_erogazione'  doesn't
    # so we can easly find a correlation between the 2 columns and fill the missing values
    missing_codice = df['codice_provincia_erogazione'].isnull()
    corresponding_provincia = df.loc[missing_codice, 'provincia_erogazione']
    # print(corresponding_provincia.unique())
    df['codice_provincia_erogazione'].fillna('NA', inplace=True)

    # doing the same thing for 'codice_provincia_residenza' and 'provincia_residenza'
    missing_codice2 = df['codice_provincia_residenza'].isnull()
    corresponding_provincia2 = df.loc[missing_codice2, 'provincia_residenza']
    # print(corresponding_provincia2.unique())
    df['codice_provincia_residenza'].fillna('NA', inplace=True)

    calculate_precentage_missing_values_in_df(df)
    print('------------------------------------')


if __name__ == '__main__':
    data_cleaning(file)
