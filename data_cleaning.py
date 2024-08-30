import pandas as pd
import numpy as np
import scipy.stats as stats


class DataCleaning:
    def __init__(self, file):
        self.file = file
        self.df = pd.read_parquet(file)

    def show_head(self):
        print(self.df.head())
        print(self.df[['duration', 'quarter', 'year']].head())

    def calculate_precentage_missing_values_in_df(self):
        print(f'if no missing values are printed, then there are no missing values in the dataset')
        for column in self.df.columns:
            missing_values = self.df[column].isnull().sum()
            total_values = self.df[column].shape[0]
            percentage = (missing_values / total_values) * 100
            if percentage > 0:
                print(f'Percentage of missing values in {column} is {np.round(percentage, 2)}%')
        print(f'DONE!')

    def handle_last_column(self):
        # Substitute "data_disdetta" with boolean values
        self.df['data_disdetta'] = self.df['data_disdetta'].notnull()

    def remove_insignificant_columns(self):
        # removing all records with missing values in the column 'comune_residenza'
        # a 0.03% of the total records will be removed (not a big deal)
        self.df.dropna(subset=['comune_residenza'], inplace=True)

        # 'ora_inizio_erogazione' and 'ora_fine_erogazione' columns are not useful anymore
        # since now we have the 'duration' column
        self.df.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)  # inplace=True to modify the original dataframe

    def add_relevant_columns(self):
        # adding relevant colmuns to the dataframe
        # converting columns to datetime format making sure to handle different time zones
        self.df['ora_inizio_erogazione'] = pd.to_datetime(self.df['ora_inizio_erogazione'], utc=True, errors='coerce')
        self.df['ora_fine_erogazione'] = pd.to_datetime(self.df['ora_fine_erogazione'], utc=True, errors='coerce')
        self.df['data_erogazione'] = pd.to_datetime(self.df['data_erogazione'], utc=True, errors='coerce')

        self.df['duration'] = (self.df['ora_fine_erogazione'] - self.df['ora_inizio_erogazione'])
        self.df['quarter'] = self.df['data_erogazione'].dt.quarter
        self.df['year'] = self.df['data_erogazione'].dt.year

    def impute_missing_values(self):

        # ----- duration column -----
        # mean and std of standard distribution of the duration column
        vector_distrib = self.df['duration'].dropna()
        mu, std = stats.norm.fit(vector_distrib.dt.total_seconds())
        # print(f'Mean: {mu}, Std: {std}')
        # creating a vector of random values with the same mean and std of the duration column
        # the size of the vector is equal to the number of missing values in the duration column
        random_durations = np.random.normal(loc=mu, scale=std, size=self.df['duration'].isnull().sum())
        # replacing the missing values with the random values
        self.df.loc[self.df['duration'].isnull(), 'duration'] = random_durations

        # ----- codice_provincia_residenza column -----
        missing_codice = self.df['codice_provincia_erogazione'].isnull()
        corresponding_provincia = self.df.loc[missing_codice, 'provincia_erogazione']
        # print(corresponding_provincia.unique())
        self.df['codice_provincia_erogazione'].fillna('NA', inplace=True)

        # ----- codice_provincia_erogazione column -----
        missing_codice2 = self.df['codice_provincia_residenza'].isnull()
        corresponding_provincia2 = self.df.loc[missing_codice2, 'provincia_residenza']
        # print(corresponding_provincia2.unique())
        self.df['codice_provincia_residenza'].fillna('NA', inplace=True)
