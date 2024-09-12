import pandas as pd
import numpy as np
import scipy.stats as stats

"""
nvidia-smi --> to check driver and CUDA version 

"""


class DataCleaning:
    def __init__(self):
        pass

    @staticmethod
    def show_head(df: pd.DataFrame) -> pd.DataFrame:
        print(df.head())
        print(df[['duration', 'semester', 'year', 'age']].head())
        print(df.info())
        return df

    @staticmethod
    def calculate_precentage_missing_values_in_df(df: pd.DataFrame):
        print(f'if no missing values are printed, then there are no missing values in the dataset')
        for column in df.columns:
            missing_values = df[column].isnull().sum()
            total_values = df[column].shape[0]
            percentage = (missing_values / total_values) * 100
            if percentage > 0:
                print(f'Percentage of missing values in {column} is {np.round(percentage, 2)}%')
        print(f'DONE!')

    @staticmethod
    def handle_last_column(df: pd.DataFrame) -> pd.DataFrame:
        # Substitute "data_disdetta" with boolean values
        df['data_disdetta'] = df['data_disdetta'].notnull()
        return df

    @staticmethod
    def add_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # adding relevant colmuns to the dataframe
        # converting columns to datetime format making sure to handle different time zones
        df['ora_inizio_erogazione'] = pd.to_datetime(df['ora_inizio_erogazione'], utc=True, errors='coerce')
        df['ora_fine_erogazione'] = pd.to_datetime(df['ora_fine_erogazione'], utc=True, errors='coerce')
        df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], utc=True, errors='coerce')
        df['data_nascita'] = pd.to_datetime(df['data_nascita'], utc=True, errors='coerce')

        df['duration'] = (df['ora_fine_erogazione'] - df['ora_inizio_erogazione'])
        # Keep only rows where the duration is a valid Timedelta
        df = df[df['duration'].apply(lambda x: isinstance(x, pd.Timedelta))]
        # turning the duration column into minutes
        df['duration_minutes'] = df['duration'].apply(lambda x: x.total_seconds() / 60)

        # NB QUARTER IS 3 MONTHS PERIOD

        df['semester'] = (df['data_erogazione'].dt.month - 1) // 6 + 1
        print(df['semester'].unique())
        df['year'] = df['data_erogazione'].dt.year
        df['age'] = (df['data_erogazione'] - df['data_nascita'])
        df['age'] = np.floor(df['age'].dt.days / 365)
        return df

    @staticmethod
    def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:

        # ----- duration column -----
        # mean and std of standard distribution of the duration column
        vector_distrib = df['duration'].dropna()
        mu, std = stats.norm.fit(vector_distrib.dt.total_seconds())
        # print(f'Mean: {mu}, Std: {std}')
        # creating a vector of random values with the same mean and std of the duration column
        # the size of the vector is equal to the number of missing values in the duration column
        random_durations = np.random.normal(loc=mu, scale=std, size=df['duration'].isnull().sum())
        random_durations = pd.to_timedelta(random_durations, unit='s')  # Explicitly cast to timedelta64[ns]
        # replacing the missing values with the random values
        df.loc[df['duration'].isnull(), 'duration'] = random_durations

        # ----- codice_provincia_residenza column -----
        missing_codice = df['codice_provincia_erogazione'].isnull()
        corresponding_provincia = df.loc[missing_codice, 'provincia_erogazione']
        # print(corresponding_provincia.unique())
        df.loc[:, 'codice_provincia_erogazione'] = df['codice_provincia_erogazione'].fillna('NA')

        # ----- codice_provincia_erogazione column -----
        missing_codice2 = df['codice_provincia_residenza'].isnull()
        corresponding_provincia2 = df.loc[missing_codice2, 'provincia_residenza']
        # print(corresponding_provincia2.unique())
        df.loc[:, 'codice_provincia_residenza'] = df['codice_provincia_residenza'].fillna('NA')

        return df

    @staticmethod
    def handle_cancelled_appointments(df: pd.DataFrame) -> pd.DataFrame:
        # handling cancelled appointments
        # if the appointment has been cancelled, then the 'data_disdetta' column is True
        # otherwise the appointment was not cancelled
        # wherever data_disdetta is True duration must be 0
        df = df[df['data_disdetta'] == False]
        return df
