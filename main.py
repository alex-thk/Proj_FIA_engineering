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
