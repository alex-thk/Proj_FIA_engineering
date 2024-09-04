import pandas as pd


class DataReduction:
    def __init__(self):
        pass

    @staticmethod
    def remove_insignificant_columns(df: pd.DataFrame, list_of_cols) -> pd.DataFrame:
        df.drop(columns=list_of_cols, inplace=True)
        return df
