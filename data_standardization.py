import numpy as np
import pandas as pd


class FeatureScaler:

    def __init__(self):
        pass

    @staticmethod
    def standardize(df: pd.DataFrame, list_of_cols: list) -> pd.DataFrame:
        for col in list_of_cols:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df
