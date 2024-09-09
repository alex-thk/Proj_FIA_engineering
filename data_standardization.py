import numpy as np
import pandas as pd


class FeatureScaler:

    def __init__(self):
        pass

    @staticmethod
    def standardize_(df: pd.DataFrame, list_of_cols: list):
        """
        Standardize the columns in a DataFrame.
        """
        for col in list_of_cols:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
        return df
