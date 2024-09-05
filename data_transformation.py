import pandas as pd


class DataTransformation:
    def __init__(self):
        pass

    @staticmethod
    def transform_col(df: pd.DataFrame, column, mapping, new_col_name) -> pd.DataFrame:
        df[column + '_tuple'] = list(zip(df[column], df[column].map(mapping)))
        df[column] = df[column + '_tuple']
        # lambda function extracts the second element of the tuple and apply applies it to the column
        df[new_col_name] = df[column].apply(lambda x: x[1])
        return df

    @staticmethod
    def create_dummies(df: pd.DataFrame, column) -> pd.DataFrame:
        df = pd.get_dummies(df, columns=[column])
        # print(.df.info())
        return df
