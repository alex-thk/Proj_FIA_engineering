import pandas as pd


class DataTransformation:
    def __init__(self, df):
        self.df = df

    def transform_col(self, column, mapping):
        self.df[column + '_tuple'] = list(zip(self.df[column], self.df[column].map(mapping)))
        self.df[column] = self.df[column + '_tuple']
        print(self.df[column].head())

    def create_dummies(self, column):
        dummies = pd.get_dummies(self.df[column], prefix=column)
        self.df = pd.concat([self.df, dummies], axis=1)
        self.df.drop(columns=[column], inplace=True)