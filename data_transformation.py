import pandas as pd


class DataTransformation:
    def __init__(self, df):
        self.df = df

    def transform_col(self, column, mapping, new_col_name):
        self.df[column + '_tuple'] = list(zip(self.df[column], self.df[column].map(mapping)))
        self.df[column] = self.df[column + '_tuple']
        # lambda function extracts the second element of the tuple and apply applies it to the column
        self.df[new_col_name] = self.df[column].apply(lambda x: x[1])

        print(self.df[[column, new_col_name]].head())

    def create_dummies(self, column):
        dummies = pd.get_dummies(self.df[column], prefix=column)
        self.df = pd.concat([self.df, dummies], axis=1)
        self.df.drop(columns=[column], inplace=True)