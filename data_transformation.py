import pandas as pd


class DataTransformation:
    def __init__(self, file):
        self.file = file
        self.df = pd.read_parquet(file)

"""    def transform_string_column_to_number(self, column_name, tuple_col):
        self.df[tuple_col] = self.df[column_name].apply(lambda x: (x, abs(hash(x)) % (10 ** 8)))  # hashing the string
        self.df.insert(self.df.columns.get_loc(column_name) + 1, tuple_col, self.df.pop(tuple_col))
        print(self.df.info())
        print(self.df[[column_name, tuple_col]].head())
"""
