import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm


class outlier_detector():
    def __init__(self):
        pass

    @staticmethod
    def detect_and_drop_outliers(df: pd.DataFrame, column, lb=-np.inf, ub=np.inf) -> pd.DataFrame:
        filtered_df = df[(df[column] >= lb) & (df[column] <= ub)]
        return filtered_df
