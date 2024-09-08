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

    @staticmethod
    def standardize(df: pd.DataFrame, list_of_cols: list) -> tuple[pd.DataFrame, dict, list]:
        """
        Standardize the columns in a DataFrame saving the mean and std for each column.
        """
        stats = {}  # Dizionario per salvare media e std
        for col in list_of_cols:
            mean = df[col].mean()
            std = df[col].std()
            stats[col] = {'mean': mean, 'std': std}  # Salva media e std
            df[col] = (df[col] - mean) / std
        return df, stats, list_of_cols

    @staticmethod
    def inverse_standardize_centroids(best_centroids: np.ndarray, stats: dict, list_of_cols: list) -> np.ndarray:
        original_centroids = np.copy(best_centroids)  # Crea una copia dei centroidi per non modificarli direttamente
        for i, col in enumerate(list_of_cols):
            mean = stats[col]['mean']
            std = stats[col]['std']
            original_centroids[:, i] = (best_centroids[:, i] * std) + mean  # Applica l'operazione inversa
        return original_centroids
