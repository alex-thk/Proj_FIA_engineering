import pandas as pd
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes


class ClusteringModel:
    """
    Class for clustering data using KMeans or KPrototypes
    """

    def __init__(self, algorithm, n_clusters):
        self.algorithm = algorithm  # string that can be 'kmeans' or 'kprototypes'
        self.n_clusters = n_clusters  # number of clusters
        self.model = None  # KMeans or KPrototypes instance

        if self.algorithm == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42)  # instance of KMeans
        elif self.algorithm == 'kprototypes':
            self.model = KPrototypes(n_clusters=n_clusters, init='Huang')  # instance of KPrototypes
        else:
            raise ValueError("Algoritmo non supportato: scegli tra 'kmeans' o 'kprototypes'")

    def fit(self, x, categorical=None):
        """
        Fit the clustering model to the data
        :param x: dataframe that represents the data to cluster, with the variable incremento
        :param categorical:
        :return: numpy array with cluster labels (1D) with the same order of the input data
        """
        if self.algorithm == 'kmeans':
            if not isinstance(x, pd.DataFrame):
                raise ValueError("Per KMeans, X deve essere un DataFrame numerico")
            self.model.fit(x)
            return self.model.labels_

        elif self.algorithm == 'kprototypes':
            if not isinstance(x, pd.DataFrame):
                raise ValueError("Per KPrototypes, X deve essere un DataFrame")
            if categorical is None:
                raise ValueError("Per KPrototypes, devi specificare gli indici delle colonne categoriche")
            self.model.fit(x, categorical=categorical)
            return self.model.labels_

    def fit_opt(self, x, categorical=None):
        """
        Fit the clustering model to the data
        :param x: dataframe that represents the data to cluster
        :param categorical: indices of categorical columns (for KPrototypes)
        :return: dictionary with clustering results and additional info
        """
        results = {}

        if self.algorithm == 'kmeans':
            if not isinstance(x, pd.DataFrame):
                raise ValueError("Per KMeans, X deve essere un DataFrame numerico")

            self.model.fit(x)
            results['labels'] = self.model.labels_
            results['centroids'] = self.model.cluster_centers_
            results['data'] = x

        elif self.algorithm == 'kprototypes':
            if not isinstance(x, pd.DataFrame):
                raise ValueError("Per KPrototypes, X deve essere un DataFrame")
            if categorical is None:
                raise ValueError("Per KPrototypes, devi specificare gli indici delle colonne categoriche")

            self.model.fit(x, categorical=categorical)
            results['labels'] = self.model.labels_
            results['centroids'] = self.model.cluster_centers_
            results['data'] = x
            results['categorical'] = categorical

        return results


# Test del KMeans protetto da __name__
if __name__ == "__main__":
    # Creiamo dei dati di esempio per testare KMeans
    data = {
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [1.0, 1.5, 3.5, 4.5, 5.5]
    }
    df = pd.DataFrame(data)

    # Istanziare il modello KMeans
    kmeans_model = ClusteringModel(algorithm='kmeans', n_clusters=2)

    # Eseguire il fit sui dati e stampare i cluster risultanti
    labels = kmeans_model.fit(df)
    print("Cluster assegnati:", labels)
