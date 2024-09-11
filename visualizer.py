import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics

class Visualizer:

    def __init__(self):
        pass

    # NB for the scatter plots
    # list_of_cols = [x, y, z, color] in the case of 3d
    # list_of_cols = [x, y, incremento, cluster] in the case of 2d

    @staticmethod
    def scatter_plot_3d(df: pd.DataFrame, list_of_cols: list):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(df[list_of_cols[0]], df[list_of_cols[1]], df[list_of_cols[2]], c=df[list_of_cols[3]], marker='o')

        ax.set_title('3D Scatter plot')
        ax.set_xlabel(list_of_cols[0])
        ax.set_ylabel(list_of_cols[1])
        ax.set_zlabel(list_of_cols[2])

        # Creating a legend
        handles, labels = scatter.legend_elements()
        legend_labels = [f'Cluster {int(label)}' for label in df[list_of_cols[3]].unique()]
        ax.legend(handles, legend_labels, title="Clusters")

        plt.show()

    @staticmethod
    def scatter_plot_2d(df: pd.DataFrame, list_of_cols: list):
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        axs = axs.flatten()
        inc_vals = df[list_of_cols[2]].unique()
        print(inc_vals)

        for i, incremento in enumerate(inc_vals.tolist()):
            # only including the rows where the incremento is equal to the current incremento (inc_vals)
            df_filtered = df[df[list_of_cols[2]] == incremento]
            scatter = axs[i].scatter(df_filtered[list_of_cols[0]], df_filtered[list_of_cols[1]],
                                     c=df_filtered[list_of_cols[3]], marker='o')
            axs[i].set_title(f'2D Scatter plot for incremento = {incremento}')
            axs[i].set_xlabel(list_of_cols[0])
            axs[i].set_ylabel(list_of_cols[1])

            # Creating a legend
            handles, labels = scatter.legend_elements()
            legend_labels = [f'Cluster {int(label)}' for label in df[list_of_cols[3]].unique()]
            axs[i].legend(handles, legend_labels, title="Clusters")

            """axs[i].set_xlim(-4, 4)
            axs[i].set_ylim(-4, 4)"""

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv('final_dataset_with_clusters.csv')
    list_of_cols = ['duration_minutes', 'age', 'incremento', 'cluster']
    Visualizer.scatter_plot_3d(df, list_of_cols)
    Visualizer.scatter_plot_2d(df, list_of_cols)