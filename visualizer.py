import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:

    def __init__(self):
        pass

    @staticmethod
    def scatter_plots(df: pd.DataFrame, list_of_cols: list):

        # 3D Scatter Plot
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        scatter_3d = ax_3d.scatter(df[list_of_cols[0]], df[list_of_cols[1]], df[list_of_cols[2]],
                                   c=df[list_of_cols[3]], marker='o')
        ax_3d.set_title('3D Scatter plot')
        ax_3d.set_xlabel(list_of_cols[0])
        ax_3d.set_ylabel(list_of_cols[1])
        ax_3d.set_zlabel(list_of_cols[2])

        # Create a legend for 3D plot
        handles, labels = scatter_3d.legend_elements()
        legend_labels = [f'Cluster {int(label)}' for label in sorted(df[list_of_cols[3]].unique())]
        ax_3d.legend(handles, legend_labels, title="Clusters")

        plt.tight_layout()
        plt.show()

        # 2D Scatter Plot with subplots for each 'incremento' value
        fig_2d, axs = plt.subplots(2, 2, figsize=(15, 12))
        axs = axs.flatten()
        inc_vals = df[list_of_cols[2]].unique()

        for i, incremento in enumerate(inc_vals.tolist()):
            df_filtered = df[df[list_of_cols[2]] == incremento]
            scatter_2d = axs[i].scatter(df_filtered[list_of_cols[0]], df_filtered[list_of_cols[1]],
                                        c=df_filtered[list_of_cols[3]], marker='o')
            axs[i].set_title(f'2D Scatter plot for incremento = {incremento}')
            axs[i].set_xlabel(list_of_cols[0])
            axs[i].set_ylabel(list_of_cols[1])

            # Create a legend for each subplot
            handles, labels = scatter_2d.legend_elements()
            legend_labels = [f'Cluster {int(label)}' for label in sorted(df[list_of_cols[3]].unique())]
            axs[i].legend(handles, legend_labels, title="Clusters")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    df = pd.read_csv('final_dataset_with_clusters.csv')
    list_of_cols = ['duration_minutes', 'age', 'incremento', 'cluster']
    Visualizer.scatter_plots(df, list_of_cols)