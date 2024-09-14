import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:

    def __init__(self):
        pass

    @staticmethod
    def scatter_plots(df: pd.DataFrame, list_of_cols: list):

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
        unique_clusters = sorted(df[list_of_cols[3]].unique())
        color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}  # dict comprehension
        print(color_map)

        # 3D Scatter Plot
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.scatter(df[list_of_cols[0]], df[list_of_cols[1]], df[list_of_cols[2]],
                                   c=df[list_of_cols[3]].map(color_map), marker='o')
        ax_3d.set_title('3D Scatter plot')
        ax_3d.set_xlabel(list_of_cols[0])
        ax_3d.set_ylabel(list_of_cols[1])
        ax_3d.set_zlabel(list_of_cols[2])

        # Create a legend for 3D plot
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cluster], markersize=10) for
                   cluster in unique_clusters]
        legend_labels = [f'Cluster {int(cluster)}' for cluster in unique_clusters]
        ax_3d.legend(handles, legend_labels, title="Clusters")

        plt.tight_layout()
        plt.show()

        # 2D Scatter Plot with subplots for each 'incremento' value
        fig_2d, axs = plt.subplots(2, 2, figsize=(15, 12))
        axs = axs.flatten()
        inc_vals = df[list_of_cols[2]].unique()

        for i, incremento in enumerate(inc_vals.tolist()):
            df_filtered = df[df[list_of_cols[2]] == incremento]
            axs[i].scatter(df_filtered[list_of_cols[0]], df_filtered[list_of_cols[1]],
                                        c=df_filtered[list_of_cols[3]].map(color_map), marker='o')
            axs[i].set_title(f'2D Scatter plot for incremento = {incremento}')
            axs[i].set_xlabel(list_of_cols[0])
            axs[i].set_ylabel(list_of_cols[1])

            # Create a legend for each subplot
            axs[i].legend(handles, legend_labels, title="Clusters")

        plt.tight_layout()
        plt.show()
