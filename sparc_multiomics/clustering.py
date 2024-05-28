import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage


def hierarchical_clustering(input_data, number_of_clusters=3, show_plot=False):
    """
    Perform hierarchical clustering on the input data and return the cluster labels.
    :param input_data: The input data to cluster
    :param number_of_clusters: The number of clusters to form, default is 3
    :param show_plot: If True, show the dendrogram plot
    :return: The cluster labels
    """
    Z = linkage(input_data, method="ward")

    if show_plot:
        plt.figure(figsize=(12, 6))
        dendrogram(Z)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.xticks([])
        plt.ylabel("Distance")
        plt.show()

    return fcluster(Z, t=number_of_clusters, criterion="maxclust")
