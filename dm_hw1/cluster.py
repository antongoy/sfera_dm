import csv
import argparse
import sys
import itertools

import numpy as np
import pylab as pl


def parse():
    """
    Parse args
    :return: namespace with parse args
    """
    my_parser = argparse.ArgumentParser(prog="cluster", description="Hierarichical clustering")
    my_parser.add_argument("-d", choices=["max", "error"], nargs=1,
                           default="error", dest="type_criterion", help='choice criterion type')
    return my_parser.parse_args()


def vectors_normalization(data_set):
    """
    Perform the normalization of all objects
    :param data_set: the set of objects
    """
    (n_vectors, n_coord) = data_set.shape

    for i in range(n_coord):
        max_in_col = data_set[:, i].max()
        data_set[:, i] = data_set[:, i] / float(max_in_col)


def cluster_median(data_set, cluster):
    """
    Get cluster median
    :param data_set: the set of objects
    :param cluster: dictionary of cluster
    """
    (n_objects, n_attributes) = data_set.shape

    median = np.zeros(n_attributes)

    n_cur_cluster = len(cluster)

    for i in range(n_cur_cluster):
        median = median + data_set[cluster[i]]

    return (1.0 / n_attributes) * median


def error_criterion(data_set, clusters, cl1, cl2):
    """
    Compute d_e criterion
    :param data_set: the set of objects
    :param clusters: dictionary of clusters
    :param cl1: index of the first cluster
    :param cl2: index of the second cluster
    """
    median1 = cluster_median(data_set, clusters[cl1])
    median2 = cluster_median(data_set, clusters[cl2])

    n1 = float(len(clusters[cl1]))
    n2 = float(len(clusters[cl2]))

    return ((float(n1 * n2) / float(n1 + n2)) ** 0.5) * np.linalg.norm(median1 - median2)


def max_criterion(data_set, clusters, cl1, cl2):
    n_cluster1 = len(clusters[cl1])
    n_cluster2 = len(clusters[cl2])

    d_max = -sys.maxint - 1

    for i in range(n_cluster1):
        for j in range(i, n_cluster2):
            dist = np.linalg.norm(data_set[i] - data_set[j])
            if dist > d_max:
                d_max = dist

    return d_max


def find_best_clusters(data_set, clusters, distances, criterion_function):
    """
    This function find two best clusters for the merging
    :param data_set: set of objects
    :param clusters: list of clusters
    :param distances: distances between clusters
    :return: two best clusters
    """
    n_cur_clusters = len(clusters)
    best_pair = (-1, -1)
    best_value = sys.maxint

    for i, j in itertools.combinations(range(n_cur_clusters), 2):
        if distances[i][j] == -1:
            distances[i][j] = criterion_function(data_set, clusters, i, j)
        if distances[i][j] < best_value:
            best_pair = (i, j)
            best_value = distances[i][j]

    return best_pair


def merge_clusters(cluster1, cluster2, clusters, distances):
    """
    Merge cluster1 and cluster2
    :param cluster1: first cluster
    :param cluster2: second cluster
    :param clusters:
    :param distances:
    """
    clusters[cluster1].extend(clusters[cluster2])
    clusters.pop(cluster2)

    n_cur_clusters = len(clusters)

    for i in range(n_cur_clusters):
        distances[i].pop(cluster2)
        distances[cluster1][i] = -1


def hierarichical_clustering(data_set, n_clusters, criterion):
    """
    My clustering
    :param data_set: set of objects
    :param n_clusters: the needed number of clusters
    """
    n_objects = len(data_set)
    n_cur_clusters = n_objects

    distances = []
    for i in range(n_cur_clusters):
        distances.append([])
        for j in range(n_cur_clusters):
            distances[i].append(-1)

    print "Initialization of clusters..."

    clusters = []
    for i in range(n_cur_clusters):
        clusters.append([])
        clusters[i].append(i)

    if criterion == 'max':
        print "Select max_criterion..."
        criterion_function = globals()['max_criterion']
    else:
        print "Select error_criterion..."
        criterion_function = globals()['error_criterion']

    print "Starting main loop..."

    while n_cur_clusters > n_clusters:
        (cluster1, cluster2) = find_best_clusters(data_set, clusters, distances, criterion_function)
        merge_clusters(cluster1, cluster2, clusters, distances)
        n_cur_clusters -= 1

    return clusters


def cluster_diameter(data_set, cluster):
    n_cluster = len(cluster)
    diameter = -sys.maxint - 1

    if n_cluster == 1:
        return 0

    for i in range(n_cluster):
        for j in range(i + 1, n_cluster):
            dist = np.linalg.norm(data_set[i] - data_set[j])
            if dist > diameter:
                diameter = dist

    return diameter


def quality_of_clustering(data_set, clusters):
    n_clusters = len(clusters)
    avg = 0.

    for cluster in clusters:
        avg += cluster_diameter(data_set, cluster)

    avg /= n_clusters
    return avg


def rand_index(data_set, benchmark_clusters, out_clusters):
    a = 0
    b = 0
    n = data_set.shape[0]

    for x, y in itertools.combinations(range(n), 2):
        for i, cluster in enumerate(out_clusters):
            if x in cluster:
                out_cluster_x = i
            if y in cluster:
                out_cluster_y = i
        for i, cluster in enumerate(benchmark_clusters):
            if x in cluster:
                bench_cluster_x = i
            if y in cluster:
                bench_cluster_y = i
        if out_cluster_x == out_cluster_y and bench_cluster_x == bench_cluster_y:
            a += 1

        if out_cluster_x != out_cluster_y and bench_cluster_x != bench_cluster_y:
            b += 1

    return float(a + b) / ((n * (n-1)) / 2)


def main():
    """
    Main function

    """
    args = parse()

    print "## Welcome to the realization of Hierarchical Clustering ##\n"

    with open('data_set.csv', 'rb') as data_file:
        print "Open data_set.csv file..."
        print "Read from data_set.csv..."

        reader = csv.reader(data_file)
        flag_start_file = 0
        first_column = 0
        data = []

        for row in reader:
            if flag_start_file == 0:
                flag_start_file = 1
                continue

            row.pop(first_column)
            data.append(row)

        data_file.close()

        print "Get benchmark clustering..."

        data_set = np.array(data, dtype=np.float)

        benchmark_clusters = np.array(get_benchmark_clustering(data_set))

        data_set = np.delete(data_set, 0, axis=1)
        
        qualities = np.zeros(9)
        rand_indexes = np.zeros(9)
        for i in range(1, 10):
            print "Run clustering for", i, "clusters..."
            output_clusters = hierarichical_clustering(data_set, i, args.type_criterion[0])
            qualities[i-1] = quality_of_clustering(data_set, output_clusters)
            rand_indexes[i-1] = rand_index(data_set, benchmark_clusters, output_clusters)

        x = np.arange(9) + 1
        qualities = np.log(qualities)

        pl.subplot(121)
        pl.plot(x, qualities, 'r*', lw=1, ls='-')
        pl.title("Average diameter of clusters (log)")
        pl.xlabel("Number of clusters")
        pl.ylabel("Avg diameters")

        pl.subplot(122)
        pl.plot(x, rand_indexes, 'g*', lw=1, ls='-')
        pl.title("Rand Index")
        pl.xlabel("Number of clusters")
        pl.ylabel("Rand indexes")

        pl.show()


def get_benchmark_clustering(data_set):
    """
    Get benchmark clusterng
    :param data_set:
    :type data_set: list(list)
    """
    clusters = []
    regions = []

    for obj in data_set:
        if not (obj[0] in regions):
            regions.append(obj[0])

    for i in range(len(regions)):
        clusters.append([])
    i = 0
    for obj in data_set:
        clusters[int(obj[0]) - 1].append(i)
        i += 1

    return clusters


if __name__ == '__main__':
    main()