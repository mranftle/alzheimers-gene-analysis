import csv
import math
import itertools
import numpy as np
import pandas as pd
from scipy.spatial import distance as pair_wise_distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
'''
CLEANING DATA INTO PROPER FORMAT
'''
all_patient_data = pd.read_csv("allPatients.csv")
classes = list(all_patient_data['Classes'])
all_patient_data = all_patient_data.drop('Classes', axis=1)
all_patient_data.replace(to_replace="?", value="NaN")
avg = all_patient_data.values.mean()
all_patient_data.replace(to_replace="NaN", value="avg")

clf = RandomForestClassifier(n_estimators=200, criterion='entropy')

clf = clf.fit(all_patient_data, classes)

genes_dict = dict()
for tree in clf.estimators_:
    i=0
    for feature in tree.feature_importances_:
        if feature > 0:
            if i in genes_dict.keys():
                genes_dict[i] = genes_dict[i] + 1
            else:
                genes_dict[i] = 1
        i=i+1

genes_dict_sorted = sorted(genes_dict, key=genes_dict.get, reverse=True)
most_frequent = genes_dict_sorted[:200]

filtered_genes = []

gene_names = list(all_patient_data.columns.values)
for i in most_frequent:
    filtered_genes.append(gene_names[i])

new_feature_set = pd.DataFrame([all_patient_data[x] for x in all_patient_data if x in filtered_genes]).T
new_feature_set_array= new_feature_set.values

# default euclidean distance
kmeanE2 = KMeans(n_clusters=2).fit(new_feature_set)
kmeanE3 = KMeans(n_clusters=3).fit(new_feature_set)
kmeanE4 = KMeans(n_clusters=4).fit(new_feature_set)

# use cosine similarity as distance measure
def cosine_similarity_as_distance_measure(X):
    return cosine_similarity(X)

KMeans.euclidean_distances = cosine_similarity_as_distance_measure
kmeanC2 = KMeans(n_clusters=2).fit(new_feature_set)
kmeanC3 = KMeans(n_clusters=3).fit(new_feature_set)
kmeanC4 = KMeans(n_clusters=4).fit(new_feature_set)

# general euclidian distance
def euclidian_distance_measure(a,b):
    return pair_wise_distance.euclidean(a,b)

# cosine sim
def cosine_similarity_measure(a,b):
    return 1 - pair_wise_distance.cosine(a, b)

#minimum distance between cluster points
def singleLink(cluster1, cluster2, dist_measure):
    min_dist = float("inf")
    for x in cluster1:
        for y in cluster2:
            dist = dist_measure(x,y)
            if dist < min_dist:
                min_dist = dist
    return min_dist

#maximum distance between cluster points
def completeLink(cluster1, cluster2, dist_measure):
    max_dist = float("-inf")
    for x in cluster1:
        for y in cluster2:
            dist = dist_measure(x,y)
            if dist > max_dist:
                max_dist = dist
    return max_dist

#average distance between cluster points
def averagePairDistance(cluster1, cluster2, dist_measure):
    totalSum = 0
    totalCount = 0
    for x in cluster1:
        for y in cluster2:
            totalSum += (math.ceil(dist_measure(x,y)*100)/100)
            totalCount += 1
    if totalCount == 0 or totalSum == 0:
        return 0
    print totalSum, totalCount, dist_measure.__name__
    return totalSum/totalCount

#centroid distance between cluster points
def centroidDistance(cluster1, cluster2, dist_measure):
    cent = 0
    for x in cluster1:
        for y in cluster2:
            cent+= dist_measure(x.mean(), y.mean())
    return cent

# generic distance function for getting distance between clusters
# params: n_clusters, dist_measure, dist_btw_func
def getDistBtwClusters(kmean, dist_measure, dist_between_func, feature_array):
    all_dists = {}
    num_clusters = kmean.get_params()['n_clusters']
    combos = list(itertools.combinations(range(num_clusters), 2))
    cluster_points = {i: feature_array[np.where(kmean.labels_ == i)] for i in range(kmean.n_clusters)}
    for c in combos:
        key = str(c[0]) + '-' + str(c[1]) + ',' + str(num_clusters) + ',' + dist_measure.__name__ + ',' + dist_between_func.__name__
        all_dists[key] = abs(dist_between_func(cluster_points[c[0]],cluster_points[c[1]], dist_measure))
    # print all_dists
    return all_dists

output = dict()

# k =2 euclidian
output.update(getDistBtwClusters(kmeanE2, euclidian_distance_measure, singleLink, new_feature_set_array))

output.update(getDistBtwClusters(kmeanE2, euclidian_distance_measure, completeLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanE2, euclidian_distance_measure, averagePairDistance, new_feature_set_array))
output.update(getDistBtwClusters(kmeanE2, euclidian_distance_measure, centroidDistance, new_feature_set_array))

# k = 3 euclidian
output.update(getDistBtwClusters(kmeanE3, euclidian_distance_measure, singleLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanE3, euclidian_distance_measure, completeLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanE3, euclidian_distance_measure, averagePairDistance, new_feature_set_array))
output.update(getDistBtwClusters(kmeanE3, euclidian_distance_measure, centroidDistance, new_feature_set_array))

# k = 4 euclidian
output.update(getDistBtwClusters(kmeanE4, euclidian_distance_measure, singleLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanE4, euclidian_distance_measure, completeLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanE4, euclidian_distance_measure, averagePairDistance, new_feature_set_array))
output.update(getDistBtwClusters(kmeanE4, euclidian_distance_measure, centroidDistance, new_feature_set_array))

# k =2 cosine_similarity
output.update(getDistBtwClusters(kmeanC2, cosine_similarity_measure, singleLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanC2, cosine_similarity_measure, completeLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanC2, cosine_similarity_measure, averagePairDistance, new_feature_set_array))
output.update(getDistBtwClusters(kmeanC2, cosine_similarity_measure, centroidDistance, new_feature_set_array))

# k = 3 cosine_similarity
output.update(getDistBtwClusters(kmeanC3, cosine_similarity_measure, singleLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanC3, cosine_similarity_measure, completeLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanC3, cosine_similarity_measure, averagePairDistance, new_feature_set_array))
output.update(getDistBtwClusters(kmeanC3, cosine_similarity_measure, centroidDistance, new_feature_set_array))

# k = 4 cosine_similarity
output.update(getDistBtwClusters(kmeanC4, cosine_similarity_measure, singleLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanC4, cosine_similarity_measure, completeLink, new_feature_set_array))
output.update(getDistBtwClusters(kmeanC4, cosine_similarity_measure, averagePairDistance, new_feature_set_array))
output.update(getDistBtwClusters(kmeanC4, cosine_similarity_measure, centroidDistance, new_feature_set_array))


# output to csv
with open('prob2results.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in output.items():
       writer.writerow([key, value])