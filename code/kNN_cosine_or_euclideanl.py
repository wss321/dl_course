# -----------------------------------------
# kNN: k Nearest Neighbors  

# Input:      newInput: vector to compare to existing dataset (1xN)  
#             dataSet:  size m data set of known vectors (NxM)  
#             labels:   data set labels (1xM vector)  
#             k:        number of neighbors to use for comparison   

# Output:     the most popular class labels
# -----------------------------------------

from numpy import *
import math
import numpy as np


# # create a dataset which contains 4 samples with 2 classes
# def create_DataSet():
#     # create a matrix: each row as a sample
#     group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']  # four samples and two classes
#     return group, labels


def cosine_distance(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    v1_sq = np.inner(v1, v1)
    v2_sq = np.inner(v2, v2)
    dis = 1 - np.inner(v1, v2) / math.sqrt(v1_sq * v2_sq)
    return dis


def euclidean_distance(v1, v2):
    dis = np.sum(np.square(v1 - v2))
    dis = np.squeeze(dis)
    return dis


# classify using kNN
def kNNClassify(new_input, data_set, labels_of_dataset, k):
    # global distance
    distance = np.zeros(data_set.shape[0])
    for i in range(data_set.shape[0]):
        distance[i] = cosine_distance(new_input, data_set[i])
    sort_dis_idx = argsort(distance)

    class_count = {}
    for i in range(k):
        vote_label = labels_of_dataset[sort_dis_idx[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    max_count = 0
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            max_index = key
    return max_index
