#!/usr/bin/env python3

import numpy as np
from collections import Counter

def Euclidean_distance(coo1, coo2):
    coo1 = np.array(coo1)
    coo2 = np.array(coo2)
    return np.sqrt(np.dot(coo1-coo2,(coo1-coo2).T))

def kNN_algorithm(data, query, k, distancef=Euclidean_distance, data_type='regression'):
    # find distance between query and dataset
    distances = [distancef(data_point[0], query) for data_point in data]

    # put into dictionary with appropriate weight
    labels = [data_point[1] for data_point in data]
    data_dict = dict(zip(distances,labels))

    # sort dictionary by distance
    data_dict_sort = dict(sorted(data_dict.items(), key=lambda item: item[0]))

    # choose k nearest neighbours
    result = [value for (key,value) in data_dict_sort.items()][:k]
    if data_type == 'regression':
        return np.mean(result)
    elif data_type == 'classification':
        return Counter(result).most_common(1)[0][0]
    else:
        return result
