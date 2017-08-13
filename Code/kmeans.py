import os
import numpy as np
from sklearn.cluster import KMeans
from organize import organizeIntoClusters
from variables import *
from operator import itemgetter

file_list = []
file_num = []
X = []

f = open(features_file, 'r')

for line in f:
    data = line.split()
    file_list.append(data[0])
    name, ext = os.path.splitext(data[0])
    file_num.append(int(name))
    features = np.array(data[1:len(data)])
    X.append(features.astype(float))

f.close()

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)

f = open(results_file, 'w')

sorted_list = [list(x) for x in zip(*sorted(zip(file_num, file_list, kmeans.labels_), key=lambda pair: pair[0]))]

for i in range(0, len(sorted_list[0])): 
    f.write("%s %d\n" %(sorted_list[1][i], sorted_list[2][i]))
     
f.close()

# uncomment for visualization of results
organizeIntoClusters()




