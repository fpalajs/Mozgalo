import os
import numpy as np
from operator import itemgetter
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from organize import organizeIntoClusters
from variables import *

def column(matrix, i):
    return [row[i] for row in matrix] 

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
    X.append([data[0], features.astype(float)])

f.close()

num_features = column(X,1)[1]

# remove anomalies
clf = IsolationForest(max_samples=100, random_state=1, n_estimators = 7, contamination = 0.1  )
clf.fit(column(X,1))
anomaly_table = clf.predict(column(X,1))

anomaly_detected = []
i = 0

for item in anomaly_table:
    if item == -1:
        anomaly_detected.append(file_list[i])
    i += 1
       
filtered_file_list = set(file_list) - set(anomaly_detected)
Z = list(filter( lambda x: x[0] in filtered_file_list, X))
Y = list(filter( lambda x: x[0] in set(anomaly_detected), X))
X = Z
anomaly_label = np.zeros( (len(Y),1))
print(len(Y))
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(column(X,1))

# assign labels
for i,x in enumerate(kmeans.labels_):
    X[i].insert(2,x)

#for j in range(0,len(anomaly_detected) ):
anomaly_label = (kmeans.predict(column(Y,1)))

centroid_list = []

for i in range(0, num_clusters):
    centroid = np.zeros(len(num_features))
    features_inside_cluster = column(list(filter(lambda x: x[2] == i, X)), 1)
    
    for j in features_inside_cluster:
        centroid = centroid + j
        centroid = centroid/len(num_features)
        centroid_list.append(centroid)
        
avg_distance_list = []
    
for i in range(0, num_clusters):
    avg_distance = 0
    features_inside_cluster = column(list(filter(lambda x: x[2] == i, X)), 1)
    
    for j in features_inside_cluster:
        avg_distance = avg_distance + np.sqrt(np.sum((centroid_list[i]- j)**2))
        avg_distance_list.append(avg_distance/len(features_inside_cluster)*(len(features_inside_cluster)/float(len(X))))
        
    print("Cluster " + str(i) + " has " + str(len(features_inside_cluster)) + " elements.")
    print("Average distance from centroid for cluster " + str(i) + ": " + str(avg_distance/len(features_inside_cluster)*(len(features_inside_cluster)/float(len(X)))))
    
      
print("Average value of average distances: " + str(np.std(np.array(avg_distance_list))))
print("Standard deviation of average distances: " + str(np.mean(np.array(avg_distance_list))))

f = open(results_file, 'w')

#file_lenght = len(kmeans.labels_)
file_list = column(X,0)

for i in range(0, len(X)):
    f.write("%s %d\n" %(file_list[i], kmeans.labels_[i]))
for i in range(0,len(Y)):
    f.write("%s %d\n" %(anomaly_detected[i],anomaly_label[i] ))
f.close()

# uncomment for visualization of results
organizeIntoClusters()
