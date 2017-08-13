import os
import numpy as np
import datetime
from shutil import copyfile
from variables import *

def organizeIntoClusters():

    dest_dir = results_current + '/results-' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '/'
    
    names = []
    clusters = []

    f = open(results_file, 'r')
	
    for line in f:
        data = line.split()
        names.append(data[0])
        clusters.append(data[1])
	
    f.close()

    uniqueClusters = np.unique(clusters)
	
    for cluster in uniqueClusters:
        os.makedirs(dest_dir + cluster)

    for i in range(0, len(names)):
        copyfile(images_dir + names[i], dest_dir + clusters[i] + '/' + names[i])
