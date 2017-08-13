import os
from shutil import copyfile
from variables import *

os.makedirs(cnn_current)
os.makedirs(train_dir)
os.makedirs(validate_dir)

f = open(labels_file, 'w')

for i in range(0, num_surr):
    label = 'label_' + str(i)
    f.write(label + '\n')
    os.makedirs(train_dir + label)
    os.makedirs(validate_dir + label)
    
    for j in range(0, num_aug):
        name = str(i*num_aug+j) + '.jpg'
        if j < num_aug//2 :
            copyfile(aug_dir + name,  train_dir   + label + '/' + name)
        else :
            copyfile(aug_dir + name, validate_dir + label + '/' + name)

f.close()
