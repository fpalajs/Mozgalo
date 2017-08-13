import os
import glob
from scipy import misc 
from variables import *

os.chdir(images_dir)
os.makedirs(test_dir)

for filename in glob.glob('*.jpg') + glob.glob('*.png') + glob.glob('*.tiff') + glob.glob('*gif'):
    im = misc.imread(filename, flatten = False, mode = 'RGB')
    im = misc.imresize(im, [patch_size, patch_size], 'bilinear')
    im = im[15:15+height, 15:15+width]
    path = test_dir + filename
    misc.imsave(path, im)
