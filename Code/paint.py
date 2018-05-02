import os
import glob
from scipy import misc
from matplotlib import gridspec
import matplotlib.pyplot as plt
from variables import *

# change number of images in row and column
examples_height = 5
examples_width  = 5

visual_fil = '/home/fpalajs/Mozgalo_save/Results/Small/100/a/'
os.chdir(visual_fil)
image_list = []

filename = glob.glob('*.jpg') + glob.glob('*.png') + glob.glob('*.gif') + glob.glob('*.tiff')

for i in range(0, examples_height * examples_width):
    im = misc.imread(filename[i])
    im = misc.imresize(im, [64, 64], 'bilinear')
    image_list.append(im)
    
fig = plt.figure(figsize = (examples_height+1, examples_width+1))

gs = gridspec.GridSpec(examples_height, examples_width,
                         wspace=0.0, hspace=0.0, 
                         top=1.-0.5/(examples_height+1), bottom=0.5/(examples_height+1), 
                         left=0.5/(examples_width+1), right=1-0.5/(examples_width+1)) 

for i in range(examples_height):
    for j in range(examples_width):
        ax = plt.subplot(gs[i,j])
        ax.imshow(image_list[i*examples_height+j])
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
            
plt.show()

