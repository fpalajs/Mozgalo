import os
import sys
import glob
import numpy as np
import imgaug
import matplotlib
import random
from scipy import misc
from variables import *


def getAugSet(images):
    n = len(images)
    perm = np.random.permutation(list(range(n)))
    augSet = [images[i] for i in perm[0 : num_surr] ]
    return augSet


def RandomList(a, b, n):
    randList = []
    for i in range(0, n):
        randList.append(random.uniform(a,b))
    return randList


def Contrast(image, power, multiply, add):
    im = np.copy(image)

    for i in range(0, patch_size):
        for j in range(0, patch_size):
            for k in range(1, 3):
                im[i,j,k] = (im[i,j,k]**power)*multiply + add;
               
    return im


def ChangeHue(image, add):
    im = np.copy(image)
    
    for i in range(0, patch_size):
        for j in range(0, patch_size):
            im[i,j,1] += add
            
    return im


def transform1(image):
    contrast = random.randint(0, 128)
    factor = 1.0 *(259 * (contrast + 255)) / (255 * (259 - contrast))
    im = np.zeros((patch_size, patch_size, 3))

    for i in range(0, patch_size):
        for j in range(0, patch_size):
            im[i,j,0] = min(255, max(0, factor * (image[i,j,0]-128) + 128)) 
            im[i,j,1] = min(255, max(0, factor * (image[i,j,1]-128) + 128))
            im[i,j,2] = min(255, max(0, factor * (image[i,j,2]-128) + 128))

    return im

    
def transform2(image):
    a = random.uniform(0.5, 2)
    b = random.uniform(0.5, 2)
    c = random.uniform(0.5, 2)
    im = np.zeros((patch_size, patch_size, 3))
    
    for i in range(0, patch_size):
        for j in range(0, patch_size):
            im[i,j,0] = image[i,j,0] * a;
            im[i,j,1] = image[i,j,1] * b;
            im[i,j,2] = image[i,j,2] * c;
            
    return im

   
def AugmentPatch(image):
    image_list = []
    
    powerElements       = RandomList( 0.5,   2, num_aug)
    multiplyElements    = RandomList( 0.7, 1.4, num_aug)
    addingElements      = RandomList(-0.1, 0.1, num_aug)
    
    addHue              = RandomList(-0.1, 0.1, num_aug)
    multiply            = RandomList( 0.7, 1.4, num_aug)
    rotation            = RandomList( -20,  20, num_aug)

    translateList       = RandomList( -15,  15, num_aug)
    translateList       = np.array(translateList)
    translateList       = translateList.astype(int)

    fliplrProbability   = random.uniform(0.1, 0.3)

    for i in range(0, num_aug):
        im = np.copy(image)

        #if i % 2:
        #im = ChangeHue(im, 0.3)
            
        #if i % 5 > 1 :
        #im = transform1(im)
            
        #if i % 3 :
        #im = transform2(im)
        
        im = matplotlib.colors.rgb_to_hsv(1.0*im/255)
        im = ChangeHue(im, 0.5)
        #im = Contrast(im, powerElements[i], multiplyElements[i], addingElements[i])

        #seq = imgaug.augmenters.Affine(scale = 1.4, translate_px = 0, rotate = 0)
        #seq = imgaug.augmenters.Affine(scale = multiply[i], translate_px = translateList[i], rotate = rotation[i])
        #im = seq.augment_image(im)
                
        im = matplotlib.colors.hsv_to_rgb(im)
        
        # take patch of interest
        translate_px = translateList[i]
        #im = im[24+translate_px:24+translate_px + height, 24+translate_px :24+translate_px + width, :]
        im = im[24:24 + height, 24:24+width, :]
        image_list.append(im)
     
    #seq = imgaug.augmenters.Fliplr(fliplrProbability)
    #image_list = seq.augment_images(image_list)
    
    return image_list


####################################################################################################

# set working directory
images_dirr = '/home/fpalajs/Mozgalo_save/temp/'
aug_dirr = '/home/fpalajs/Mozgalo_save/color/'
os.chdir(images_dirr)

# don't overwrite existing augmentations
if os.path.exists(aug_dirr):
    print('Directory ' + aug_dirr + ' already exists!\n')
    sys.exit()

os.mkdir(aug_dirr)

# load dataset
image_list = []

for filename in glob.glob('*.jpg') + glob.glob('*.png') + glob.glob('*.tiff') + glob.glob('*gif'):
    im = misc.imread(filename, mode = 'RGB')
    im = misc.imresize(im, [patch_size, patch_size], 'bilinear')
    image_list.append(im)    
    
# make augmentations
aug = []
augSet = getAugSet(image_list)

for i in range(0, num_surr):
    print(i)
    aug = AugmentPatch(augSet[i])
    for j in range(0, num_aug):
        path = aug_dirr + str(i*num_aug+j) + '.jpg'
        misc.imsave(path, aug[j])  
