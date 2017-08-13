import os
import glob
from scipy import misc
import tensorflow as tf
from big_CNN import *

#reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# interactive session allows inteleaving of building and running steps
sess = tf.InteractiveSession()

# Evaluation functions
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# initialize the variables
sess.run(tf.global_variables_initializer())

os.chdir(test_dir)

image_list = []
file_list = []

for filename in glob.glob('*.jpg') + glob.glob('*.png') + glob.glob('*.tiff') + glob.glob('*gif'):
    im = misc.imread(filename,mode = 'RGB')
    file_list.append(filename)
    im.shape = (1, height*width*3)
    image_list.append(im)  

saver = tf.train.Saver()
saver = tf.train.import_meta_graph(cnn_model + '.meta')
saver.restore(sess, cnn_model)

#os.makedirs(results_current)
f = open(features_file, 'w')
                    
for i in range(0, len(file_list)):
    result = sess.run(h_fc1_drop, feed_dict={x: image_list[i], keep_prob: 1.0})
    f.write(file_list[i] + ' ' + ' '.join(str(e) for e in  result[0]) + '\n')

f.close()
