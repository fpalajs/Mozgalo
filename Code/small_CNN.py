import tensorflow as tf
from variables import *

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Input layer
x  = tf.placeholder(tf.float32, [None, height*width*3], name='x')   
y_ = tf.placeholder(tf.float32, [None, num_surr],  name='y_')
x_image = tf.reshape(x, [-1, height, width, 3]) 

# Convolutional layer 1
W_conv1 = weight_variable([5, 5, 3, 64]) 
b_conv1 = bias_variable([64]) 

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
W_fc1 = weight_variable([(height//4)*(width//4)*64 , 128]) 
b_fc1 = bias_variable([128])

h_pool1_flat = tf.reshape(h_pool2, [-1, (height//4)*(width//4)*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([128, num_surr])
b_fc2 = bias_variable([num_surr])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')
