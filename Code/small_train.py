import os
import tensorflow as tf
from load_train_image import getImage
from small_CNN import *

# reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# associate the "label" and "image" objects with the corresponding features read from 
# a single example in the training data file
label, image = getImage(cnn_current + '/train-00000-of-00001')

# and similarly for the validation data
vlabel, vimage = getImage(cnn_current + '/validation-00000-of-00001')

# associate the "label_batch" and "image_batch" objects with a randomly selected batch
# of labels and images respectively
imageBatch, labelBatch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size_,
    capacity=2000,
    min_after_dequeue=1000)

# and similarly for the validation data 
vimageBatch, vlabelBatch = tf.train.shuffle_batch(
    [vimage, vlabel], batch_size=batch_size_,
    capacity=2000,
    min_after_dequeue=1000)

# interactive session allows inteleaving of building and running steps
sess = tf.InteractiveSession()

# evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y+1e-8), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# training algorithm
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# save trained model
saver = tf.train.Saver()

# initialize the variables
sess.run(tf.global_variables_initializer())

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

for step in range(train_steps):
    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
    
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    
    if (step+1)%20 == 0:
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        train_accuracy = sess.run(accuracy, feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
        valid_accuracy = sess.run(accuracy, feed_dict={x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g, validation accuracy %g"%(step+1, train_accuracy, valid_accuracy))

# finalise
coord.request_stop()
coord.join(threads)

# save the variables to disk
if not os.path.exists(cnn_models):
    os.mkdir(cnn_models)
save_path = saver.save(sess, cnn_model)
print("Model saved in file: %s" % save_path)
