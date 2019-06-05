#wrappers for the layers

import tensorflow as tf
# import numpy as np

#CHECKED
#define the conv2D model
def conv(input,num_input_channels,num_filters,filter_size,padding = 'SAME',bias=True):
    
    with tf.name_scope('conv'):
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.random_normal(shape, stddev=0.1),name='w')
        

        # TensorFlow operation conv2D
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=padding)


        if bias:
            # Create new biases, one for each filter
            biases = tf.Variable(tf.constant(0.05, shape=[num_filters]),name='b')

            # Add the biases to the results of the convolution.
            layer = tf.nn.bias_add(layer,biases)
            tf.summary.histogram('biases',biases)

        # return layer, weights
        tf.summary.histogram("weights",weights)
        tf.summary.histogram("activation",layer) 
        return layer

#CHECKED
#define the max pool model
def pool2d(input):
    
    with tf.name_scope('max_pool'):
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        return layer

def pool_center_lower(input):
    
    with tf.name_scope('avg_pool'):
        #Tensorflow operation for convolution 
        layer = tf.nn.avg_pool(input, ksize=[1, 9, 9, 1], strides=[1, 8, 8, 1], padding='VALID')
        # layer = tf.layers.AveragePooling2D()
        return layer
