#wrappers for the layers

import tensorflow as tf
# import numpy as np

#CHECKED
#define the conv2D model
# def conv(input,num_input_channels,num_filters,filter_size,pad = 'SAME',bias=True,activation=None):

def conv(input,num_filters,filter_size,pad = 'SAME',bias=True,act=None):
    with tf.name_scope('conv'):
        # # Shape of the filter-weights for the convolution
        # shape = [filter_size, filter_size, num_input_channels, num_filters]

        # # Create new weights (filters) with the given shape
        # weights = tf.Variable(tf.random_normal(shape),name='w')
        

        # # TensorFlow operation conv2D
        # layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=padding)


        # if bias:
        #     # Create new biases, one for each filter
        #     biases = tf.Variable(tf.constant(0.05, shape=[num_filters]),name='b')

        #     # Add the biases to the results of the convolution.
        #     layer = tf.nn.bias_add(layer,biases)
        #     tf.summary.histogram('biases',biases)

        # # return layer, weights
        # tf.summary.histogram("weights",weights)
        # tf.summary.histogram("activation",layer) 

        #using the layers module and replacing the wrappers 
        layer = tf.layers.conv2d(inputs=input,filters=num_filters,kernel_size=[filter_size,filter_size],padding=pad,activation=act,use_bias=bias)

        return layer

#CHECKED
#define the max pool model
def pool2d(inp):
    
    with tf.name_scope('max_pool'):
        # TensorFlow operation for convolution
        # layer = tf.nn.max_pool(value=input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        layer = tf.layers.max_pooling2d(inputs = inp,pool_size=[3, 3], strides=2)
        return layer

def pool_center_lower(inp):
    
    with tf.name_scope('avg_pool'):
        #Tensorflow operation for convolution 
        # layer = tf.nn.avg_pool(input, ksize=[1, 9, 9, 1], strides=[1, 8, 8, 1], padding='VALID')
        layer = tf.layers.average_pooling2d(inputs=inp,pool_size=9,strides=8)
        return layer
