import numpy as np
np.random.seed(2017)
import tensorflow as tf
tf.set_random_seed(0)
import os

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def max_pool(x, name, f_size=2, stride=2, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, f_size, f_size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding, name=name)


def avg_pool(x, name, f_size=2, stride=2, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.avg_pool(x, ksize=[1, f_size, f_size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding, name=name)


def conv(x, f_size, num_filters, stride, name,
         padding='SAME', reuse=False):
    initializer = tf.contrib.layers.xavier_initializer()
    input_channels = int(x.get_shape()[-1])

    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride, stride, 1],
                                         padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        weights = tf.get_variable('weights', initializer=initializer([f_size,
                                                                      f_size,
                                                                      input_channels,
                                                                      num_filters]))
        biases = tf.get_variable('biases', shape=[num_filters])

        conv = convolve(x, weights)
        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        # Apply relu function
        relu = tf.nn.relu(bias)

        return relu

def fc(x, num_in, num_out, name, relu=True, reuse=False):
    initializer = tf.contrib.layers.xavier_initializer()
    # Create tf variables for the weights and biases

    with tf.variable_scope(name, reuse=reuse) as scope:
        weights = tf.get_variable('weights', initializer=initializer([num_in, num_out]),
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act
