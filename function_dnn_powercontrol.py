# ###############################################
# This file contains functions for deep neural network. 
# Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.0.0 and Numpy 1.12.0 support.
#
# References: 
# [1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nikos D. Sidiropoulos. 
# "Learning to optimize: Training deep neural networks for wireless resource management." 
# in proceedings of IEEE 18th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), 2017.
# 
# [2] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nikos D. Sidiropoulos, 
# "Learning to Optimize: Training Deep Neural Networks for Interference Management," 
# in IEEE Transactions on Signal Processing, vol. 66, no. 20, pp. 5438-5453, 15 Oct.15, 2018.
#
# version 1.0 -- February 2017. Written by Haoran Sun (hrsun AT iastate.edu) and Xiangyi Chen (xiangyi AT iastate.edu)
# ###############################################

from __future__ import print_function
import numpy as np
import scipy.io as sio
import time

import tensorflow as tfn
tf = tfn.compat.v1
tf.disable_v2_behavior()

# Functions for deep neural network weights initialization
def ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1]) / np.sqrt(n_input)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]) / np.sqrt(n_hidden_1)),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3]) / np.sqrt(n_hidden_2)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_3, n_output])) / n_hidden_3,
    }
    biases = {
        'b1': tf.Variable(tf.ones([n_hidden_1]) * 0.1),
        'b2': tf.Variable(tf.ones([n_hidden_2]) * 0.1),
        'b3': tf.Variable(tf.ones([n_hidden_3]) * 0.1),
        'out': tf.Variable(tf.ones([n_output]) * 0.1),
    }
    return weights, biases

# Functions for deep neural network structure construction
def multilayer_perceptron(x, weights, biases,input_keep_prob,hidden_keep_prob):
    x = tf.nn.dropout(x, input_keep_prob)                         # dropout layer
    
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])   # x = wx+b
    layer_1 = tf.nn.relu(layer_1)                                 # x = max(0, x)
    layer_1 = tf.nn.dropout(layer_1, hidden_keep_prob)            # dropout layer

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, hidden_keep_prob)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, hidden_keep_prob)

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    out_layer = tf.nn.relu6(out_layer) / 6
    return out_layer


# Functions for deep neural network training
def train(X, Y,location, training_epochs=300, batch_size=1000, LR= 0.001, n_hidden_1 = 200,n_hidden_2 = 80,n_hidden_3 = 80, traintestsplit = 0.01, LRdecay=0):
    num_total = X.shape[1]                        # number of total samples
    num_val = int(num_total * traintestsplit)     # number of validation samples
    num_train = num_total - num_val               # number of training samples
    n_input = X.shape[0]                          # input size
    n_output = Y.shape[0]                         # output size
    X_train = np.transpose(X[:, 0:num_train])     # training data
    Y_train = np.transpose(Y[:, 0:num_train])     # training label
    X_val = np.transpose(X[:, num_train:num_total]) # validation data
    Y_val = np.transpose(Y[:, num_train:num_total]) # validation label

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])
    is_train = tf.placeholder("bool")
    learning_rate = tf.placeholder(tf.float32, shape=[])
    total_batch = int(num_total / batch_size)
    print('train: %d ' % num_train, 'validation: %d ' % num_val)

    input_keep_prob = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)
    weights, biases = ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    pred = multilayer_perceptron(x, weights, biases, input_keep_prob, hidden_keep_prob)
    cost = tf.reduce_mean(tf.square(pred - y))    # cost function: MSE
    optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost) # training algorithms: RMSprop
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    MSETime=np.zeros((training_epochs,3))
    with tf.Session() as sess:
        sess.run(init)
        start_time = time.time()
        for epoch in range(training_epochs):
            for i in range(total_batch):
                idx = np.random.randint(num_train,size=batch_size)
                if LRdecay==1:
                    _, c = sess.run([optimizer, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                  input_keep_prob: 1, hidden_keep_prob: 1,
                                                                  learning_rate: LR/(epoch+1), is_train: True})
                elif LRdecay==0:
                    _, c = sess.run([optimizer, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                      input_keep_prob: 1, hidden_keep_prob: 1,
                                                                      learning_rate: LR, is_train: True})
            MSETime[epoch, 0]= c
            MSETime[epoch, 1]= sess.run(cost, feed_dict={x: X_val, y: Y_val, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})
            MSETime[epoch, 2]= time.time() - start_time
            if epoch%(int(training_epochs/10))==0:
                print('epoch:%d, '%epoch, 'train:%0.2f%%, '%(c*100), 'validation:%0.2f%%.'%(MSETime[epoch, 1]*100))

        print("training time: %0.2f s" % (time.time() - start_time))
        sio.savemat('MSETime_%d_%d_%d' % (n_output, batch_size, LR*10000) , {'train': MSETime[:,0], 'validation': MSETime[:,1], 'time': MSETime[:,2]})
        saver.save(sess, location)
    return 0

# Functions for deep neural network testing
def test(X, model_location, save_name, n_input, n_output, n_hidden_1 = 200, n_hidden_2 = 80, n_hidden_3 = 80, binary=0):
    tf.reset_default_graph()
    x = tf.placeholder("float", [None, n_input])
    is_train = tf.placeholder("bool")
    input_keep_prob = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)
    weights, biases = ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    pred = multilayer_perceptron(x, weights, biases, input_keep_prob, hidden_keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_location)
        start_time = time.time()
        y_pred = sess.run(pred, feed_dict={x: np.transpose(X), input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})
        testtime = time.time() - start_time
        # print("testing time: %0.2f s" % testtime)
        if binary==1:
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
        sio.savemat(save_name, {'pred': y_pred})
    return testtime
