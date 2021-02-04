#  #################################################################
#  Python code to reproduce our works on DNN research for SPAWC 2017.
#
#  This file contains the whole process from data generation, training, testing to plotting 
#  for 10 users' IC case, even though such process done on a small dataset of 25000 samples, 
#  94% accuracy can still be easily attained in less than 100 iterations.
#
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.0.0 and Numpy 1.12.0 support.
#
#  References: 
#   [1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nikos D. Sidiropoulos. 
#       "LEARNING TO OPTIMIZE: TRAINING DEEP NEURAL NETWORKS FOR WIRELESS RESOURCE MANAGEMENT."
#
#  version 1.0 -- February 2017
#  Written by Haoran Sun (hrsun AT iastate.edu)
#  #################################################################

import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import function_wmmse_powercontrol as wf   # import our function file
import function_dnn_powercontrol as df     # import our function file

K = 10                     # number of users
num_H = 25000              # number of training samples
num_test = 5000            # number of testing  samples
training_epochs = 100      # number of training epochs
trainseed = 0              # set random seed for training set
testseed = 7               # set random seed for test set

# Problem Setup
print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n'%(K, num_H, training_epochs))

# Generate Training Data
Xtrain, Ytrain, wtime = wf.generate_Gaussian(K, num_H, seed=trainseed)

# Training Deep Neural Networks
print('train DNN ...')
# Save & Load model from this path 
model_location = "./DNNmodel/model_demo.ckpt"
df.train(Xtrain, Ytrain, model_location, training_epochs=training_epochs, traintestsplit = 0.2, batch_size=200)

# Generate Testing Data
X, Y, wmmsetime = wf.generate_Gaussian(K, num_test, seed=testseed)

# Testing Deep Neural Networks
dnntime = df.test(X, model_location, "Prediction_%d" % K , K * K, K, binary=1)
print('wmmse time: %0.3f s, dnn time: %0.3f s, time speed up: %0.1f X' % (wmmsetime, dnntime, wmmsetime / dnntime))

# Evaluate Performance of DNN and WMMSE
H = np.reshape(X, (K, K, X.shape[1]), order="F")
NNVbb = sio.loadmat('Prediction_%d' % K)['pred']
wf.perf_eval(H, Y, NNVbb, K)

# Plot figures
train = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['train']
time = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['time']
val = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['validation']
plt.figure(0)
plt.plot(time.T, val.T,label='validation')
plt.plot(time.T, train.T,label='train')
plt.legend(loc='upper right')
plt.xlabel('time (seconds)')
plt.ylabel('Mean Square Error')
plt.savefig('MSE_train.eps', format='eps', dpi=1000)
plt.show()
