#  #################################################################
#  Python code to reproduce our works on DNN research for SPAWC 2017.
#
#  This file contains the testing stage for Table I: Gaussian IC case in the paper, 
#  the testing are based on the pre-trained models. To train models from scrach, 
#  please follow the instructions in the paper and reference the demo.py for data generation.
#
#  Codes have been tested successfully on Python 3.6.0 with TensorFlow 1.0.0 and Numpy 1.12.0 support.
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
#  version 1.0 -- February 2017
#  Written by Haoran Sun (hrsun AT iastate.edu)
#  #################################################################

import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import function_wmmse_powercontrol as wf   # import our function file
import function_dnn_powercontrol as df     # import our function file



for K in [10, 20, 30]:
    # Problem Setup, K: number of users.
    print('Gaussian IC Case: K=%d' % K)
    
    # Load model from this path, ! Please modify this path ! 
    model_location = "./DNNmodel/model_%d.ckpt" % (K)
    
    # Generate Testing Data
    num_test = 1000     # number of testing samples
    X, Y, wmmsetime = wf.generate_Gaussian(K, num_test, seed=7)
    
    # Testing Deep Neural Networks
    dnntime = df.test(X, model_location, "Prediction_%d" % K , K * K, K, binary=1)
    print('wmmse time: %0.3f s, dnn time: %0.3f s, time speed up: %0.1f X' % (wmmsetime, dnntime, wmmsetime / dnntime))
    
    # Evaluate Performance of DNN and WMMSE
    H = np.reshape(X, (K, K, X.shape[1]), order="F")
    NNVbb = sio.loadmat('Prediction_%d' % K)['pred']
    wf.perf_eval(H, Y, NNVbb, K)
