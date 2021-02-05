# ###############################################
# This file includes functions to perform the WMMSE algorithm [2].
# Codes have been tested successfully on Python 3.6.0 with Numpy 1.12.0 support.
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
# [3] Qingjiang Shi, Meisam Razaviyayn, Zhi-Quan Luo, and Chen He.
# "An iteratively weighted MMSE approach to distributed sum-utility maximization for a MIMO interfering broadcast channel."
# IEEE Transactions on Signal Processing 59, no. 9 (2011): 4331-4340.
#
# version 1.0 -- February 2017. Written by Haoran Sun (hrsun AT iastate.edu)
# ###############################################

import numpy as np
import math
import time
import scipy.io as sio
import matplotlib.pyplot as plt

# Functions for objective (sum-rate) calculation
def obj_IA_sum_rate(H, p, var_noise, K):
    y = 0.0
    for i in range(K):
        s = var_noise
        for j in range(K):
            if j!=i:
                s = s+H[i,j]**2*p[j]
        y = y+math.log2(1+H[i,i]**2*p[i]/s)
    return y

# Functions for WMMSE algorithm
def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-3:
            break

    p_opt = np.square(b)
    return p_opt

# Functions for performance evaluation
def perf_eval(H, Py_p, NN_p, K, var_noise=1):
    num_sample = H.shape[2]
    pyrate = np.zeros(num_sample)
    nnrate = np.zeros(num_sample)
    mprate = np.zeros(num_sample)
    rdrate = np.zeros(num_sample)
    for i in range(num_sample):
        pyrate[i] = obj_IA_sum_rate(H[:, :, i], Py_p[:, i], var_noise, K)
        nnrate[i] = obj_IA_sum_rate(H[:, :, i], NN_p[i, :], var_noise, K)
        mprate[i] = obj_IA_sum_rate(H[:, :, i], np.ones(K), var_noise, K)
        rdrate[i] = obj_IA_sum_rate(H[:, :, i], np.random.rand(K,1), var_noise, K)
    print('Sum-rate: WMMSE: %0.3f, DNN: %0.3f, Max Power: %0.3f, Random Power: %0.3f'%(sum(pyrate)/num_sample, sum(nnrate)/num_sample, sum(mprate)/num_sample, sum(rdrate)/num_sample))
    print('Ratio: DNN: %0.3f%%, Max Power: %0.3f%%, Random Power: %0.3f%%\n' % (sum(nnrate) / sum(pyrate)* 100, sum(mprate) / sum(pyrate) * 100, sum(rdrate) / sum(pyrate) * 100))

    plt.figure('%d'%K)
    plt.style.use('seaborn-deep')
    data = np.vstack([pyrate, nnrate]).T
    bins = np.linspace(0, max(pyrate), 50)
    plt.hist(data, bins, alpha=0.7, label=['WMMSE', 'DNN'])
    plt.legend(loc='upper right')
    plt.xlim([0, 8])
    plt.xlabel('sum-rate')
    plt.ylabel('number of samples')
    plt.savefig('Histogram_%d.eps'%K, format='eps', dpi=1000)
    plt.show()
    return 0

# Functions for data generation, Gaussian IC case
def generate_Gaussian(K, num_H, Pmax=1, Pmin=0, seed=2017):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pini = Pmax*np.ones(K)
    var_noise = 1
    X=np.zeros((K**2,num_H))
    Y=np.zeros((K,num_H))
    total_time = 0.0
    for loop in range(num_H):
        CH = 1/np.sqrt(2)*(np.random.randn(K,K)+1j*np.random.randn(K,K))
        H=abs(CH)
        X[:,loop] = np.reshape(H, (K**2,), order="F")
        H=np.reshape(X[:,loop], (K,K), order="F")
        mid_time = time.time()
        Y[:,loop] = WMMSE_sum_rate(Pini, H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time
    # print("wmmse time: %0.2f s" % total_time)
    return X, Y, total_time

# Functions for data generation, Gaussian IC half user case
def generate_Gaussian_half(K, num_H, Pmax=1, Pmin=0, seed=2017):
    print('Generate Testing Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pini = Pmax * np.ones(K)
    var_noise = 1
    X = np.zeros((K ** 2 * 4, num_H))
    Y = np.zeros((K * 2, num_H))
    total_time = 0.0
    for loop in range(num_H):
        CH = 1 / np.sqrt(2) * (np.random.randn(K, K) + 1j * np.random.randn(K, K))
        H = abs(CH)
        mid_time = time.time()
        Y[0: K, loop] = WMMSE_sum_rate(Pini, H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time
        OH = np.zeros((K * 2, K * 2))
        OH[0: K, 0:K] = H
        X[:, loop] = np.reshape(OH, (4 * K ** 2,), order="F")

    # print("wmmse time: %0.2f s" % total_time)
    return X, Y, total_time

# Functions for data generation, IMAC case
def generate_IMAC(num_BS, num_User, num_H, Pmax=1, var_noise = 1):
    # Load Channel Data
    CH = sio.loadmat('IMAC_%d_%d_%d' % (num_BS, num_User, num_H))['X']
    Temp = np.reshape(CH, (num_BS, num_User * num_BS, num_H), order="F")
    H = np.zeros((num_User * num_BS, num_User * num_BS, num_H))
    for iter in range(num_BS):
        H[iter * num_User:(iter + 1) * num_User, :, :] = Temp[iter, :, :]

    # Compute WMMSE output
    Y = np.zeros((num_User * num_BS, num_H))
    Pini = Pmax * np.ones(num_User * num_BS)
    start_time = time.time()
    for loop in range(num_H):
        Y[:, loop] = WMMSE_sum_rate(Pini, H[:, :, loop], Pmax, var_noise)
    wmmsetime=(time.time() - start_time)
    # print("wmmse time: %0.2f s" % wmmsetime)
    return CH, Y, wmmsetime, H

