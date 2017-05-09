import function_wmmse_powercontrol as wf
import function_dnn_powercontrol as df
import scipy.io as sio
import numpy as np

for K in [10, 20, 30]:
    print('Gaussian IC Case: K=%d' % K)
    model_location = "/Users/SUN/Desktop/SPAWC2017-master/DNNmodel/model_%d.ckpt" % (K)
    num_test = 1000
    X, Y, wmmsetime = wf.generate_Gaussian(K, num_test, seed=7)
    dnntime = df.test(X, model_location, "Prediction_%d" % K , K * K, K, binary=1)
    print('wmmse time: %0.3f s, dnn time: %0.3f s, time speed up: %0.1f X' % (wmmsetime, dnntime, wmmsetime / dnntime))
    H = np.reshape(X, (K, K, X.shape[1]), order="F")
    NNVbb = sio.loadmat('Prediction_%d.mat' % K)['pred']
    wf.perf_eval(H, Y, NNVbb, K)