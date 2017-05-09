import function_wmmse_powercontrol as wf
import function_dnn_powercontrol as df
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

K=10
num_H=25000
training_epochs=100
trainseed = 0
testseed = 7
print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n'%(K, num_H, training_epochs))
Xtrain, Ytrain, wtime = wf.generate_Gaussian(K, num_H, seed=trainseed)

print('train DNN ...')
model_location = "/Users/SUN/Desktop/SPAWC2017-master/DNNmodel/model_demo.ckpt"
df.train(Xtrain, Ytrain, model_location, training_epochs=training_epochs, traintestsplit = 0.2, batch_size=200)

num_test = 5000
X, Y, wmmsetime = wf.generate_Gaussian(K, num_test, seed=testseed)
dnntime = df.test(X, model_location, "Prediction_%d" % K , K * K, K, binary=1)
print('wmmse time: %0.3f s, dnn time: %0.3f s, time speed up: %0.1f X' % (wmmsetime, dnntime, wmmsetime / dnntime))

H = np.reshape(X, (K, K, X.shape[1]), order="F")
NNVbb = sio.loadmat('Prediction_%d.mat' % K)['pred']
wf.perf_eval(H, Y, NNVbb, K)

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