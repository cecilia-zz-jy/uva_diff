# Imports
from __future__ import division

from matplotlib import pyplot as plt

from utils import raw_spike_time, magno_spike_time
# if you have a CUDA-enabled GPU the set the GPU flag as True
GPU=False

import time
import numpy as np
if GPU:
    import cupy as cp  # You need to have a CUDA-enabled GPU to use this package!
else:
    cp=np

# Parameter setting
thr = [10]  # The threshold of hidden and output neurons [100]
lr = [.1]  # The learning rate of hidden and ouput neurons
lamda = [0.001]  # The regularization penalty for hidden and ouput neurons
b = [5]  # The upper bound of wight initializations for hidden and ouput neurons
a = [0]  # The lower bound of wight initializations for hidden and ouput neurons
Nepoch = 50  # The maximum number of training epochs
NumOfClasses = 12000  # Number of classes
Nlayers = 1  # Number of layers
Dropout = [0]
tmax = 256  # Simulatin time
GrayLevels = 255  # Image GrayLevels

# General settings
loading = False  # Set it as True if you want to load a pretrained model
LoadFrom = "weights.npy"  # The pretrained model
saving = False  # Set it as True if you want to save the trained model
best_perf = 0
Nnrn = [NumOfClasses]  # Number of neurons at hidden and output layers

# images_test = []  # To keep test images
# labels_test = []  # To keep test labels
W = []  # To hold the weights of hidden and output layers
firingTime = []  # To hold the firing times of hidden and output layers
Spikes = []  # To hold the spike trains of hidden and output layers
X = []  # To be used in converting firing times to spike trains
target = cp.zeros([NumOfClasses])  # To keep the target firing times of current image
FiringFrequency = []  # to count number of spikes each neuron emits during an epoch

raw_video = "D:/learning/Retina/original_120_100_synced.avi"
label_video = "D:/learning/Retina/magno_120_100.avi"
count = 2
images = raw_spike_time(raw_video, count, tmax, GrayLevels)
labels = magno_spike_time(label_video, count, tmax, GrayLevels)


images = cp.asarray(images)
labels = cp.asarray(labels)
# images_test = cp.asarray(images_test)
# labels_test = cp.asarray(labels_test)

# Building the model
layerSize = [[images[0].shape[0], images[0].shape[1]], [NumOfClasses, 1]]
x = cp.mgrid[0:layerSize[0][0], 0:layerSize[0][1]]  # To be used in converting raw image into a spike image
SpikeImage = cp.zeros((layerSize[0][0], layerSize[0][1], tmax + 1))  # To keep spike image (30, 40, 257) 308400

# Initializing the network
np.random.seed(0)
for layer in range(Nlayers):
    W.append(cp.asarray(
        (b[layer] - a[layer]) * np.random.random_sample((Nnrn[layer], layerSize[layer][0], layerSize[layer][1])) + a[
            layer]))  # 权重初始化
    firingTime.append(cp.asarray(np.zeros(Nnrn[layer])))
    Spikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1))))  # (1200, 1, 257)
    X.append(cp.asarray(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]]))
if loading:
    if GPU:
        W = np.load(LoadFrom, allow_pickle=True)
    else:
        for i in range(len(W)):
            W[i] = cp.asnumpy(W[i])
# (30, 40, 257)  (1200, 1, 257)
SpikeList = [SpikeImage] + Spikes  # 初始数据[SpikeImage]+通过网络层后的数据Spikes

# Start learning
for epoch in range(Nepoch):
    start_time = time.time()
    correct = cp.zeros(NumOfClasses)
    # FiringFrequency = cp.zeros((NhidenNeurons))
    # print(len(images))  # 2
    # Start an epoch
    for iteration in range(len(images)):
        # converting input image into spiking image
        SpikeImage[:, :, :] = 0
        # 没看懂？？将已经编码过的图像 images[iteration] 的脉冲信息复制到 SpikeImage 中的相应位置
        SpikeImage[x[0], x[1], images[iteration]] = 1
        # print(SpikeImage.shape)  # (30, 40, 257)
        # Feedforward path
        for layer in range(Nlayers):
            Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)  # Computing the voltage
            Voltage[:, tmax] = thr[layer] + 1  # Forcing the fake spike
            firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(
                float) + 1  # Findign the first threshold crossing
            # print('firingTime[layer]', firingTime[layer])
            firingTime[layer][firingTime[layer] > tmax] = tmax  # Forcing the fake spike
            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(
                int)] = 1  # converting firing times to spikes
        # 记录了每个隐藏层神经元在当前训练周期中发放脉冲的次数
        # FiringFrequency = FiringFrequency + (firingTime[0] < tmax)  # FiringFrequency is used to find dead neurons
        # 可视化
        # if epoch == 49:
        f = firingTime[0].reshape(100, 120)
        plt.subplot(2, 1, 1)
        plt.imshow(f, cmap='viridis', vmax=255, vmin=0)
        plt.colorbar(extend='min')
        plt.subplot(2, 1, 2)
        plt.imshow(labels[0, :, :], cmap='viridis', vmax=255, vmin=0)
        plt.colorbar(extend='min')
        plt.tight_layout()
        plt.suptitle(f'Frames Epoch {epoch}')
        plt.show()
        # Computing the relative target firing times
        # 计算目标的发放时间
        target = labels.flatten()
        # labels = labels.flatten()
        # Backward path
        layer = Nlayers - 1  # Output layer

        delta_o = (target - firingTime[layer]) / tmax  # Error in the ouput layer
#
#         # Gradient normalization
#         norm = cp.linalg.norm(delta_o)
#         if (norm != 0):
#             delta_o = delta_o / norm
# #
#         if Dropout[layer] > 0:
#             firingTime[layer][cp.asarray(np.random.permutation(Nnrn[layer])[:Dropout[layer]])] = tmax
#
# #         # Updating hidden-output weights
#         hasFired_o = firingTime[layer - 1] < firingTime[layer][:,cp.newaxis]  # To find which hidden neurons has fired before the ouput neurons
#         W[layer][:, :, 0] -= (delta_o[:, cp.newaxis] * hasFired_o * lr[layer])  # Update hidden-ouput weights
#         W[layer] -= lr[layer] * lamda[layer] * W[layer]  # Weight regularization
# #
#         # Backpropagating error to hidden neurons
#         delta_h = (cp.multiply(delta_o[:, cp.newaxis] * hasFired_o, W[layer][:, :, 0])).sum(
#             axis=0)  # Backpropagated errors from ouput layer to hidden layer
#
#         layer = Nlayers - 2  # Hidden layer
#
        # Gradient normalization
        norm = cp.linalg.norm(delta_o)
        if (norm != 0):
            delta_h = delta_o / norm
        # Updating input-hidden weights
        hasFired_h = images[iteration] < firingTime[layer][:, cp.newaxis,
                                         cp.newaxis]  # To find which input neurons has fired before the hidden neurons
        W[layer] -= lr[layer] * delta_o[:, cp.newaxis, cp.newaxis] * hasFired_h  # Update input-hidden weights
        W[layer] -= lr[layer] * lamda[layer] * W[layer]  # Weight regularization

    # # Evaluating on train samples
    # correct = 0
    # for iteration in range(len(images)):
    #     SpikeImage[:, :, :] = 0
    #     SpikeImage[x[0], x[1], images[iteration]] = 1
    #     for layer in range(Nlayers):
    #         Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
    #         Voltage[:, tmax] = thr[layer] + 1
    #         firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(float) + 1
    #         firingTime[layer][firingTime[layer] > tmax] = tmax
    #         Spikes[layer][:, :, :] = 0
    #         Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
    #     # Compare firingTime with labels
    #     # print("firingTime[Nlayers - 1].shape:", firingTime[Nlayers - 1].shape)
    #     # print("labels[iteration].shape:", labels[iteration].shape)
    #     correct += cp.sum(firingTime[Nlayers - 1] == labels[iteration])
    #
    # trainPerf = correct / 12000
    #
    # print('epoch= ', epoch, 'Perf_train= ', trainPerf)
    # print("--- %s seconds ---" % (time.time() - start_time))
