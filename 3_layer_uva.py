
# Imports
from __future__ import division

import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataloader import UvaDataset
from utils import raw_spike_time, magno_spike_time
from utils import van_rossum_dist
import torch.nn.functional as F
# if you have a CUDA-enabled GPU the set the GPU flag as True
GPU=False

import time
import numpy as np
if GPU:
    import cupy as cp  # You need to have a CUDA-enabled GPU to use this package!
else:
    cp=np

# Parameter setting
# thr = [30, 30]  # The threshold of hidden and output neurons
lr = [.02, .02]  # The learning rate of hidden and ouput neurons
lamda = [0.000001, 0.000001]  # The regularization penalty for hidden and ouput neurons
b = [1, 10]  # The upper bound of wight initializations for hidden and ouput neurons
a = [-1, -1]  # The lower bound of wight initializations for hidden and ouput neurons
Nepoch = 20  # The maximum number of training epochs
NumOfClasses = 12000   # Number of classes
Nlayers = 2  # Number of layers
NhidenNeurons = 12000   # Number of hidden neurons
Dropout = [0, 0]
tmax = 256  # Simulatin time
GrayLevels = 255  # Image GrayLevels
gamma = 3  # The gamma parameter in the relative firingTarget firing calculation

# General settings
loading = False  # Set it as True if you want to load a pretrained model
LoadFrom = "weights.npy"  # The pretrained model
saving = True # Set it as True if you want to save the trained model
van_best = float('inf')
mse_best = float('inf')
mae_best = float('inf')
Nnrn = [NhidenNeurons, NumOfClasses]   # Number of neurons at hidden and output layers


# Loss
testVanDis = []
testMseLoss = []
testMaeLoss = []
Bce_criterion = torch.nn.BCELoss()
Mse_criterion = torch.nn.MSELoss()

W = []  # To hold the weights of hidden and output layers
firingTime = []  # To hold the firing times of hidden and output layers
Spikes = []  # To hold the spike trains of hidden and output layers
X = []  # To be used in converting firing times to spike trains
TargetSpikes = []
firingTarget = cp.zeros([NumOfClasses])  # To keep the firingTarget firing times of current image
FiringFrequency = []  # to input_count number of spikes each neuron emits during an epoch
# train video
train_raw = [
    # big : large and clear drone video
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_train_01.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_02.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_03.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_train_04.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_05.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_06.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_train_07.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_08.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_09.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_train_10.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_11.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_12.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_train_13.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_14.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_15.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_train_16.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_17.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_18.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_train_19.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_train_20.avi',
    # small : small and clear drone video
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_train_01.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_02.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_03.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_train_04.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_05.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_06.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_train_07.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_08.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_09.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_train_10.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_11.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_12.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_train_13.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_14.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_15.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_train_16.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_17.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_18.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_train_19.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_train_20.avi',

]
train_label = [
    # big label
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_01.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_02.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_03.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_04.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_05.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_06.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_07.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_08.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_09.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_10.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_11.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_12.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_13.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_14.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_15.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_16.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_17.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_18.avi',
    '/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_19.avi','/data3/zjy/S4NN/train/magno_dataset/big/uva_magno_20.avi',
    # small label
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_01.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_02.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_03.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_04.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_05.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_06.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_07.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_08.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_09.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_10.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_11.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_12.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_13.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_14.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_15.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_16.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_17.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_18.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_19.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_20.avi',
]

# val video
test_raw = [
    '/data3/zjy/S4NN/magno_val/big/uva_val_01.avi','/data3/zjy/S4NN/magno_val/big/uva_val_02.avi','/data3/zjy/S4NN/magno_val/big/uva_val_03.avi',
    '/data3/zjy/S4NN/magno_val/small/uva_val_01.avi','/data3/zjy/S4NN/magno_val/small/uva_val_02.avi','/data3/zjy/S4NN/magno_val/small/uva_val_03.avi',
]
test_label = [
    '/data3/zjy/S4NN/magno_val/big/uva_magno_val_01.avi','/data3/zjy/S4NN/magno_val/big/uva_magno_val_02.avi','/data3/zjy/S4NN/magno_val/big/uva_magno_val_03.avi',
    '/data3/zjy/S4NN/magno_val/small/uva_magno_val_01.avi','/data3/zjy/S4NN/magno_val/small/uva_magno_val_02.avi','/data3/zjy/S4NN/magno_val/small/uva_magno_val_03.avi'
]

traindataset = UvaDataset(train_raw, train_label)
testdataset =UvaDataset(test_raw, test_label)
# 创建 DataLoader
traindata_loader = DataLoader(traindataset, batch_size=1, shuffle=True)
images, labels = next(iter(traindata_loader))  # image-->(1,2,100,120) labels-->（1，100，120）
images = cp.asarray(images[0,:,:,:])
labels = cp.asarray(labels)

testdata_loader = DataLoader(testdataset, batch_size=1, shuffle=True)

# Building the model
layerSize = [[images[0].shape[0], images[0].shape[1]], [NhidenNeurons, 1], [NumOfClasses, 1]]
x = cp.mgrid[0:layerSize[0][0], 0:layerSize[0][1]]  # To be used in converting raw image into a spike image
SpikeImage = cp.zeros((layerSize[0][0], layerSize[0][1], tmax + 1))  # To keep spike image


baseThr = 0
de_factor = 0.5
SpikeList = [SpikeImage] + Spikes  # first data[SpikeImage]+network before Spikes
DynamicThr = np.full((NumOfClasses,tmax + 1), baseThr)

output_dir = '/data3/zjy/S4NN/output/3_layer/plt_1/'

# Initializing the network
np.random.seed(0)
for layer in range(Nlayers):
    W.append(cp.asarray((b[layer] - a[layer]) * np.random.random_sample((Nnrn[layer], layerSize[layer][0], layerSize[layer][1])) + a[layer]))
    firingTime.append(cp.asarray(np.zeros(Nnrn[layer])))
    Spikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1))))
    X.append(cp.asarray(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]]))
if loading:
    if GPU:
        W = np.load(LoadFrom, allow_pickle=True)
    else:
        for i in range(len(W)):
            W[i] = cp.asnumpy(W[i])
SpikeList = [SpikeImage] + Spikes

# Start learning
for epoch in range(Nepoch):
    start_time = time.time()
    i = 0
    # Start an epoch
    for raw_frames, label_frame in traindata_loader:
        images = cp.asarray(raw_frames[0, :, :, :])
        labels = cp.asarray(label_frame)
    # Start an epoch
        for iteration in range(len(images)):
            # converting input image into spiking image
            SpikeImage[:, :, :] = 0

            SpikeImage[x[0], x[1], images[iteration]] = 1
            DynamicThr = np.full((NumOfClasses, tmax + 1), baseThr)
            # Feedforward path
            for layer in range(Nlayers):
                Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)  # Computing the voltage
                for n in range(NumOfClasses):
                    spk_time_n = images[iteration].flatten()
                    spk_time = spk_time_n[n]
                    Thr = baseThr + spk_time * de_factor
                    # Thr = baseThr + np.exp(spk_time / 100 )
                    DynamicThr[n,:] = Thr
                    # print(DynamicThr)
                Voltage[:, tmax] = DynamicThr[:,0] + 1  # Forcing the fake spike
                firingTime[layer] = cp.argmax(Voltage > DynamicThr, axis=1).astype(
                        float) + 1  # Findign the first threshold crossing
                firingTime[layer][firingTime[layer] > tmax] = tmax  # Forcing the fake spike
                Spikes[layer][:, :, :] = 0
                Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(
                    int)] = 1  # converting firing times to spikes  (12000, 1, 257)
            firingTarget = labels.flatten()
            # Backward path
            layer = Nlayers - 1  # Output layer
            delta_o = []
            # anti-STDP
            for i in range(len(firingTime[layer])):
                if (firingTarget[i] - firingTime[layer][i]) > 0:
                    delta = ((firingTarget[i] - firingTime[layer][i]) ** 2) / (tmax**2)
                    delta_o.append(delta)
                else:
                    delta = -((firingTarget[i] - firingTime[layer][i]) ** 2) / (tmax**2)
                    delta_o.append(delta)
            # xi = []  # STDP
            # for i in range(len(firingTime[layer])):
            #     if firingTarget[i] != tmax:
            #         if (firingTarget[i] - firingTime[layer][i]) >= 0:
            #             xi_i = 1
            #         else:
            #             xi_i = -1
            #     else:
            #         xi_i = 5
            #     xi.append(xi_i)
            # xi = cp.asarray(xi)
            # delta_o = ((firingTarget - firingTime[layer]) ** 2) / tmax
            delta_o = cp.asarray(delta_o)
            # delta_o = (firingTarget - firingTime[layer]) / tmax  # Error in the ouput layer

            # Gradient normalization
            norm = cp.linalg.norm(delta_o)
            if (norm != 0):
                delta_o = delta_o / norm
            if Dropout[layer] > 0:
                firingTime[layer][cp.asarray(np.random.permutation(Nnrn[layer])[:Dropout[layer]])] = tmax

            # Updating hidden-output weights
            hasFired_o = firingTime[layer - 1] < firingTime[layer][:,
                                                 cp.newaxis]  # To find which hidden neurons has fired before the ouput neurons
            # anti-STDP
            # W[layer] -= lr[layer] * delta_o[:, cp.newaxis, cp.newaxis]* xi[:, cp.newaxis, cp.newaxis] * hasFired_h  # Update input-hidden weights
            # STDP
            W[layer][:, :, 0] -= (delta_o[:, cp.newaxis] * hasFired_o * lr[layer])  # Update hidden-ouput weights
            W[layer] -= lr[layer] * lamda[layer] * W[layer]  # Weight regularization

            # Backpropagating error to hidden neurons
            delta_h = (cp.multiply(delta_o[:, cp.newaxis] * hasFired_o, W[layer][:, :, 0])).sum(
                axis=0)  # Backpropagated errors from ouput layer to hidden layer
            layer = Nlayers - 2  # Hidden layer
            # Gradient normalization
            norm = cp.linalg.norm(delta_h)
            if (norm != 0):
                delta_h = delta_h / norm
            # Updating input-hidden weights
            hasFired_h = images[iteration] < firingTime[layer][:, cp.newaxis,
                                             cp.newaxis]  # To find which input neurons has fired before the hidden neurons
            # anti-STDP
            W[layer] -= lr[layer] * delta_h[:, cp.newaxis, cp.newaxis] * hasFired_h  # Update input-hidden weights
            W[layer] -= lr[layer] * lamda[layer] * W[layer]  # Weight regularization

    # Evaluating on test samples
    # test
    test_Van = 0
    test_MAE = 0
    test_MSE = 0
    for raw_frames, label_frame in testdata_loader:
        test_images = cp.asarray(raw_frames[0, :, :, :])
        labels = cp.asarray(label_frame)
        for iteration in range(len(test_images)):
            SpikeImage[:, :, :] = 0
            SpikeImage[x[0], x[1], test_images[iteration]] = 1
            SpikeList = [SpikeImage] + Spikes
            DynamicThr = np.full((NumOfClasses, tmax + 1), baseThr)
            for layer in range(Nlayers):
                Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
                for n in range(NumOfClasses):
                    spk_time_n = test_images[iteration].flatten()
                    spk_time = spk_time_n[n]
                    Thr = baseThr + spk_time * de_factor
                    # Thr = baseThr + np.exp(spk_time / 100 )
                    DynamicThr[n,:] = Thr

                Voltage[:, tmax] = DynamicThr[:,0] + 1
                firingTime[layer] = cp.argmax(Voltage > DynamicThr, axis=1).astype(float) + 1
                firingTime[layer][firingTime[layer] > tmax] = tmax
                Spikes[layer][:, :, :] = 0
                Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
                # TargetSpikes[layer][X[layer][0], X[layer][1], firingTarget.reshape(Nnrn[layer], 1).astype(int)] = 1
            firingTarget = labels.flatten()
            if epoch in [10,5,16,18,19]:
                # PLOT
                f = firingTime[0].reshape(100, 120)
                plt.subplot(2, 1, 1)
                plt.imshow(f, cmap='viridis', vmax=255, vmin=0)
                plt.colorbar(extend='min')
                plt.subplot(2, 1, 2)
                plt.imshow(labels[0, :, :], cmap='viridis', vmax=255, vmin=0)
                plt.colorbar(extend='min')
                plt.suptitle(f'2 Frame Thr=0+L0.5 Epoch {epoch} - 3LAYER')
                filename = f'epoch_{epoch}_image_{epoch + i}.png'
                plt.savefig(os.path.join(output_dir, filename))
                i += 1
                plt.show()
                plt.close()
                # plt.show()

            # Compare firingTime with labels
            for i in range(NumOfClasses):
                output_spike = int(firingTime[Nlayers - 1][i])
                target_spike = int(firingTarget[i])
                dis = van_rossum_dist(output_spike, target_spike, 100, tmax + 1)
                test_Van += dis
            mse = Mse_criterion(torch.tensor(firingTime[Nlayers - 1]), torch.tensor(firingTarget))
            test_MSE += mse
            mae = F.l1_loss(torch.tensor(firingTime[Nlayers - 1]), torch.tensor(firingTarget))
            test_MAE += mae
    test_Van /= NumOfClasses
    test_MSE /= NumOfClasses
    test_MAE /= NumOfClasses

    print('test_epoch= ', epoch)
    print('test_Van= ', test_Van)
    print('test_MSE= ', test_MSE)
    print('test_MAE= ', test_MAE)
    print("--- %s seconds ---" % (time.time() - start_time))

    # # Evaluating on train samples
    # train_Van = 0
    # train_MAE = 0
    # train_MSE = 0
    # for raw_frames, label_frame in traindata_loader:
    #     train_images = cp.asarray(raw_frames[0, :, :, :])
    #     labels = cp.asarray(label_frame)
    #     for iteration in range(len(train_images)):
    #         SpikeImage[:, :, :] = 0
    #         SpikeImage[x[0], x[1], train_images[iteration]] = 1
    #         SpikeList = [SpikeImage] + Spikes
    #         for layer in range(Nlayers):
    #             Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
    #             Voltage[:, tmax] = thr[layer] + 1
    #             firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(float) + 1
    #             firingTime[layer][firingTime[layer] > tmax] = tmax
    #             Spikes[layer][:, :, :] = 0
    #             Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
    #         firingTarget = labels.flatten()
    #         for i in range(NumOfClasses):
    #             output_spike = int(firingTime[Nlayers - 1][i])
    #             target_spike = int(firingTarget[i])
    #             dis = van_rossum_dist(output_spike, target_spike, 100, tmax + 1)
    #             train_Van += dis
    #         mse = Mse_criterion(torch.tensor(firingTime[Nlayers - 1]), torch.tensor(firingTarget))
    #         train_MSE += mse
    #         mae = F.l1_loss(torch.tensor(firingTime[Nlayers - 1]), torch.tensor(firingTarget))
    #         train_MAE += mae
    #
    # train_Van /= NumOfClasses
    # train_MSE /= NumOfClasses
    # train_MAE /= NumOfClasses
    # print('train_epoch= ', epoch)
    # print('train_Van= ', train_Van)
    # print('train_MSE= ', train_MSE)
    # print('train_MAE= ', train_MAE)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # To save the weights
    if saving:
        np.save("/data3/zjy/S4NN/clear_background/log1/3weights", W, allow_pickle=True)
        print("Dy-W done.")
        if test_Van < van_best:
            np.save("/data3/zjy/S4NN/clear_background/log1/3weights_van_best", W, allow_pickle=True)
            van_best = test_Van
        if test_MSE < mse_best:
            np.save("/data3/zjy/S4NN/clear_background/log1/3weights_mse_best", W, allow_pickle=True)
            mse_best = test_MSE
        if test_MAE < mae_best:
            np.save("/data3/zjy/S4NN/clear_background/log1/3weights_mae_best", W, allow_pickle=True)
            mae_best = test_MAE

    # # To find and reset dead neurons
    # ResetCheck = FiringFrequency < 0.001 * len(images)
    # ToReset = [i for i in range(NhidenNeurons) if ResetCheck[i]]
    # for i in ToReset:
    #     W[0][i] = cp.asarray((b[0] - a[0]) * np.random.random_sample((layerSize[0][0], layerSize[0][1])) + a[0])  # r
