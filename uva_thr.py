# Imports
from __future__ import division

import math
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import raw_spike_time, magno_spike_time, calculate_dynamic_threshold
from utils import van_rossum_dist
from dataloader import UvaDataset
# if you have a CUDA-enabled GPU the set the GPU flag as True
GPU=False

import time
import numpy as np
if GPU:
    import cupy as cp  # You need to have a CUDA-enabled GPU to use this package!
else:
    cp=np

# Parameter setting
thr = [30]  # The threshold of hidden and output neurons [100]
lr = [0.02]  # The learning rate of hidden and ouput neurons
lamda = [0.000001]  # The regularization penalty for hidden and ouput neurons
b = [1]  # The upper bound of wight initializations for hidden and ouput neurons
a = [-1]  # The lower bound of wight initializations for hidden and ouput neurons
Nepoch = 20  # The maximum number of training epochs
NumOfClasses = 12000  # Number of classes
Nlayers = 1  # Number of layers
Dropout = [1]
tmax = 256  # Simulation time
GrayLevels = 255  # Image GrayLevels
tau = 20
delta_o =[]
# General settings
loading = False  # Set it as True if you want to load a pretrained model
LoadFrom = "weights.npy"  # The pretrained model
saving = True # Set it as True if you want to save the trained model
van_best = float('inf')
mse_best = float('inf')
mae_best = float('inf')
Nnrn = [NumOfClasses]  # Number of neurons at hidden and output layers


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
    # cloud : background with cloud
    '/data3/zjy/S4NN/train/magno_dataset/cloud/uva_train_01.avi','/data3/zjy/S4NN/train/magno_dataset/cloud/uva_train_02.avi','/data3/zjy/S4NN/train/magno_dataset/cloud/uva_train_03.avi',
    '/data3/zjy/S4NN/train/magno_dataset/cloud/uva_train_04.avi','/data3/zjy/S4NN/train/magno_dataset/cloud/uva_train_05.avi','/data3/zjy/S4NN/train/magno_dataset/cloud/uva_train_06.avi',
    # city : background with city (hard)
    '/data3/zjy/S4NN/train/magno_dataset/city/uva_train_01.avi','/data3/zjy/S4NN/train/magno_dataset/city/uva_train_02.avi','/data3/zjy/S4NN/train/magno_dataset/city/uva_train_03.avi',
    '/data3/zjy/S4NN/train/magno_dataset/city/uva_train_04.avi','/data3/zjy/S4NN/train/magno_dataset/city/uva_train_05.avi','/data3/zjy/S4NN/train/magno_dataset/city/uva_train_06.avi',
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
    # cloud label
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_01.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_02.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_03.avi',
    '/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_04.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_05.avi','/data3/zjy/S4NN/train/magno_dataset/small/uva_magno_06.avi',
    # city label
    '/data3/zjy/S4NN/train/magno_dataset/city/uva_magno_01.avi','/data3/zjy/S4NN/train/magno_dataset/city/uva_magno_02.avi','/data3/zjy/S4NN/train/magno_dataset/city/uva_magno_03.avi',
    '/data3/zjy/S4NN/train/magno_dataset/city/uva_magno_04.avi','/data3/zjy/S4NN/train/magno_dataset/city/uva_magno_05.avi','/data3/zjy/S4NN/train/magno_dataset/city/uva_magno_06.avi',

]

# val video
test_raw = [
    '/data3/zjy/S4NN/magno_val/big/uva_val_01.avi','/data3/zjy/S4NN/magno_val/big/uva_val_02.avi','/data3/zjy/S4NN/magno_val/big/uva_val_03.avi',
    '/data3/zjy/S4NN/magno_val/small/uva_val_01.avi','/data3/zjy/S4NN/magno_val/small/uva_val_02.avi','/data3/zjy/S4NN/magno_val/small/uva_val_03.avi',
    '/data3/zjy/S4NN/magno_val/cloud/uva_val_01.avi','/data3/zjy/S4NN/magno_val/cloud/uva_val_02.avi','/data3/zjy/S4NN/magno_val/cloud/uva_val_03.avi',
    '/data3/zjy/S4NN/magno_val/city/uva_val_01.avi','/data3/zjy/S4NN/magno_val/city/uva_val_02.avi','/data3/zjy/S4NN/magno_val/city/uva_val_03.avi',
]
test_label = [
    '/data3/zjy/S4NN/magno_val/big/uva_magno_val_01.avi','/data3/zjy/S4NN/magno_val/big/uva_magno_val_02.avi','/data3/zjy/S4NN/magno_val/big/uva_magno_val_03.avi',
    '/data3/zjy/S4NN/magno_val/small/uva_magno_val_01.avi','/data3/zjy/S4NN/magno_val/small/uva_magno_val_02.avi','/data3/zjy/S4NN/magno_val/small/uva_magno_val_03.avi',
    '/data3/zjy/S4NN/magno_val/cloud/uva_magno_val_01.avi','/data3/zjy/S4NN/magno_val/cloud/uva_magno_val_02.avi','/data3/zjy/S4NN/magno_val/cloud/uva_magno_val_03.avi',
    '/data3/zjy/S4NN/magno_val/city/uva_magno_val_01.avi','/data3/zjy/S4NN/magno_val/city/uva_magno_val_02.avi','/data3/zjy/S4NN/magno_val/city/uva_magno_val_03.avi'
]

traindataset = UvaDataset(train_raw, train_label)
testdataset = UvaDataset(test_raw, test_label)
# DataLoader
traindata_loader = DataLoader(traindataset, batch_size=1, shuffle=True)
print("train datasets done.")
images, labels = next(iter(traindata_loader))  # image-->(1,2,100,120) labels-->（1，100，120）
images = cp.asarray(images[0,:,:,:])
labels = cp.asarray(labels)

testdata_loader = DataLoader(testdataset, batch_size=1, shuffle=False)
print("test datasets done.")
# Building the model
layerSize = [[images[0].shape[0], images[0].shape[1]], [NumOfClasses, 1]]
x = cp.mgrid[0:layerSize[0][0], 0:layerSize[0][1]]  # To be used in converting raw image into a spike image
SpikeImage = cp.zeros((layerSize[0][0], layerSize[0][1], tmax + 1))  # To keep spike image (30, 40, 257) 308400

# Initializing the network
np.random.seed(0)
for layer in range(Nlayers):
    W.append(cp.asarray(
        (b[layer] - a[layer]) * np.random.random_sample((Nnrn[layer], layerSize[layer][0], layerSize[layer][1])) + a[
            layer]))  # random Weight
    firingTime.append(cp.asarray(np.zeros(Nnrn[layer])))
    Spikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1))))  # (12000, 1, 257)
    # TargetSpikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1))))  # (12000, 1, 257)
    X.append(cp.asarray(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]]))
# if loading:
#     if GPU:
#         W = np.load(LoadFrom, allow_pickle=True)
#     else:
#         for i in range(len(W)):
#             W[i] = cp.asnumpy(W[i])
# (30, 40, 257)  (1200, 1, 257)
baseThr = 0
de_factor = 0.8
SpikeList = [SpikeImage] + Spikes  # first data[SpikeImage]+network before Spikes
DynamicThr = np.full((NumOfClasses,tmax + 1), baseThr)

output_dir = '/data3/zjy/S4NN/output/2_layer_mult_data/plt_1/'

# Start learning
for epoch in range(Nepoch):
    start_time = time.time()
    i = 0
    # Evaluating on train samples
    # Start an epoch
    for raw_frames, label_frame in traindata_loader:
        images = cp.asarray(raw_frames[0, :, :, :])
        labels = cp.asarray(label_frame)
        for iteration in range(len(images)):
            # converting input image into spiking image
            SpikeImage[:, :, :] = 0
            # images[iteration] --> SpikeImage
            SpikeImage[x[0], x[1], images[iteration]] = 1
            DynamicThr = np.full((NumOfClasses, tmax + 1), baseThr)
            # print(SpikeImage.shape)  # (30, 40, 257)
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
                # print(firingTime[layer])
                firingTime[layer][firingTime[layer] > tmax] = tmax  # Forcing the fake spike
                Spikes[layer][:, :, :] = 0
                Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(
                    int)] = 1  # converting firing times to spikes  (12000, 1, 257)
            # FiringFrequency = FiringFrequency + (firingTime[0] < tmax)  # FiringFrequency is used to find dead neurons

            # plot
            # f = firingTime[0].reshape(100, 120)
            # plt.subplot(2, 1, 1)
            # plt.imshow(f, cmap='viridis',vmax=255,vmin=0)
            # plt.colorbar(extend='min')
            # plt.subplot(2, 1, 2)
            # plt.imshow(labels[0, :, :], cmap='viridis',vmax=255,vmin=0)
            # plt.colorbar(extend='min')
            # # plt.tight_layout()
            # plt.suptitle(f'Frames Epoch {epoch}')
            # plt.show()

            # Computing the relative firingTarget firing times
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
            # # xi = []  # STDP
            # # for i in range(len(firingTime[layer])):
            # #     if firingTarget[i] != tmax:
            # #         if (firingTarget[i] - firingTime[layer][i]) >= 0:
            # #             xi_i = 1
            # #         else:
            # #             xi_i = -1
            # #     else:
            # #         xi_i = 5  # add lateral inhibition
            # #     xi.append(xi_i)
            # # xi = cp.asarray(xi)
            # # delta_o = ((firingTarget - firingTime[layer]) ** 2) / tmax
            delta_o = cp.asarray(delta_o)
            # delta_o = (firingTarget - firingTime[layer]) / tmax

            norm = cp.linalg.norm(delta_o)
            if (norm != 0):  # grad normalization
                delta_o = delta_o / norm
            # Dropout
            if Dropout[layer] > 0:
                firingTime[layer][cp.asarray(np.random.permutation(Nnrn[layer])[Dropout[layer]])] = tmax
            # Updating input-hidden weights
            hasFired_h = images[iteration] < firingTime[layer][:, cp.newaxis,
                                             cp.newaxis]  # To find which input neurons has fired before the hidden neurons
            # STDP
            # W[layer] -= lr[layer] * delta_o[:, cp.newaxis, cp.newaxis]* xi[:, cp.newaxis, cp.newaxis] * hasFired_h  # Update input-hidden weights
            # anti-STDP
            W[layer] -= lr[layer] * delta_o[:, cp.newaxis, cp.newaxis] * hasFired_h
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
                plt.suptitle(f'2 Frame Thr=0+L0.8 Epoch {epoch}-4')
                filename = f'epoch_{epoch}_image_{epoch + i}_L8.png'
                plt.savefig(os.path.join(output_dir, filename))
                i += 1
                plt.close()
                # plt.show()

            # Compare firingTime with labels
            # for layer in range(Nlayers):
            #     for i in range(NumOfClasses):
            #         output_spike = torch.from_numpy(Spikes[layer][i]).type(torch.int)
            #         target_spike = torch.from_numpy(TargetSpikes[layer][i]).type(torch.int)
            # distance = van_rossum_distance(output_spike, target_spike, 1.0)
            # VanDis += torch.sum(distance).numpy()
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
    testVanDis.append(test_Van)
    test_MSE /= NumOfClasses
    testMseLoss.append(test_MSE)
    test_MAE /= NumOfClasses
    testMaeLoss.append(test_MAE)

    print('test_epoch= ', epoch)
    print('test_Van= ', test_Van)
    print('test_MSE= ', test_MSE)
    print('test_MAE= ', test_MAE)
    print("--- %s seconds ---" % (time.time() - start_time))
    # To save the weights
    if saving:
        np.save("/data3/zjy/S4NN/mult_datasets/log1/weights", W, allow_pickle=True)
        print("Dy-W done.")
        if test_Van < van_best:
            np.save("/data3/zjy/S4NN/mult_datasets/log1/weights_van_best", W, allow_pickle=True)
            van_best = test_Van
        if test_MSE < mse_best:
            np.save("/data3/zjy/S4NN/mult_datasets/log1/weights_mse_best", W, allow_pickle=True)
            mse_best = test_MSE
        if test_MAE < mae_best:
            np.save("/data3/zjy/S4NN/mult_datasets/log1/weights_mae_best", W, allow_pickle=True)
            mae_best = test_MAE

# plt.plot(range(1,21), Mse, label = 'MSE')
# plt.xlabel('Epoch')
# plt.ylabel('MSE')
# plt.title('MSE')
# plt.show()
#
# plt.plot(range(1,21), VanDis, label ='VanDis')
# plt.xlabel('Epoch')
# plt.ylabel('VanDis')
# plt.title('VanDis')
# plt.show()
#
# plt.plot(range(1,21), MaeLoss, label ='MaeLoss')
# plt.xlabel('Epoch')
# plt.ylabel('MaeLoss')
# plt.title('MaeLoss')
# plt.show()

