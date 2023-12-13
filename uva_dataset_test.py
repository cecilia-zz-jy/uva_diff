# Imports
from __future__ import division

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import raw_spike_time, magno_spike_time, van_rossum_distance
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
lr = [0.01]  # The learning rate of hidden and ouput neurons
lamda = [0.01]  # The regularization penalty for hidden and ouput neurons
b = [1]  # The upper bound of wight initializations for hidden and ouput neurons
a = [0]  # The lower bound of wight initializations for hidden and ouput neurons
Nepoch = 20  # The maximum number of training epochs
NumOfClasses = 12000  # Number of classes
Nlayers = 1  # Number of layers
Dropout = [1]
tmax = 256  # Simulatin time
GrayLevels = 255  # Image GrayLevels
tau = 20
delta_o =[]
# General settings
loading = False  # Set it as True if you want to load a pretrained model
LoadFrom = "weights.npy"  # The pretrained model
saving = False  # Set it as True if you want to save the trained model
best_perf = 0
Nnrn = [NumOfClasses]  # Number of neurons at hidden and output layers


# 误差
VanDistacne = []
Mse = []
VanRosTimepoint = []
MaeLoss = []
Bce_criterion = torch.nn.BCELoss()
Mse_criterion = torch.nn.MSELoss()

# images_test = []  # To keep test images
# labels_test = []  # To keep test labels
W = []  # To hold the weights of hidden and output layers
firingTime = []  # To hold the firing times of hidden and output layers
Spikes = []  # To hold the spike trains of hidden and output layers
X = []  # To be used in converting firing times to spike trains
TargetSpikes = []
firingTarget = cp.zeros([NumOfClasses])  # To keep the firingTarget firing times of current image
FiringFrequency = []  # to input_count number of spikes each neuron emits during an epoch


# 示例视频文件列表
train_raw = [
    # "/data2/S4NN-master/train/dot_train.avi"
    "/data2/S4NN-master/train/original_120_100_synced.avi"
]
train_label = [
    # "/data2/S4NN-master/train/dotmagno_train.avi"
    "/data2/S4NN-master/train/magno_120_100.avi"
]
test_raw = [
    # "/data2/S4NN-master/test/dot_test.avi"
    "/data2/S4NN-master/test/test_raw.avi"
]
test_label = [
    # "/data2/S4NN-master/test/dotmagno_test.avi"
    "/data2/S4NN-master/test/magno_test.avi"
]

traindataset = UvaDataset(train_raw, train_label)
testdataset =UvaDataset(test_raw, test_label)
# 创建 DataLoader
traindata_loader = DataLoader(traindataset, batch_size=1, shuffle=True)
images, labels = next(iter(traindata_loader))  # image-->(1,2,100,120) labels-->（1，100，120）
images = cp.asarray(images[0,:,:,:])
labels = cp.asarray(labels)

testdata_loader = DataLoader(testdataset, batch_size=1, shuffle=True)
# print(labels)
# print(labels.shape)
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
SpikeList = [SpikeImage] + Spikes  # 初始数据[SpikeImage]+通过网络层后的数据Spikes

# Start learning
for epoch in range(Nepoch):
    start_time = time.time()
    # Evaluating on train samples
    VanDis = 0
    Van = 0
    MAE = 0
    MSE = 0
    # Start an epoch
    for raw_frames, label_frame in traindata_loader:
        images = cp.asarray(raw_frames[0, :, :, :])
        labels = cp.asarray(label_frame)


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

                firingTime[layer][firingTime[layer] > tmax] = tmax  # Forcing the fake spike
                # firingTime[layer][firingTime[layer] > 200 ] = tmax
                Spikes[layer][:, :, :] = 0
                Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(
                    int)] = 1  # converting firing times to spikes  (12000, 1, 257)

            # 记录了每个隐藏层神经元在当前训练周期中发放脉冲的次数
            # FiringFrequency = FiringFrequency + (firingTime[0] < tmax)  # FiringFrequency is used to find dead neurons

            # # 可视化
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
            # 计算目标的发放时间
            firingTarget = labels.flatten()

            # Backward path
            layer = Nlayers - 1  # Output layer
            delta_o = []
            xi = []  # STDP
            for i in range(len(firingTime[layer])):
                if firingTarget[i] != tmax:
                    if (firingTarget[i] - firingTime[layer][i]) >= 0:
                        xi_i = 1
                    else:
                        xi_i = -1
                else:
                    xi_i = 10  # 增加侧抑制
                xi.append(xi_i)
            xi = cp.asarray(xi)
            # for i in range(len(firingTime[layer])):
            #
            #     if (firingTarget[i] - firingTime[layer][i]) > 0:
            #         delta = ((firingTarget[i] - firingTime[layer][i]) ** 2) / tmax
            #         delta_o.append(delta)
            #     else:
            #         delta = -((firingTarget[i] - firingTime[layer][i]) ** 2) / tmax
            #         delta_o.append(delta)
            delta_o = ((firingTarget - firingTime[layer]) ** 2) / tmax
            delta_o = cp.asarray(delta_o)

            # norm = cp.linalg.norm(delta_o)
            # if (norm != 0):  # 梯度归一化、可以去掉
            #     delta_o = delta_o / norm

            if Dropout[layer] > 0:  # 防止过拟合
                firingTime[layer][cp.asarray(np.random.permutation(Nnrn[layer])[Dropout[layer]])] = tmax
            # Updating input-hidden weights
            hasFired_h = images[iteration] < firingTime[layer][:, cp.newaxis,
                                             cp.newaxis]  # To find which input neurons has fired before the hidden neurons
            W[layer] -= lr[layer] * delta_o[:, cp.newaxis, cp.newaxis]* xi[:, cp.newaxis, cp.newaxis] * hasFired_h  # Update input-hidden weights
            W[layer] -= lr[layer] * lamda[layer] * W[layer]  # Weight regularization

    # test
    for raw_frames, label_frame in testdata_loader:
        test_images = cp.asarray(raw_frames[0, :, :, :])
        labels = cp.asarray(label_frame)
        for iteration in range(len(test_images)):
            SpikeImage[:, :, :] = 0
            SpikeImage[x[0], x[1], test_images[iteration]] = 1
            for layer in range(Nlayers):
                Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
                Voltage[:, tmax] = thr[layer] + 1
                firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(float) + 1
                firingTime[layer][firingTime[layer] > tmax] = tmax
                Spikes[layer][:, :, :] = 0
                Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
                # TargetSpikes[layer][X[layer][0], X[layer][1], firingTarget.reshape(Nnrn[layer], 1).astype(int)] = 1

            # 可视化
            f = firingTime[0].reshape(100, 120)
            plt.subplot(2, 1, 1)
            plt.imshow(f, cmap='viridis',vmax=255,vmin=0)
            plt.colorbar(extend='min')
            plt.subplot(2, 1, 2)
            plt.imshow(labels[0, :, :], cmap='viridis',vmax=255,vmin=0)
            plt.colorbar(extend='min')
            # plt.tight_layout()
            plt.suptitle(f'Frames Epoch {epoch}')
            plt.show()
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
                # MSE += Mse_criterion(output_spike, target_spike)
                Van += dis

            # MAE = Bce_criterion(torch.tensor(firingTime[Nlayers - 1]), torch.tensor(firingTarget))
            # print(MAE)
            # MAE = Bce_criterion(firingTime[Nlayers - 1], firingTarget)
            # MSE += cp.sum(((firingTime[Nlayers - 1] - labels[iteration]) / tmax) ** 2)
            # MSE += cp.sum((firingTime[Nlayers - 1] - firingTarget) ** 2)
            mse = Mse_criterion(torch.tensor(firingTime[Nlayers - 1]), torch.tensor(firingTarget))
            MSE += mse
            mae = F.l1_loss(torch.tensor(firingTime[Nlayers - 1]), torch.tensor(firingTarget))
            MAE += mae


    # VanRossum
    # trainVanDis = VanDis / NumOfClasses
    VanRos_timepoint = Van / NumOfClasses
    # VanDistacne.append(trainVanDis)
    VanRosTimepoint.append(VanRos_timepoint)

    # Mse
    MSE /= NumOfClasses
    MSE = MSE.item()
    Mse.append(MSE)

    # Bce
    MAE /= NumOfClasses
    MAE = MAE.item()
    MaeLoss.append(MAE)

    print('epoch=', epoch)
    # print('VanDis num:', VanDis, 'VanDistance= ', trainVanDis)
    print('VanRos_timepoint= ', VanRos_timepoint)
    print('MSE= ', MSE)
    print('MAE= ', MAE)
    print("--- %s seconds ---" % (time.time() - start_time))


# plt.plot(range(1,21), VanDistacne, label = 'VanDistacne')
# plt.xlabel('Epoch')
# plt.ylabel('VanDistacne')
# # plt.ylim(0,1)
# # plt.legend()
# plt.title('VanDistacne')
# plt.show()

plt.plot(range(1,21), Mse, label = 'MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
# plt.legend()
# plt.ylim(0,1)
plt.title('MSE')
plt.show()

plt.plot(range(1,21), VanRosTimepoint, label = 'VanRosTimepoint')
plt.xlabel('Epoch')
plt.ylabel('VanRosTimepoint')
# plt.legend()
# plt.ylim(0,1)
plt.title('VanRosTimepoint')
plt.show()

plt.plot(range(1,21), MaeLoss, label ='MaeLoss')
plt.xlabel('Epoch')
plt.ylabel('MaeLoss')
# plt.legend()
# plt.ylim(0,1)
plt.title('MaeLoss')
plt.show()
