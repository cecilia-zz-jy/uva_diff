import cv2
import numpy as np
import torch


def raw_spike_time(video, specified_frame_count, tmax, GrayLevels):
    video_capture = cv2.VideoCapture(video)
    processed_frames = 0  # 用于记录已处理的帧数
    s_frames = []

    while processed_frames < specified_frame_count:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 从头开始处理每一帧
        frame = np.array(frame)
        s_frames.append(np.floor((GrayLevels - frame.reshape(100, 120)) * tmax / GrayLevels).astype(int))

        processed_frames += 1

    video_capture.release()
    return s_frames

def magno_spike_time(video, specified_frame_count, tmax, GrayLevels):
    video_capture = cv2.VideoCapture(video)
    output_frames_list = []  # 存储帧的窗口
    frame_count = 0

    while frame_count < specified_frame_count + 1:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count > 1:  # 跳过第一帧
            frame = np.array(frame)
            output_frames_list.append(np.floor((GrayLevels - frame.reshape(100, 120)) * tmax / GrayLevels).astype(int))

        frame_count += 1

    video_capture.release()
    return output_frames_list

# raw_video = "D:/learning/video/moving_dot.mp4"
# label_video = "D:/learning/video/black_magno.avi"
# input_count = 2
# tmax = 256  # Simulatin time
# GrayLevels = 255  # Image GrayLevels
#
# s_spk = raw_spike_time(raw_video, input_count, tmax, GrayLevels)
# print(s_spk)
# l_spk = magno_spike_time(label_video, input_count, tmax, GrayLevels)
# print(l_spk)

def van_rossum_distance(tensor1, tensor2, tau):
    i = torch.cumsum(torch.exp(-tensor1 / tau) * (tensor1 > 0).float(), dim=0)
    j = torch.cumsum(torch.exp(-tensor2 / tau) * (tensor2 > 0).float(), dim=0)
    diff = i - j
    squared_diff = diff ** 2
    integral = torch.sum(squared_diff) * (1.0 / tau)
    distance = 1.0 / tau * integral
    return distance

def van_rossum_dist(spike_time1, spike_time2, tau, total_time_steps):
    """
    计算两个spiketimes对应的spiketrains之间的van Rossum距离。

    参数:
    spike_time1, spike_time2 (int): 两个spiketrains的spike时间步。
    tau (float): 时间常数，用于控制指数衰减的速率。
    total_time_steps (int): 总的时间步数。

    返回:
    float: 两个spiketrains之间的van Rossum距离。
    """
    # 创建两个空的spiketrains
    train1 = np.zeros(total_time_steps)
    train2 = np.zeros(total_time_steps)

    # 在指定的时间步上添加spike
    train1[spike_time1] = 1
    train2[spike_time2] = 1

    # 创建时间轴
    time_axis = np.arange(total_time_steps)

    # 转换spiketrains为连续的时间函数
    continuous_train1 = np.zeros(total_time_steps)
    continuous_train2 = np.zeros(total_time_steps)

    for t in time_axis:
        continuous_train1 += train1[t] * np.exp(-(time_axis - t) / tau)
        continuous_train2 += train2[t] * np.exp(-(time_axis - t) / tau)

    # 计算欧氏距离
    distance = np.sqrt(np.sum((continuous_train1 - continuous_train2) ** 2))

    return distance
