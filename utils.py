import cv2
import numpy as np


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

    while frame_count < specified_frame_count:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count > 0:  # 跳过第一帧
            frame = np.array(frame)
            output_frames_list.append(np.floor((GrayLevels - frame.reshape(100, 120)) * tmax / GrayLevels).astype(int))

        frame_count += 1

    video_capture.release()
    return output_frames_list
