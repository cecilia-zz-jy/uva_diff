import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class UvaDataset(Dataset):
    def __init__(self, raw_videos, label_videos):
        assert len(raw_videos) == len(label_videos), "Raw videos and label videos list must be of the same length"
        self.raw_videos = raw_videos
        self.label_videos = label_videos
        self.lengths = self.calculate_lengths()

    def calculate_lengths(self):
        lengths = []
        for video in self.raw_videos:
            cap = cv2.VideoCapture(video)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            lengths.append(length - 1)  # -1 because we are taking two frames at a time
        return lengths

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, global_idx):
        video_idx, local_idx = self.map_global_index_to_video(global_idx)
        # 为了获取重叠的帧对，我们需要从 local_idx 和 local_idx + 1 获取帧
        raw_frames = self.load_frames(self.raw_videos[video_idx], local_idx, 2)

        # 对于 label，我们只需要从 local_idx 获取一帧
        label_frame = self.load_frames(self.label_videos[video_idx], local_idx, 1)

        return raw_frames, label_frame[0]

    def map_global_index_to_video(self, global_idx):
        running_sum = 0
        for i, length in enumerate(self.lengths):
            if running_sum + length > global_idx:
                return i, global_idx - running_sum
            running_sum += length
        raise IndexError("Global index out of range")

    def load_frames(self, video, start_idx, num_frames):
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.array(frame)
                GrayLevels = 255
                tmax = 256
                s_frame = np.floor((GrayLevels - frame.reshape(100, 120)) * tmax / GrayLevels).astype(int)
                frames.append(s_frame)
            else:
                break

        cap.release()
        return frames

# 示例视频文件列表
raw_videos = [
    "D:/learning/Retina/original_120_100_synced.avi"
    # "D:/learning/Retina/original_120_100_synced_2.avi",
    # ... 更多视频文件
]

label_videos = [
    "D:/learning/Retina/magno_120_100.avi"
    # "D:/learning/Retina/magno_120_100_2.avi",
    # ... 更多视频文件
]

# 创建数据集实例
dataset = UvaDataset(raw_videos, label_videos)

# 创建 DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 在训练循环中使用 DataLoader
for raw_frames, label_frame in data_loader:
    # 这里可以对 raw_frames 和 label_frame 进行处理和训练
    ra = np.asarray(raw_frames)
    le = np.asarray(label_frame)
    print(ra)
    print(le)
