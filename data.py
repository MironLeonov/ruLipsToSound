import os
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torch
import numpy as np
import librosa


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.frames_dir = root_dir + '/pics'
        self.videos = os.listdir(self.frames_dir)

        self.audio_dir = root_dir + '/audio'
        self.audios = os.listdir(self.audio_dir)

        assert len(self.videos) == len(self.audios)

    def __len__(self):
        return len(self.videos)

    def get_frames(self, idx):
        video_dir = self.frames_dir + '/' + self.videos[idx]

        images_paths = glob(video_dir + "/*.jpg")


        frames = []
        for img_path in images_paths:
            frame = Image.open(img_path)
            frames.append(frame)

        frames_tr = []
        for frame in frames:
            frame = self.transform(frame)
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            frames_tr.append(frame)

        if len(frames_tr) > 0:
            frames_tr = torch.stack(frames_tr)

        return frames_tr

    def get_mel(self, idx):
        audio_path = self.audio_dir + '/' + self.videos[idx] + '.wav'

        audio, sample_rate = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y = audio, sr=int(16000), n_fft=int(400), hop_length=int(160), n_mels=int(80))
        mel_spec = torch.tensor(mel[:, :100])


        return mel_spec

    def __getitem__(self, idx):

        mel_spec = self.get_mel(idx)
        frames = self.get_frames(idx)

        return mel_spec, frames