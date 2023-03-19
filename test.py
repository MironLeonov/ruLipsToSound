from video_encoder import VideoEncoder
from audio_encoder import AudioEncoder
from decoder import Decoder
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from data import VideoDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings
import torch.optim as optim


import torch.nn.functional as F


warnings.simplefilter(action='ignore', category=UserWarning)

test_transformer = transforms.Compose([
    transforms.ToTensor(),
])

kl_loss = nn.KLDivLoss(reduction="sum", log_target=False)
mse_loss = nn.MSELoss()


test_ds = VideoDataset(root_dir='data/features_data', transform=test_transformer)
loader = DataLoader(test_ds, batch_size=int(1))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

v_enc = VideoEncoder().to(device)
a_enc = AudioEncoder().to(device)
decoder = Decoder().to(device)

A_optimizer = optim.Adam([{'params': a_enc.parameters()}, {'params':decoder.parameters()}], lr=0.001, betas=(0.9, 0.999))
V_optimizer = optim.Adam(v_enc.parameters(), lr=0.001, betas=(0.9, 0.999))

a_scores = open('a_scores.txt', 'w')
v_scores = open('v_scores.txt', 'w')
os.makedirs('checkpoints', exist_ok=True)
EPOCHS = 100

if __name__ == '__main__': 
    train = True
    for epoch in range(EPOCHS): 
        progress_bar = tqdm(enumerate(loader), total=len(loader))
        for idx, batch in enumerate(loader):
            real_mel_spec = batch[0].to(device)
            
            video_frames = batch[1].to(device)

            with torch.no_grad(): 
                audio_embed = a_enc(real_mel_spec)
            
            video_embed = v_enc(video_frames)

            v_loss = kl_loss(video_embed, audio_embed)
            V_optimizer.zero_grad()
            v_loss.backward()
            V_optimizer.step()

            output = decoder(a_enc(real_mel_spec))

            a_loss = mse_loss(output, real_mel_spec)

            A_optimizer.zero_grad()
            a_loss.backward()
            A_optimizer.step()

            if idx % 5 == 0 or idx == len(loader):
                progress_bar.set_description(f"[{epoch + 1}/{EPOCHS}][{idx + 1}/{len(loader)}] "
                                             f"video_loss: {v_loss} audio_loss: {a_loss} ")
                v_scores.write(f'Epoch: {epoch} Iteration: {idx} video_loss: {v_loss}\n')
                a_scores.write(
                    f'Epoch: {epoch} Iteration: {idx} audio_loss: {a_loss}\n')

        if (epoch + 1) % 5 == 0:
            torch.save(a_enc, f'checkpoints/AudioEncoder_epoch_{epoch}.pth')
            torch.save(v_enc, f'checkpoints/VideoEncoder_epoch_{epoch}.pth')
            torch.save(decoder, f'checkpoints/Decoder_epoch_{epoch}.pth')
            print(f'{epoch} Model saved.')



