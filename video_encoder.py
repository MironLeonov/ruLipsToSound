import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class VideoEncoder(nn.Module):


    def __init__(self) -> None:
        super(VideoEncoder, self).__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.requires_grad_(False)


        self.lstm = nn.LSTM(input_size=10,
                              hidden_size=512, num_layers=1,
                              batch_first=True)
        self.conv1 = nn.Conv3d(in_channels=25, out_channels=100, kernel_size=(1, 3, 3))
        self.conv2 = nn.Conv3d(in_channels=100, out_channels=100, kernel_size=(1, 4, 4), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(in_channels=100, out_channels=100, kernel_size=3, stride=2)
        self.linear = nn.Linear(in_features=484, out_features=512)


        


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std) 
        sample = mu + (eps * std)
        data = sample.clone()

        data = data.view(sample.size(0), -1)
        data -= data.min(1, keepdim=True)[0]
        data /= data.max(1, keepdim=True)[0]
        data = data.view(sample.size(0), sample.size(1), sample.size(2))

        return data



    def forward(self, images): 
        batch_size, numbers_of_frames, c, h, w = images.shape

        x = self.conv1(images)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (-1, 100, 484))
        out = self.linear(x)

        out = out.view(-1, 100, 2, 256)
        mu = out[:, :, 0, :]
        log_var = out[:, :, 1, :]

        res = self.reparameterize(mu, log_var)

        return res