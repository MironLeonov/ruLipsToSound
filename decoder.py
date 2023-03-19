import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self) -> None:
        super(Decoder, self).__init__()


        self.conv1 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7)
        self.conv4 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7)
        self.conv5 = nn.Conv1d(in_channels=100, out_channels=80, kernel_size=14)


    def forward(self, x): 

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x