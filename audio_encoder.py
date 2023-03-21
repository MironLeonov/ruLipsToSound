import torch
import torch.nn as nn


class AudioEncoder(nn.Module): 

    def __init__(self) -> None:
        super(AudioEncoder, self).__init__()

        self.conv1 = nn.Conv1d(80, 100, 3, padding='same')

        self.conv2 = nn.Conv1d(100, 100, 3, padding='same')

        self.conv3 = nn.Conv1d(100, 256, 3, padding='same')

        self.conv4 = nn.Conv1d(256, 512, 3, padding='same')

        self.relu = nn.ReLU()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std) 
        sample = mu + (eps * std)
        data = sample.clone()


        # data = data.reshape(sample.size(0), -1)
        # data -= data.min(1, keepdim=True)[0]
        # data /= data.max(1, keepdim=True)[0]
        # data = data.view(sample.size(0), sample.size(1), sample.size(2))

        return data


    def forward(self, x): 
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        
        out = torch.transpose(x, 1, 2)

        out = out.view(-1, 100, 2, 256)
        mu = out[:, :, 0, :]
        log_var = out[:, :, 1, :]

        res = self.reparameterize(mu, log_var)
        
        return res