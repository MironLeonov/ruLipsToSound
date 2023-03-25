import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class AudioEncoder(nn.Module): 

    def __init__(self) -> None:
        super(AudioEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(80, 100, 3, padding='same'),
            nn.LeakyReLU(),

            nn.Conv1d(100, 100, 3, padding='same'),
            nn.LeakyReLU(),

            nn.Conv1d(100, 256, 3, padding='same'),
            nn.LeakyReLU(), 

            nn.Conv1d(256, 512, 3, padding='same'),
            nn.LeakyReLU()
        )

        self.conv_layers.apply(init_weights)

        self.gru = nn.GRU(input_size = 512, hidden_size = 256, batch_first = True, bidirectional = True)



    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std) 
        sample = mu + (eps * std)
        data = sample.clone()

        return data


    def forward(self, x): 
        x = self.conv_layers(x)
        out = torch.transpose(x, 1, 2)
 
        out, hn = self.gru(out)

        out = out.view(-1, 100, 2, 256)
        mu = out[:, :, 0, :]
        log_var = out[:, :, 1, :]

        res = self.reparameterize(mu, log_var)
        
        return res