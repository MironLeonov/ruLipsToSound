import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)



class Decoder(nn.Module):

    def __init__(self) -> None:
        super(Decoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3), 
            nn.LeakyReLU(), 

            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, stride=2),
            nn.LeakyReLU(), 

            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7),
            nn.LeakyReLU(), 

            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7),
            nn.LeakyReLU(), 

            nn.Conv1d(in_channels=100, out_channels=80, kernel_size=14),
            nn.LeakyReLU()
        )

        self.conv_layers.apply(init_weights)

        self.lstm = nn.LSTM(input_size = 256, hidden_size = 128, batch_first = True, bidirectional = True)


    def forward(self, x): 
        x, _ = self.lstm(x)
        x = self.conv_layers(x)
        return x