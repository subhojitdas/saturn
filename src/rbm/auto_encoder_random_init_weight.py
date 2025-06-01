import torch
import torch.nn as nn


class DeepAutoEncoderWithRandomInitWeight(nn.Module):
    def __init__(self):
        super().__init__()
        #encoders
        self.enc1 = nn.Linear(784, 500)
        self.enc2 = nn.Linear(500, 250)
        self.enc3 = nn.Linear(250, 125)
        self.enc4 = nn.Linear(125, 2)
        #decoders # parameters are transpose of enc
        self.dec1 = nn.Linear(2, 125)
        self.dec2 = nn.Linear(125, 250)
        self.dec3 = nn.Linear(250, 500)
        self.dec4 = nn.Linear(500, 784)

    def forward(self, x):
        x = torch.sigmoid(self.enc1(x))
        x = torch.sigmoid(self.enc2(x))
        x = torch.sigmoid(self.enc3(x))
        code = torch.sigmoid(self.enc4(x))

        x = torch.sigmoid(self.dec1(code))
        x = torch.sigmoid(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        x = torch.sigmoid(self.dec4(x))
        return x

