import torch
import torch.nn as nn


class DeepAutoEncoder(nn.Module):
    """
        I am trying to replicate the experiment that Prof. Hinton and Salakhutdinov did in 2006
        from the paper: "Reducing the Dimensionality of Data with Neural Networks"

        The initial weights of the auto encoder are from the pre-training of RBMs with stochastic binary neurons

    """
    def __init__(self, rbm1, rbm2, rbm3, rbm4):
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

        self.enc1.weight.data = rbm1.W.data.T
        self.enc1.bias.data = rbm1.c.data
        self.enc2.weight.data = rbm2.W.data.T
        self.enc2.bias.data = rbm2.c.data
        self.enc3.weight.data = rbm3.W.data.T
        self.enc3.bias.data = rbm3.c.data
        self.enc4.weight.data = rbm4.W.data.T
        self.enc4.bias.data = rbm4.c.data

        self.dec1.weight.data = rbm4.W.data
        self.dec1.bias.data = rbm4.b.data
        self.dec2.weight.data = rbm3.W.data
        self.dec2.bias.data = rbm3.b.data
        self.dec3.weight.data = rbm2.W.data
        self.dec3.bias.data = rbm2.b.data
        self.dec4.weight.data = rbm1.W.data
        self.dec4.bias.data = rbm1.b.data

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
