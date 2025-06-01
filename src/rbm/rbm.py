import torch.nn as nn
import torch
import torch.nn.functional as F


class RBM(nn.Module):

    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_visible))
        self.c = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        prob = torch.sigmoid(v @ self.W + self.c)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(h @ self.W.t() + self.b)
        return prob, torch.bernoulli(prob)

    def cd_step(self, v0):
        """
        Contrastive Divergence => comparing two distributions and get their difference measure
        """
        h0_prob, h0 = self.sample_h(v0)
        v1_prob, v1 = self.sample_v(h0)
        h1_prob, h1 = self.sample_h(v1)

        positive_grad = v0.t() @ h0_prob
        negative_grad = v1_prob.t() @ h1_prob

        self.W.grad = -(positive_grad - negative_grad) / v0.size(0)
        self.b.grad = -(v0 - v1_prob).mean(dim=0)
        self.c.grad = -(h0_prob - h1_prob).mean(dim=0)

        return F.binary_cross_entropy(v1_prob, v0, reduction='mean')

    def forward(self, v):
        return self.sample_h(v)[0]



