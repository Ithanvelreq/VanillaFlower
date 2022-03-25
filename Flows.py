import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import *
# from nf.utils import unconstrained_RQS

# supported non-linearities: note that the function must be invertible
functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) + \
                            (x < 0).type(torch.FloatTensor) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor) + \
                     (x < 0).type(torch.FloatTensor) * torch.exp(x)
}


class StepOfFlow(nn.Module):
    """
    Step of flow as defined in [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim, cond_dim, keep_first_for_coupling):
        super().__init__()
        self.actnorm = ActNorm(dim)
        self.one_by_one_conv = OneByOneConv(dim)
        self.conditional_coupling = ConditionalCoupling(dim, cond_dim, keep_first_for_coupling)
        self.flows = nn.ModuleList([self.actnorm, self.one_by_one_conv, self.conditional_coupling])

    def forward(self, flow_in, conditioning, reverse=False):
        batch_size, channels, dim = flow_in.shape
        log_det = torch.zeros(batch_size).to(flow_in.device)
        if not reverse:
            for flow in self.flows:
                flow_in, ld = flow.forward(flow_in, conditioning)
                log_det += ld
        else:
            for flow in self.flows[::-1]:
                flow_in, ld = flow.forward(flow_in, conditioning, reverse=True)
                log_det += ld
        return flow_in, log_det

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class ActNorm(nn.Module):
    """
    ActNorm layer.
    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype=torch.float))
        self.sigma = nn.Parameter(torch.zeros(dim, dtype=torch.float))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, flow_in, conditioning, reverse=False):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.mu.squeeze().data.copy_(flow_in.transpose(0, 1).flatten(1).mean(1)).view_as(self.sigma)
            self.sigma.squeeze().data.copy_(flow_in.transpose(0, 1).flatten(1).std(1, False) + 1e-6).view_as(self.mu)
            self.initialized += 1
        if not reverse:
            latent_variables = (flow_in - self.mu) / self.sigma
            log_det = torch.sum(torch.log(self.sigma)) * flow_in.shape[-1]  # sum(log(s)) * dim
            return latent_variables, log_det  # do we return + or - logdet???
        else:
            data_vector = flow_in * self.sigma + self.mu
            log_det = -torch.sum(torch.log(self.sigma)) * flow_in.shape[-1]  # sum(log(s)) * dim
            return data_vector, log_det  # do we return + or - logdet???


class OneByOneConv(nn.Module):
    """
    Invertible 1x1 convolution.
    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        W, _ = sp.linalg.qr(np.random.randn(dim, dim))
        P, L, U = sp.linalg.lu(W)
        self.P = nn.Parameter(torch.tensor(P, dtype=torch.float))
        self.L = nn.Parameter(torch.tensor(L, dtype=torch.float))
        self.S = nn.Parameter(torch.tensor(np.diag(U), dtype=torch.float))
        self.U = nn.Parameter(torch.triu(torch.tensor(U, dtype=torch.float), diagonal=1))

    def forward(self, flow_input, conditioning, reverse=False):
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim)).to(flow_input.device)
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        if not reverse:
            z = flow_input @ W
            log_det = torch.sum(torch.log(torch.abs(self.S)))
            return z, log_det
        else:
            W_inv = torch.inverse(W)
            x = flow_input @ W_inv
            log_det = -torch.sum(torch.log(torch.abs(self.S)))
            return x, log_det


class ConditionalCoupling(nn.Module):
    def __init__(self, vector_dim, condition_dim, keep_first):
        super().__init__()
        self.fcnn = FCNN(vector_dim//2+condition_dim, 2, (vector_dim//2+condition_dim)//2)  #nn.Linear(vector_dim//2+condition_dim, 2)
        self.keep_first = keep_first

    def forward(self, flow_input, conditioning, reverse=False):

        # z1, z2 = split(z)
        flow_input1, flow_input2 = flow_input[:, :, :flow_input.shape[2] // 2], flow_input[:, :, flow_input.shape[2] // 2:]
        if not self.keep_first:
            flow_input2, flow_input1 = flow_input1, flow_input2

        h = self.fcnn(torch.cat([flow_input2, conditioning], 2))

        t, h_scale = h[:, :, 0], h[:, :, 1]
        if h.shape[1] == 1:
            h_scale = h_scale[:, None, :]
            t = t[:, None, :]
        scale = torch.nn.functional.softplus(h_scale)

        if not reverse:
            flow_output2 = (flow_input2 * scale) + t
            flow_output1 = flow_input1
            logdet = flatten_sum(torch.log(scale))  # This may be bogus

        else:
            flow_output2 = (flow_input2 - t) / scale
            flow_output1 = flow_input1
            logdet = flatten_sum(torch.log(scale))  # This may be bogus
        if self.keep_first:
            flow_output = torch.cat((flow_output1, flow_output2), 2)
        else:
            flow_output = torch.cat((flow_output2, flow_output1), 2)
        return flow_output, logdet


class Squeeze(nn.Module):
    def __init__(self, factor=2):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(
            factor, int
        ), "no point of using this if factor <= 1"
        self.factor = factor

    def squeeze(self, x):
        batch_size, channels, vector_dim = x.size()
        assert vector_dim % self.factor == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(batch_size, channels, vector_dim // self.factor, self.factor)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, channels * self.factor, vector_dim // self.factor)

        return x

    def unsqueeze(self, x):
        bs, c, h = x.size()
        assert c >= 2 and c % 2 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor, self.factor, h)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(bs, c // self.factor, h * self.factor)
        return x

    def forward(self, x, reverse=False):
        if len(x.size()) != 3:
            raise NotImplementedError
        if not reverse:
            return self.squeeze(x)
        else:
            return self.unsqueeze(x)
