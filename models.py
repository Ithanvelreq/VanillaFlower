import torch
import torch.nn as nn
from Flows import StepOfFlow


class NormalizingFlowModel(nn.Module):

    def __init__(self, prior, dim, cond_dim, K, L):
        super().__init__()
        self.prior = prior
        steps = []
        keep_first_for_coupling = True
        for i in range(K):
            steps.append(StepOfFlow(dim, cond_dim, keep_first_for_coupling))
            keep_first_for_coupling = not keep_first_for_coupling
        self.steps_of_flows = nn.ModuleList(steps)

    def forward(self, x, conditioning):
        batch_size, channels, dim = x.shape
        log_det = torch.zeros(batch_size).to(x.device)
        for flow in self.steps_of_flows:
            x, ld = flow.forward(x, conditioning)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z, conditioning):
        batch_size, channels, dim = z.shape
        log_det = torch.zeros(batch_size).to(z.device)
        for flow in self.steps_of_flows[::-1]:
            z, ld = flow.forward(z, conditioning, reverse=True)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples, conditioning):
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z[:, None, :], conditioning)
        return x


class NormalizingFlowModelTester(nn.Module):

    def __init__(self, prior, dim, cond_dim, layer_to_test):
        super().__init__()
        self.prior = prior
        self.layer = layer_to_test

    def forward(self, x, conditioning):
        batch_size, channels, dim = x.shape
        log_det = torch.zeros(batch_size).to(x.device)
        x, ld = self.layer.forward(x, conditioning)
        log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z, conditioning):
        batch_size, channels, dim = z.shape
        log_det = torch.zeros(batch_size).to(z.device)
        z, ld = self.layer.forward(z, conditioning, reverse=True)
        log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples, conditioning):
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z[:, None, :], conditioning)
        return x
