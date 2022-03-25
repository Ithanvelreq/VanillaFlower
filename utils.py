import torch

def split(feature):
    """
    Splits the input feature tensor into two halves along the channel dimension.
    Channel-wise masking.
    Args:
        feature: Input tensor to be split.
    Returns:
        Two output tensors resulting from splitting the input tensor into half
        along the channel dimension.
    """
    C = feature.size(1)
    return feature[:, : C // 2, ...], feature[:, C // 2:, ...]


def flatten_sum(logps):
    while len(logps.size()) > 1:
        logps = logps.sum(dim=-1)
    return logps

def one_hot_MNIST(labels):
    encoded_labels = torch.zeros(labels.shape[0], 10)
    for i in range(0, labels.shape[0]):
        encoded_labels[i, labels[i]] = 1
    return encoded_labels
