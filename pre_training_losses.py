import torch
import torch.nn.functional as F

def triangular_loss(Zmix, Z1, Z2, lamda):
    return torch.mean(torch.sum(torch.square(Zmix - (lamda * Z1 + (1 - lamda) * Z2)), dim=1))

def off_diagonal(x):
    """https://github.com/facebookresearch/vicreg
    for the covariance loss."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def covariance_loss(Z1, Z2):
    """https://github.com/facebookresearch/vicreg
    for the covariance loss."""
    batch_size = Z1.size()[0]
    num_features = Z1.size()[1]
    cov1 = (Z1.T @ Z1) / (batch_size - 1)
    cov2 = (Z2.T @ Z2) / (batch_size - 1)
    return off_diagonal(cov1).pow_(2).sum().div(num_features) + off_diagonal(cov2).pow_(2).sum().div(num_features)

def variance_loss(Z1, Z2):
    """https://github.com/facebookresearch/vicreg
    for the variance loss."""
    std1 = torch.sqrt(Z1.var(dim=0) + 0.0001)
    std2 = torch.sqrt(Z2.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std1)) / 2 + torch.mean(F.relu(1 - std2)) / 2
    return std_loss

def consistency_loss(Z1, Z2):
    return F.mse_loss(Z1, Z2)

def mean_loss(Es, Et):
    return torch.sqrt(torch.sum(torch.square(torch.mean(torch.square(Es), dim=0) - torch.mean(torch.square(Et), dim=0))))