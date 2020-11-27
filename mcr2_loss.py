# adapted from https://github.com/ryanchankh/mcr2

import torch
import torch.nn as nn
from easydict import EasyDict

class MaximalCodingRateReduction(nn.Module):
    def __init__(self, num_classes, gam1=1.0, gam2=1.0, eps=0.5):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.num_classes = num_classes

    def compute_discrim_loss_empirical(self, Z):
        """Empirical Discriminative Loss."""
        m, d = Z.shape
        I = torch.eye(d).cuda()
        scalar = d / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * Z.T @ Z)
        return logdet / 2.

    def compute_compress_loss_empirical(self, Z, Pi):
        """Empirical Compressive Loss."""
        # Pi shape (num_classes, num_samples)
        num_samples, d = Z.shape
        I = torch.eye(d).cuda()  # shape (d, d)

        # TODO: add 1e-8 only where needed for safety
        num_class_samples = Pi.sum(-1) + 1e-8  # shape (self.num_classes)
        scale = d / (num_class_samples * self.eps)  # shape (self.num_classes)
        selected_Z = Z.unsqueeze(0) * Pi.unsqueeze(2)  # shape (self.num_classes, num_samples, d)

        compress_loss = 0.
        ZtPiZ = torch.zeros(self.num_classes, d, d).cuda()
        for i in range(self.num_classes):
            ZtPiZ[i] = torch.sparse.mm(selected_Z[i].T.to_sparse(), selected_Z[i])  # shape (d, d)
        log_det = torch.logdet(I + scale.view(-1, 1, 1) * ZtPiZ)  # shape (self.num_classes)
        compress_loss = torch.sum(log_det * num_class_samples / num_samples)  # shape (1)
        return compress_loss / 2., ZtPiZ.detach()

    def compute_discrim_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def forward(self, Z, Y, compute_theo=False):
        """

        :param Z: shape (num_samples, d)
        :param Y: shape (num_samples)
        :param num_classes:
        :return: shape (1)
        """
        num_samples = Z.shape[0]
        Z_mean = torch.mean(Z, dim=0)
        # Pi shape (num_classes, num_samples)
        zeros = torch.zeros(self.num_classes, num_samples).cuda()
        Pi = zeros.scatter(dim=0, index=Y.unsqueeze(0), src=torch.ones_like(zeros)).cuda()

        discrim_loss_empi = self.compute_discrim_loss_empirical(Z)
        compress_loss_empi, ZtPiZ = self.compute_compress_loss_empirical(Z, Pi)
        total_loss_empi = self.gam2 * -discrim_loss_empi + compress_loss_empi
        ret = EasyDict(
            loss=total_loss_empi,
            ZtPiZ=ZtPiZ,
            Z_mean=Z_mean,
            discrim_loss=discrim_loss_empi.item(),
            compress_loss=compress_loss_empi.item()
        )

        if compute_theo:
            W = Z.T
            Pi = label_to_membership(Y.cpu().numpy(), self.num_classes)
            Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
            ret.discrim_loss_theo = self.compute_discrim_loss_theoretical(W)
            ret.compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)

        return ret

import numpy as np

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot

def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.
    Parameters:
        targets (np.ndarray): matrix with one hot labels
    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)
    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi

if __name__ == '__main__':
    # Verify optimized empirical computation against original theoretical implementation
    from itertools import product

    num_samples = [10, 100, 1000]
    dim_z = [16, 64, 128]
    eps = [.5]
    num_classes = [10]
    for ns, dz, eps, nc in product(num_samples, dim_z, eps, num_classes):
        mcr2 = MaximalCodingRateReduction(num_classes=nc, eps=eps)

        discrim_errors = []
        compress_errors = []
        for i in range(10):
            Z = torch.rand(size=(ns, dz)).cuda()
            Z = Z / torch.norm(Z, dim=1, keepdim=True)
            Y = torch.randint(0, nc-1, size=(ns,)).cuda()

            ret = mcr2(Z, Y, compute_theo=True)
            d_err = torch.abs(ret.discrim_loss_theo - ret.discrim_loss).item()
            c_err = torch.abs(ret.compress_loss_theo - ret.compress_loss).item()
            discrim_errors.append(d_err)
            compress_errors.append(c_err)

        d_err = sum(discrim_errors) / len(discrim_errors)
        c_err = sum(compress_errors) / len(compress_errors)
        print(f'ns{ns} dz{dz} eps{eps:.1} nc{nc}\t d_err{d_err:.3}\t c_err{c_err:.3}')

    """
    RESULTS:
    ns100 dz16 eps0.5 nc10  	 d_err1.91e-07	 c_err1.43e-07
    ns100 dz64 eps0.5 nc10  	 d_err5.72e-07	 c_err3.81e-07
    ns100 dz128 eps0.5 nc10 	 d_err9.54e-07	 c_err7.63e-07
    ns1000 dz16 eps0.5 nc10 	 d_err2.38e-07	 c_err3.34e-07
    ns1000 dz64 eps0.5 nc10      d_err1.05e-06	 c_err7.63e-07
    ns1000 dz128 eps0.5 nc10	 d_err1.91e-06	 c_err9.54e-07
    ns10000 dz16 eps0.5 nc10	 d_err5.25e-07	 c_err3.81e-07
    ns10000 dz64 eps0.5 nc10	 d_err1.62e-06	 c_err6.68e-07
    ns10000 dz128 eps0.5 nc10	 d_err1.72e-06	 c_err9.54e-07
    """

