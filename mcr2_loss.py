# adapted from https://github.com/ryanchankh/mcr2

import torch
import torch.nn as nn

class MaximalCodingRateReduction(nn.Module):
    def __init__(self, num_classes, gam1=1.0, gam2=1.0, eps=0.5):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.num_classes = num_classes

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        d, m = W.shape
        I = torch.eye(d).cuda()
        scalar = d / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W @ W.T)
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        # Pi shape (num_classes, num_samples)
        num_samples, d = W.shape
        I = torch.eye(d).cuda()  # shape (d, d)

        # TODO: add 1e-8 only where needed for safety
        num_class_samples = Pi.sum(-1) + 1e-8  # shape (self.num_classes)
        scale = d / (num_class_samples * self.eps)  # shape (self.num_classes)
        selected_W = W.unsqueeze(0) * Pi.unsqueeze(2)  # shape (self.num_classes, num_samples, d)
        compress_loss = 0.
        for i in range(self.num_classes):
            W_2 = torch.sparse.mm(selected_W[i].T.to_sparse(), selected_W[i])  # shape (d, d)
            log_det = torch.logdet(I + scale[i] * W_2)  # shape (1)
            compress_loss = compress_loss + log_det * num_class_samples[i] / num_samples
        return compress_loss / 2.
        # W_2 = torch.sparse.mm(selected_W.to_sparse(), selected_W.permute(0, 2, 1))  # shape (self.num_classes, d, d)
        # log_det = torch.logdet(I + scale.view(-1, 1, 1) * W_2)  # shape (self.num_classes)
        # compress_loss = torch.sum(log_det * num_class_samples / num_samples)  # shape (1)

    def compute_discrimn_loss_theoretical(self, W):
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

    def forward(self, X, Y):
        """

        :param X: shape (num_samples, d)
        :param Y: shape (num_samples)
        :param num_classes:
        :return: shape (1)
        """
        num_samples = X.shape[1]

        # W shape (d, num_samples)
        W = X.T
        # Pi shape (num_classes, num_samples)
        Z = torch.zeros(self.num_classes, num_samples).cuda()
        Pi = Z.scatter(dim=0, index=Y.unsqueeze(0), src=torch.ones_like(Z))
        Pi = Pi.cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        # discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        # compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)

        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return total_loss_empi
                # [discrimn_loss_empi.item(), compress_loss_empi.item()],
                # [discrimn_loss_theo.item(), compress_loss_theo.item()]
