import torch
import torch.nn as nn
import wandb
from mcr2_loss import MaximalCodingRateReduction
import pytorch_lightning as pl
from models.classifiers import *
from easydict import EasyDict


def to_wandb_im(x):
    if len(x.shape) == 3:
        x = x.permute(1, 2, 0)
    return x.cpu().numpy()


# TODO: generalize this to a module which utilizes generic encoders
class MainModel(pl.LightningModule):
    def __init__(self, encoder, num_classes, feat_dim, loss, task, lr, arch, class_labels, mcr2_bg_acc_threshhold=0.95, bg_encoder=None, bg_weight=1.0, **unused_kwargs):
        super(MainModel, self).__init__()
        self.bg_encoder = bg_encoder
        self.bg_weight = bg_weight
        self.encoder = encoder
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss = loss
        self.task = task
        self.encode_arch = arch
        self.lr = lr
        self.class_labels = class_labels
        self.bg_criterion = None
        self.mcr2_bg_acc_threshhold = mcr2_bg_acc_threshhold
        self.mcr2_starts = False

        if self.loss == 'mcr2_bg':
            self.bg_criterion = nn.CrossEntropyLoss()
        if self.loss == 'mcr2' or self.loss == 'mcr2_bg':
            self.criterion = MaximalCodingRateReduction(num_classes)
            self.classifier = None
            self.reset_agg()
        elif self.loss == 'ce' and self.task == 'classify':
            self.criterion = nn.CrossEntropyLoss()
            self.classifier = nn.Linear(feat_dim, num_classes)
        elif self.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
            self.classifier = nn.Conv2d(feat_dim, num_classes, kernel_size=1, padding=0)
        self.accuracy = pl.metrics.Accuracy()

    def reset_agg(self):
        self.__ZtPiZ = torch.zeros(self.num_classes, self.feat_dim, self.feat_dim).cuda()
        self.__Z_mean = torch.zeros(self.feat_dim).cuda()
        self.__num_batches = 0.

    @property
    def ZtPiZ(self):
        return self.__ZtPiZ / self.__num_batches

    @property
    def Z_mean(self):
        return self.__Z_mean / self.__num_batches

    def forward(self, x, labels=None, log=False, log_img=False):
        """

        :param x:
        :param labels:
        :param log: str with logging prefix
        :param log_img: bool
        :return:
        """
        metrics = EasyDict()
        feats = self.encoder(x)
        preds = None

        if self.task == 'classify':
            labels, _ = labels.view(labels.shape[0], -1).max(-1)
            if self.encode_arch == 'cnn':
                feats = feats.view(*feats.shape[:-2], -1).mean(-1)

        if self.loss == 'mcr2':
            feats = feats / torch.norm(feats, dim=1, keepdim=True)

            Z = feats.transpose(0, 1).reshape(feats.shape[1], -1).T
            Y = labels.view(-1)
            mcr_ret = self.criterion(Z, Y)
            self.__ZtPiZ += mcr_ret.ZtPiZ
            self.__Z_mean += mcr_ret.Z_mean
            self.__num_batches += 1

            mrc2_loss = mcr_ret.loss
            preds = self.classifier(Z).view(x.shape[0], *x.shape[-2:]) if self.classifier else None

            metrics.update(
                discrim_loss=mcr_ret.discrim_loss,
                compress_loss=mcr_ret.compress_loss,
                ZtPiZ_mean=torch.mean(mcr_ret.ZtPiZ),
                Z_mean=torch.mean(mcr_ret.Z_mean),
            )
        elif self.loss == 'mcr2_bg':
            mcr2_loss = 0

            if self.bg_encoder:
                bg_logits = self.bg_encoder(x)
            else:
                bg_logits, feats = feats[:, :2, :, :], feats[:, 2:, :, :]

            bg_mask = torch.argmax(bg_logits, 1).reshape(-1).detach()
            bg_logits = bg_logits.transpose(0, 1).reshape(bg_logits.shape[1], -1).T
            bg_labels = (labels.reshape(-1) == 0).type(torch.long)
            bg_acc = self.accuracy(bg_mask, bg_labels)
            
            bg_loss = self.bg_criterion(bg_logits, bg_labels)

            if self.mcr2_starts:
                Z = feats.transpose(0, 1).reshape(feats.shape[1], -1).T
                Z = Z[bg_mask == 1]
                Z = Z / torch.norm(Z, dim=1, keepdim=True)
                Y = labels.view(-1)
                Y = Y[bg_mask == 1]
                
                mcr_ret = self.criterion(Z, Y)
                self.__ZtPiZ += mcr_ret.ZtPiZ
                self.__Z_mean += mcr_ret.Z_mean
                self.__num_batches += 1

                mrc2_loss = mcr_ret.loss
                preds = self.classifier(Z).view(x.shape[0], *x.shape[-2:]) if self.classifier else None

                metrics.update(
                    discrim_loss=mcr_ret.discrim_loss,
                    compress_loss=mcr_ret.compress_loss,
                    ZtPiZ_mean=torch.mean(mcr_ret.ZtPiZ),
                    Z_mean=torch.mean(mcr_ret.Z_mean),
                )

            metrics.update(
                bg_loss=bg_loss,
                bg_acc=bg_acc
            )
            loss = self.bg_weight * bg_loss + mcr2_loss
        else:
            logits = self.classifier(feats)
            loss = self.criterion(logits, labels) if labels is not None else None
            preds = logits.argmax(dim=1)

        metrics.loss = loss
        if preds is not None and labels is not None:
            metrics.acc = self.accuracy(preds, labels)

        # All logging ops
        if log:
            for k in metrics:
                m = metrics[k]
                if k in {'loss' or 'acc'}:
                    self.log(f'{log}_{k}', m, prog_bar=True)
                else:
                    self.log(f'{log}_{k}', m)

            if log_img:
                if self.task == 'semseg':
                    imgs = []
                    for i in range(10):
                        masks = {}
                        if labels is not None:
                            masks["ground_truth"] = {"mask_data": to_wandb_im(labels[i]), "class_labels": self.class_labels}
                        if preds is not None:
                            masks["predictions"] = {"mask_data": to_wandb_im(preds[i]), "class_labels": self.class_labels}
                        imgs.append(wandb.Image(to_wandb_im(x[i]), masks=masks, caption=f'idx{i}'))
                    self.logger.experiment.log({f'{log}_imgs': imgs}, commit=False)

        return EasyDict(
            loss=loss,
            feats=feats.detach(),
            preds=preds,
            metrics=metrics,
        )

    def training_step(self, batch, batch_idx):
        # x      shape (batch_size, C,     H, W)
        # feats  shape (batch_size, feat_dim, H, W)
        # labels shape (batch_size, H, W)
        x, labels = batch
        labels = torch.squeeze(labels)
        ret = self(x, labels, log='train', log_img=(batch_idx % 50 == 10))

        return ret.loss

    def training_epoch_end(self, outputs):
        ...

    # TODO: might need to re-compute classifier over validation epoch
    # TODO: --OR-- compute classifier for each validation batch?
    # TODO: try other classifiers of Z
    def on_validation_epoch_start(self):
        if self.loss == 'mcr2':
            self.classifier = FastNearestSubspace(self.ZtPiZ, self.Z_mean,
                                                  num_classes=self.num_classes,
                                                  n_components=self.feat_dim // self.num_classes)
            self.reset_agg()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, labels = batch
            labels = torch.squeeze(labels)
            ret = self(x, labels, log='val', log_img=(batch_idx % 10 == 0))
            
            bg_val_acc = ret.metrics.bg_acc if 'bg_acc' in ret.metrics else 0.0
            self.mcr2_starts = self.mcr2_starts or bg_val_acc >= self.mcr2_bg_acc_threshhold

            # TODO: agg over epoch
            # TODO: confusion matrix
        return

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)