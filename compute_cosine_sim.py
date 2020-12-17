import matplotlib
matplotlib.use('TkAgg')

import torch, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
from data.digit_semseg import DigitSS
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset
from models.main_model import MainModel
from models import unet
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer


def main(batch_size, num_samples, checkpoint_path, outfile):
    val_dataset = DigitSS(train=False, cifar_bg=False)
    class_labels = val_dataset.class_labels
    # val_dataset = Subset(val_dataset, indices=list(range(num_samples)))
    dataloader_args = EasyDict(batch_size=batch_size, shuffle=False, num_workers=1)
    val_dataloader = DataLoader(val_dataset, **dataloader_args)

    model = MainModel.load_from_checkpoint(checkpoint_path, class_labels=class_labels)
    model.cuda()
    model.eval()

    all_Z = []
    all_Y = []

    for i, data in enumerate(val_dataloader, 0):
        x, labels = data
        x = x.cuda()
        labels = labels.cuda()
        labels = torch.squeeze(labels)

        feats = model(x, labels).feats
        feats = feats / torch.norm(feats, dim=1, keepdim=True)
        Z = feats.transpose(0, 1).reshape(feats.shape[1], -1).T
        Y = labels.view(-1)

        all_Z.append(Z)
        all_Y.append(Y)

    Z = torch.cat(all_Z)[:num_samples]
    Y = torch.cat(all_Y)[:num_samples]
    

    idx = torch.argsort(Y)
    Z = Z[idx]
    Z_t = Z.transpose(0, 1)

    del model
    del idx
    torch.cuda.empty_cache()
    cosine_sim = torch.matmul(Z, Z_t).cpu().numpy()

    with open(outfile, 'wb') as f:
        pickle.dump(cosine_sim, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, default=500)
    parser.add_argument('--num_samples', '-n', type=int, default=25000)
    parser.add_argument('--checkpoint_path', '-cpt', type=str, default='wandb/latest-run/files/model-epoch=0.ckpt')
    parser.add_argument('--outfile', '-o', type=str, default='cosine_sim.pkl')
    args = vars(parser.parse_args())
    main(**args)