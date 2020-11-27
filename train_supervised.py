from data.mnist_semseg import MnistSS

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import argparse
from easydict import EasyDict


def main():
    parser = argparse.ArgumentParser(description='Supervised Learning')

    parser.add_argument('--data', type=str, default='mnist', help='mnist or mnist_bg')
    parser.add_argument('--es', type=int, default=10, help='num epochs (default: 10)')
    parser.add_argument('--bs', type=int, default=1000, help='batch size (default: 1000)')

    parser.add_argument('--arch', type=str, default='cnn', help='')
    parser.add_argument('--task', type=str, default='classify', help='semseg or classify')
    parser.add_argument('--loss', type=str, default='mcr2', help='mcr2 or ce (cross-entropy)')
    parser.add_argument('--eps', type=float, default=.5, help='mcr2 eps param')
    parser.add_argument('--fd', type=int, default=128, help='dimension of feature dimension (default: 128)')

    args = EasyDict(vars(parser.parse_args()))

    if 'mnist' in args.data:
        train_dataset = MnistSS(train=True, cifar_bg='bg' in args.data)
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
        val_dataset = MnistSS(train=False, cifar_bg='bg' in args.data)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
    else:
        raise NotImplementedError(args.data)

    im_channels = 1 if 'mnist' in args.data else 3
    if args.arch == 'cnn':
        from models import cnn
        encoder = cnn.get_mnist_semseg(in_c=im_channels, feat_dim=args.fd)
        model = cnn.CNN(encoder, 11, dim_z=args.fd, loss=args.loss, task=args.task)
    else:
        raise NotImplementedError(args.model)

    logger = WandbLogger(project='mcr2-semseg', config=args)
    trainer = pl.Trainer(gpus=1, max_epochs=args.es, logger=logger)

    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()
