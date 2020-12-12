from data.mnist_semseg import MnistSS
from models import cnn, resnet
from models.main_model import MainModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import argparse
from easydict import EasyDict


def main():
    parser = argparse.ArgumentParser(description='Supervised Learning')
    parser.add_argument('--name', type=str, default=None, help='Name of run')

    parser.add_argument('--data', type=str, default='mnist', help='mnist or mnist_bg')
    parser.add_argument('--es', type=int, default=10, help='num epochs (default: 10)')
    parser.add_argument('--bs', type=int, default=1000, help='batch size (default: 1000)')

    parser.add_argument('--arch', type=str, choices=['cnn', 'resnet10', 'resnet18', 'unet'], default='cnn', help='What encoder to use')
    parser.add_argument('--task', type=str, default='classify', help='semseg or classify')
    parser.add_argument('--loss', type=str, choices=['mcr2', 'ce'], default='mcr2', help='mcr2 or ce (cross-entropy)')
    parser.add_argument('--eps', type=float, default=.5, help='mcr2 eps param')
    parser.add_argument('-fd', '--feat_dim', type=int, default=128, help='dimension of features (default: 128)')
    parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')

    parser.add_argument('--debug', action='store_true', default=False)

    args = EasyDict(vars(parser.parse_args()))

    if 'mnist' in args.data:
        train_dataset = MnistSS(train=True, cifar_bg='bg' in args.data)
        val_dataset = MnistSS(train=False, cifar_bg='bg' in args.data)
        im_channels = 1
    else:
        raise NotImplementedError(args.data)

    dataloader_args = EasyDict(batch_size=args.bs, shuffle=False, num_workers=0 if args.debug else 4)
    train_dataloader = DataLoader(train_dataset, **dataloader_args)
    val_dataloader = DataLoader(val_dataset, **dataloader_args)

    if args.arch == 'cnn':
        encoder = cnn.get_mnist_semseg(in_c=im_channels, feat_dim=args.feat_dim)
    elif args.arch == 'resnet10':
        encoder = resnet.get_mnist_resnet(in_c=im_channels, feat_dim=args.feat_dim, depth="10")
    elif args.arch == 'resnet18':
        encoder = resnet.get_mnist_resnet(in_c=im_channels, feat_dim=args.feat_dim, depth="18")
    elif args.arch == 'unet':
        from models import cnn
        encoder = cnn.get_mnist_unet(in_c=im_channels, feat_dim=args.feat_dim)
    else:
        raise NotImplementedError(args.model)
    model = MainModel(encoder, 11, **args, class_labels=train_dataset.class_labels)

    logger = WandbLogger(project='mcr2-semseg', config=args)
    trainer = pl.Trainer(gpus=1, max_epochs=args.es, logger=logger, auto_select_gpus=True, log_every_n_steps=10)

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
