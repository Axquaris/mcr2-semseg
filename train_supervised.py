from data.digit_semseg import DigitSS
from data.bdd100k_semseg import BddSS
from models import cnn, resnet, unet
from models.main_model import MainModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import argparse
from easydict import EasyDict
import torch

def main():
    parser = argparse.ArgumentParser(description='Supervised Learning')
    parser.add_argument('--name', type=str, default=None, help='Name of run')
    parser.add_argument('--entity', type=str, default=None, help='Wandb username or team')

    parser.add_argument('--data', type=str, choices=['mnist', 'bdd', 'qmnist', 'usps', 'digits'],
                        default='mnist', help='digits is for both qmnist and usps')
    parser.add_argument('--bg', action='store_true', default=False)
    parser.add_argument('--es', type=int, default=100, help='num epochs (default: 10)')
    parser.add_argument('--bs', type=int, default=1000, help='batch size (default: 1000)')

    parser.add_argument('--arch', type=str, choices=['cnn', 'resnet10', 'resnet18', 'unet', 'unet_bg', 'unet_bg_separate'], default='cnn', help='What encoder to use')
    parser.add_argument('--task', type=str, default='classify', help='semseg or classify')
    parser.add_argument('--loss', type=str, choices=['mcr2', 'ce', 'hybrid'], default='mcr2', help='')
    parser.add_argument('--eps', type=float, default=.5, help='mcr2 eps param')
    parser.add_argument('-fd', '--feat_dim', type=int, default=128, help='dimension of features (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1., help='learning rate decay')
    parser.add_argument('--bg_weight', type=float, default=1., help="Relative weight of bg loss to mcr2 loss")

    parser.add_argument('--debug', action='store_true', default=False)

    args = EasyDict(vars(parser.parse_args()))
    for k in args.keys():
        if type(args[k]) == str:
            args[k] = args[k].lower()

    val_sets = None
    dataloader_args = EasyDict(batch_size=args.bs, shuffle=False, num_workers=0 if args.debug else 4)
    if args.data in ['mnist', 'qmnist', 'usps', 'digits']:
        if args.data == 'mnist':
            train_dataloader = DataLoader(DigitSS(train=True, cifar_bg=args.bg), **dataloader_args)
            val_dataloader = DataLoader(DigitSS(train=False, cifar_bg=args.bg), **dataloader_args)
        else:
            digit_sources_train = ['qmnist', 'usps'] if args.data == 'digits' else args.data
            train_dataloader = DataLoader(DigitSS(train=True, cifar_bg=args.bg, digit_sources=digit_sources_train), **dataloader_args)
            val_datasets = [DigitSS(train=False, cifar_bg=args.bg, digit_sources=['qmnist']),
                            DigitSS(train=False, cifar_bg=args.bg, digit_sources=['usps'])]
            val_dataloader = [DataLoader(ds, **dataloader_args) for ds in val_datasets]
            val_sets = ['qmnist', 'usps']

        if args.bg:
            im_channels = 3
        else:
            im_channels = 1
        # early_stopper = pl.callbacks.EarlyStopping(monitor='val_qmnist_acc', min_delta=.008, patience=5)
    elif args.data == 'bdd':
        train_dataloader = DataLoader(BddSS(train=True), **dataloader_args)
        val_dataloader = DataLoader(BddSS(train=False), **dataloader_args)
        im_channels = 3
        # early_stopper = pl.callbacks.EarlyStopping(monitor='val_acc', min_delta=.008, patience=2)
    else:
        raise NotImplementedError(args.data)

    bg_encoder = None
    if args.arch == 'cnn':
        encoder = cnn.get_mnist_semseg(in_c=im_channels, feat_dim=args.feat_dim)
    elif args.arch == 'resnet10':
        encoder = resnet.get_mnist_resnet(in_c=im_channels, feat_dim=args.feat_dim, depth="10")
    elif args.arch == 'resnet18':
        encoder = resnet.get_mnist_resnet(in_c=im_channels, feat_dim=args.feat_dim, depth="18")
    elif args.arch == 'unet':
        encoder = unet.UNet(n_channels=im_channels, feat_dim=args.feat_dim)
    elif args.arch == 'unet_bg':
        encoder = unet.UNet(n_channels=im_channels, feat_dim=args.feat_dim + 2)
        args['loss'] = 'mcr2_bg'
    elif args.arch == 'unet_bg_separate':
        encoder = unet.UNet(n_channels=im_channels, feat_dim=args.feat_dim)
        bg_encoder = unet.UNet(n_channels=im_channels, feat_dim=2)
        args['loss'] = 'mcr2_bg'
    else:
        raise NotImplementedError(args.model)

    model = MainModel(encoder, 11, bg_encoder=bg_encoder, class_labels=train_dataloader.dataset.class_labels,
                      val_sets=val_sets, **args)

    logger = WandbLogger(project='mcr2-semseg', config=args, name=args.name, entity=args.entity)
    # checkpointer = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, dirpath=logger.experiment.dir, prefix='model')
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, dirpath=logger.experiment.dir, prefix='model')

    trainer = pl.Trainer(gpus=1, max_epochs=args.es, logger=logger, auto_select_gpus=True, log_every_n_steps=10,
                         auto_lr_find=False, benchmark=True, terminate_on_nan=True,
                         callbacks=[checkpointer], limit_val_batches=10)
    # trainer.tune(model, train_dataloader)
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
