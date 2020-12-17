CUDA_VISIBLE_DEVICES=4 python3 train_supervised.py --task semseg --arch unet --loss ce --bs 500 --name "ce qmnist" --data qmnist --feat_dim 128 --lr 0.003 --entity axquaris &
CUDA_VISIBLE_DEVICES=5 python3 train_supervised.py --task semseg --arch unet --loss mcr2 --bs 500 --name "mcr2 qmnist" --data qmnist --feat_dim 128 --lr 0.003 --entity axquaris &
CUDA_VISIBLE_DEVICES=6 python3 train_supervised.py --task semseg --arch unet --loss ce --bs 500 --name "ce digits" --data digits --feat_dim 128 --lr 0.003 --entity axquaris &
CUDA_VISIBLE_DEVICES=7 python3 train_supervised.py --task semseg --arch unet --loss mcr2 --bs 500 --name "mcr2 digits" --data digits --feat_dim 128 --lr 0.003 --entity axquaris
