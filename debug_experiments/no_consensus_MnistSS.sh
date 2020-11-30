# Commit: https://github.com/Axquaris/mcr2-semseg/commit/35d794d4a836fb10b7193adca3315281c97394c1
python train_supervised.py \
  --name no_consensus_mcr2 \
  --data mnist --es 10 --bs 500 \
  --arch cnn --task semseg --loss mcr2 --eps .5 \
  --fd 128

python train_supervised.py \
  --name no_consensus_crossentropy \
  --data mnist --es 10 --bs 500 \
  --arch cnn --task semseg --loss ce \
  --fd 128