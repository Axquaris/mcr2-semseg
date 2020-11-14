from data.mnist_semseg import MnistSS
from models import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader


def main():
    train_dataset = MnistSS(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=500, shuffle=True, num_workers=4)
    val_dataset = MnistSS(train=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

    model = cnn.get_mnist_semseg()
    trainer = pl.Trainer(gpus=1, max_epochs=5)

    trainer.fit(model, train_dataloader)

    # Nearest subspace classifier needs training (latent) features and labels to construct projections
    clf = NearestSubpace(train_latent_features, train_lables, num_classes, n_components)

    # forward takes in test (latent) features
    test_preds = cfl(test_latent_features)



if __name__ == "__main__":
    main()
