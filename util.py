from matplotlib.pyplot import plt
import numpy as np
import torch


def plot_data(X):
    """
    Generic function to plot the images in a grid
    of num_plot x num_plot
    :param X:
    :return:
    """
    plt.figure()
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for i in range(num_plot):
        for j in range(num_plot):
            idx = np.random.randint(0, X.shape[0])
            ax[i, j].imshow(X[idx])
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)  # No horizontal space between subplots
    f.subplots_adjust(wspace=0)


def cos_dist_matrix(model, dataloader):
    with torch.no_grad:
        dataset = dataloader.dataset

        features = []
        labels = []
        attrs = []
        for batch in dataloader:
            img, label, attr = batch
            feats = model.encoder(img)[..., ::4, ::4]
            feats = feats / torch.norm(feats, dim=1, keepdim=True)
            Z = feats.transpose(0, 1).reshape(feats.shape[1], -1).T
            features.append(Z)

            label = label[..., ::4, ::4]
            Y = label.view(-1)
            labels.append(Y)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        ordered_features = []
        for idx in dataset.class_labels:
            ordered_features.append(features[labels == idx])
        ordered_features = torch.cat(ordered_features, dim=0)
        return ordered_features @ ordered_features.T

