import torch
import torch.nn as nn
import numpy as np

class NearestSubspace(nn.Module):

    def __init__(self, train_features, train_labels, num_classes, n_components=5):
        super(NearestSubspace, self).__init__()
        # All array inputs should be tensors
        self.num_classes = num_classes
        self.n_components = n_components
        self.feature_dimension = train_features.shape[1]
        self.projections = self._get_projections(train_features, train_labels)
        self.mean = torch.mean(train_features, axis=0)


    def forward(self, test_features):
        scores = []
        test_features = test_features - self.mean
        for i in range(self.num_classes):
            residual = torch.matmul(torch.eye(self.feature_dimension) - self.projections[i], test_features.transpose(1, 0))
            scores_i = torch.norm(residual, p=2, dim=0).numpy()
            scores.append(scores_i)
        
        test_preds = np.argmin(scores, axis=0)

        return test_preds

    def _get_projections(self, features, labels):
        projections = []
        features_grouped = self._group_by_label(features.numpy(), labels.numpy())

        for features in features_grouped:
            features = torch.tensor(features).cuda()
            x_t_x = torch.matmul(features.transpose(1, 0), features)
            eigvals, eigvecs = torch.symeig(torch.matmul(features.transpose(1, 0), features), eigenvectors=True)
            _, indices = torch.topk(eigvals, self.n_components)
            eigvecs = eigvecs[:, indices]
            projection = torch.matmul(eigvecs, eigvecs.transpose(1, 0))
            projections.append(projection)

        return projections

    def _group_by_label(self, features, labels):
        # Takes in numpy arrays and outputs numpy arrays
        grouped = []
        for i in range(self.num_classes):
            idx = np.where(labels == i)[0]
            features_i = features[idx,:]
            grouped.append(features_i)

        return grouped

if __name__ == '__main__':
    # Simple test to make sure nothing breaks

    num_classes = 3
    num_train_samples = 20
    num_test_samples = 7
    latent_dim_size = 10

    train_feats = torch.tensor(np.random.rand(num_train_samples, latent_dim_size), dtype=torch.float32)
    train_labels = torch.tensor(np.random.randint(num_classes, size=num_train_samples), dtype=torch.float32)
    test_feats = torch.tensor(np.random.rand(num_test_samples, latent_dim_size), dtype=torch.float32)

    clf = NearestSubspace(train_feats, train_labels, num_classes)

    preds = clf(test_feats)

    assert preds.shape == (num_test_samples,)






        