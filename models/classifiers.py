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

class FastNearestSubspace(nn.Module):
    def __init__(self, feat_cov, feat_mean, num_classes, n_components=5):
        super(FastNearestSubspace, self).__init__()
        # All array inputs should be tensors
        self.num_classes = num_classes
        self.n_components = n_components
        self.feat_mean = feat_mean
        self.feat_dim = feat_mean.shape[0]

        self.projections = self._get_projections(feat_cov)


    def forward(self, features):
        """
        Classifies features using projection matrix
        :param features: shape (num_samples, dim_z)
        :return:
        """
        features_ = features - self.feat_mean

        # self.projections shape (num_classes, dim_z, dim_z)
        # features_        shape (num_samples, dim_z)
        residual = features_.unsqueeze(0) @ self.projections.transpose(-1, -2)
        # residual         shape (num_samples, num_classes, dim_z)
        scores = torch.norm(residual, p=2, dim=-1)
        # scores           shape (num_samples, num_classes)
        test_preds = torch.argmin(scores, dim=0)

        return test_preds

    def _get_projections(self, feat_cov):
        """

        :param feat_cov: Feature covariance* matricies
            shape (num_classes, dim_z, dim_z)
        :return: returns projection matrix penalizing other class subspace activations
            shape (num_classes, dim_z, dim_z)
        """
        eigvals, eigvecs = torch.symeig(feat_cov, eigenvectors=True)
        _, indices = torch.topk(eigvals, self.n_components)
        top_eigvecs = []
        for i in range(self.num_classes):  # TODO: batch this https://discuss.pytorch.org/t/batched-index-select/9115/7
            top_eigvecs.append(torch.index_select(eigvecs[i], 1, indices[i]))
        top_eigvecs = torch.stack(top_eigvecs)

        projections = top_eigvecs @ top_eigvecs.transpose(-1, -2)
        return torch.eye(self.feat_dim).cuda() - projections

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






        