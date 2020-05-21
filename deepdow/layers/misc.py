"""miscellaneous layers."""

import torch
import torch.nn as nn


class Cov2Corr(nn.Module):
    """Conversion from covariance matrix to correlation matrix."""

    def forward(self, covmat):
        """Convert.

        Parameters
        ----------
        covmat : torch.Tensor
            Covariance matrix of shape (n_samples, n_assets, n_assets).

        Returns
        -------
        corrmat : torch.Tensor
            Correlation matrix of shape (n_samples, n_assets, n_assets).

        """
        n_samples, n_assets, _ = covmat.shape
        stds = torch.sqrt(torch.diagonal(covmat, dim1=1, dim2=2))
        stds_ = stds.view(n_samples, n_assets, 1)

        corr = covmat / torch.matmul(stds_, stds_.permute(0, 2, 1))

        return corr


class CovarianceMatrix(nn.Module):
    """Covariance matrix or its square root.

    Parameters
    ----------
    sqrt : bool
        If True, then returning the square root.

    shrinkage_strategy : None or {'diagonal', 'identity', 'scaled_identity'}
        Strategy of combining the sample covariance matrix with some more stable matrix.

    shrinkage_coef : float or None
        If ``float`` then in the range [0, 1] representing the weight of the convex combination. If `shrinkage_coef=1`
        then using purely the sample covariance matrix. If `shrinkage_coef=0` then using purely the stable matrix.
        If None then needs to be provided dynamically when performing forward pass.
    """

    def __init__(self, sqrt=True, shrinkage_strategy='diagonal', shrinkage_coef=0.5):
        """Construct."""
        super().__init__()

        self.sqrt = sqrt

        if shrinkage_strategy is not None:
            if shrinkage_strategy not in {'diagonal', 'identity', 'scaled_identity'}:
                raise ValueError('Unrecognized shrinkage strategy {}'.format(shrinkage_strategy))

        self.shrinkage_strategy = shrinkage_strategy
        self.shrinkage_coef = shrinkage_coef

    def forward(self, x, shrinkage_coef=None):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, n_assets).

        shrinkage_coef : None or torch.Tensor
            If None then using the `self.shrinkage_coef` supplied at construction for each sample. Otherwise a
            tensor of shape `(n_shapes,)`.

        Returns
        -------
        covmat : torch.Tensor
            Of shape (n_samples, n_assets, n_assets).

        """
        n_samples = x.shape[0]
        dtype, device = x.dtype, x.device

        if not ((shrinkage_coef is None) ^ (self.shrinkage_coef is None)):
            raise ValueError('Not clear which shrinkage coefficient to use')

        if shrinkage_coef is not None:
            shrinkage_coef_ = shrinkage_coef  # (n_samples,)
        else:
            shrinkage_coef_ = self.shrinkage_coef * torch.ones(n_samples, dtype=dtype, device=device)

        wrapper = self.compute_sqrt if self.sqrt else lambda h: h

        return torch.stack([wrapper(self.compute_covariance(x[i].T.clone(),
                                                            shrinkage_strategy=self.shrinkage_strategy,
                                                            shrinkage_coef=shrinkage_coef_[i]))
                            for i in range(n_samples)], dim=0)

    @staticmethod
    def compute_covariance(m, shrinkage_strategy=None, shrinkage_coef=0.5):
        """Compute covariance matrix for a single sample.

        Parameters
        ----------
        m : torch.Tensor
            Of shape (n_assets, n_channels).

        shrinkage_strategy : None or {'diagonal', 'identity', 'scaled_identity'}
            Strategy of combining the sample covariance matrix with some more stable matrix.

        shrinkage_coef : torch.Tensor
            A ``torch.Tensor`` scalar (probably in the range [0, 1]) representing the weight of the
            convex combination.

        Returns
        -------
        covmat_single : torch.Tensor
            Covariance matrix of shape (n_assets, n_assets).

        """
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)  # !!!!!!!!!!! INPLACE
        mt = m.t()

        s = fact * m.matmul(mt)  # sample covariance matrix

        if shrinkage_strategy is None:
            return s

        elif shrinkage_strategy == 'identity':
            identity = torch.eye(len(s), device=s.device, dtype=s.dtype)

            return shrinkage_coef * s + (1 - shrinkage_coef) * identity

        elif shrinkage_strategy == 'scaled_identity':
            identity = torch.eye(len(s), device=s.device, dtype=s.dtype)
            scaled_identity = identity * torch.diag(s).mean()

            return shrinkage_coef * s + (1 - shrinkage_coef) * scaled_identity

        elif shrinkage_strategy == 'diagonal':
            diagonal = torch.diag(torch.diag(s))

            return shrinkage_coef * s + (1 - shrinkage_coef) * diagonal

    @staticmethod
    def compute_sqrt(m):
        """Compute the square root of a single positive definite matrix.

        Parameters
        ----------
        m : torch.Tensor
            Tensor of shape `(n_assets, n_assets)` representing the covariance matrix - needs to be PSD.

        Returns
        -------
        m_sqrt : torch.Tensor
            Tensor of shape `(n_assets, n_assets)` representing the square root of the covariance matrix.

        """
        _, s, v = m.svd()

        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]  # pragma: no cover
            v = v[..., :common]  # pragma: no cover
            if unbalanced:  # pragma: no cover
                good = good[..., :common]  # pragma: no cover
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))  # pragma: no cover

        return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


class KMeans(torch.nn.Module):
    """K-means algorithm.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to look for.

    init : str, {'random, 'k-means++', 'manual'}
        How to initialize the clusters at the beginning of the algorithm.

    n_init : int
        Number of times the algorithm is run. The best clustering is determined based on the
        potential (sum of distances of all points to the centroids).

    max_iter : int
        Maximum number of iterations of the algorithm. Note that if `norm(new_potential - old_potential) < tol`
        then stop prematurely.

    tol : float
        If `abs(new_potential - old_potential) < tol` then algorithm stopped irrespective of the `max_iter`.

    random_state : int or None
        Setting randomness.

    verbose : bool
        Control level of verbosity.
    """

    def __init__(self, n_clusters=5, init='random', n_init=1, max_iter=30, tol=1e-5, random_state=None, verbose=False):
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        if self.init not in {'manual', 'random', 'k-means++'}:
            raise ValueError('Unrecognized initialization {}'.format(self.init))

    def initialize(self, x, manual_init=None):
        """Initialize the k-means algorithm.

        Parameters
        ----------
        x : torch.Tensor
            Feature matrix of shape `(n_samples, n_features)`.

        manual_init : None or torch.Tensor
            If not None then expecting a tensor of shape `(n_clusters, n_features)`. Note that for this feature
            to be used one needs to set `init='manual'` in the constructor.

        Returns
        -------
        cluster_centers : torch.Tensor
            Tensor of shape `(n_clusters, n_features)` representing the initial cluster centers.

        """
        n_samples, n_features = x.shape
        device, dtype = x.device, x.dtype

        # Note that normalization to probablities is done automatically within torch.multinomial
        if self.init == 'random':
            p = torch.ones(n_samples, dtype=dtype, device=device)
            # centroid_samples = torch.randperm(n_samples).to(device=device)[:self.n_clusters]
            centroid_samples = torch.multinomial(p, num_samples=self.n_clusters, replacement=False)
            cluster_centers = x[centroid_samples]

        elif self.init == 'k-means++':
            p = torch.ones(n_samples, dtype=dtype, device=device)
            cluster_centers_l = []
            centroid_samples_l = []

            while len(cluster_centers_l) < self.n_clusters:
                centroid_sample = torch.multinomial(p, num_samples=1, replacement=False)

                if centroid_sample in centroid_samples_l:
                    continue  # pragma: no cover
                centroid_samples_l.append(centroid_sample)

                cluster_center = x[[centroid_sample]]  # (1, n_features)
                cluster_centers_l.append(cluster_center)
                p = self.compute_distances(x, cluster_center).view(-1)

            cluster_centers = torch.cat(cluster_centers_l, dim=0)

        elif self.init == 'manual':
            if not torch.is_tensor(manual_init):
                raise TypeError('The manual_init needs to be a torch.Tensor')

            if manual_init.shape[0] != self.n_clusters:
                raise ValueError('The number of manually provided cluster centers is different from n_clusters')

            if manual_init.shape[1] != x.shape[1]:
                raise ValueError('The feature size of manually provided cluster centers is different from the input')

            cluster_centers = manual_init.to(dtype=dtype, device=device)

        return cluster_centers

    def forward(self, x, manual_init=None):
        """Perform clustering.

        Parameters
        ----------
        x : torch.Tensor
            Feature matrix of shape `(n_samples, n_features)`.

        manual_init : None or torch.Tensor
            If not None then expecting a tensor of shape `(n_clusters, n_features)`. Note that for this feature
            to be used one needs to set `init='manual'` in the constructor.

        Returns
        -------
        cluster_ixs : torch.Tensor
            1D array of lenght `n_samples` representing to what cluster each sample belongs.

        cluster_centers : torch.tensor
            Tensor of shape `(n_clusters, n_features)` representing the cluster centers.

        """
        n_samples, n_features = x.shape
        if n_samples < self.n_clusters:
            raise ValueError('The number of samples is lower than the number of clusters.')

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        lowest_potential = float('inf')
        lowest_potential_cluster_ixs = None
        lowest_potential_cluster_centers = None

        for run in range(self.n_init):
            cluster_centers = self.initialize(x, manual_init=manual_init)
            previous_potential = float('inf')

            for it in range(self.max_iter):
                distances = self.compute_distances(x, cluster_centers)  # (n_samples, n_clusters)

                # E step
                cluster_ixs = torch.argmin(distances, dim=1)  # (n_samples,)

                # M step
                cluster_centers = torch.stack([x[cluster_ixs == i].mean(dim=0) for i in range(self.n_clusters)], dim=0)

                # stats
                current_potential = distances.gather(1, cluster_ixs.view(-1, 1)).sum()

                if abs(current_potential - previous_potential) < self.tol or it == self.max_iter - 1:
                    if self.verbose:
                        print('Run: {}, n_iters: {}, stop_early: {}, potential: {:.3f}'.format(run,
                                                                                               it,
                                                                                               it != self.max_iter - 1,
                                                                                               current_potential))
                    break

                previous_potential = current_potential

            if current_potential < lowest_potential:
                lowest_potential = current_potential
                lowest_potential_cluster_ixs = cluster_ixs.clone()
                lowest_potential_cluster_centers = cluster_centers.clone()

        if self.verbose:
            print('Lowest potential: {}'.format(lowest_potential))

        return lowest_potential_cluster_ixs, lowest_potential_cluster_centers

    @staticmethod
    def compute_distances(x, cluster_centers):
        """Compute squared distances of samples to cluster centers.

        Parameters
        ----------
        x : torch.tensor
            Tensor of shape `(n_samples, n_features)`.

        cluster_centers : torch.tensor
            Tensor of shape `(n_clusters, n_features)`.

        Returns
        -------
        distances : torch.tensor
            Tensor of shape `(n_samples, n_clusters)` that provides for each sample (row) the squared distance
            to a given cluster center (column).

        """
        x_n = (x ** 2).sum(dim=1).view(-1, 1)  # (n_samples, 1)
        c_n = (cluster_centers ** 2).sum(dim=1).view(1, -1)  # (1, n_clusters)

        distances = x_n + c_n - 2 * torch.mm(x, cluster_centers.permute(1, 0))  # (n_samples, n_clusters)

        return torch.clamp(distances, min=0)


class MultiplyByConstant(torch.nn.Module):
    """Multiplying constant.

    Parameters
    ----------
    dim_size : int
        Number of input channels. We learn one constant per channel. Therefore `dim_size=n_trainable_parameters`.

    dim_ix : int
        Which dimension to apply the multiplication to.
    """

    def __init__(self, dim_size=1, dim_ix=1):
        super().__init__()

        self.dim_size = dim_size
        self.dim_ix = dim_ix
        self.constant = torch.nn.Parameter(torch.ones(self.dim_size), requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1})

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (d_0, d_1, ..., d_{N-1}).

        """
        if self.dim_size != x.shape[self.dim_ix]:
            raise ValueError('The size of dimension {} is {} which is different than {}'.format(self.dim_ix,
                                                                                                x.shape[self.dim_ix],
                                                                                                self.dim_size))
        view = [self.dim_size if i == self.dim_ix else 1 for i in range(x.ndim)]
        return x * self.constant.view(view)
