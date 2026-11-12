import numpy as np
import torch
from scipy.linalg import cholesky, solve_triangular


def _is_torch(x):
    return isinstance(x, torch.Tensor)


def _to_numpy(x):
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _torch_vector(x_np, reference):
    return torch.as_tensor(x_np, dtype=reference.dtype, device=reference.device)


def _torch_matrix(x_np, reference):
    return torch.as_tensor(x_np, dtype=reference.dtype, device=reference.device)


class DistTransform:
    def __init__(self, name):
        self.name = name

    def transform(self, sample):
        raise NotImplementedError("Transform must implement transform method")

    def inverse_transform(self, sample):
        raise NotImplementedError("Transform must implement inverse_transform method")

    def transform_score(self, score):
        raise NotImplementedError("Transform must implement transform_score method")

    def inverse_transform_score(self, score):
        raise NotImplementedError("Transform must implement inverse_transform_score method")

    @property
    def log_abs_det_jacobian(self):
        return 0.5 * self.log_det_cov

    def wrap_target_score(self, target_score):
        def score_in_transformed_space(sample):
            original_sample = self.inverse_transform(sample)
            score_original = target_score(original_sample)
            return self.transform_score(score_original)

        return score_in_transformed_space


class LinearFullCov(DistTransform):
    """
    Affine Gaussian standardization.

    If cov = L @ L.T, then
        y = L^{-1} (z - mean)
        z = mean + L y.

    Scores follow the row-batch convention used in this repository:
        score_y = score_z @ L
        score_z = score_y @ L^{-1}.
    """

    def __init__(self, mean, cov):
        super().__init__('LinearFullCov')
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        assert self.mean.ndim == 1
        assert self.cov.ndim == 2
        assert self.cov.shape == (self.mean.shape[0], self.mean.shape[0])

        self.L = cholesky(self.cov, lower=True)
        self.log_det_cov = np.linalg.slogdet(self.cov)[1]

    def transform(self, sample):
        assert sample.ndim == 2
        if _is_torch(sample):
            mean = _torch_vector(self.mean, sample)
            L = _torch_matrix(self.L, sample)
            centered = sample - mean[None, :]
            return torch.linalg.solve_triangular(
                L, centered.T, upper=False
            ).T

        sample_np = _to_numpy(sample)
        centered = sample_np - self.mean[None, :]
        return solve_triangular(self.L, centered.T, lower=True).T
    
    def transform_score(self, score):
        if _is_torch(score):
            L = _torch_matrix(self.L, score)
            return score @ L

        score_np = _to_numpy(score)
        return score_np @ self.L

    def inverse_transform_score(self, score):
        if _is_torch(score):
            L = _torch_matrix(self.L, score)
            return torch.linalg.solve_triangular(
                L.T, score.T, upper=True
            ).T

        score_np = _to_numpy(score)
        return solve_triangular(self.L.T, score_np.T, lower=False).T
    
    def inverse_transform(self, sample):
        assert sample.ndim == 2
        if _is_torch(sample):
            mean = _torch_vector(self.mean, sample)
            L = _torch_matrix(self.L, sample)
            return sample @ L.T + mean[None, :]

        sample_np = _to_numpy(sample)
        return sample_np @ self.L.T + self.mean[None, :]


class LinearDiagonalCov(DistTransform):
    def __init__(self, mean, cov):
        super().__init__('LinearDiagonalCov')
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        assert self.mean.ndim == 1
        assert self.cov.ndim == 1
        assert self.cov.shape == self.mean.shape

        self.std = np.sqrt(self.cov)
        self.log_det_cov = np.sum(np.log(self.cov))

    def transform(self, sample):
        assert sample.ndim == 2
        if _is_torch(sample):
            mean = _torch_vector(self.mean, sample)
            std = _torch_vector(self.std, sample)
            return (sample - mean[None, :]) / std[None, :]

        sample_np = _to_numpy(sample)
        return (sample_np - self.mean[None, :]) / self.std[None, :]

    def transform_score(self, score):
        if _is_torch(score):
            std = _torch_vector(self.std, score)
            return score * std[None, :]

        score_np = _to_numpy(score)
        return score_np * self.std[None, :]

    def inverse_transform_score(self, score):
        if _is_torch(score):
            std = _torch_vector(self.std, score)
            return score / std[None, :]

        score_np = _to_numpy(score)
        return score_np / self.std[None, :]
    
    def inverse_transform(self, sample):
        assert sample.ndim == 2
        if _is_torch(sample):
            mean = _torch_vector(self.mean, sample)
            std = _torch_vector(self.std, sample)
            return sample * std[None, :] + mean[None, :]

        sample_np = _to_numpy(sample)
        return sample_np * self.std[None, :] + self.mean[None, :]


class Central(DistTransform):
    def __init__(self, mean):
        super().__init__('Central')
        self.mean = np.asarray(mean, dtype=np.float64)
        assert self.mean.ndim == 1

        self.log_det_cov = 0.0

    def transform(self, sample):
        assert sample.ndim == 2
        if _is_torch(sample):
            mean = _torch_vector(self.mean, sample)
            return sample - mean[None, :]

        sample_np = _to_numpy(sample)
        return sample_np - self.mean[None, :]

    def transform_score(self, score):
        return score

    def inverse_transform_score(self, score):
        return score
    
    def inverse_transform(self, sample):
        assert sample.ndim == 2
        if _is_torch(sample):
            mean = _torch_vector(self.mean, sample)
            return sample + mean[None, :]

        sample_np = _to_numpy(sample)
        return sample_np + self.mean[None, :]


GaussianStandardizer = LinearFullCov


def make_gaussian_standardizer(mean, cov, diagonal=False):
    if diagonal:
        return LinearDiagonalCov(mean, cov)
    return LinearFullCov(mean, cov)


