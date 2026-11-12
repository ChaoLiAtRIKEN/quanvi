import math
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily, Distribution, Independent


DTYPE = torch.float64
EPS = torch.tensor(1e-8, dtype=DTYPE)


def _resolve_device(device):
    """Normalize a device spec (None / str / torch.device) to torch.device."""
    if device is None:
        return torch.device('cpu')
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class _RingDistribution(Distribution):
    """
    From Codex:
    - There is a mismatch between sample() and log_prob(). When sampling r, there is torch.abs().
    - This is not a standard way of implementing the ring distribution.
    """
    def __init__(self, ring_radius, sigma, validate_args=None):
        self.ring_radius = torch.tensor(ring_radius, dtype=DTYPE)
        self.sigma = torch.tensor(sigma, dtype=DTYPE)
        batch_shape = torch.Size([])
        event_shape = torch.Size([2])
        super(RingDistribution, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self):
        raise NotImplementedError("RingDistribution does not have a closed-form mean")

    @property
    def variance(self):
        raise NotImplementedError("RingDistribution does not have a closed-form variance")

    def log_prob(self, value):
        # value is expected to be at least 2-dimensional with last dim size 2 (x, y)
        r = torch.norm(value, dim=-1)
        # Gaussian radial component centered at ring_radius
        log_radial = -0.5 * torch.square((r - self.ring_radius) / self.sigma)
        # Use torch.tensor to ensure constants are float64
        two_pi = torch.tensor(2 * math.pi, dtype=DTYPE)
        # Jacobian correction for the polar-to-Cartesian transformation
        log_jacobian = -torch.log(two_pi) - torch.log(r + EPS)
        return log_radial + log_jacobian

    def sample(self, sample_shape=torch.Size()):
        # Handle sample_shape
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])
        elif isinstance(sample_shape, (list, tuple)):
            sample_shape = torch.Size(sample_shape)
        
        # Total number of samples needed
        total_samples = sample_shape.numel() if sample_shape.numel() > 0 else 1
        
        # Sample the radial coordinate using a normal distribution
        r = torch.abs(torch.normal(
            mean=self.ring_radius.expand(total_samples),
            std=self.sigma.expand(total_samples)
        ))
        
        # Sample the angle uniformly between 0 and 2*pi
        theta = torch.rand(total_samples, dtype=DTYPE) * 2 * math.pi
        
        # Convert polar coordinates to Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        samples = torch.stack([x, y], dim=-1)
        
        # Reshape to match the requested sample_shape
        if sample_shape.numel() > 0:
            samples = samples.view(sample_shape + torch.Size([2]))
        else:
            samples = samples.squeeze(0)
            
        return samples

    def rsample(self, sample_shape=torch.Size()):
        # For reparameterized sampling (supports gradients)
        return self.sample(sample_shape)


class RingDistribution(Distribution):
    def __init__(self, ring_radius, sigma, validate_args=None, device=None):
        device = _resolve_device(device)
        self.device = device
        self.ring_radius = torch.tensor(ring_radius, dtype=DTYPE, device=device)
        self.sigma = torch.tensor(sigma, dtype=DTYPE, device=device)
        if torch.any(self.ring_radius < 0):
            raise ValueError("ring_radius must be nonnegative.")
        if torch.any(self.sigma <= 0):
            raise ValueError("sigma must be positive.")
        batch_shape = torch.Size([])
        event_shape = torch.Size([2])
        super(RingDistribution, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self):
        raise NotImplementedError("RingDistribution does not have a closed-form mean")

    @property
    def variance(self):
        raise NotImplementedError("RingDistribution does not have a closed-form variance")

    def log_prob(self, value):
        r = torch.norm(value, dim=-1)
        radius = self.ring_radius.to(dtype=value.dtype, device=value.device)
        sigma = self.sigma.to(dtype=value.dtype, device=value.device)

        energy = -0.5 * torch.square((r - radius) / sigma)
        log_normalizer = math.log(2 * math.pi) + torch.log(
            self._radial_integral(dtype=value.dtype, device=value.device)
        )
        return energy - log_normalizer

    def _radial_integral(self, dtype=DTYPE, device=None):
        radius = self.ring_radius.to(dtype=dtype, device=device)
        sigma = self.sigma.to(dtype=dtype, device=device)
        scaled_radius = radius / sigma
        return (
            radius * sigma * math.sqrt(math.pi / 2.0)
            * (1.0 + torch.erf(scaled_radius / math.sqrt(2.0)))
            + sigma.square() * torch.exp(-0.5 * scaled_radius.square())
        )

    def _radial_cdf(self, r):
        radius = self.ring_radius.to(dtype=r.dtype, device=r.device)
        sigma = self.sigma.to(dtype=r.dtype, device=r.device)
        scaled_radius = radius / sigma
        scaled_r = (r - radius) / sigma

        numerator = (
            sigma.square()
            * (
                torch.exp(-0.5 * scaled_radius.square())
                - torch.exp(-0.5 * scaled_r.square())
            )
            + radius * sigma * math.sqrt(math.pi / 2.0)
            * (
                torch.erf(scaled_r / math.sqrt(2.0))
                + torch.erf(scaled_radius / math.sqrt(2.0))
            )
        )
        cdf = numerator / self._radial_integral(dtype=r.dtype, device=r.device)
        return torch.clamp(cdf, min=0.0, max=1.0)

    def sample(self, sample_shape=torch.Size()):
        # Handle sample_shape
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])
        elif isinstance(sample_shape, (list, tuple)):
            sample_shape = torch.Size(sample_shape)
        
        # Total number of samples needed
        total_samples = sample_shape.numel() if sample_shape.numel() > 0 else 1

        device = self.device
        # The Cartesian density is proportional to
        # exp(-0.5 * ((||x|| - R) / sigma)^2), so the radial density carries
        # the polar Jacobian factor r.
        u = torch.rand(total_samples, dtype=DTYPE, device=device).clamp(
            min=torch.finfo(DTYPE).tiny,
            max=1.0 - torch.finfo(DTYPE).eps,
        )
        low = torch.zeros(total_samples, dtype=DTYPE, device=device)
        upper = torch.clamp(self.ring_radius + 12.0 * self.sigma, min=12.0 * self.sigma)
        high = torch.ones(total_samples, dtype=DTYPE, device=device) * upper
        for _ in range(64):
            mid = 0.5 * (low + high)
            cdf_mid = self._radial_cdf(mid)
            low = torch.where(cdf_mid < u, mid, low)
            high = torch.where(cdf_mid >= u, mid, high)
        r = 0.5 * (low + high)
        
        # Sample the angle uniformly between 0 and 2*pi
        theta = torch.rand(total_samples, dtype=DTYPE, device=device) * 2 * math.pi
        
        # Convert polar coordinates to Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        samples = torch.stack([x, y], dim=-1)
        
        # Reshape to match the requested sample_shape
        if sample_shape.numel() > 0:
            samples = samples.view(sample_shape + torch.Size([2]))
        else:
            samples = samples.squeeze(0)
            
        return samples

    def rsample(self, sample_shape=torch.Size()):
        # For reparameterized sampling (supports gradients)
        return self.sample(sample_shape)
    

class FunnelDistribution(Distribution):
    """
    Neal's Funnel distribution: x1 ~ N(0, sigma^2), x_i ~ N(0, exp(x1)), i = 2, ..., d
    This is a challenging distribution for sampling due to the strong correlation
    between x1 and the scale of other dimensions.
    
    Args:
        dim: Dimensionality of the distribution
        sigma: Standard deviation of the first dimension (default: 3.0, giving variance 9)
    """
    def __init__(self, sigma, dim=2, validate_args=None, device=None):
        device = _resolve_device(device)
        self.device = device
        self.dim = dim
        self.sigma = torch.tensor(sigma, dtype=DTYPE, device=device)
        batch_shape = torch.Size([])
        event_shape = torch.Size([dim])
        super(FunnelDistribution, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self):
        # Mean is zero for all dimensions
        return torch.zeros(self.dim, dtype=DTYPE, device=self.device)

    @property
    def variance(self):
        # Variance is not well-defined due to the hierarchical structure
        raise NotImplementedError("FunnelDistribution does not have a closed-form variance")

    def log_prob(self, value):
        """
        Compute log probability of the funnel distribution.
        log p(x) = log p(x1) + sum_i log p(x_i | x1)
        where x1 ~ N(0, sigma^2) and x_i ~ N(0, exp(x1)) for i > 1
        """
        x1 = value[..., 0]  # First dimension
        x_rest = value[..., 1:]  # Remaining dimensions
        
        # Log probability of x1 ~ N(0, sigma^2)
        log_prob_x1 = -0.5 * torch.square(x1 / self.sigma) - 0.5 * math.log(2 * math.pi) - torch.log(self.sigma)
        
        # Log probability of x_i ~ N(0, exp(x1)) for i > 1
        # log p(x_i | x1) = -0.5 * x_i^2 / exp(x1) - 0.5 * log(2π * exp(x1))
        #                 = -0.5 * x_i^2 * exp(-x1) - 0.5 * log(2π) - 0.5 * x1
        log_prob_rest = -0.5 * torch.sum(torch.square(x_rest) * torch.exp(-x1.unsqueeze(-1)), dim=-1)
        log_prob_rest -= 0.5 * (self.dim - 1) * math.log(2 * math.pi)
        log_prob_rest -= 0.5 * (self.dim - 1) * x1
        
        return log_prob_x1 + log_prob_rest

    def sample(self, sample_shape=torch.Size()):
        # Handle sample_shape
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])
        elif isinstance(sample_shape, (list, tuple)):
            sample_shape = torch.Size(sample_shape)
        
        # Total number of samples needed
        total_samples = sample_shape.numel() if sample_shape.numel() > 0 else 1

        device = self.device
        # Sample x1 ~ N(0, sigma^2). torch.normal does not take a device kwarg
        # when called with size=, so generate on a zero-mean tensor of the right
        # device first.
        mean_x1 = torch.zeros(total_samples, dtype=DTYPE, device=device)
        std_x1 = self.sigma.expand(total_samples)
        x1 = torch.normal(mean=mean_x1, std=std_x1)

        # Sample x_i ~ N(0, exp(x1)) for i > 1
        std_rest = torch.exp(0.5 * x1).unsqueeze(-1)  # Standard deviation for remaining dims
        x_rest = torch.normal(
            mean=torch.zeros(total_samples, self.dim - 1, dtype=DTYPE, device=device),
            std=std_rest.expand(total_samples, self.dim - 1),
        )
        
        # Combine samples
        samples = torch.cat([x1.unsqueeze(-1), x_rest], dim=-1)
        
        # Reshape to match the requested sample_shape
        if sample_shape.numel() > 0:
            samples = samples.view(sample_shape + torch.Size([self.dim]))
        else:
            samples = samples.squeeze(0)
            
        return samples

    def rsample(self, sample_shape=torch.Size()):
        # For reparameterized sampling (supports gradients)
        return self.sample(sample_shape)


def XShape(scale, device=None):
    # Two Gaussian components rotated to form an 'X' using PyTorch
    device = _resolve_device(device)

    angle = torch.tensor(45.0 * (math.pi / 180), dtype=DTYPE, device=device)  # 45 degrees
    rotation_matrix = torch.stack([
        torch.stack([torch.cos(angle), -torch.sin(angle)]),
        torch.stack([torch.sin(angle), torch.cos(angle)])
    ])

    mean1 = torch.zeros(2, dtype=DTYPE, device=device)
    cov1 = torch.diag(torch.tensor([1.0 * scale, 0.1 * scale], dtype=DTYPE, device=device))
    cov1_rotated = rotation_matrix @ cov1 @ rotation_matrix.T

    mean2 = torch.zeros(2, dtype=DTYPE, device=device)
    cov2 = torch.diag(torch.tensor([1.0 * scale, 0.1 * scale], dtype=DTYPE, device=device))
    rotation_matrix_neg = torch.stack([
        torch.stack([torch.cos(-angle), -torch.sin(-angle)]),
        torch.stack([torch.sin(-angle), torch.cos(-angle)])
    ])
    cov2_rotated = rotation_matrix_neg @ cov2 @ rotation_matrix_neg.T

    means = torch.stack([mean1, mean2], dim=0)  # shape [2, 2]
    covs = torch.stack([cov1_rotated, cov2_rotated], dim=0)  # shape [2, 2, 2]

    mix = Categorical(torch.tensor([0.5, 0.5], dtype=DTYPE, device=device))
    comp = Independent(MultivariateNormal(means, covs), 0)
    x_distribution = MixtureSameFamily(mix, comp)
    return x_distribution


class SinhArcsinhGaussian(Distribution):
    r"""Multivariate Gaussian sinh–arcsinh distribution.

    Y = sinh( (asinh(Z) + skewness) / tailscale ),
    with  Z ~ N(0, Σ).

    Args:
        covariance_matrix : Float tensor `[d, d]`
            Positive‑definite covariance Σ of the base Gaussian.
        skewness : Float tensor `[d]`
            Vector *s* controlling skew.
        tailscale : Float tensor `[d]`
            Vector *τ*  (>0) controlling tail‑weight.
        validate_args : bool
            Propagate shape checks to runtime.
    """
    def __init__(self, covariance_matrix, skewness, tailscale, validate_args=None, device=None):
        device = _resolve_device(device)
        self.device = device
        self.covariance_matrix = torch.as_tensor(covariance_matrix, dtype=DTYPE).to(device)
        self.skewness = torch.as_tensor(skewness, dtype=DTYPE).to(device)
        self.tailscale = torch.as_tensor(tailscale, dtype=DTYPE).to(device)
        
        # Base multivariate normal distribution
        self.base_dist = MultivariateNormal(
            loc=torch.zeros(self.covariance_matrix.shape[-1], dtype=DTYPE, device=device),
            covariance_matrix=self.covariance_matrix
        )
        
        batch_shape = torch.Size([])
        event_shape = torch.Size([self.covariance_matrix.shape[-1]])
        super(SinhArcsinhGaussian, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self):
        raise NotImplementedError("SinhArcsinhGaussian does not have a closed-form mean")

    @property
    def variance(self):
        raise NotImplementedError("SinhArcsinhGaussian does not have a closed-form variance")

    def _sinh_arcsinh_transform(self, z):
        """Apply sinh-arcsinh transform: y = sinh((asinh(z) + skewness) / tailscale)"""
        asinh_z = torch.asinh(z)
        transformed = torch.sinh((asinh_z + self.skewness) / self.tailscale)
        return transformed

    def _sinh_arcsinh_inverse(self, y):
        """Inverse sinh-arcsinh transform: z = sinh(asinh(y) * tailscale - skewness)"""
        asinh_y = torch.asinh(y)
        transformed = torch.sinh(asinh_y * self.tailscale - self.skewness)
        return transformed

    def _log_abs_det_jacobian(self, z):
        """Compute log absolute determinant of Jacobian for the transform"""
        # For y = sinh((asinh(z) + s) / τ), the Jacobian diagonal element is:
        # dy/dz = cosh((asinh(z) + s) / τ) * (1 / τ) * (1 / sqrt(1 + z^2))
        
        asinh_z = torch.asinh(z)
        arg = (asinh_z + self.skewness) / self.tailscale
        
        cosh_term = torch.cosh(arg)
        inv_tailscale = 1.0 / self.tailscale
        inv_sqrt_term = 1.0 / torch.sqrt(1.0 + z**2)
        
        jacobian_diag = cosh_term * inv_tailscale * inv_sqrt_term
        log_abs_det = torch.sum(torch.log(torch.abs(jacobian_diag)), dim=-1)
        
        return log_abs_det

    def log_prob(self, value):
        """Compute log probability using change of variables formula"""
        # Transform back to base space
        z = self._sinh_arcsinh_inverse(value)
        
        # Base log probability
        base_log_prob = self.base_dist.log_prob(z)
        
        # Jacobian correction (inverse transform)
        # For inverse transform z = g^(-1)(y), we need |det(dg^(-1)/dy)|
        # But we computed |det(dg/dz)|, so we need the reciprocal
        log_abs_det_jacobian = self._log_abs_det_jacobian(z)
        
        return base_log_prob - log_abs_det_jacobian

    def sample(self, sample_shape=torch.Size()):
        """Sample from the distribution"""
        # Sample from base distribution
        z = self.base_dist.sample(sample_shape)
        
        # Apply sinh-arcsinh transform
        y = self._sinh_arcsinh_transform(z)
        
        return y

    def rsample(self, sample_shape=torch.Size()):
        """Reparameterized sample (supports gradients)"""
        # Sample from base distribution (reparameterized)
        z = self.base_dist.rsample(sample_shape)
        
        # Apply sinh-arcsinh transform
        y = self._sinh_arcsinh_transform(z)
        
        return y


def GMM3(device=None):
    device = _resolve_device(device)
    sigma = torch.eye(2, dtype=DTYPE, device=device)
    sigma[0, 1] = 0.1
    sigma[1, 0] = 0.1

    g1 = MultivariateNormal(
        loc=torch.tensor([-1.0, 1.0], dtype=DTYPE, device=device), covariance_matrix=sigma * 0.5
    )
    g2 = MultivariateNormal(
        loc=torch.tensor([1.1, 1.1], dtype=DTYPE, device=device), covariance_matrix=sigma
    )
    g3 = MultivariateNormal(
        loc=torch.tensor([-1.0, -1.0], dtype=DTYPE, device=device), covariance_matrix=sigma
    )
    mix = Categorical(torch.tensor([0.4, 0.3, 0.3], dtype=DTYPE, device=device))
    target = MixtureSameFamily(
        mix, Independent(
            torch.distributions.MultivariateNormal(
                torch.stack([g1.mean, g2.mean, g3.mean], dim=0),
                torch.stack([g1.covariance_matrix, g2.covariance_matrix, g3.covariance_matrix], dim=0)
            ), 0
        )
    )
    return target
