
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

_TRANSFORMS = [
    # Repeated maps cover the paper's augmented targets through D=100.
    lambda x:  1.5 * torch.sin(x),
    lambda x:  1.5 * torch.cos(x),
    lambda x: -1.5 * torch.sin(x),
    lambda x: -1.5 * torch.cos(x),

    lambda x:  1.5 * torch.sin(2 * x),
    lambda x:  1.5 * torch.cos(2 * x),
    lambda x: -1.5 * torch.sin(2 * x),
    lambda x: -1.5 * torch.cos(2 * x),

    lambda x:  1.5 * (torch.sigmoid(4 * x) - 0.5),
    lambda x: -1.5 * (torch.sigmoid(4 * x) - 0.5),

    lambda x:  1.5 * torch.tanh(2 * x),
    lambda x: -1.5 * torch.tanh(2 * x),

    lambda x:  1.5 * torch.tanh(4 * x),
    lambda x: -1.5 * torch.tanh(4 * x),

    lambda x:  1.5 * torch.sin(x) * torch.tanh(x),
    lambda x: -1.5 * torch.sin(x) * torch.tanh(x),

    lambda x:  1.5 * torch.cos(x) * torch.tanh(x),
    lambda x: -1.5 * torch.cos(x) * torch.tanh(x),

    lambda x:  1.5 * x / (1 + x**2),
    lambda x: -1.5 * x / (1 + x**2),
    lambda x:  1.5 * torch.sin(x),
    lambda x:  1.5 * torch.cos(x),
    lambda x: -1.5 * torch.sin(x),
    lambda x: -1.5 * torch.cos(x),

    lambda x:  1.5 * torch.sin(2 * x),
    lambda x:  1.5 * torch.cos(2 * x),
    lambda x: -1.5 * torch.sin(2 * x),
    lambda x: -1.5 * torch.cos(2 * x),

    lambda x:  1.5 * (torch.sigmoid(4 * x) - 0.5),
    lambda x: -1.5 * (torch.sigmoid(4 * x) - 0.5),

    lambda x:  1.5 * torch.tanh(2 * x),
    lambda x: -1.5 * torch.tanh(2 * x),

    lambda x:  1.5 * torch.tanh(4 * x),
    lambda x: -1.5 * torch.tanh(4 * x),

    lambda x:  1.5 * torch.sin(x) * torch.tanh(x),
    lambda x: -1.5 * torch.sin(x) * torch.tanh(x),

    lambda x:  1.5 * torch.cos(x) * torch.tanh(x),
    lambda x: -1.5 * torch.cos(x) * torch.tanh(x),

    lambda x:  1.5 * x / (1 + x**2),
    lambda x: -1.5 * x / (1 + x**2),

    lambda x:  1.5 * torch.sin(x),
    lambda x:  1.5 * torch.cos(x),
    lambda x: -1.5 * torch.sin(x),
    lambda x: -1.5 * torch.cos(x),

    lambda x:  1.5 * torch.sin(2 * x),
    lambda x:  1.5 * torch.cos(2 * x),
    lambda x: -1.5 * torch.sin(2 * x),
    lambda x: -1.5 * torch.cos(2 * x),

    lambda x:  1.5 * (torch.sigmoid(4 * x) - 0.5),
    lambda x: -1.5 * (torch.sigmoid(4 * x) - 0.5),

    lambda x:  1.5 * torch.tanh(2 * x),
    lambda x: -1.5 * torch.tanh(2 * x),

    lambda x:  1.5 * torch.tanh(4 * x),
    lambda x: -1.5 * torch.tanh(4 * x),

    lambda x:  1.5 * torch.sin(x) * torch.tanh(x),
    lambda x: -1.5 * torch.sin(x) * torch.tanh(x),

    lambda x:  1.5 * torch.cos(x) * torch.tanh(x),
    lambda x: -1.5 * torch.cos(x) * torch.tanh(x),

    lambda x:  1.5 * x / (1 + x**2),
    lambda x: -1.5 * x / (1 + x**2),
    
    lambda x:  1.5 * torch.sin(x),
    lambda x:  1.5 * torch.cos(x),
    lambda x: -1.5 * torch.sin(x),
    lambda x: -1.5 * torch.cos(x),

    lambda x:  1.5 * torch.sin(2 * x),
    lambda x:  1.5 * torch.cos(2 * x),
    lambda x: -1.5 * torch.sin(2 * x),
    lambda x: -1.5 * torch.cos(2 * x),

    lambda x:  1.5 * (torch.sigmoid(4 * x) - 0.5),
    lambda x: -1.5 * (torch.sigmoid(4 * x) - 0.5),

    lambda x:  1.5 * torch.tanh(2 * x),
    lambda x: -1.5 * torch.tanh(2 * x),

    lambda x:  1.5 * torch.tanh(4 * x),
    lambda x: -1.5 * torch.tanh(4 * x),

    lambda x:  1.5 * torch.sin(x) * torch.tanh(x),
    lambda x: -1.5 * torch.sin(x) * torch.tanh(x),

    lambda x:  1.5 * torch.cos(x) * torch.tanh(x),
    lambda x: -1.5 * torch.cos(x) * torch.tanh(x),

    lambda x:  1.5 * x / (1 + x**2),
    lambda x: -1.5 * x / (1 + x**2),

    lambda x:  1.5 * torch.sin(x),
    lambda x:  1.5 * torch.cos(x),
    lambda x: -1.5 * torch.sin(x),
    lambda x: -1.5 * torch.cos(x),

    lambda x:  1.5 * torch.sin(2 * x),
    lambda x:  1.5 * torch.cos(2 * x),
    lambda x: -1.5 * torch.sin(2 * x),
    lambda x: -1.5 * torch.cos(2 * x),

    lambda x:  1.5 * (torch.sigmoid(4 * x) - 0.5),
    lambda x: -1.5 * (torch.sigmoid(4 * x) - 0.5),

    lambda x:  1.5 * torch.tanh(2 * x),
    lambda x: -1.5 * torch.tanh(2 * x),

    lambda x:  1.5 * torch.tanh(4 * x),
    lambda x: -1.5 * torch.tanh(4 * x),

    lambda x:  1.5 * torch.sin(x) * torch.tanh(x),
    lambda x: -1.5 * torch.sin(x) * torch.tanh(x),

    lambda x:  1.5 * torch.cos(x) * torch.tanh(x),
    lambda x: -1.5 * torch.cos(x) * torch.tanh(x),

    lambda x:  1.5 * x / (1 + x**2),
    lambda x: -1.5 * x / (1 + x**2),
]


class FlowTransform(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def transform(self, sample):
        return self.forward(sample)

    def compute_jacobian(self, sample):
        raise NotImplementedError("FlowTransform must implement compute_jacobian method")

    def transform_score(self, sample, score_x, score_z):
        """
        Transform score function using the chain rule:
        ∇_x log p(x, z) = s(x) + J^T ∇_x log p(z | x) = s(x) - J^T ∇_z log p(z | x)
        where J is the Jacobian dh/dx
        """
        jacobians = self.compute_jacobian(sample)  # [n_samples, output_dim, input_dim]
        if jacobians.ndim == 3:
            transformed_score = score_x - contract('noi,no->ni', jacobians, score_z)
        else:
            transformed_score = score_x - jacobians * score_z
        return transformed_score


class MLPTransform(FlowTransform):
    """
    Multi-layer perceptron transform with automatic Jacobian computation using PyTorch.
    Args:
        layers: List of layer sizes, e.g., [input_dim, hidden_dim, output_dim]
        activation: Activation function ('tanh', 'relu', 'sigmoid', 'elu', 'linear')
        seed: Random seed for weight initialization
    """
    def __init__(self, layers, activation='tanh', scale=2.0, seed=42, dtype=torch.float64):
        super().__init__('MLPTransform')
        self.layers = layers
        self.activation = activation
        self.scale = scale
        self.seed = seed
        torch.manual_seed(seed)
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]).to(dtype) for i in range(len(layers)-1)
        ])
        self._init_weights()
        for p in self.parameters():
            p.requires_grad = False

    def _init_weights(self):
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def _activation_fn(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'elu':
            return F.elu(x)
        elif self.activation == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        h = x
        for i, linear in enumerate(self.linears):
            h = linear(h)
            if i < len(self.linears) - 1:
                h = self._activation_fn(h)
        return h * self.scale

    def compute_jacobian(self, sample):
        assert sample.ndim == 2
        n_samples = sample.shape[0]
        
        x = sample.clone().detach().requires_grad_(True)
        jacobians = []
        for i in range(n_samples):
            xi = x[i:i+1].clone().detach().requires_grad_(True)
            zi = self.forward(xi)
            jacobian = []
            for j in range(zi.shape[1]):
                grad = torch.autograd.grad(zi[0, j], xi, retain_graph=True, create_graph=True)[0]
                jacobian.append(grad[0].detach())
            jacobian = torch.stack(jacobian, dim=0)  # [output_dim, input_dim]
            jacobians.append(jacobian)
        return torch.stack(jacobians, dim=0).detach()  # [n_samples, output_dim, input_dim]


class ElementwiseTransform(FlowTransform):
    """
    Element-wise transform with Markov chain structure:
        p(x, z) = p(x1, x2) * prod_{i=1}^{N} p(z_i | z_{i-1}),  z_0 = x_2
    where p(z_i | z_{i-1}) = N(z_i; h_i(z_{i-1}), sigma^2).

    Each h_i is a scalar-to-scalar function applied along the chain.
    Args:
        transform_fns: List of transform functions or function names
    """
    def __init__(self, transform_fns):
        super().__init__('ElementwiseTransform')
        self.transform_fns = []
        self.derivative_fns = []
        for fn in transform_fns:
            if fn == 'cos' or fn == np.cos:
                self.transform_fns.append(torch.cos)
                self.derivative_fns.append(lambda x: -torch.sin(x))
            elif fn == 'sin' or fn == np.sin:
                self.transform_fns.append(torch.sin)
                self.derivative_fns.append(lambda x: torch.cos(x))
            elif fn == 'exp' or fn == np.exp:
                self.transform_fns.append(torch.exp)
                self.derivative_fns.append(lambda x: torch.exp(x))
            elif fn == 'log' or fn == np.log:
                self.transform_fns.append(torch.log)
                self.derivative_fns.append(lambda x: 1.0 / x)
            elif fn == 'tanh' or fn == np.tanh:
                self.transform_fns.append(torch.tanh)
                self.derivative_fns.append(lambda x: 1.0 - torch.tanh(x)**2)
            elif fn == 'square':
                self.transform_fns.append(lambda x: x**2)
                self.derivative_fns.append(lambda x: 2.0 * x)
            elif fn == 'identity' or fn is None:
                self.transform_fns.append(lambda x: x)
                self.derivative_fns.append(lambda x: torch.ones_like(x))
            elif callable(fn):
                self.transform_fns.append(fn)
                self.derivative_fns.append(None)  # Will compute numerically
            else:
                raise ValueError(f"Unknown transform function: {fn}")

    def forward(self, z_prev):
        """
        Apply each h_i element-wise: result[:, i] = h_i(z_prev[:, i]).
        Args:
            z_prev: [batch, N] where z_prev[:, i] is z_{i-1} for the i-th link
                    (z_prev[:, 0] = x_2 = z_0)
        Returns:
            [batch, N] chain means
        """
        assert z_prev.ndim == 2
        x = torch.as_tensor(z_prev, dtype=torch.float64)
        transformed_cols = []
        for i, fn in enumerate(self.transform_fns):
            col = x[:, i]
            transformed_cols.append(fn(col).unsqueeze(1))
        return torch.cat(transformed_cols, dim=1)

    def compute_jacobian(self, z_prev):
        """
        Compute diagonal Jacobian: result[:, i] = h_i'(z_prev[:, i]).
        Args:
            z_prev: [batch, N]
        Returns:
            [batch, N] derivatives
        """
        assert z_prev.ndim == 2
        x = z_prev.clone().detach().requires_grad_(True)
        n_samples = x.shape[0]
        jacobian_diag = []
        for i, (fn, deriv_fn) in enumerate(zip(self.transform_fns, self.derivative_fns)):
            col = x[:, i]
            if deriv_fn is not None:
                deriv = deriv_fn(col)
            else:
                deriv = []
                for j in range(n_samples):
                    xi = col[j].clone().detach().requires_grad_(True)
                    out = fn(xi)
                    grad = torch.autograd.grad(out, xi, retain_graph=True)[0]
                    deriv.append(grad)
                deriv = torch.tensor(deriv, dtype=torch.float64)
            jacobian_diag.append(deriv)
        jacobian_diag = torch.stack(jacobian_diag, dim=1)
        return jacobian_diag.detach()

    def compute_score(self, xz, score_x, sigma_sq):
        """
        Compute the full score for the Markov chain augmented distribution:
            p(x, z) = p(x1, x2) * prod_i N(z_i; h_i(z_{i-1}), sigma_sq)

        Args:
            xz: [batch, 2+N] full sample (x1, x2, z1, ..., zN)
            score_x: [batch, 2] score of base distribution nabla log p(x1, x2)
            sigma_sq: scalar variance
        Returns:
            [batch, 2+N] full score
        """
        N = len(self.transform_fns)
        x2 = xz[:, 1]       # [batch]
        z = xz[:, 2:]        # [batch, N]  (z_1, ..., z_N)

        # z_prev[:, i] = z_i for i=0,...,N-1 where z_0 = x_2
        z_prev = torch.cat([x2.unsqueeze(1), z[:, :-1]], dim=1)  # [batch, N]

        h_vals = self.forward(z_prev)           # [batch, N]: h_{i+1}(z_i)
        h_derivs = self.compute_jacobian(z_prev) # [batch, N]: h_{i+1}'(z_i)

        sigma_inv = 1.0 / sigma_sq

        # From parent: nabla_{z_i} log p(z_i | z_{i-1}) = sigma^{-2}(h_i(z_{i-1}) - z_i)
        score_from_parent = sigma_inv * (h_vals - z)  # [batch, N]

        # From child: h_{i+1}'(z_i) * sigma^{-2}(z_{i+1} - h_{i+1}(z_i))
        score_from_child = torch.zeros_like(z)  # [batch, N]
        if N > 1:
            # For z_i (i=1,...,N-1): child uses h_{i+1} at z_i
            # h_derivs[:, 1:] = h_{i+1}'(z_i) for i=1,...,N-1
            # h_vals[:, 1:]   = h_{i+1}(z_i)  for i=1,...,N-1
            # z[:, 1:]        = z_{i+1}        for i=1,...,N-1
            score_from_child[:, :N-1] = h_derivs[:, 1:] * sigma_inv * (z[:, 1:] - h_vals[:, 1:])

        # x_2 child contribution: h_1'(x_2) * sigma^{-2}(z_1 - h_1(x_2))
        score_x2_child = h_derivs[:, 0] * sigma_inv * (z[:, 0] - h_vals[:, 0])

        score = torch.zeros_like(xz)
        score[:, 0] = score_x[:, 0]                           # x_1
        score[:, 1] = score_x[:, 1] + score_x2_child          # x_2
        score[:, 2:] = score_from_parent + score_from_child    # z_1, ..., z_N
        return score


class CompositeTransform(FlowTransform):
    """
    Compose multiple transforms: h_n(...h_2(h_1(x))...)
    Args:
        transforms: List of transform objects to apply in sequence
    """
    def __init__(self, transforms):
        super().__init__('CompositeTransform')
        self.transforms = transforms

    def forward(self, sample):
        result = sample
        for transform in self.transforms:
            result = transform.forward(result)
        return result

    def transform_score(self, score, sample):
        intermediate_samples = [sample]
        current_sample = sample
        for transform in self.transforms:
            current_sample = transform.forward(current_sample)
            intermediate_samples.append(current_sample)
        current_score = score
        for i in reversed(range(len(self.transforms))):
            transform = self.transforms[i]
            input_sample = intermediate_samples[i]
            current_score = transform.transform_score(current_score, input_sample)
        return current_score


class TransformedDistribution:
    """
    Markov chain augmented distribution:
        p(x, z) = p(x1, x2) * prod_{i=1}^{N} N(z_i; h_i(z_{i-1}), sigma_sq)
    where z_0 = x_2.
    """
    def __init__(self, base_distribution, ndim, transform_fns=None, sigma_sq=1.0):
        self.base_distribution = base_distribution
        self.ndim = ndim
        self.N = ndim - 2  # number of augmented dimensions
        if transform_fns is None:
            assert self.N <= len(_TRANSFORMS), "Not enough default transforms for the given ndim"
            transform_fns = _TRANSFORMS[:self.N]
        self.transform_fns = ElementwiseTransform(transform_fns)
        self.sigma_sq = sigma_sq

    def log_prob(self, value):
        x, z = value[:, :2], value[:, 2:]
        # Build chain: z_prev[:, i] = z_{i-1}, z_0 = x_2
        z_prev = torch.cat([x[:, 1:2], z[:, :-1]], dim=1)  # [batch, N]
        h_vals = self.transform_fns(z_prev)  # [batch, N]
        # sum of log N(z_i; h_i(z_{i-1}), sigma_sq) for i=1,...,N
        logp = self.base_distribution.log_prob(x) \
            - 0.5 * self.N * np.log(2 * np.pi * self.sigma_sq) \
            - 0.5 / self.sigma_sq * ((z - h_vals) ** 2).sum(dim=1)
        return logp

    def sample(self, sample_shape=torch.Size()):
        x = self.base_distribution.sample(sample_shape)
        z_list = []
        z_prev = x[:, 1]  # z_0 = x_2
        sigma_std = np.sqrt(self.sigma_sq)
        for i in range(self.N):
            mean = self.transform_fns.transform_fns[i](z_prev)
            z_i = mean + sigma_std * torch.randn_like(mean)
            z_list.append(z_i)
            z_prev = z_i
        z = torch.stack(z_list, dim=1)
        xz = torch.cat([x, z], dim=-1)
        return xz


class LongRangeTransformedDistribution(TransformedDistribution):
    """
    Controlled departure from the original first-order Markov target.

    Coordinates up to x_L retain the original local dependency.  For i > L,

        mean(x_i | x_<i)
          = sqrt(1 - strength^2) * h_i(x_{i-1})
            + strength * h_i(x_{i-L}).

    Hence strength=0 reproduces the original target, while strength=1 replaces
    each affected local edge with a pure lag-L edge.  The squared coefficients
    sum to one so intermediate strengths do not artificially attenuate the
    conditional-mean signal as a convex average would.
    """

    def __init__(
        self,
        base_distribution,
        ndim,
        dependency_lag,
        long_range_strength,
        transform_fns=None,
        sigma_sq=1.0,
    ):
        super().__init__(
            base_distribution=base_distribution,
            ndim=ndim,
            transform_fns=transform_fns,
            sigma_sq=sigma_sq,
        )

        if isinstance(dependency_lag, bool) or not isinstance(dependency_lag, int):
            raise TypeError("dependency_lag must be an integer.")
        if dependency_lag < 2 or dependency_lag >= ndim:
            raise ValueError(
                f"dependency_lag must satisfy 2 <= dependency_lag < ndim; "
                f"got dependency_lag={dependency_lag}, ndim={ndim}."
            )
        if not 0.0 <= long_range_strength <= 1.0:
            raise ValueError("long_range_strength must lie in [0, 1].")
        if sigma_sq <= 0:
            raise ValueError("sigma_sq must be positive.")

        self.dependency_lag = dependency_lag
        self.long_range_strength = float(long_range_strength)

    @property
    def _num_unchanged_augmented(self):
        # The base distribution supplies x_1 and x_2.  Among the augmented
        # coordinates x_3, ..., x_D, the unchanged prefix is x_3, ..., x_L.
        return self.dependency_lag - 2

    def _conditional_means(self, value):
        """Compute means for x_3, ..., x_D from a two-dimensional sample batch."""
        local_parents = value[:, 1:-1]
        local_means = self.transform_fns(local_parents)

        prefix = self._num_unchanged_augmented
        if prefix == self.N:
            return local_means

        long_means = []
        for augmented_idx in range(prefix, self.N):
            full_coordinate_idx = augmented_idx + 2
            long_parent_idx = full_coordinate_idx - self.dependency_lag
            transform_fn = self.transform_fns.transform_fns[augmented_idx]
            long_means.append(
                transform_fn(value[:, long_parent_idx]).unsqueeze(1)
            )
        long_means = torch.cat(long_means, dim=1)

        strength = self.long_range_strength
        local_weight = (1.0 - strength ** 2) ** 0.5
        mixed_means = (
            local_weight * local_means[:, prefix:]
            + strength * long_means
        )
        return torch.cat([local_means[:, :prefix], mixed_means], dim=1)

    def log_prob(self, value):
        x, z = value[:, :2], value[:, 2:]
        means = self._conditional_means(value)
        return (
            self.base_distribution.log_prob(x)
            - 0.5 * self.N * np.log(2 * np.pi * self.sigma_sq)
            - 0.5 / self.sigma_sq * ((z - means) ** 2).sum(dim=1)
        )

    def sample(self, sample_shape=torch.Size()):
        x = self.base_distribution.sample(sample_shape)
        coordinates = [x[:, 0], x[:, 1]]
        sigma_std = np.sqrt(self.sigma_sq)
        strength = self.long_range_strength
        local_weight = (1.0 - strength ** 2) ** 0.5

        for augmented_idx in range(self.N):
            full_coordinate_idx = augmented_idx + 2
            transform_fn = self.transform_fns.transform_fns[augmented_idx]
            local_mean = transform_fn(coordinates[-1])

            if full_coordinate_idx >= self.dependency_lag:
                long_parent_idx = full_coordinate_idx - self.dependency_lag
                long_mean = transform_fn(coordinates[long_parent_idx])
                mean = local_weight * local_mean + strength * long_mean
            else:
                mean = local_mean

            coordinate = mean + sigma_std * torch.randn_like(mean)
            coordinates.append(coordinate)

        return torch.stack(coordinates, dim=1)


