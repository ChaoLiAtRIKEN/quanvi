import math
import numpy as np
from scipy.special import eval_hermitenorm
from typing import Sequence

ein_str = "abcdefghijklmnopqrstuvxyz"


def _normalize_num_basis_functions(num_basis_functions, dim):
    if isinstance(num_basis_functions, (int, np.integer)):
        if int(num_basis_functions) <= 0:
            raise ValueError("num_basis_functions must be positive.")
        return (int(num_basis_functions),) * dim

    if isinstance(num_basis_functions, np.ndarray):
        num_basis_functions = num_basis_functions.tolist()

    if isinstance(num_basis_functions, Sequence) and not isinstance(num_basis_functions, (str, bytes)):
        counts = tuple(int(k) for k in num_basis_functions)
        if len(counts) != dim:
            raise ValueError(
                f"Expected {dim} per-dimension basis counts, got {len(counts)}."
            )
        if any(k <= 0 for k in counts):
            raise ValueError("All per-dimension basis counts must be positive.")
        return counts

    raise TypeError("num_basis_functions must be an int or a sequence of ints.")


class BasisFunctions:
    def __init__(
        self,
        dim: int,
        num_basis_functions,
    ):
        """
        Args:
            dim (int): Dimensionality of the input.
            num_basis_functions (int or Sequence[int]): Number of basis
                functions to use. An int applies the same count to every
                dimension; a sequence provides one count per dimension.
        """
        self.dim = int(dim)
        self.num_basis_functions_per_dim = _normalize_num_basis_functions(
            num_basis_functions, self.dim
        )
        self.uniform_num_basis_functions = (
            len(set(self.num_basis_functions_per_dim)) == 1
        )
        self.num_basis_functions = (
            self.num_basis_functions_per_dim[0]
            if self.uniform_num_basis_functions
            else self.num_basis_functions_per_dim
        )
        self.max_num_basis_functions = max(self.num_basis_functions_per_dim)
        self.total_num_basis_functions = math.prod(self.num_basis_functions_per_dim)

    def basis_size(self, d):
        if not isinstance(d, int) or not 0 <= d < self.dim:
            raise ValueError(f"Dimension index must be in [0, {self.dim}).")
        return self.num_basis_functions_per_dim[d]

    def transform(self, x):
        raise NotImplementedError

    def transform_dim(self, x, d):
        raise NotImplementedError

    def gradient_transform(self, x, k):
        raise NotImplementedError

    def gradient_transform_dim(self, x, d, k):
        raise NotImplementedError

    def partial_gradients(self, x, k=None):
        if not self.uniform_num_basis_functions:
            raise ValueError(
                "partial_gradients is only defined for uniform basis counts. "
                "Use transform_dim and gradient_transform_dim for per-dimension counts."
            )
        phi_x = self.transform(x)
        if k is not None:
            assert isinstance(k, int) and k < self.num_basis_functions
            phi_x[:, :, k] = self.gradient_transform(x, k)
            return phi_x
        else:
            phi_x_list = []
            for k in range(self.num_basis_functions):
                phi_x_k = np.copy(phi_x)
                phi_x_k[:, :, k] = self.gradient_transform(x, k)
                phi_x_list.append(phi_x_k)
            return phi_x_list


class HermitePolynomials(BasisFunctions):
    def __init__(
        self,
        dim: int,
        num_basis_functions,
    ):
        """
        Args:
            dim (int): Dimensionality of the input.
            num_basis_functions (int or Sequence[int]): Number of basis
                functions to use globally or per dimension.
        """
        super().__init__(dim, num_basis_functions)

    @staticmethod
    def _weights(num_basis_functions):
        return np.array([
            math.pow(math.sqrt(2 * math.pi) * math.factorial(k), -0.5)
            for k in range(num_basis_functions)
        ])

    def _transform_with_count(self, x, num_basis_functions):
        x = np.asarray(x)
        out = [eval_hermitenorm(k, x) for k in range(num_basis_functions)]
        weights = self._weights(num_basis_functions)
        weight_shape = (1,) * x.ndim + (num_basis_functions,)
        return (
            weights.reshape(weight_shape)
            * np.sqrt(np.exp(-np.square(x) / 2))[..., None]
            * np.stack(out, axis=-1)
        )

    def transform(self, x):
        x = np.asarray(x)
        if self.uniform_num_basis_functions:
            weights = self._weights(self.num_basis_functions)
            self.weights = weights.reshape((1, 1, self.num_basis_functions))
            return self._transform_with_count(x, self.num_basis_functions)

        if x.ndim == 1:
            x = x[None, :]
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected the last input dimension to be {self.dim}, got {x.shape[-1]}."
            )
        return [
            self.transform_dim(x[..., d], d)
            for d in range(self.dim)
        ]

    def transform_dim(self, x, d):
        return self._transform_with_count(np.asarray(x), self.basis_size(d))
    
    def gradient_transform(self, x, k):
        assert isinstance(k, int) and k < self.max_num_basis_functions
        x = np.asarray(x)
        if self.uniform_num_basis_functions:
            return self._gradient_with_count(x, self.num_basis_functions, k)

        if x.ndim == 1:
            x = x[None, :]
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected the last input dimension to be {self.dim}, got {x.shape[-1]}."
            )
        out = np.zeros_like(x, dtype=np.result_type(x, float))
        for d, count in enumerate(self.num_basis_functions_per_dim):
            if k < count:
                out[..., d] = self.gradient_transform_dim(x[..., d], d, k)
        return out

    def _gradient_with_count(self, x, num_basis_functions, k):
        assert isinstance(k, int) and k < num_basis_functions
        weight = self._weights(k + 1)[k] * np.sqrt(np.exp(-np.square(x) / 2))
        if k == 0:
            return weight * (-0.5 * x * eval_hermitenorm(k, x))
        return weight * (
            -0.5 * x * eval_hermitenorm(k, x)
            + k * eval_hermitenorm(k - 1, x)
        )

    def gradient_transform_dim(self, x, d, k):
        return self._gradient_with_count(np.asarray(x), self.basis_size(d), k)


class FourierBasis(BasisFunctions):
    def __init__(
        self,
        dim: int,
        num_basis_functions,
    ):
        """
        Args:
            dim (int): Dimensionality of the input.
            num_basis_functions (int or Sequence[int]): Number of Fourier
                frequencies to use globally or per dimension.
        """
        fourier_counts = _normalize_num_basis_functions(num_basis_functions, dim)
        basis_counts = tuple(2 * k + 1 for k in fourier_counts)
        super().__init__(dim, basis_counts)
        self.num_fourier_per_dim = fourier_counts
        self.num_fourier = (
            fourier_counts[0]
            if len(set(fourier_counts)) == 1
            else fourier_counts
        )

    def transform(self, x):
        x = np.asarray(x)
        if not self.uniform_num_basis_functions:
            if x.ndim == 1:
                x = x[None, :]
            if x.shape[-1] != self.dim:
                raise ValueError(
                    f"Expected the last input dimension to be {self.dim}, got {x.shape[-1]}."
                )
            return [
                self.transform_dim(x[..., d], d)
                for d in range(self.dim)
            ]

        out = [np.ones_like(x)]
        for k in range(self.num_fourier_per_dim[0]):
            out.append(np.cos((k + 1) * x))
            out.append(np.sin((k + 1) * x))
        out = np.stack(out, axis=-1)
        assert out.shape[-1] == self.num_basis_functions
        return out

    def transform_dim(self, x, d):
        x = np.asarray(x)
        out = [np.ones_like(x)]
        for k in range(self.num_fourier_per_dim[d]):
            out.append(np.cos((k + 1) * x))
            out.append(np.sin((k + 1) * x))
        return np.stack(out, axis=-1)

    def gradient_transform(self, x, k):
        assert isinstance(k, int) and k < self.max_num_basis_functions
        x = np.asarray(x)
        if not self.uniform_num_basis_functions:
            if x.ndim == 1:
                x = x[None, :]
            if x.shape[-1] != self.dim:
                raise ValueError(
                    f"Expected the last input dimension to be {self.dim}, got {x.shape[-1]}."
                )
            out = np.zeros_like(x, dtype=np.result_type(x, float))
            for d, count in enumerate(self.num_basis_functions_per_dim):
                if k < count:
                    out[..., d] = self.gradient_transform_dim(x[..., d], d, k)
            return out

        return self.gradient_transform_dim(x, 0, k)

    def gradient_transform_dim(self, x, d, k):
        assert isinstance(k, int) and k < self.basis_size(d)
        x = np.asarray(x)
        if k == 0:
            return np.zeros_like(x)

        frequency = (k + 1) // 2
        if k % 2 == 1:
            return -frequency * np.sin(frequency * x)
        return frequency * np.cos(frequency * x)
