import torch
import inspect
from functools import wraps
import numpy as np
from utils.basis_function import HermitePolynomials
from tensor_factory import get_each_core, conj_transpose
import opt_einsum as oe

_QUERY_BUFFER_BYTES = 64 * 1024 ** 2

def _compatible_inference_init(init):
    """Support old entry points for unprojected (R=K) models."""
    signature = inspect.signature(init)
    @wraps(init)
    def wrapped(self, *args, **kwargs):
        if len(args) > 4 and not isinstance(args[4], torch.dtype):
            names = ("D", "E", "K", "R", "Nk", "dtype", "Inquire_B", "state",
                     "einsum_builder", "core_tensor", "proj_matrix", "optimizer",
                     "device", "transform")
            if len(args) > len(names):
                raise TypeError("Too many positional arguments for Inference")
            for name, value in zip(names, args):
                if name in kwargs:
                    raise TypeError(f"Inference got multiple values for argument '{name}'")
                kwargs[name] = value
            args = ()
        rank = kwargs.pop("R", None)
        projection = kwargs.pop("proj_matrix", None)
        bound = signature.bind(self, *args, **kwargs)
        K = bound.arguments["K"]
        if (rank is not None and rank != K) or (
            projection is not None and not np.array_equal(projection, np.eye(K))
        ):
            raise ValueError("This memory-optimized Inference supports R=K and identity "
                             "proj_matrix only; a nonidentity projection cannot be discarded.")
        init(*bound.args, **bound.kwargs)
        self.R = K
        self.proj_matrix = projection
    return wrapped


class Inference:
    @_compatible_inference_init
    def __init__(self, D, E, K, Nk, dtype, Inquire_B, state, einsum_builder, core_tensor, optimizer="greedy",device=None,transform=None):
        self.D = D
        self.E = E
        self.K = K
        self.Nk = Nk
        self.Inquire_B = Inquire_B
        self.dtype = dtype
        self.state = state
        self.einsum_builder = einsum_builder
        self.core_tensor = core_tensor
        self.Num_gates = self.D + self.E - self.Nk + 1
        self.optimizer = optimizer
        self.device = device if device is not None else torch.device('cpu')
        self.m = torch.eye(self.K, dtype=self.dtype, device=self.device)
        self.transform = transform
        # The basis stores metadata only, so build it once and reuse it.
        self.basis = HermitePolynomials(dim=self.D, num_basis_functions=self.K)


    def as_dtype(self, new_dtype):
        new_state = self.state.detach().to(new_dtype)
        new_core = [c.detach().to(new_dtype) for c in self.core_tensor]
        return Inference(
            D=self.D, E=self.E, K=self.K, Nk=self.Nk,
            dtype=new_dtype, Inquire_B=self.Inquire_B,
            state=new_state, einsum_builder=self.einsum_builder,
            core_tensor=new_core,
            optimizer=self.optimizer, device=self.device, transform=self.transform
        )


    def measurement_factors(self, x_batch_np=None, marginal_list=None):
        if isinstance(x_batch_np, torch.Tensor):
            x = x_batch_np.detach().cpu().numpy()
        else:
            x = x_batch_np
        if x is None:
            x = np.random.uniform(low=-5, high=5, size=(1, self.D)) # For whole probability calculation
        phi_x = self.basis.transform(x)  # (Inquire_B, D, K)
        Lfactor = torch.as_tensor(phi_x, dtype=self.dtype,device=self.device).permute(1,2,0)  # (D, K, Inquire_B)
        Factor  = oe.contract('dab,dcb->dacb', Lfactor, Lfactor, optimize = self.optimizer)  # (D, K, K, Inquire_B)
        # marginal part
        if marginal_list is not None and len(marginal_list) > 0:
            eye = torch.eye(self.K, dtype=self.dtype, device=self.device)
            identity_block = eye.unsqueeze(-1).expand(self.K, self.K, x.shape[0])  # (K, K, Inquire_B)
            Factor[marginal_list] = identity_block
        return Factor

    def _measurement_factors_with_grad(self, x_batch_np):
        if isinstance(x_batch_np, torch.Tensor):
            x = x_batch_np.detach().cpu().numpy()
        else:
            x = np.asarray(x_batch_np)

        phi_x = self.basis.transform(x)  # (B, D, K)
        dphi_x = np.stack(
            [self.basis.gradient_transform(x, k) for k in range(self.K)],
            axis=-1,
        )  # (B, D, K)

        phi = torch.as_tensor(phi_x, dtype=self.dtype, device=self.device).permute(1, 2, 0)
        dphi = torch.as_tensor(dphi_x, dtype=self.dtype, device=self.device).permute(1, 2, 0)

        factor = oe.contract('dab,dcb->dacb', phi, phi, optimize=self.optimizer)
        d_factor = (
            oe.contract('dab,dcb->dacb', dphi, phi, optimize=self.optimizer)
            + oe.contract('dab,dcb->dacb', phi, dphi, optimize=self.optimizer)
        )
        return factor, d_factor
    
    def _use_amplitude_queries(self):
        # The exact amplitude path is independent of Nk. Use it whenever the
        # complete state fits the fixed budget; otherwise retain the streaming
        # tensor-network path so large-D models stay memory bounded.
        return self.K ** (self.D + self.E) * self.m.element_size() <= 16 * 1024 ** 2

    def _wavefunction(self):
        """Exact full state, used only when it fits the small-state budget."""
        n = self.D + self.E
        psi = self.state
        for _ in range(n - 1):
            psi = torch.tensordot(psi, self.state, dims=0)
        # The next gate's outputs feed the previous gate's inputs.
        for i in reversed(range(self.Num_gates)):
            sites = list(range(i, i + self.Nk))
            psi = torch.tensordot(psi, self.core_tensor[i], dims=(sites, list(range(self.Nk))))
            order = [j for j in range(n) if j not in sites] + sites
            psi = psi.permute([order.index(j) for j in range(n)])
        return psi

    def _amplitude_probabilities(self, x, measured, psi):
        """Exact density; sum squared amplitudes over unmeasured/environment sites."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if not measured:
            return psi.abs().square().sum().expand(len(x))
        order = list(measured) + [d for d in range(self.D + self.E) if d not in measured]
        matrix = psi.permute(order).reshape(self.K ** len(measured), -1)
        # Bound both basis-product and output-amplitude buffers.
        per_row = (matrix.shape[0] + matrix.shape[1]) * self.m.element_size()
        limit = max(1, _QUERY_BUFFER_BYTES // per_row)
        parts = []
        for start in range(0, len(x), limit):
            stop = min(start + limit, len(x))
            features = torch.ones((stop-start, 1), dtype=self.dtype, device=self.device)
            for d in measured:
                phi = torch.as_tensor(self.basis.transform_dim(x[start:stop, d], d),
                                      dtype=self.dtype, device=self.device)
                features = (features[:, :, None] * phi[:, None, :]).reshape(stop-start, -1)
            amplitude = features @ matrix
            parts.append(amplitude.abs().square().sum(dim=1))
        return torch.cat(parts) if parts else psi.real.new_empty((0,))

    def _inquire_batch_limit(self):
        # The current query path has intermediates of about N*K**(2*Nk-1).
        # Budget one such intermediate at 64 MiB; total process memory is higher.
        bytes_per_row = self.K ** (2 * self.Nk - 1) * self.m.element_size()
        return max(1, _QUERY_BUFFER_BYTES // bytes_per_row)

    def inquire_process(self, core_tensor, state, Factor):
        batch = Factor.shape[-1]
        limit = self._inquire_batch_limit()
        if batch <= limit:
            return self._inquire_process_once(core_tensor, state, Factor)
        # Rows are independent. Preserve dtype and autograd; do not normalize
        # each chunk separately or change the number/order of query points.
        return torch.cat([
            self._inquire_process_once(core_tensor, state, Factor[..., start:start + limit])
            for start in range(0, batch, limit)
        ], dim=0)

    def _inquire_process_once(self, core_tensor, state, Factor):
        D = self.D
        E = self.E
        Nk = self.Nk
        einsum_builder = self.einsum_builder
        m = self.m  # identity matrix for the extra qubits
        Num_gates = self.Num_gates

        # i == 0
        operation = []
        if Num_gates ==1:
            for i in range(D + E):
                operation.append(state)
        else:
            operation.append(state)
        operation.append(core_tensor[0])
        operation.extend(Factor[:Nk])
        operation.append(conj_transpose(core_tensor[0],Nk))
        if Num_gates ==1:
            for i in range(D + E):
                operation.append(torch.conj(state))
        else:
            operation.append(torch.conj(state))
        for i in range(Nk-D):
            operation.append(m)

        einsum_str = einsum_builder.first_core_inquire_new(D, E , Nk, Num_gates)
        F = oe.contract(einsum_str, *operation, optimize = self.optimizer)

        # i in (1, D - Nk):
        for i in range(1, D - Nk + 1):
            Lcore, Rcore = get_each_core(D,E,Nk,core_tensor,state,i,None,None)
            einsum_str = einsum_builder.cp_inquire_new(Nk, i==0, i==D-Nk, i==D+E-Nk)
            F = oe.contract(einsum_str, F, Lcore, Rcore, Factor[Nk+i-1], optimize = self.optimizer)

        # i in (D - Nk + 1, D + E - Nk):
        for i in range(max(D - Nk + 1,1), Num_gates):
            Lcore, Rcore = get_each_core(D,E,Nk,core_tensor,state,i,None,None)
            einsum_str = einsum_builder.extra_inquire_new(Nk, i == D + E - Nk)
            F = oe.contract(einsum_str, F, Lcore, Rcore, m, optimize = self.optimizer)
        return F
    
    def whole_prob(self, core_tensor, state):
        marginal_list = [i for i in range(self.D)]
        factor = self.measurement_factors(x_batch_np=None, marginal_list=marginal_list)
        whole_prob_result = self.inquire_process(core_tensor, state, factor)
        return whole_prob_result
    
    def inquire_result(self, core_tensor, state, Factor):
        unnormalized_result = self.inquire_process(core_tensor, state, Factor)
        whole_prob = self.whole_prob(core_tensor, state)
        normalized_result = unnormalized_result/whole_prob
        return normalized_result
    
    def conditional(self, core_tensor, state, marginal_list):
        factor = self.measurement_factors(x_batch_np=None)
        inquire_result = self.inquire_result(core_tensor, state, factor)
        factor = self.measurement_factors(x_batch_np=None, marginal_list=marginal_list)
        marginal_result = self.inquire_result(core_tensor, state, factor)
        conditional_result = inquire_result / marginal_result
        return conditional_result
    
    def sample_basis(self, n_samples, n_grid=1000, x_range=(-5, 5), chunk_size=None):
        """
        Autoregressive sampling using Inverse CDF method.
        Args:
            n_samples (int): Number of samples to generate.
            n_grid (int): Number of grid points for numerical integration.
            x_range (tuple): The range [min, max] for integration/sampling.
            chunk_size (int | None): If set, draw samples in chunks of this size so that
                peak batch is chunk_size * n_grid (reduces memory vs n_samples * n_grid).
        """
        if chunk_size is None or chunk_size <= 0 or chunk_size >= n_samples:
            return self._sample_basis_once(n_samples, n_grid, x_range)
        parts = []
        for start in range(0, n_samples, chunk_size):
            m = min(chunk_size, n_samples - start)
            parts.append(self._sample_basis_once(m, n_grid, x_range))
        return np.vstack(parts)

    def _sample_basis_once(self, n_samples, n_grid, x_range):
        if n_samples * n_grid > self._inquire_batch_limit():
            return self._sample_basis_bounded(n_samples, n_grid, x_range)
        samples = np.zeros((n_samples, self.D))
        grid = np.linspace(x_range[0], x_range[1], n_grid)
        grid_width = grid[1] - grid[0]
        original_B = self.Inquire_B
        current_batch_size = n_samples * n_grid
        self.Inquire_B = current_batch_size
        with torch.no_grad():
            for d in range(self.D):
                if d > 0:
                    past_part = np.repeat(samples[:, :d], n_grid, axis=0)
                else:
                    past_part = np.zeros((current_batch_size, 0))
                current_part = np.tile(grid, n_samples).reshape(-1, 1)
                future_part = np.zeros((current_batch_size, self.D - d - 1))
                x_input = np.hstack([past_part, current_part, future_part])
                marginal_list = list(range(d + 1, self.D))
                Factor = self.measurement_factors(x_batch_np=x_input, marginal_list=marginal_list)
                pdf_flat = self.inquire_result(self.core_tensor, self.state, Factor)
                pdf_val = pdf_flat.detach().cpu().numpy().reshape(n_samples, n_grid)
                pdf_val = np.real(pdf_val)
                cdf_val = np.cumsum(pdf_val, axis=1) * grid_width
                normalization_const = cdf_val[:, -1:] + 1e-12
                cdf_normalized = cdf_val / normalization_const
                u = np.random.rand(n_samples)
                new_samples_d = np.zeros(n_samples)
                for i in range(n_samples):
                    new_samples_d[i] = np.interp(u[i], cdf_normalized[i, :], grid)
                samples[:, d] = new_samples_d
        self.Inquire_B = original_B
        return samples

    @torch.no_grad()
    def _sample_basis_amplitudes(self, n_samples, n_grid, x_range, psi):
        """Exact conditional K x K densities; reuse each prefix across its grid."""
        samples = np.zeros((n_samples, self.D))
        grid = np.linspace(x_range[0], x_range[1], n_grid)
        width = grid[1] - grid[0]
        n = self.D + self.E
        for d in range(self.D):
            # Preserve one normalization probe and n_samples uniforms per site.
            normalizer = self.whole_prob(self.core_tensor, self.state)
            uniforms = np.random.rand(n_samples)
            grid_phi = torch.as_tensor(self.basis.transform_dim(grid, d),
                                       dtype=self.dtype, device=self.device)
            matrix = psi.reshape(self.K ** d, self.K ** (n-d))
            per_row = (matrix.shape[0] + matrix.shape[1] + n_grid*self.K) * self.m.element_size()
            rows = max(1, _QUERY_BUFFER_BYTES // per_row)
            # At the first site every sample has the same empty prefix.
            count = 1 if d == 0 else n_samples
            for start in range(0, count, rows):
                stop = min(start + rows, count)
                if d == 0:
                    amplitude = psi.reshape(1, self.K, -1)
                else:
                    features = torch.ones((stop-start, 1), dtype=self.dtype, device=self.device)
                    for j in range(d):
                        phi = torch.as_tensor(self.basis.transform_dim(samples[start:stop, j], j),
                                              dtype=self.dtype, device=self.device)
                        features = (features[:, :, None] * phi[:, None, :]).reshape(stop-start, -1)
                    amplitude = (features @ matrix).reshape(stop-start, self.K, -1)
                # Trace all future coordinates and environment axes exactly.
                density = amplitude @ amplitude.mH
                pdf = oe.contract('gk,bkl,gl->bg', grid_phi, density, grid_phi,
                                  optimize=self.optimizer) / normalizer
                cdf = np.cumsum(pdf.real.cpu().numpy(), axis=1) * width
                cdf = cdf / (cdf[:, -1:] + 1e-12)
                if d == 0:
                    samples[:, d] = np.interp(uniforms, cdf[0], grid)
                else:
                    for row in range(stop-start):
                        samples[start+row, d] = np.interp(uniforms[start+row], cdf[row], grid)
        return samples

    @torch.no_grad()
    def _sample_basis_bounded(self, n_samples, n_grid, x_range):
        """Bound sampling buffers without changing the original RNG order.

        Loop over dimensions first, just like the original sampler. Draw the
        normalization probe and all uniforms once per dimension, not per chunk.
        """
        samples = np.zeros((n_samples, self.D))
        grid = np.linspace(x_range[0], x_range[1], n_grid)
        width = grid[1] - grid[0]
        if self._use_amplitude_queries():
            return self._sample_basis_amplitudes(n_samples, n_grid, x_range, self._wavefunction())
        rows = max(1, self._inquire_batch_limit() // n_grid)
        original_B = self.Inquire_B
        try:
            for d in range(self.D):
                normalizer = self.whole_prob(self.core_tensor, self.state)
                uniforms = np.random.rand(n_samples)
                for start in range(0, n_samples, rows):
                    stop = min(start + rows, n_samples)
                    size = stop - start
                    self.Inquire_B = size * n_grid
                    x = np.zeros((size, n_grid, self.D))
                    x[:, :, :d] = samples[start:stop, None, :d]
                    x[:, :, d] = grid
                    factors = self.measurement_factors(
                        x.reshape(-1, self.D), marginal_list=list(range(d + 1, self.D))
                    )
                    pdf = self.inquire_process(self.core_tensor, self.state, factors)
                    pdf = pdf / normalizer
                    pdf = pdf.real.cpu().numpy().reshape(size, n_grid)
                    cdf = np.cumsum(pdf, axis=1) * width
                    cdf = cdf / (cdf[:, -1:] + 1e-12)
                    for row in range(size):
                        samples[start + row, d] = np.interp(uniforms[start + row], cdf[row], grid)
            return samples
        finally:
            self.Inquire_B = original_B

    def sample(self, n_samples, n_grid=1000, x_range=(-5, 5), chunk_size=None):
        samples = self.sample_basis(
            n_samples, n_grid=n_grid, x_range=x_range, chunk_size=chunk_size
        )
        if self.transform is not None:
            samples = self.transform.inverse_transform(samples)
        return samples
    
    def log_prob(self, value, marginal_list=None):
        reshape = False
        if value.ndim > 2:
            value_shape = value.shape
            value = value.reshape(-1, value_shape[-1])
            reshape = True

        if self.transform is not None:
            value = self.transform.transform(value)

        original_B = self.Inquire_B
        current_batch_size = value.shape[0]
        self.Inquire_B = current_batch_size
        try:
            if self._use_amplitude_queries():
                marginalized = (set() if marginal_list is None else
                                set(np.arange(self.D)[marginal_list].reshape(-1).tolist()))
                measured = [d for d in range(self.D) if d not in marginalized]
                prob = self._amplitude_probabilities(value, measured, self._wavefunction())
                # Retain the original normalization and RNG probe behavior.
                prob = prob / self.whole_prob(self.core_tensor, self.state)
            else:
                Factors = self.measurement_factors(x_batch_np=value, marginal_list=marginal_list)
                prob = self.inquire_result(self.core_tensor, self.state, Factors)
            prob = torch.real(prob).to(torch.float64)        # Promote to float64 before taking the logarithm.
            eps = torch.finfo(prob.dtype).tiny               # Use the promoted probability dtype.
            log_p = torch.log(torch.clamp(prob, min=eps))
            if self.transform is not None:
                log_p = log_p - 0.5 * self.transform.log_det_cov
            if reshape:
                log_p = log_p.reshape(value_shape[:-1])
        finally:
            self.Inquire_B = original_B
        return log_p

    def score_basis(self, value):
        input_is_tensor = isinstance(value, torch.Tensor)
        reshape = False
        if value.ndim > 2:
            value_shape = value.shape
            value = value.reshape(-1, value_shape[-1])
            reshape = True

        original_B = self.Inquire_B
        self.Inquire_B = value.shape[0]
        try:
            factors, d_factors = self._measurement_factors_with_grad(value)
            prob = torch.real(self.inquire_process(self.core_tensor, self.state, factors)).to(torch.float64)
            eps = torch.finfo(prob.dtype).tiny
            prob = torch.clamp(prob, min=eps)

            score_components = []
            for d in range(self.D):
                factors_d = factors.clone()
                factors_d[d] = d_factors[d]
                d_prob = torch.real(
                    self.inquire_process(self.core_tensor, self.state, factors_d)
                ).to(torch.float64)
                score_components.append(d_prob / prob)

            score = torch.stack(score_components, dim=-1)
            if reshape:
                score = score.reshape(*value_shape[:-1], self.D)
        finally:
            self.Inquire_B = original_B

        if input_is_tensor:
            return score.to(dtype=value.dtype, device=value.device)
        return score.detach().cpu().numpy()

    def score(self, value):
        input_is_tensor = isinstance(value, torch.Tensor)
        if self.transform is not None:
            value_basis = self.transform.transform(value)
            score_basis = self.score_basis(value_basis)
            score = self.transform.inverse_transform_score(score_basis)
        else:
            score = self.score_basis(value)

        if input_is_tensor and not isinstance(score, torch.Tensor):
            return torch.as_tensor(score, dtype=value.dtype, device=value.device)
        return score

