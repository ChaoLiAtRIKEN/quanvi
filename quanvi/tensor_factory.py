from typing import Any, List, Optional

import numpy as np
from utils.basis_function import HermitePolynomials
from utils.utils import get_pytorch_score
import torch

def init_core_tensor(D,E,K, Nk, Dtype, is_identity=False, device=None):
    device = device if device is not None else torch.device('cpu')
    shape = (K,) * (Nk*2)
    if is_identity:
        core_tensor = torch.nn.ParameterList([ 
            torch.nn.Parameter(torch.eye(K**Nk, dtype=Dtype, device=device).reshape(shape)) #cfloat is complex number field
            for _ in range(D+E-Nk+1)
        ])
    else:
        core_tensor = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(shape, dtype=Dtype, device=device)) #cfloat is complex number field
            for _ in range(D+E-Nk+1)
         ])
        with torch.no_grad():
            for p in core_tensor:
                A = p.reshape(K**Nk, K**Nk)     
                Q, R = torch.linalg.qr(A) 
                # Diagonal phase normalization to avoid distribution bias
                diag = torch.diagonal(R)
                phases = diag / (diag.abs() + 1e-12)
                D = torch.diag(phases.conj())
                Q = Q @ D
                M = Q
                p.copy_(M.reshape(shape)) #copy_ is a pytorch function, a inplace operation to modify the original tensor
    return core_tensor

def conj_transpose(tensor, Nk):
    ndim = tensor.ndim
    start = ndim - Nk
    perm = list(range(start, ndim)) + list(range(0, start))
    return tensor.permute(*perm).conj()

def get_each_core(D,E,Nk,core_tensor,state,idx,Lfactor,Rfactor):
    '''
    This function computes a pair of left and right core contractions for a given index.
    Except for the first core, which will be handled separately. (Only process cores from 1 to the last one)
    '''
    # contract with state

    F = torch.tensordot(state, core_tensor[idx], dims=([0],[0]))
    if idx == D + E - Nk:   #last core
        for j in range(1,Nk):
            F = torch.tensordot(state, F, dims=([0],[0]))
    
    # contract with cp or extra matrix
    if (1<= idx <= D - Nk) and (Lfactor is not None):       #Cp related cores
        Fl = torch.tensordot(F, Lfactor[Nk+idx-1], dims = ([-1],[0]))
        Fr = torch.tensordot(F, Rfactor[Nk+idx-1], dims = ([-1],[0]))
        left_core = Fl
        right_core = conj_transpose(Fr, Nk)
    else:                       #Extra qubits related cores
        left_core = F
        right_core = conj_transpose(F, Nk)
    return left_core, right_core

class InitM:
    class M:
        @classmethod
        def build(cls, phi_x, dphi_x, sp_x, x):
            m = cls()
            m.phi_x  = phi_x
            m.dphi_x = dphi_x
            m.sp_x   = sp_x
            m.x      = x
            return m
        
    def __init__(self, D, K, B, seed, target_score, dtype, device=None,x_range=(-5, 5),transform=None,):
        self.D = D
        self.K = K
        self.B = B
        self.seed = seed
        self.target_score = target_score
        self.dtype = dtype
        self.device = device if device is not None else torch.device('cpu')
        self.x_range = x_range
        # When transform is provided, training operates in standardized space z̃:
        # samples / phi / dphi all live in z̃ space, and target_score is queried
        # in the original space x = transform.inverse_transform(z̃) and then
        # pulled back to z̃ space with transform.transform_score.
        self.transform = transform

    def _sample_x(self, mini_batch):
        # The local operators are always estimated against the uniform proposal
        # (see LossCalculator._build_local_m), so x is drawn uniformly too.
        np.random.seed(self.seed + mini_batch)  # deterministic across runs
        return np.random.uniform(
            low=self.x_range[0],
            high=self.x_range[1],
            size=(self.B, self.D),
        )

    def get_mini_batch(self, mini_batch):
        np.random.seed(self.seed+mini_batch)  # Ensure different mini-batches get different samples
        # z lives in the QuanVI training space (== standardized space when transform
        # is set, otherwise original space).
        z = self._sample_x(mini_batch)

        if self.transform is not None:
            x_orig = self.transform.inverse_transform(z)
        else:
            x_orig = z

        basis = HermitePolynomials(dim=self.D, num_basis_functions=self.K)
        phi_x = basis.transform(z)

        dphi_x = []
        for k in range(self.K):
            dx = basis.gradient_transform(z, k)
            dx = np.asarray(dx).reshape(self.B, self.D)
            dphi_x.append(dx)
        dphi_x = np.stack(dphi_x, axis=0)
        dphi_x = dphi_x.transpose(1,2,0)

        if self.transform is not None:
            sp_x_orig = self.target_score(x_orig)
            sp_x_orig = self.transform.transform_score(sp_x_orig)
        else:
            sp_x_orig = self.target_score(x_orig)
        if isinstance(sp_x_orig, torch.Tensor):
            sp_x_orig = sp_x_orig.detach().cpu().numpy()
        sp_x = np.asarray(sp_x_orig, dtype=np.float64)

        real_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64

        phi_x  = torch.as_tensor(phi_x,  dtype=self.dtype,      device=self.device)         # complex64
        dphi_x = torch.as_tensor(dphi_x, dtype=self.dtype,      device=self.device)         # complex64
        sp_x   = torch.as_tensor(sp_x,   dtype=real_dtype,      device=self.device)         # float32 referred with complex64

        # x field is stored in the SAME space as phi/dphi (i.e. z̃ when transform
        # is set), so any downstream consumer that mixes M.x with M.phi_x stays
        # consistent. Original-space coordinates can be recovered via
        # self.transform.inverse_transform(M.x) when needed.
        return InitM.M.build(phi_x, dphi_x, sp_x, z)


class LocalMBuilder:
    def __init__(
        self,
        D: int,
        K: int,
        proposal_dist: Any,
        basis_function: Any,
        transform: Optional[Any] = None,
        x_range: Optional[tuple] = None,
    ):
        self.D = D
        self.K = K
        self.proposal_dist = proposal_dist
        self.basis_function = basis_function
        self.transform = transform
        # (lo, hi) for per-dimension uniform proposal. When provided and
        # importance_weight=True, inv_pi is computed from the LOCAL marginal
        # (n_sites-dim uniform) instead of the full D-dim joint, which is the
        # statistically correct choice because each local g_d depends only on
        # n_sites coordinates.
        self.x_range = x_range

    def build_M_loc_3local_new(
        self,
        target_score: callable,
        num_samples: int,
        symmetrize: bool = True,
        dtype: np.dtype = np.float64,
        importance_weight: bool = True,
    ) -> List[np.ndarray]:
        """
        Estimate a 3-local decomposition of M using importance sampling, where each
        local block is estimated from an independent resampling of the proposal.

        Assumption:
            The score s_d(x) = d/dx_d log p(x) depends only on local variables:
                - d = 0:        (x_0, x_1)
                - 1 <= d <= D-2:(x_{d-1}, x_d, x_{d+1})
                - d = D-1:      (x_{D-2}, x_{D-1})

        Sampling convention:
            The input `num_samples` denotes the total sampling budget from the proposal.
            This budget is split across all local blocks. For each local term, we draw
            a fresh batch of samples independently from `proposal_dist`.

            If D = self.dim, then the number of local blocks is D:
                - 1 left boundary block
                - D - 2 interior blocks
                - 1 right boundary block

            Therefore, each block is estimated using approximately `num_samples / D`
            proposal samples.

        Returns:
            m_list: list of local matrices, reshaped so every leg has size K
                (the number of basis functions == the core tensor bond dimension).
                - m_list[0]      has shape (K, K, K, K)
                - m_list[d]      has shape (K, K, K, K, K, K) for 1 <= d <= D-2
                - m_list[D - 1]  has shape (K, K, K, K)

        Notes:
            Each local block is estimated by importance sampling:
                M_loc^(d) ~= (1 / N_loc) sum_n [ g_d(x_n) g_d(x_n)^T / pi(x_n) ],
                x_n ~ proposal_dist

            where N_loc is the number of resampled proposal points assigned to the
            corresponding local block.

        Warning:
            The exact importance weight for each local block may need adjustment
            depending on the factorization structure of the proposal distribution
            and the locality assumption used in the decomposition.
        """
        if self.D < 2:
            raise ValueError("build_M_loc_3local assumes dim >= 2.")

        D = self.D
        # Feature dimension: legs of m have size K, which is also the core
        # tensor bond dimension.
        K = self.K

        # Total number of local blocks:
        #   1 left boundary + (D - 2) interior + 1 right boundary = D
        num_blocks = D

        # Split the total proposal budget across local terms.
        # Each local term is estimated from an independent resampling.
        samples_per_block = max(1, num_samples // num_blocks)

        def _draw_samples_and_features(n_sites: int):
            """
            Draw a fresh batch from the proposal and compute the ingredients needed
            for local block construction.

            n_sites: number of physical coordinates that the local g_d actually
            depends on (2 for boundary blocks, 3 for interior). Used to decide
            the correct importance weight.
            """
            x = self.proposal_dist.sample((samples_per_block,))  # (N_loc, D)

            if self.transform is not None:
                x_original = self.transform.inverse_transform(x)
                d_log_p_x = target_score(x_original)
                d_log_p_x = self.transform.transform_score(d_log_p_x)
            else:
                d_log_p_x = target_score(x)

            # Importance weight:
            #   - g_d depends only on n_sites (2 or 3) coordinates, so the right
            #     MC estimator divides by the LOCAL marginal p_local(x_local),
            #     not by the full D-dim joint p_full(x). For per-dim uniform on
            #     [lo, hi] this is just (hi - lo) ** n_sites -> a constant that
            #     does NOT depend on D, so m stays at a stable scale.
            #   - When importance_weight is False, inv_pi_x = 1 and m estimates
            #     E_pi[g g^T] (the InitM-style scale, no reweighting).
            #   - When importance_weight is True but x_range is not provided, we
            #     fall back to the original full-D log_prob path (kept for
            #     backward compatibility; numerically unstable at larger D).
            if importance_weight:
                if self.x_range is not None:
                    lo = float(self.x_range[0])
                    hi = float(self.x_range[1])
                    local_vol = (hi - lo) ** int(n_sites)
                    inv_pi_x = np.full((samples_per_block,), local_vol, dtype=dtype)
                else:
                    log_pi_x = self.proposal_dist.log_prob(x).detach().cpu().numpy()  # (N_loc,)
                    inv_pi_x = np.exp(-log_pi_x).astype(dtype)
            else:
                inv_pi_x = np.ones((samples_per_block,), dtype=dtype)

            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            if isinstance(d_log_p_x, torch.Tensor):
                d_log_p_x = d_log_p_x.detach().cpu().numpy()
            d_log_p_x = np.asarray(d_log_p_x).astype(dtype)

            phi_x = self.basis_function.transform(x_np).astype(dtype)  # (N_loc, D, K)

            d_phi_x = np.stack(
                [
                    self.basis_function.gradient_transform(x_np, k)
                    for k in range(K)
                ],
                axis=-1,
            ).astype(dtype)  # (N_loc, D, K)

            return x_np, d_log_p_x, phi_x, d_phi_x, inv_pi_x

        def _weighted_gram(g: np.ndarray, inv_pi_x: np.ndarray) -> np.ndarray:
            """
            Parameters
            ----------
            g : np.ndarray
                Shape (N_loc, K)
            inv_pi_x : np.ndarray
                Shape (N_loc,)

            Returns
            -------
            np.ndarray
                Weighted Gram matrix of shape (K, K)
            """
            gw = g * np.sqrt(inv_pi_x)[:, None]
            m = (gw.T @ gw) / float(g.shape[0])
            if symmetrize:
                m = 0.5 * (m + m.T)
            return m.astype(dtype, copy=False)

        def _pair_feature(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """
            Construct pairwise Kronecker features.

            Parameters
            ----------
            a, b : np.ndarray
                Shape (N_loc, K)

            Returns
            -------
            np.ndarray
                Shape (N_loc, K^2), ordered as a \\otimes b
            """
            return np.einsum("ni,nj->nij", a, b).reshape(a.shape[0], K * K)

        def _triple_feature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
            """
            Construct triple Kronecker features.

            Parameters
            ----------
            a, b, c : np.ndarray
                Shape (N_loc, K)

            Returns
            -------
            np.ndarray
                Shape (N_loc, K^3), ordered as a \\otimes b \\otimes c
            """
            return np.einsum("ni,nj,nk->nijk", a, b, c).reshape(a.shape[0], K * K * K)

        m_list: List[np.ndarray] = []

        # ----- left boundary block: d = 0, support on (0, 1)
        _, d_log_p_x, phi_x, d_phi_x, inv_pi_x = _draw_samples_and_features(n_sites=2)
        phi0 = phi_x[:, 0, :]
        phi1 = phi_x[:, 1, :]
        dphi0 = d_phi_x[:, 0, :]
        s0 = d_log_p_x[:, 0]

        Phi_loc = _pair_feature(dphi0, phi1)
        Phi_base = _pair_feature(phi0, phi1)
        g0 = 2.0 * Phi_loc - s0[:, None] * Phi_base
        m_list.append(_weighted_gram(g0, inv_pi_x).reshape(K, K, K, K))

        # ----- interior blocks: d = 1, ..., D - 2, support on (d-1, d, d+1)
        for d in range(1, D - 1):
            _, d_log_p_x, phi_x, d_phi_x, inv_pi_x = _draw_samples_and_features(n_sites=3)

            phi_l = phi_x[:, d - 1, :]
            phi_c = phi_x[:, d, :]
            phi_r = phi_x[:, d + 1, :]
            dphi_c = d_phi_x[:, d, :]
            sd = d_log_p_x[:, d]

            Phi_loc = _triple_feature(phi_l, dphi_c, phi_r)
            Phi_base = _triple_feature(phi_l, phi_c, phi_r)
            gd = 2.0 * Phi_loc - sd[:, None] * Phi_base
            m_list.append(_weighted_gram(gd, inv_pi_x).reshape(K, K, K, K, K, K))

        # ----- right boundary block: d = D - 1, support on (D - 2, D - 1)
        _, d_log_p_x, phi_x, d_phi_x, inv_pi_x = _draw_samples_and_features(n_sites=2)
        phi_l = phi_x[:, D - 2, :]
        phi_r = phi_x[:, D - 1, :]
        dphi_r = d_phi_x[:, D - 1, :]
        s_last = d_log_p_x[:, D - 1]

        Phi_loc = _pair_feature(phi_l, dphi_r)
        Phi_base = _pair_feature(phi_l, phi_r)
        g_last = 2.0 * Phi_loc - s_last[:, None] * Phi_base
        m_list.append(_weighted_gram(g_last, inv_pi_x).reshape(K, K, K, K))

        self.M_loc = m_list
        return m_list
    
        
