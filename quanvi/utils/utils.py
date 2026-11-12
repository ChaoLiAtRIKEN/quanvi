import math
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import opt_einsum as oe
from posteriordb import PosteriorDatabase
import bridgestan as bs
import json

from .target_distribution import (
    XShape, RingDistribution, FunnelDistribution, SinhArcsinhGaussian,
    GMM3
)
from .distribution_transform import (
    LongRangeTransformedDistribution,
    TransformedDistribution,
)


def set_target_score(
    target_name,
    ndim,
    device=None,
    dependency_lag=5,
    long_range_strength=0.0,
):
    # Resolve device once so every component lives on the same device.
    if device is None:
        device = torch.device('cpu')
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    # For gaussian
    cov_matrix = torch.eye(ndim, dtype=torch.float64, device=device)
    indices = torch.arange(ndim - 1, device=device)
    cov_matrix[indices + 1, indices] = 0.2
    cov_matrix[indices, indices + 1] = 0.2
    cov_matrix.diagonal().fill_(1.0)
    identity_cov_matrix = torch.eye(ndim, dtype=torch.float64, device=device)
    long_range_suffix = '_long_range'
    use_long_range = target_name.endswith(long_range_suffix)
    base_target_name = (
        target_name[:-len(long_range_suffix)]
        if use_long_range
        else target_name
    )
    target_distribution = {
        'xshape': XShape(3.0, device=device),
        'ring': RingDistribution(3.0, 0.5, device=device),
        'funnel': FunnelDistribution(1.2, device=device),
        'gmm3': GMM3(device=device),
        'gaussian': torch.distributions.MultivariateNormal(
            loc=torch.zeros(ndim, dtype=torch.float64, device=device),
            covariance_matrix = torch.linalg.inv(cov_matrix)
        ),
        'identity_gaussian': torch.distributions.MultivariateNormal(
            loc=torch.zeros(ndim, dtype=torch.float64, device=device),
            covariance_matrix = torch.linalg.inv(identity_cov_matrix)
        ),
    }[base_target_name]
    if base_target_name == 'gaussian' or base_target_name == 'identity_gaussian':
        if use_long_range:
            raise ValueError(
                "Long-range augmentation is only available for xshape, ring, "
                "funnel, and gmm3."
            )
        return get_pytorch_score(target_distribution), target_distribution

    if ndim == 2:
        if use_long_range:
            raise ValueError("A long-range target requires ndim > 2.")
        return get_pytorch_score(target_distribution), target_distribution

    if use_long_range:
        target_distribution = LongRangeTransformedDistribution(
            base_distribution=target_distribution,
            ndim=ndim,
            dependency_lag=dependency_lag,
            long_range_strength=long_range_strength,
        )
    else:
        target_distribution = TransformedDistribution(
            base_distribution=target_distribution, ndim=ndim,
        )

    # Autograd score: directly differentiate log_prob of the full augmented distribution
    target_score = get_pytorch_score(target_distribution)
    return target_score, target_distribution


def get_pytorch_score(distribution):
    with torch.no_grad():
        _probe = distribution.sample(torch.Size([1]))
        target_dtype = _probe.dtype
        # The distribution's parameters live on this device; log_prob requires
        # inputs on the same device, so we coerce x to it inside score_fn.
        target_device = _probe.device
    def score_fn(x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=target_dtype, device=target_device)
        else:
            x = x.to(device=target_device, dtype=target_dtype)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            logp = distribution.log_prob(x)
            (grad,) = torch.autograd.grad(
                logp.real.sum(), x,
                create_graph=False,
                retain_graph=False
            )
        return grad.detach()
    return score_fn

def get_stein_score_from_stan(model):
    def score_fn(x):
        assert x.ndim == 2
        score = []
        for i in range(x.shape[0]):
            grad = model.log_density_gradient(x[i], jacobian=True)[1]
            score.append(grad)
        return np.stack(score)
    return score_fn


def sample_fisher_divergence(score_p, score_q, samples): 
    score_px = _as_2d_numpy_array(score_p(samples), "score_p")
    score_qx = _as_2d_numpy_array(score_q(samples), "score_q")
    assert score_px.ndim == 2 and score_qx.ndim == 2
    if score_px.shape != score_qx.shape:
        raise ValueError(f"score shape mismatch: {score_px.shape} vs {score_qx.shape}")

    diff = score_px - score_qx
    finite_rows = np.isfinite(diff).all(axis=1)
    if not finite_rows.all():
        warnings.warn(
            f"Dropping {np.size(finite_rows) - np.count_nonzero(finite_rows)} non-finite score rows "
            "from Fisher divergence estimate.",
            RuntimeWarning,
        )
        diff = diff[finite_rows]
    if diff.shape[0] == 0:
        raise ValueError("No finite score rows available for Fisher divergence estimate.")

    return np.mean(np.linalg.norm(diff, ord=2, axis=-1) ** 2)


def sample_fisher_divergence_pytorch(score_p, score_q, samples):

    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples)

    score_px = score_p(samples)
    score_qx = score_q(samples)
    
    assert score_px.dim() == 2 and score_qx.dim() == 2
    if score_px.shape != score_qx.shape:
        raise ValueError(f"score shape mismatch: {score_px.shape} vs {score_qx.shape}")
    
    diff = score_px - score_qx
    finite_rows = torch.isfinite(diff).all(dim=1)
    if not bool(finite_rows.all()):
        warnings.warn(
            f"Dropping {diff.shape[0] - int(finite_rows.sum().item())} non-finite score rows "
            "from Fisher divergence estimate.",
            RuntimeWarning,
        )
        diff = diff[finite_rows]
    if diff.shape[0] == 0:
        raise ValueError("No finite score rows available for Fisher divergence estimate.")

    squared_norm = torch.linalg.norm(diff, ord=2, dim=-1).pow(2)
    
    return torch.mean(squared_norm)


def _as_numpy_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_2d_numpy_array(x, name):
    x = _as_numpy_array(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {x.shape}.")
    return x.astype(np.float64, copy=False)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def kl_divergence(p, q, num_samples=10000, return_stats=False):
    samples = p.sample((num_samples,))
    logp = p.log_prob(samples)
    logq = q.log_prob(samples)

    if not isinstance(logp, torch.Tensor):
        logp = torch.as_tensor(logp, dtype=torch.float64)
    if not isinstance(logq, torch.Tensor):
        logq = torch.as_tensor(logq, dtype=logp.dtype, device=logp.device)
    else:
        # p and q may live on different devices (e.g. CPU target vs GPU
        # inference engine); align before arithmetic.
        logq = logq.to(device=logp.device, dtype=logp.dtype)

    terms = (logp - logq).reshape(-1)
    finite = torch.isfinite(terms)
    num_dropped = int(terms.numel() - finite.sum().item())
    if num_dropped > 0:
        warnings.warn(
            f"Dropping {num_dropped} non-finite KL terms from {terms.numel()} samples.",
            RuntimeWarning,
        )
        terms = terms[finite]
    if terms.numel() == 0:
        raise ValueError("No finite KL terms available.")

    kl = torch.mean(terms)
    if not return_stats:
        return kl.item()

    if terms.numel() > 1:
        std = torch.std(terms, unbiased=True)
        se = std / math.sqrt(terms.numel())
    else:
        std = torch.zeros((), dtype=terms.dtype, device=terms.device)
        se = torch.full((), float("nan"), dtype=terms.dtype, device=terms.device)

    return {
        "kl": kl.item(),
        "std": std.item(),
        "se": se.item(),
        "n": int(terms.numel()),
        "num_dropped": num_dropped,
    }


def get_posterior_db_model(model_name):
    names = {
        'kidscore': 'kidiq-kidscore_momhs',
        'logearn': 'earnings-logearn_logheight_male',
        'gpregr': 'gp_pois_regr-gp_regr',
        'garch': 'garch-garch11',
        'ark': 'arK-arK',
        'logmesquite': 'mesquite-logmesquite_logvas',
        'school': 'eight_schools-eight_schools_noncentered',
        'nes': 'nes1972-nes',  # 10
        'diamonds': 'diamonds-diamonds',
        'gp_pois': 'gp_pois_regr-gp_pois_regr',
        'mcycle': 'mcycle_gp-accel_gp',
        'hmm': 'hmm_example-hmm_example',
        'bball0': 'bball_drive_event_0-hmm_drive_0',
        'bball1': 'bball_drive_event_1-hmm_drive_1',
        'arma': 'arma-arma11',
        'blr': 'sblri-blr',
    }
    assert model_name in names.keys(), f"Model {model_name} not supported."
    # Resolve this path from the repository because posterior runs may start
    # from a separate output directory.

    _repo_root = Path(__file__).resolve().parents[2]
    posterior_path = os.environ.get(
        "POSTERIOR_DB_PATH",
        str(_repo_root / "data" / "posteriordb" / "posterior_database"),
    )
    if not Path(posterior_path).is_dir():
        raise FileNotFoundError(
            f"PosteriorDatabase path not found: {posterior_path!r}. "
            "Set POSTERIOR_DB_PATH to your posterior_database directory, "
            "or run from a cwd where data/posteriordb/posterior_database exists under the repo root."
        )
    posterior_database = PosteriorDatabase(posterior_path)

    posterior = posterior_database.posterior(names[model_name])
    stan_file = posterior.model.code_file_path("stan")
    data = posterior.data.values()    

    bs_model = bs.StanModel.from_stan_file(stan_file, data)

    return posterior, bs_model


def _as_jsonable(x):
    """Convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    return x


def _call_log_density_gradient(bs_model, theta_unc, jacobian=True):
    """
    BridgeStan APIs differ slightly across versions/languages.
    Try keyword jacobian first, then fall back to positional/default.
    """
    theta_unc = np.ascontiguousarray(theta_unc, dtype=np.float64)
    try:
        return bs_model.log_density_gradient(theta_unc, jacobian=jacobian)
    except TypeError:
        # older signature: log_density_gradient(theta_unc) with default jacobian
        return bs_model.log_density_gradient(theta_unc)


def _unconstrain_draw(bs_model, draw_con_dict):
    """
    Convert a single constrained draw (dict form) into theta_unc.
    Tries param_unconstrain(dict) then param_unconstrain_json(str).
    """
    # ensure jsonable
    draw_con_dict = {k: _as_jsonable(v) for k, v in draw_con_dict.items()}

    if hasattr(bs_model, "param_unconstrain"):
        try:
            # Some versions accept dict-like constrained params
            return bs_model.param_unconstrain(draw_con_dict)
        except Exception:
            pass

    if hasattr(bs_model, "param_unconstrain_json"):
        return bs_model.param_unconstrain_json(json.dumps(draw_con_dict))

    raise RuntimeError("BridgeStan model does not expose param_unconstrain / param_unconstrain_json.")


def _normalize_index_name(name: str) -> str:
    """
    Normalize names like:
      beta[1]      -> beta.1
      Sigma[2, 3]  -> Sigma.2.3
      beta.1       -> beta.1  (unchanged)
    BridgeStan uses dot-separated indexing for containers.
    """
    s = str(name).replace(" ", "")
    s = s.replace("[", ".").replace("]", "")
    s = s.replace(",", ".")
    # collapse accidental double dots
    while ".." in s:
        s = s.replace("..", ".")
    return s


def _posterior_data_values(posterior):
    """Return PosteriorDB data values as a plain mapping when available."""
    if posterior is None or not hasattr(posterior, "data"):
        return {}

    data_obj = posterior.data
    if not hasattr(data_obj, "values"):
        return {}

    values = data_obj.values
    return values() if callable(values) else values


def _gp_exp_quad_cholesky(x, alpha, rho):
    """
    Cholesky factor matching Stan's gp_exp_quad_cov(x, alpha, rho)
    with the small diagonal nugget used by gp_pois_regr.stan.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    diff = x[:, None] - x[None, :]
    cov = float(alpha) ** 2 * np.exp(-0.5 * (diff / float(rho)) ** 2)
    cov[np.diag_indices_from(cov)] += 1e-10
    cov = 0.5 * (cov + cov.T)
    return np.linalg.cholesky(cov)


def _derive_gp_pois_f_tilde(chain, bs_names_norm, posterior):
    """
    PosteriorDB stores gp_pois_regr reference draws for transformed parameter f,
    while BridgeStan expects constrained parameters rho, alpha, f_tilde.
    Reconstruct f_tilde from f = L_cov * f_tilde draw-by-draw.
    """
    key_map = {_normalize_index_name(k): k for k in chain.keys()}
    missing = [n for n in bs_names_norm if n not in key_map]
    f_tilde_missing = [n for n in missing if n.startswith("f_tilde.")]
    if not f_tilde_missing:
        return chain

    data = _posterior_data_values(posterior)
    if "x" not in data or "alpha" not in key_map or "rho" not in key_map:
        return chain

    suffixes = [name[len("f_tilde."):] for name in f_tilde_missing]
    f_keys = [key_map.get(f"f.{suffix}") for suffix in suffixes]
    if any(k is None for k in f_keys):
        return chain

    x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    if len(f_keys) != x.size:
        return chain

    alpha = np.asarray(chain[key_map["alpha"]], dtype=np.float64).reshape(-1)
    rho = np.asarray(chain[key_map["rho"]], dtype=np.float64).reshape(-1)
    f = np.column_stack([
        np.asarray(chain[k], dtype=np.float64).reshape(-1) for k in f_keys
    ])
    if f.shape[0] != alpha.size or f.shape[0] != rho.size:
        raise RuntimeError(
            "Cannot reconstruct gp_pois_regr f_tilde: reference draw columns "
            "have inconsistent lengths."
        )

    f_tilde = np.empty_like(f, dtype=np.float64)
    for j in range(f.shape[0]):
        if alpha[j] <= 0 or rho[j] <= 0:
            raise RuntimeError(
                "Cannot reconstruct gp_pois_regr f_tilde from non-positive "
                f"alpha/rho at draw {j}: alpha={alpha[j]}, rho={rho[j]}."
            )
        try:
            L_cov = _gp_exp_quad_cholesky(x, alpha[j], rho[j])
            f_tilde[j] = np.linalg.solve(L_cov, f[j])
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Cannot reconstruct gp_pois_regr f_tilde at draw {j}: {e}"
            ) from e

    if not np.isfinite(f_tilde).all():
        raise RuntimeError("Reconstructed gp_pois_regr f_tilde contains non-finite values.")

    chain_view = dict(chain)
    for col, name in enumerate(f_tilde_missing):
        chain_view[name] = f_tilde[:, col]
    return chain_view


def _chain_with_derived_reference_params(chain, bs_names_norm, posterior=None):
    """
    Return a shallow chain copy with constrained Stan parameters reconstructed
    when PosteriorDB stores only an equivalent transformed parameter.
    """
    chain = _derive_gp_pois_f_tilde(chain, bs_names_norm, posterior)

    key_map = {_normalize_index_name(k): k for k in chain.keys()}
    missing = [n for n in bs_names_norm if n not in key_map]
    theta_trans_missing = [n for n in missing if n.startswith("theta_trans.")]
    if not theta_trans_missing:
        return chain

    if "mu" not in key_map or "tau" not in key_map:
        return chain

    chain_view = dict(chain)
    mu = np.asarray(chain[key_map["mu"]], dtype=np.float64)
    tau = np.asarray(chain[key_map["tau"]], dtype=np.float64)
    derived = False

    for name in theta_trans_missing:
        suffix = name[len("theta_trans."):]
        theta_key = key_map.get(f"theta.{suffix}")
        if theta_key is None:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            theta_trans = (np.asarray(chain[theta_key], dtype=np.float64) - mu) / tau
        if not np.isfinite(theta_trans).all():
            raise RuntimeError(
                "Failed to reconstruct theta_trans from PosteriorDB reference draws "
                "using theta_trans = (theta - mu) / tau."
            )
        chain_view[name] = theta_trans
        derived = True

    return chain_view if derived else chain


def _posterior_chain_key_map(chain, bs_model, posterior=None):
    try:
        bs_names = list(bs_model.param_names())
    except Exception as e:
        raise RuntimeError(f"Failed to read BridgeStan param_names(): {e}")

    bs_names_norm = [_normalize_index_name(n) for n in bs_names]
    chain = _chain_with_derived_reference_params(chain, bs_names_norm, posterior=posterior)
    key_map = {_normalize_index_name(k): k for k in chain.keys()}
    missing = [n for n in bs_names_norm if n not in key_map]
    if missing:
        avail = list(chain.keys())
        raise RuntimeError(
            "reference_draws keys do not match BridgeStan param_names().\n"
            f"Missing (first 10): {missing[:10]}\n"
            f"Available keys (first 20): {avail[:20]}\n"
            "Hint: PosteriorDB often uses bracket indexing like beta[1], while "
            "BridgeStan uses dot indexing like beta.1."
        )

    return bs_names, bs_names_norm, key_map, chain


def _unconstrained_param_names(bs_model, dim):
    """
    Return names aligned with BridgeStan's unconstrained parameter vector.

    Constrained parameter names can be longer than the unconstrained vector for
    constrained containers such as simplex[2], so plots/statistics should use
    these names instead.
    """
    if hasattr(bs_model, "param_unc_names"):
        try:
            names = list(bs_model.param_unc_names())
            if len(names) == dim:
                return names
        except Exception:
            pass

    return [f"theta_unc[{i}]" for i in range(dim)]


def posterior_reference_samples_unconstrained(
    posterior,
    bs_model,
    *,
    chain_idx: int = 0,
    max_draws: int | None = 1000,
    seed: int | None = None,
):
    """
    Return PosteriorDB reference draws in BridgeStan unconstrained space.

    The returned variable names are aligned with columns of the unconstrained
    sample matrix, not the original constrained parameter vector.

    This intentionally does not mutate posterior.reference_draws(); mutating those
    cached dictionaries can make later Fisher-divergence calls unconstrain an
    already-unconstrained draw.
    """
    ref = posterior.reference_draws()
    if chain_idx >= len(ref):
        raise IndexError(f"chain_idx={chain_idx} is out of range for {len(ref)} chains.")

    chain = ref[chain_idx]
    if not isinstance(chain, dict) or len(chain) == 0:
        raise RuntimeError(f"Reference chain {chain_idx} is empty or invalid.")

    bs_names, bs_names_norm, key_map, chain = _posterior_chain_key_map(
        chain, bs_model, posterior=posterior
    )
    first_key = next(iter(chain.keys()))
    S = len(chain[first_key])

    if max_draws is None or max_draws >= S:
        draw_ids = np.arange(S)
    elif seed is None:
        draw_ids = np.arange(max_draws)
    else:
        rng = np.random.default_rng(seed)
        draw_ids = np.sort(rng.choice(S, size=max_draws, replace=False))

    samples = []
    for j in draw_ids:
        theta_con = np.array(
            [chain[key_map[name_norm]][j] for name_norm in bs_names_norm],
            dtype=np.float64,
        )
        theta_unc = bs_model.param_unconstrain(np.ascontiguousarray(theta_con))
        samples.append(np.asarray(theta_unc, dtype=np.float64))

    samples = np.stack(samples, axis=0)
    return samples, _unconstrained_param_names(bs_model, samples.shape[1])


def fisher_div(
    posterior,
    bs_model,
    score_q,
    *,
    jacobian: bool = True,
    max_draws_per_chain: int | None = None,
    seed: int = 0,
    transform=None,
    fisher_space: str = "unconstrained",
):
    """
    Fisher divergence estimate:
      E_{p(theta_unc)} || score_p(theta_unc) - score_q(theta_unc) ||^2

    - PosteriorDB reference_draws are treated as CONSTRAINED draws.
    - We map them to unconstrained space via bs_model.param_unconstrain(theta_con).
    - score_p is computed by bs_model.log_density_gradient(theta_unc, jacobian=...),
      whose input is UNCONSTRAINED parameters.
    - By default the squared norm is also computed in that unconstrained space.
      If fisher_space="standardized", transform must be the affine standardizer
      used by q, and both score vectors are mapped to the standardized coordinates
      before taking the norm. This matches the Fisher metric for y=T(theta_unc).
    """
    valid_spaces = {"unconstrained", "standardized"}
    if fisher_space not in valid_spaces:
        raise ValueError(f"fisher_space must be one of {sorted(valid_spaces)}, got {fisher_space!r}.")
    if fisher_space == "standardized" and transform is None:
        raise ValueError('fisher_space="standardized" requires transform.')

    ref = posterior.reference_draws()
    rng = np.random.default_rng(seed)

    div_per_chain = []

    for chain_idx, chain in enumerate(ref):
        if not isinstance(chain, dict) or len(chain) == 0:
            continue

        _, bs_names_norm, key_map, chain = _posterior_chain_key_map(
            chain, bs_model, posterior=posterior
        )

        # Determine number of draws from any one column
        first_key = next(iter(chain.keys()))
        S = len(chain[first_key])

        if max_draws_per_chain is None or max_draws_per_chain >= S:
            draw_ids = np.arange(S)
        else:
            draw_ids = rng.choice(S, size=max_draws_per_chain, replace=False)

        vals = []
        for j in draw_ids:
            # 1) constrained flat vector in BridgeStan order
            theta_con = np.array(
                [chain[key_map[name_norm]][j] for name_norm in bs_names_norm],
                dtype=np.float64,
            )
            theta_con = np.ascontiguousarray(theta_con)

            # 2) unconstrain
            theta_unc = bs_model.param_unconstrain(theta_con)
            theta_unc = np.ascontiguousarray(theta_unc, dtype=np.float64)

            # 3) posterior score in unconstrained space
            _, score_p = bs_model.log_density_gradient(theta_unc, jacobian=jacobian)
            score_p = np.asarray(score_p, dtype=np.float64)

            # 4) variational score (must be on SAME theta_unc space)
            score_qx = _as_2d_numpy_array(score_q(theta_unc[None, :]), "score_q").reshape(-1)

            if score_p.shape != score_qx.shape:
                raise ValueError(f"score shape mismatch: {score_p.shape} vs {score_qx.shape}")
            if not np.isfinite(score_p).all() or not np.isfinite(score_qx).all():
                continue
            if fisher_space == "standardized":
                    score_p = _as_2d_numpy_array(
                        transform.transform_score(score_p[None, :]),
                        "score_p_standardized",
                    ).reshape(-1)
                    score_qx = _as_2d_numpy_array(
                        transform.transform_score(score_qx[None, :]),
                        "score_q_standardized",
                    ).reshape(-1)

            diff = score_p - score_qx
            vals.append(float(np.dot(diff, diff)))

        if len(vals) == 0:
            warnings.warn(
                f"Skipping chain {chain_idx}: no finite Fisher-divergence terms.",
                RuntimeWarning,
            )
            continue

        div_per_chain.append(float(np.mean(vals)))

    if len(div_per_chain) == 0:
        raise RuntimeError("No valid reference draws found to compute fisher_div().")

    return float(np.mean(div_per_chain)), float(np.std(div_per_chain))
