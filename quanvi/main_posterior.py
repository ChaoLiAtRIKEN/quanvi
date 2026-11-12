"""Run QuanVI on a PosteriorDB target."""
from __future__ import annotations

import argparse
import cmath
import random
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from einsum_string_builder import EinsumStringBuilder
from inference import Inference
from inference_visualization import plot_theoretical_marginal_comparison_style
from loss_calculator import LossCalculator
from tensor_factory import InitM, init_core_tensor
from utils import stiefel_optimizer_complex
from utils.transforms import make_gaussian_standardizer
from utils.plot_utils import plot_lower_triangle_joint_distribution
from utils.utils import (
    fisher_div,
    get_posterior_db_model,
    posterior_reference_samples_unconstrained,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser(description="QuanVI on PosteriorDB (minimal)")
    p.add_argument("--target", type=str, default="kidscore", help="PosteriorDB short name (kidscore, school, ...)")
    p.add_argument("--E", type=int, default=0, help="Environment qubits (use 0 for posterior runs)")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--Nk", type=int, default=2)
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--Nb", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--Num_Step", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--dtype", type=str, default="cdouble", choices=["cfloat", "cdouble", "float", "double"])
    p.add_argument("--is_identity", action="store_true", default=False)
    p.add_argument("--is_inference", action="store_true", default=False)
    p.add_argument("--is_random_seed", action="store_true", default=False)
    p.add_argument("--is_gradclip", action="store_true", default=False)
    p.add_argument("--environment_inx_list", type=int, nargs="+", default=[1, 4])
    p.add_argument("--read_coretensor", type=str, default="")
    p.add_argument("--is_savem", action="store_true", default=False)
    p.add_argument("--is_minibatch", action="store_true", default=False)
    p.add_argument(
        "--sample_chunk_size",
        type=int,
        default=None,
        help="Draw AR samples in chunks of this size. Peak memory scales with "
             "chunk_size*n_grid instead of n_samples*n_grid. Default None = one shot (unchanged).",
    )
    p.add_argument("--not_train", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--standardize_mean", type=str, default="")
    p.add_argument("--standardize_cov", type=str, default="")
    p.add_argument("--standardize_diag", action="store_true", default=False)
    p.add_argument(
        "--fisher_max_draws",
        type=int,
        default=None,
        help="Per-chain cap for Fisher (default: use all draws, like baselines/eigen_vi/posterior_db.py).",
    )
    p.add_argument(
        "--fisher_space",
        type=str,
        default="unconstrained",
        choices=["auto", "unconstrained", "standardized"],
        help="Same as baselines/eigen_vi/posterior_db.py (EigenVI default: unconstrained).",
    )
    p.add_argument(
        "--plot_n_samples",
        type=int,
        default=5000,
        help="Samples for marginal/joint plots (EigenVI posterior_db uses 5000).",
    )
    p.add_argument(
        "--sample_n_grid",
        type=int,
        default=1000,
        help="Grid points for Inference.sample AR (Inference default 1000).",
    )
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def accumulate_gradients(
    m_builder, core_tensor, D, E, K, Nk, B, Dtype, state, einsum_builder,
    is_train,
):
    context = torch.enable_grad() if is_train else torch.no_grad()
    total_loss = 0.0
    with context:
        loss_calculator = LossCalculator(
            D, E, K, Nk, B, Dtype, state, einsum_builder, core_tensor,
            optimizer="greedy", init_m=m_builder,
        )
        loss = loss_calculator.get_loss()
        if is_train:
            loss.backward()
        total_loss += float(loss.detach().item())
    return total_loss


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device {args.device} but CUDA unavailable, falling back to cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "cfloat":
        Dtype = torch.cfloat
    elif args.dtype == "cdouble":
        Dtype = torch.cdouble
    elif args.dtype == "float":
        Dtype = torch.float32
    else:
        Dtype = torch.float64

    seed = args.seed
    if args.is_random_seed:
        seed = random.randint(0, 10000)
    torch.manual_seed(seed)
    set_all_seeds(seed)

    posterior, bs_model = get_posterior_db_model(args.target)
    D = int(bs_model.param_unc_num())
    E = args.E
    K = args.K
    Nk = args.Nk
    B = args.B
    Nb = args.Nb

    def target_score(x):
        is_tensor = isinstance(x, torch.Tensor)
        x_np = x.detach().cpu().numpy() if is_tensor else np.asarray(x)
        grad = []
        for i in range(x_np.shape[0]):
            theta = np.ascontiguousarray(x_np[i], dtype=np.float64)
            _, grad_i = bs_model.log_density_gradient(theta, jacobian=True)
            grad.append(grad_i)
        grad = np.stack(grad, axis=0)
        return torch.tensor(grad, dtype=torch.float64, device=device) if is_tensor else grad

    transform = None
    if args.standardize_mean or args.standardize_cov:
        if not args.standardize_mean or not args.standardize_cov:
            raise ValueError("--standardize_mean and --standardize_cov must be provided together.")
        transform = make_gaussian_standardizer(
            np.load(args.standardize_mean),
            np.load(args.standardize_cov),
            diagonal=args.standardize_diag,
        )

    print(f"[main.posterior] target={args.target} D={D} (param_unc_num) E={E} K={K} seed={seed}")

    einsum_builder = EinsumStringBuilder()
    state = torch.zeros(K, dtype=Dtype, device=device)
    if Dtype in (torch.cfloat, torch.cdouble):
        state[:] = (1.0 + 1j) / cmath.sqrt(2 * K)
    else:
        state[:] = 1.0 / (K ** 0.5)

    environment_inx_list = list(args.environment_inx_list)
    system_inx_list = []
    lenth = len(environment_inx_list)
    for i in range(E - lenth):
        environment_inx_list.append(D + i + lenth)
    for i in range(D + E):
        if i not in environment_inx_list:
            system_inx_list.append(i)

    Mbuilder = InitM(
        D=D,
        K=K,
        B=B,
        seed=seed,
        target_score=target_score,
        dtype=Dtype,
        device=device,
        transform=transform,
    )

    if args.read_coretensor:
        obj = torch.load(args.read_coretensor, map_location=device)
        core_tensor = [obj[str(i)] for i in range(len(obj))]
        print(f"Core tensor loaded from {args.read_coretensor}")
    else:
        core_tensor = init_core_tensor(D, E, K, Nk, Dtype, is_identity=args.is_identity, device=device)

    if args.is_savem:
        for _ in range(Nb):
            M_data = Mbuilder.get_mini_batch(_)
            torch.save({"phi_x": M_data.phi_x, "dphi_x": M_data.dphi_x, "sp_x": M_data.sp_x}, f"M_{_}.pt")

    if not args.is_minibatch:
        print(f"Not using mini-batch training, total samples B={B}")

    if not args.not_train:
        opt = stiefel_optimizer_complex.SGDG(
            [{"params": core_tensor, "lr": args.lr, "momentum": args.momentum, "stiefel": True}]
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=50, threshold=1e-4,
            threshold_mode="rel", cooldown=20, min_lr=1e-8,
        )
        for step in range(args.Num_Step):
            opt.zero_grad()
            loss = accumulate_gradients(
                Mbuilder,
                core_tensor,
                D,
                E,
                K,
                Nk,
                B,
                Dtype,
                state,
                einsum_builder,
                is_train=True,
            )
            if args.is_gradclip:
                torch.nn.utils.clip_grad_norm_(core_tensor, max_norm=1.0)
            opt.step()
            sched.step(loss)
            if step % 20 == 0:
                log(f"Step {step:03d}  loss = {loss:.10f}  lr = {opt.param_groups[0]['lr']:.10f}")
            if args.is_inference and step % 500 == 0:
                inf = Inference(
                    D=D, E=E, K=K, Nk=Nk, dtype=Dtype, Inquire_B=B, state=state,
                    einsum_builder=einsum_builder, core_tensor=core_tensor,
                    optimizer="greedy", device=device, transform=transform,
                )
                plot_theoretical_marginal_comparison_style(
                    inf,
                    region=[-5, 5, -5, 5],
                    sampled_data=None,
                    name=f"{args.target}_D{D}_E{E}_Step{step}",
                    gridsize=60,
                    n_grid=100,
                    levels=10,
                    scatter_n=1000,
                )

    inference_engine = Inference(
        D=D,
        E=E,
        K=K,
        Nk=Nk,
        dtype=Dtype,
        Inquire_B=B,
        state=state,
        einsum_builder=einsum_builder,
        core_tensor=core_tensor,
        optimizer="greedy",
        device=device,
        transform=transform,
    )
    samples_quanvi = inference_engine.sample(
        n_samples=args.plot_n_samples,
        n_grid=args.sample_n_grid,
        x_range=(-5, 5),
        chunk_size=args.sample_chunk_size,
    )

    fisher_space = args.fisher_space
    if fisher_space == "auto":
        fisher_space = "standardized" if transform is not None else "unconstrained"

    def score_q(x_np_2d):
        t = torch.as_tensor(x_np_2d, dtype=torch.float64, device=device)
        return inference_engine.score(t)

    fisher_kw = dict(
        posterior=posterior,
        bs_model=bs_model,
        score_q=score_q,
        seed=seed,
        transform=transform,
        fisher_space=fisher_space,
    )
    if args.fisher_max_draws is not None:
        fisher_kw["max_draws_per_chain"] = args.fisher_max_draws
    f_mean, f_std = fisher_div(**fisher_kw)
    print(f"Fisher divergence ({fisher_space} space): {f_mean} ± {f_std}")

    ref_samples, variable_names = posterior_reference_samples_unconstrained(
        posterior, bs_model, max_draws=5000,
    )
    if D > 5:
        plt.figure(figsize=(16, 8))
    else:
        plt.figure(figsize=(16, 4))
    quanvi_np = samples_quanvi.detach().cpu().numpy() if isinstance(samples_quanvi, torch.Tensor) else np.asarray(samples_quanvi)
    for i in range(D):
        if D > 5:
            plt.subplot(2, (D + 1) // 2, i + 1)
        else:
            plt.subplot(1, D, i + 1)
        plt.hist(ref_samples[:, i], bins=50, density=True, alpha=0.5, label="Reference")
        plt.hist(quanvi_np[:, i], bins=50, density=True, alpha=0.5, label="Approximation")
        plt.title(f"Marginal x[{i}]")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"marginals_QuanVI_{args.target}.pdf")
    plt.close()

    plot_lower_triangle_joint_distribution(
        ref_samples,
        quanvi_np,
        save_path=f"joint_QuanVI_{args.target}.pdf",
        variable_names=variable_names,
        title=f"QuanVI vs reference samples ({args.target})",
        learned_label="QuanVI",
    )


    if not args.read_coretensor:
        torch.save(
            {str(i): core_tensor[i] for i in range(len(core_tensor))},
            f"{args.target}_D{D}_E{E}_K{K}_QuanVI_core_tensor.pt",
        )


if __name__ == "__main__":
    t0 = datetime.now()
    main()
    print(f"Total elapsed: {datetime.now() - t0}")
