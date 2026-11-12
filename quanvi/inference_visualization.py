import numpy as np
import matplotlib.pyplot as plt
import torch
from inference import Inference

def plot_sample(sample1, sample2, gridsize, region, name, cmap="summer"):
    """
    sample1, sample2: 1D arrays
    region: [xmin, xmax, ymin, ymax]
    """
    plt.figure(figsize=(4, 4))
    plt.hexbin(sample1, sample2, gridsize=gridsize, cmap=cmap, mincnt=1, extent=region)
    plt.xlim(region[0], region[1])
    plt.ylim(region[2], region[3])
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"sample_{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_contour_theoretical_marginal(
    inference_engine,
    sampled_data,
    region,
    name,
    n_grid=100,
    levels=8,
    scatter_n=1000,
    cmap="RdBu_r",
):
    """Evaluate and plot the theoretical marginal over dimensions 0 and 1.
Dimensions 2 and above are marginalized out. When samples are provided,
overlay a random subset as a scatter plot.

region: [xmin, xmax, ymin, ymax]"""
    D = inference_engine.D
    if D < 2:
        raise ValueError(f"D must be >= 2, got D={D}")

    # --- grid ---
    x = np.linspace(region[0], region[1], n_grid)
    y = np.linspace(region[2], region[3], n_grid)
    X, Y = np.meshgrid(x, y)

    # mesh_input: (n_grid*n_grid, D), only dim0/dim1 filled
    mesh_input = np.zeros((n_grid * n_grid, D), dtype=np.float64)
    mesh_input[:, 0] = X.reshape(-1)
    mesh_input[:, 1] = Y.reshape(-1)

    marginal_list = list(range(2, D)) if D > 2 else None

    with torch.no_grad():
        log_p = inference_engine.log_prob(mesh_input, marginal_list=marginal_list)
    log_p_np = (
        log_p.detach().cpu().numpy()
        if isinstance(log_p, torch.Tensor)
        else np.asarray(log_p, dtype=np.float64)
    )
    Z = np.exp(np.real(log_p_np)).reshape(n_grid, n_grid)

    # --- scatter samples (subsample for speed/clarity) ---
    if sampled_data is not None and len(sampled_data) > 0:
        n = min(scatter_n, len(sampled_data))
        idx = np.random.choice(len(sampled_data), size=n, replace=False)
        scatter = sampled_data[idx, :2]
    else:
        scatter = None

    # --- plot ---
    plt.figure(figsize=(4, 4))
    plt.contourf(X, Y, Z, levels=levels, cmap=cmap)

    if scatter is not None:
        plt.scatter(scatter[:, 0], scatter[:, 1], color="k", s=5, alpha=0.3)

    plt.xlim(region[0], region[1])
    plt.ylim(region[2], region[3])
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"contour_{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_theoretical_marginal_comparison_style(
    inference_engine,
    region,
    sampled_data = None,
    name="marginal",
    gridsize=60,
    n_grid=100,
    levels=10,
    scatter_n=1000,
):
    """Create a sample-density plot and a theoretical marginal contour plot.

The sample plot is written as sample_{name}.pdf; the contour plot is written
as contour_{name}.pdf."""
    # sample hexbin
    if sampled_data is not None:
        plot_sample(
            sampled_data[:, 0],
            sampled_data[:, 1],
            gridsize=gridsize,
            region=region,
            name=name,
        )

    # theoretical contour
    plot_contour_theoretical_marginal(
        inference_engine=inference_engine,
        sampled_data=sampled_data,
        region=region,
        name=name,
        n_grid=n_grid,
        levels=levels,
        scatter_n=scatter_n,
    )
    