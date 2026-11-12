import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _as_2d_numpy_samples(samples):
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    else:
        samples = np.asarray(samples)

    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise ValueError(
            f"samples must be a 2D array with shape (num_samples, dim), got {samples.shape}."
        )

    finite_rows = np.isfinite(samples).all(axis=1)
    samples = samples[finite_rows]
    if samples.shape[0] == 0:
        raise ValueError("samples does not contain any finite rows.")

    return samples.astype(np.float64, copy=False)


def _subsample_rows(samples, max_points):
    if max_points is None or samples.shape[0] <= max_points:
        return samples

    idx = np.linspace(0, samples.shape[0] - 1, max_points, dtype=int)
    return samples[idx]


def plot_lower_triangle_joint_distribution(
    reference_samples,
    learned_samples,
    save_path=None,
    variable_names=None,
    title=None,
    bins=50,
    max_points=5000,
    reference_color="C0",
    learned_color="C1",
    reference_label="Reference",
    learned_label="Approximation",
    hist_alpha=0.45,
    scatter_alpha=0.35,
    scatter_size=5,
    figsize=None,
    dpi=200,
    close=True,
):
    """
    Plot a lower-triangle corner plot comparing two sets of samples.

    Diagonal panels overlay one-dimensional marginal histograms. Lower-triangle
    off-diagonal panels overlay two-dimensional sample scatter plots for each
    pair of variables.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path

    reference_samples = _as_2d_numpy_samples(reference_samples)
    learned_samples = _as_2d_numpy_samples(learned_samples)

    if reference_samples.shape[1] != learned_samples.shape[1]:
        raise ValueError(
            "reference_samples and learned_samples must have the same dimension, "
            f"got {reference_samples.shape[1]} and {learned_samples.shape[1]}."
        )

    reference_scatter = _subsample_rows(reference_samples, max_points)
    learned_scatter = _subsample_rows(learned_samples, max_points)

    _, dim = reference_samples.shape
    if variable_names is None:
        variable_names = [f"x[{i}]" for i in range(dim)]
    if len(variable_names) != dim:
        raise ValueError(
            f"variable_names must have length {dim}, got {len(variable_names)}."
        )

    ranges = []
    for i in range(dim):
        values = np.concatenate([reference_samples[:, i], learned_samples[:, i]])
        lo = float(np.min(values))
        hi = float(np.max(values))
        if np.isclose(lo, hi):
            pad = 0.5 if np.isclose(lo, 0.0) else 0.05 * abs(lo)
        else:
            pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    if figsize is None:
        panel_size = 1.8 if dim > 6 else 2.2
        figsize = (panel_size * dim, panel_size * dim)

    fig, axes = plt.subplots(dim, dim, figsize=figsize, squeeze=False)
    for row in range(dim):
        for col in range(dim):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
                continue

            if row == col:
                ax.hist(
                    reference_samples[:, col],
                    bins=bins,
                    range=ranges[col],
                    density=True,
                    color=reference_color,
                    alpha=hist_alpha,
                    label=reference_label,
                )
                ax.hist(
                    learned_samples[:, col],
                    bins=bins,
                    range=ranges[col],
                    density=True,
                    color=learned_color,
                    alpha=hist_alpha,
                    label=learned_label,
                )
                ax.set_xlim(ranges[col])
            else:
                ax.scatter(
                    reference_scatter[:, col],
                    reference_scatter[:, row],
                    s=scatter_size,
                    color=reference_color,
                    alpha=scatter_alpha,
                    linewidths=0,
                    rasterized=True,
                )
                ax.scatter(
                    learned_scatter[:, col],
                    learned_scatter[:, row],
                    s=scatter_size,
                    color=learned_color,
                    alpha=scatter_alpha,
                    linewidths=0,
                    rasterized=True,
                )
                ax.set_xlim(ranges[col])
                ax.set_ylim(ranges[row])

            if row == dim - 1:
                ax.set_xlabel(variable_names[col], fontsize=8)
            else:
                ax.set_xticklabels([])

            if col == 0 and row > 0:
                ax.set_ylabel(variable_names[row], fontsize=8)
            else:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=7, length=2)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=reference_color,
            markersize=5,
            label=reference_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=learned_color,
            markersize=5,
            label=learned_label,
        ),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)

    if title is not None:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.parent != Path("."):
            save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)

    return fig, axes

