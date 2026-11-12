# QuanVI

QuanVI is a score-based variational inference implementation. It represents the variational distribution with a trainable tensor network, uses the target score ∇x log p(x) to train the model, and provides autoregressive sampling and evaluation tools.

The repository includes experiments on synthetic targets and PosteriorDB models. Synthetic runs report an estimated KL divergence and save plots and a trained core-tensor checkpoint. PosteriorDB runs compare learned marginals and joint distributions with reference samples and report Fisher divergence.

## Requirements

Python 3.10 is recommended. The pinned dependencies are listed in `requirements.txt`.

```bash
conda create -n quanvi python=3.10 -y
conda activate quanvi
python -m pip install -r requirements.txt
```

Runs default to the CPU. To use a GPU, install a PyTorch build compatible with your CUDA setup and pass `--device cuda`.

## Synthetic targets

The built-in targets are `gaussian`, `identity_gaussian`, `xshape`, `gmm3`, `funnel`, and `ring`. For example, run X-shape in 20 dimensions for 1,000 optimization steps:

```bash
mkdir -p runs/xshape_D20_seed42
cd runs/xshape_D20_seed42
python ../../quanvi/main.py \
  --target xshape --D 20 --E 0 --K 5 --Nk 2 --B 10000 \
  --Num_Step 1000 --seed 42 --lr 0.1 --momentum 0.9 --dtype cfloat
```

The program writes outputs to the current working directory, including PDF visualizations and a trained core-tensor checkpoint. Training progress and the final KL estimate are printed to the terminal. Run `python ../../quanvi/main.py --help` to see all options.

## PosteriorDB

PosteriorDB experiments require a local PosteriorDB database and a working BridgeStan/Stan toolchain. By default, QuanVI looks for the database at `data/posteriordb/posterior_database` in the repository. To use a database stored elsewhere, set `POSTERIOR_DB_PATH` to its directory.

```bash
export POSTERIOR_DB_PATH=/path/to/posterior_database
mkdir -p runs/posterior_hmm_seed42
cd runs/posterior_hmm_seed42
python ../../quanvi/main_posterior.py \
  --target hmm --E 0 --K 5 --Nk 2 --B 10000 \
  --Num_Step 1000 --seed 42 --lr 0.1 --momentum 0.9 --dtype cdouble
```

Supported target aliases are `kidscore`, `logearn`, `gpregr`, `garch`, `ark`, `logmesquite`, `school`, `nes`, `diamonds`, `gp_pois`, `mcycle`, `hmm`, `bball0`, `bball1`, `arma`, and `blr`.

A PosteriorDB run writes marginal and joint comparison plots, a core-tensor checkpoint, and the Fisher-divergence summary. Optional Gaussian standardization can be enabled with `--standardize_mean` and `--standardize_cov`; use `--standardize_diag` when the covariance file contains diagonal variances.

## Reproducibility

Set `--seed` to fix the random seed. For comparable results, use the same dependency versions, device, and run parameters. Use `--sample_chunk_size` to limit peak memory during sampling; leaving it unset uses the default one-shot sampling path.

## Citation

If you use QuanVI in your work, please cite:

```bibtex
@article{cong2026quanvi,
  title   = {QuanVI: Score-based Variational Inference via Quantum Maximally Mixed States},
  author  = {Cong, Yuchen and Tao, Zerui and Li, Chao and Sun, Zhe and Zhao, Qibin},
  journal = {arXiv preprint arXiv:2609.39164},
  year    = {2026}
}