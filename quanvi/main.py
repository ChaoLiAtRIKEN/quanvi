import torch
from utils import stiefel_optimizer_complex
from einsum_string_builder import EinsumStringBuilder
from utils.utils import set_all_seeds, kl_divergence, set_target_score
from datetime import datetime
import random
from tensor_factory import init_core_tensor, InitM
from loss_calculator import LossCalculator
from inference_visualization import plot_theoretical_marginal_comparison_style
from inference import Inference
from utils.transforms import make_gaussian_standardizer
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cmath

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument('--D', type=int, default=2, help='Data dimension')
    parser.add_argument('--E', type=int, default=0, help='Number of environment qubits')
    parser.add_argument('--K', type=int, default=5, help='Number of basis functions')
    parser.add_argument('--Nk', type=int, default=2, help='Number of qubits a core tensor acts on')
    parser.add_argument('--B', type=int, default=10000, help='Number of samples')
    parser.add_argument('--Nb', type=int, default=1, help='Number of batches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--Num_Step', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--target', type=str, default='xshape', help='Target shape (e.g., ring, xshape)')
    parser.add_argument('--dependency_lag', type=int, default=5, help='Long-range dependency lag for *_long_range targets')
    parser.add_argument('--long_range_strength', type=float, default=0.0, help='Long-range coupling strength in [0, 1] for *_long_range targets')
    parser.add_argument('--dtype', type=str, default='cfloat', choices=['cfloat', 'cdouble', 'float', 'double'], help='Data type: cfloat, cdouble, float, or double')
    parser.add_argument('--is_identity', action='store_true', default=False, help='Whether the core tensor is identity')
    parser.add_argument('--is_inference', action='store_true', default=False, help='Run inference code')
    parser.add_argument('--is_random_seed', action='store_true', default=False, dest='is_random_seed', help='Enable random seed (set is_random_seed to True)')
    parser.add_argument('--is_gradclip', action='store_true', default=False, help='Enable gradient clipping')
    # Pass multiple values with --environment_inx_list, for example: 1 4 5.
    parser.add_argument('--environment_inx_list', type=int, nargs='+', default=[1, 4], help='List of environment indices')
    parser.add_argument('--read_coretensor',type=str, default='', help='Path to read core tensor from')
    parser.add_argument('--is_savem', action='store_true', default=False, help='Save M ')
    parser.add_argument('--is_minibatch', action='store_true', default=False, help='Use mini-batch training')
    parser.add_argument('--kl_B', type=int, default=10000, help='Number of samples for KL divergence computation')
    parser.add_argument('--sample_chunk_size', type=int, default=None, help='Draw AR samples in chunks of this size. Peak memory scales with chunk_size*n_grid instead of n_samples*n_grid. Default None = one shot (unchanged).')
    parser.add_argument('--not_train', action='store_true', default=False, help='Whether to run training loop')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu, cuda, cuda:0, etc.')
    parser.add_argument('--standardize_mean', type=str, default='', help='Path to .npy mean for affine Gaussian standardization')
    parser.add_argument('--standardize_cov', type=str, default='', help='Path to .npy covariance for affine Gaussian standardization')
    parser.add_argument('--standardize_diag', action='store_true', default=False, help='Treat --standardize_cov as diagonal variances')
    return parser.parse_args()
args = parse_args()
D = args.D
E = args.E
K = args.K
Nk = args.Nk
B = args.B
Nb = args.Nb
seed = args.seed
Num_Step = args.Num_Step
lr = args.lr
momentum = args.momentum
target = args.target
is_identity = args.is_identity
is_random_seed = args.is_random_seed
is_inference = args.is_inference
is_gradclip = args.is_gradclip
environment_inx_list = args.environment_inx_list
read_coretensor = args.read_coretensor
is_savem = args.is_savem
is_minibatch = args.is_minibatch
not_train = args.not_train
kl_B = args.kl_B
if kl_B is None:
    kl_B = B

if args.device.startswith('cuda') and not torch.cuda.is_available():
    print(f"[warn] requested device {args.device} but CUDA unavailable, falling back to cpu")
    device = torch.device('cpu')
else:
    device = torch.device(args.device)

if args.dtype == 'cfloat':
    Dtype = torch.cfloat
elif args.dtype == 'cdouble':    
    Dtype = torch.cdouble   
elif args.dtype == 'float':
    Dtype = torch.float32
elif args.dtype == 'double':
    Dtype = torch.float64

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

if is_random_seed:
    seed = random.randint(0, 10000)
torch.manual_seed(seed)
set_all_seeds(seed)

def accumulate_gradients(m_builder, core_tensor, is_train = False):
    # Compute the loss and backpropagate when training.
    context = torch.enable_grad() if is_train else torch.no_grad()
    total_loss =0.0
    with context:
        loss_calculator = LossCalculator(
            D, E, K, Nk, B, Dtype, state, einsum_builder, core_tensor,
            optimizer='greedy', init_m=m_builder,
        )
        loss = loss_calculator.get_loss() # Total loss
        if is_train:
            # Backpropagate immediately to accumulate gradients and free the computation graph
            loss.backward()
        total_loss = total_loss + loss.detach().item()
    return total_loss

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"This run's hyperparameters: {args}")
    target_score, target_distribution = set_target_score(
        target,
        D,
        device=device,
        dependency_lag=args.dependency_lag,
        long_range_strength=args.long_range_strength,
    )
    transform = None
    if args.standardize_mean or args.standardize_cov:
        if not args.standardize_mean or not args.standardize_cov:
            raise ValueError("--standardize_mean and --standardize_cov must be provided together.")
        transform = make_gaussian_standardizer(
            np.load(args.standardize_mean),
            np.load(args.standardize_cov),
            diagonal=args.standardize_diag,
        )
    state = torch.zeros(K, dtype=Dtype, device=device)
    if Dtype in (torch.cfloat, torch.cdouble):
        state[:] = (1.0 + 1j) / cmath.sqrt(2 * K)
    else:
        state[:] = 1.0 / (K ** 0.5)
    
    system_inx_list = []
    lenth = len(environment_inx_list)
    for i in range(E-lenth):
        environment_inx_list.append(D + i + lenth)
    for i in range(D + E):
        if i not in environment_inx_list:
            system_inx_list.append(i)
    last_system = system_inx_list[-1]   #Is used for determining the last system index for merging dimension

    einsum_builder = EinsumStringBuilder()
    Mbuilder = InitM(
        D=D, K=K, B=B, seed=seed, target_score=target_score,
        dtype=Dtype,device=device,
        transform=transform,
    )
    if read_coretensor != '':
        obj = torch.load(read_coretensor, map_location=device)
        core_tensor = [obj[str(i)] for i in range(len(obj))]
        print(f"Core tensor loaded from {read_coretensor}")
    else:
        core_tensor = init_core_tensor(D,E,K,Nk, Dtype, is_identity=is_identity, device=device)
    if is_savem:
        for _ in range(Nb):
            M_data = Mbuilder.get_mini_batch(_)
            torch.save({
                "phi_x":  M_data.phi_x,
                "dphi_x": M_data.dphi_x,
                "sp_x":   M_data.sp_x
            }, f"M_{_}.pt")
            
    if not is_minibatch:
        print(f"Not using mini-batch training, total samples B={B}")

    if not not_train:
        dict_sf = {'params':core_tensor,'lr':lr,'momentum':momentum,'stiefel':True}
        optimizer = stiefel_optimizer_complex.SGDG([dict_sf])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',          # Minimize the loss.
            factor=0.5,          # Halve the learning rate when triggered.
            patience=50,         # Wait 50 steps without improvement.
            threshold=1e-4,      # Relative improvement threshold.
            threshold_mode='rel',
            cooldown=20,         # Wait 20 steps before checking again.
            min_lr=1e-8,         # Keep the learning rate above zero.
        )        

        for step in range(Num_Step):
            optimizer.zero_grad()
            loss = accumulate_gradients(Mbuilder, core_tensor, is_train=True)
            if is_gradclip:
                torch.nn.utils.clip_grad_norm_(core_tensor, max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)
            cur_lr = optimizer.param_groups[0]['lr']
            if step % 20 == 0:
                inference_engine = Inference(D=D, E=E, K=K, Nk=Nk, dtype=Dtype, Inquire_B=B, state=state, einsum_builder=einsum_builder, core_tensor=core_tensor, optimizer="greedy",device=device,transform=transform)
                kl = kl_divergence(target_distribution, inference_engine.as_dtype(torch.cdouble), num_samples=2000) #lightweight kl
                log(f"Step {step:03d}  loss = {loss:.10f}  lr = {cur_lr:.10f}  kl = {kl:.6f}")

            if is_inference and step % 500 == 0: # Visualization inference every x steps
                inference_engine = Inference(D=D, E=E, K=K, Nk=Nk, dtype=Dtype, Inquire_B=B, state=state, einsum_builder=einsum_builder, core_tensor=core_tensor, optimizer="greedy",device=device,transform=transform)
                visualization_region = [-5, 5, -5, 5]
                kl = kl_divergence(target_distribution, inference_engine.as_dtype(torch.cdouble), num_samples=kl_B)
                print(f"KL divergence at step {step}: {kl:.6f}")
                plot_theoretical_marginal_comparison_style(inference_engine,region=visualization_region,sampled_data=None,name = f"{target}_D{D}_E{E}_Step{step}_kl{kl:.6f}",gridsize=60,n_grid=100,levels=10,scatter_n=1000,)

    # Final inference after training
    inference_engine = Inference(D=D, E=E, K=K, Nk=Nk, dtype=Dtype, Inquire_B=B, state=state, einsum_builder=einsum_builder, core_tensor=core_tensor, optimizer="greedy",device=device,transform=transform)
    sample_result = inference_engine.sample(n_samples=1000, n_grid=100, x_range=(-5, 5), chunk_size=args.sample_chunk_size)

    kl = kl_divergence(target_distribution, inference_engine.as_dtype(torch.cdouble), num_samples=kl_B)
    print(f"Final KL divergence: {kl:.6f}")

    # Final visualization for first two dimensions
    visualization_region = [-5, 5, -5, 5]
    plot_theoretical_marginal_comparison_style(inference_engine,region=visualization_region,sampled_data=sample_result,name = f"{target}_D{D}_E{E},kl{kl:.6f}",gridsize=60,n_grid=100,levels=10,scatter_n=1000)
    def plot_contour(dist, region, name):
        x = np.linspace(region[0], region[1], 100)
        y = np.linspace(region[2], region[3], 100)
        X, Y = np.meshgrid(x, y)
        grid_points = np.stack([X, Y], axis=-1)  # shape: (100, 100, 2)
        grid_points = torch.tensor(grid_points, dtype=torch.float64, device=device)

        log_probs = dist.log_prob(grid_points)  # shape: (100, 100)
        try:
            samples = dist.sample(1000, quadrature_interval=0.01)
        except:
            samples = dist.sample([1000])

        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        pdf = np.exp(log_probs)

        # Plot the contour
        plt.figure(figsize=(4, 4))
        plt.contourf(X, Y, pdf, levels=8, cmap='RdBu_r')
        plt.scatter(samples[:, 0], samples[:, 1], color='k', s=5, alpha=0.3)
        plt.xlim(region[0], region[1])
        plt.ylim(region[2], region[3])
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f"contour_{name}.pdf")
        plt.close()

    if D == 2:
        plot_contour(target_distribution, region=[-4, 4, -4, 4], name=f'Target_{args.target}_D{args.D}')
        pass
    else:
        xz_true = target_distribution.sample([5000])
        # NOTE: n_grid defaults to 1000 here, so this draws 5000*1000 rows at once.
        xz = inference_engine.sample(5000, chunk_size=args.sample_chunk_size)
        if isinstance(xz_true, torch.Tensor):
            xz_true = xz_true.detach().cpu().numpy()
        if isinstance(xz, torch.Tensor):
            xz = xz.detach().cpu().numpy()
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 3, 1)
        plt.scatter(xz_true[:, 0], xz_true[:, 1], color='r', s=5, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.title('True: x0 vs x1')

        plt.subplot(2, 3, 2)
        plt.scatter(xz_true[:, 1], xz_true[:, 2], color='r', s=5, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.title('True: x1 vs x2')

        plt.subplot(2, 3, 3)
        plt.scatter(xz_true[:, 2], xz_true[:, 3], color='r', s=5, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.title('True: x2 vs x3')

        plt.subplot(2, 3, 4)
        plt.scatter(xz[:, 0], xz[:, 1], color='k', s=5, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.title('x0 vs x1')

        plt.subplot(2, 3, 5)
        plt.scatter(xz[:, 1], xz[:, 2], color='k', s=5, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.title(f'x1 vs x2')

        plt.subplot(2, 3, 6)
        plt.scatter(xz[:, 2], xz[:, 3], color='k', s=5, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.title(f'x2 vs x3')

        plt.tight_layout()
        plt.savefig(f"scatter_QuanVI_{args.target}_D{args.D}_{args.K}.pdf")
        plt.close()
    if read_coretensor == '':
        torch.save(core_tensor.state_dict(), f"{target}_D{D}_E{E}_K{K}_Nk{Nk}_QuanVI_core_tensor.pt")
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Total elapsed: {elapsed}")


