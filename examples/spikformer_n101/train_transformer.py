import os

# Avoid large inactive CUDA segments in the direct VOPRED CuPy path.
# An explicit user setting still takes precedence.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Multi-GPU sweeps otherwise let every process create a full CPU thread pool.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import datetime
import json
import time
import random
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from torch.cuda import amp
import model as spikformer_models
from ablation_io import rng_state, restore_rng, load_model
from ablation_models import effective_masses
import utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy
import autoaugment
from torch.nn import functional as F
from spikingjelly.datasets.n_caltech101 import NCaltech101 
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from hardvs import HARDVS
from dvs_lip import DVSLip
from torch.utils.data import random_split         
# Keep this reference so local timm model registration side effects are preserved.
_ = spikformer_models


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_model_head_num_classes(model):
    linear_out_features = [
        module.out_features for module in model.modules()
        if isinstance(module, nn.Linear)
    ]
    if linear_out_features:
        return linear_out_features[-1]
    return getattr(model, 'num_classes', None)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='spikformer', help='model')
    parser.add_argument('--dataset', default='ncaltech101',
                        choices=['ncaltech101', 'dvs128gesture', 'hardvs', 'dvslip', 'dvslips'],
                        help='dataset')
    parser.add_argument('--num-classes', type=int, default=101, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--data-path', default='data/ncaltech101', help='dataset root')

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        '--finetune', default='',
        help='load model weights from a checkpoint while resetting optimizer, scheduler, epoch, and best accuracy')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        # default=True,
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True,  action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    parser.add_argument('--embed-dims', default=256, type=int,
                        help='Spikformer embedding dimension')
    parser.add_argument('--depths', default=2, type=int,
                        help='Spikformer transformer block depth')
    # parser.add_argument('--adam', default=True, action='store_true',
    #                     help='Use Adam')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    #Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--integrator', type=str, default='fdeint',
                        choices=['odeint_adjoint', 'odeint', 'fdeint_adjoint', 'fdeint'],
                        help='differential equation integrator type (default: odeint_adjoint)')

    parser.add_argument('--alpha', type=float, nargs='+', default=None,
                        help='Fractional order(s). Single value for single-term FDE, '
                             'multiple values for multi-term FDE. '
                             'Examples: --alpha 0.5 (single) or --alpha 0.3 0.7 0.9 (multi)')
    parser.add_argument('--multi_coefficient', type=float, nargs='+', default=None,
                        help='Coefficients for multi-term FDE. Required when --alpha has multiple values. '
                             'Example: --multi_coefficient 1.0 0.5 0.2')
    parser.add_argument('--learn_coefficient', action='store_true',
                        help='Make multi-term coefficients learnable')
    parser.add_argument('--coefficient-constraint', default='none',
                        choices=['none', 'softplus', 'softmax'],
                        help='Constraint applied to learnable distributed-order coefficients')
    parser.add_argument('--coefficient-update-interval', type=int, default=1,
                        help='Update shared distributed-order coefficients every N mini-batches')
    parser.add_argument('--coefficient-lr', type=float, default=None,
                        help='Independent coefficient learning rate; default uses the network group')
    parser.add_argument('--coefficient-weight-decay', type=float, default=0.0,
                        help='Weight decay for the independent coefficient optimizer group')
    parser.add_argument('--glmulti-history-grad', default='truncated',
                        choices=['truncated', 'full'],
                        help='Backpropagate through all GL voltage-history edges or truncate them')
    parser.add_argument('--history-grad-mode', default='approx',
                        choices=['approx', 'full'],
                        help='VOGL1/FAGL1 temporal-history backward mode')
    parser.add_argument('--learn-alpha', action='store_true',
                        help='Make the fractional order learnable')
    parser.add_argument('--alpha-mode', default='global', choices=['global', 'per_state', 'per_time'],
                        help='Fractional-order sharing mode')
    parser.add_argument('--alpha-constraint', default='none', choices=['none', 'softmax01'],
                        help='Constraint for learnable per-time alpha')
    parser.add_argument('--beta-mode', default='shared_per_time',
                        choices=['fixed_per_time', 'shared_scalar', 'grid', 'time_mlp', 'input_time',
                                 'shared_per_time', 'residual_grid4', 'residual_grid16'],
                        help='Sharing mode for direct variable-order LIFNeuronFDE')
    parser.add_argument('--beta-init', type=float, default=0.5,
                        help='Initial shared beta value for direct VOPRED')
    parser.add_argument('--beta-lr', type=float, default=1e-4,
                        help='Learning rate for the shared beta schedule')
    parser.add_argument('--beta-min', type=float, default=0.02)
    parser.add_argument('--beta-max', type=float, default=0.30)
    parser.add_argument('--beta-anchor', type=float, default=0.10)
    parser.add_argument('--beta-residual-scale', type=float, default=0.05,
                        help='Maximum tanh residual around beta-anchor for residual beta modes')
    parser.add_argument('--beta-residual-init', default='zero',
                        choices=['zero', 'increasing', 'decreasing',
                                 'middle_high', 'middle_low', 'zigzag'])
    parser.add_argument('--beta-anchor-weight', type=float, default=0.0)
    parser.add_argument('--beta-smooth-weight', type=float, default=0.0)
    parser.add_argument('--beta-freeze-epochs', type=int, default=0)
    parser.add_argument('--schedule-policy', default='legacy',
                        choices=['legacy', 'proportional', 'stabilized', 'fixed_beta_budget'],
                        help='Automatically scale warmup, beta learning, and mixup milestones')
    parser.add_argument('--reference-epochs', type=int, default=100)
    parser.add_argument('--beta-final-freeze-ratio', type=float, default=0.20)
    parser.add_argument('--beta-train-budget', type=int, default=95)
    parser.add_argument('--vo-solver', choices=['abm', 'l1', 'abm_pc'], default='abm')
    parser.add_argument('--vo-reset-in-drift', action='store_true')
    parser.add_argument('--vo-residual-mix', type=float, default=0.0)

    parser.add_argument('--method', type=str, default='pred', choices=[
        'euler',  # odeint only supports euler method
        'gl-f', 'gl-o', 'trap-f', 'trap-o',
        'l1-f', 'l1-o', 'pred-f', 'pred-o',  # fdeint_adjoint methods
        'pred', 'vopred', 'vogl', 'vogl1', 'gl', 'gl1', 'trap', 'l1', 'glmulti'  # fdeint methods
    ], help='method for euler solver')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='Integration step size (default: 1.0)')
    parser.add_argument('--time_interval', type=float, default=0.1,
                        help='Time interval between steps in ms (default: 1.0) for each input event')
    parser.add_argument('--memory', type=int, default=-1,
                        help='if cost much gpu memory, set save_memory to small positive int, '
                             'default: -1, which means full memory')
    parser.add_argument('--glmulti_chunk', type=int, default=96,
                        help='Chunk size for CuPy glmulti history convolution (default: 96)')
    parser.add_argument('--glmulti_detach_conv', type=int, default=1, choices=[0, 1],
                        help='Detach conv history for CuPy glmulti to reduce memory (default: 1)')
    parser.add_argument('--cupy', action='store_true',
                        help='use CuPy backend for fractional ODE solver (requires CUDA + CuPy)')
    parser.add_argument('--seed', default=2021, type=int,
                        help='global random seed for full reproducibility')
    parser.add_argument('--split-seed', default=2021, type=int,
                        help='fixed N-Caltech101 train/test split seed (default: 2021)')
    
    args = parser.parse_args()
    return args


def build_datasets(args):
    dataset_name = args.dataset.lower()
    if dataset_name == 'ncaltech101':
        data_set = NCaltech101(
            root=args.data_path,
            data_type='frame',
            frames_number=args.T,
            split_by='number',
        )
        train_ratio = 0.8
        train_size = int(len(data_set) * train_ratio)
        test_size = len(data_set) - train_size
        dataset_train, dataset_test = random_split(
            data_set,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(args.split_seed),
        )
    elif dataset_name == 'dvs128gesture':
        dataset_train = DVS128Gesture(
            root=args.data_path,
            train=True,
            data_type='frame',
            frames_number=args.T,
            split_by='number',
        )
        dataset_test = DVS128Gesture(
            root=args.data_path,
            train=False,
            data_type='frame',
            frames_number=args.T,
            split_by='number',
        )
    elif dataset_name == 'hardvs':
        dataset_train = HARDVS(
            root=args.data_path,
            train_test_val='train',
            data_type='frame',
            frames_number=args.T,
            split_by='number',
        )
        dataset_test = HARDVS(
            root=args.data_path,
            train_test_val='test',
            data_type='frame',
            frames_number=args.T,
            split_by='number',
        )
    elif dataset_name in ('dvslip', 'dvslips'):
        dataset_train = DVSLip(
            root=args.data_path,
            train=True,
            data_type='frame',
            frames_number=args.T,
            split_by='number',
        )
        dataset_test = DVSLip(
            root=args.data_path,
            train=False,
            data_type='frame',
            frames_number=args.T,
            split_by='number',
        )
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    return dataset_train, dataset_test


def process_alpha_args(args):
    if getattr(args, "alpha", None) is None:
        if getattr(args, "multi_coefficient", None) is not None:
            print("Warning: --multi_coefficient is ignored for single-term FDE (--alpha).")
        return args.alpha, None

    alpha = args.alpha
    if len(alpha) == 1:
        if args.multi_coefficient is not None:
            print("Warning: --multi_coefficient is ignored for single-term FDE (single alpha value)")
        return alpha[0], None

    multi_coefficient = args.multi_coefficient
    if multi_coefficient is None:
        raise ValueError(
            f"--multi_coefficient is required when --alpha has multiple values. "
            f"Got --alpha {alpha}"
        )
    if len(alpha) != len(multi_coefficient):
        raise ValueError(
            f"--alpha (len={len(alpha)}) and --multi_coefficient "
            f"(len={len(multi_coefficient)}) must have the same length"
        )
    print("[Multi-term FDE] Enabled:")
    print(f"  - Fractional orders (alpha): {alpha}")
    print(f"  - Coefficients: {multi_coefficient}")
    print(f"  - Learnable coefficients: {args.learn_coefficient}")
    return alpha, multi_coefficient

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None, aug=None, trival_aug=None, mixup_fn=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    data_time = torch.linspace(0, args.time_interval * (args.T),
                               args.T+1, device=args.device).float()
    options = {
        'step_size': args.step_size,
        'memory': args.memory,
        'glmulti_chunk': args.glmulti_chunk,
        'glmulti_detach_conv': bool(args.glmulti_detach_conv),
    }

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        image = image.float()  # [N, T, C, H, W]
        N,T,C,H,W = image.shape

        image = image.permute(1, 0, 2, 3, 4).contiguous()

        image = F.interpolate(
                image.reshape(-1, C, H, W),  # 展平到 [(T * N), C, H, W]
                size=(128, 128),  # 目标大小
                mode='bilinear',  # 使用双线性插值
                align_corners=False  # 避免插值边界问题
            )
        image = image.reshape(T, N, C, 128, 128)
        # print(image.shape)
        if aug != None:
            image = torch.stack([(aug(image[i])) for i in range(T)])
        if trival_aug != None:
            image = torch.stack([(trival_aug(image[i])) for i in range(T)])

        if mixup_fn is not None:
            image, target = mixup_fn(image.transpose(0, 1), target)
            image = image.transpose(0, 1).contiguous()
            target_for_compu_acc = target.argmax(dim=-1)

        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                #print(image.shape)
                output = model(image, data_time, output_time=data_time, method=args.method,
                                 options=options)

                loss = criterion(output, target)
        else:
            output = model(image, data_time, output_time=data_time, method=args.method,
                             options=options)
            loss = criterion(output, target)

        model_for_reg = model.module if hasattr(model, 'module') else model
        if hasattr(model_for_reg, 'beta_regularization'):
            loss = loss + model_for_reg.beta_regularization()

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        functional.reset_net(model)
        if mixup_fn is not None:
            acc1, acc5 = utils.accuracy(output, target_for_compu_acc, topk=(1, 5))
        else:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = target.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    data_time = torch.linspace(0, args.time_interval * (args.T),
                               args.T+1, device=args.device).float()
    options = {
        'step_size': args.step_size,
        'memory': args.memory,
        'glmulti_chunk': args.glmulti_chunk,
        'glmulti_detach_conv': bool(args.glmulti_detach_conv),
    }
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # print('(---test---)' * 4, target.shape)

            image = image.float()  # [N, T, C, H, W]
            N, T, C, H, W = image.shape
            image = image.permute(1, 0, 2, 3, 4).contiguous()

            image = F.interpolate(
                image.reshape(-1, C, H, W),  # 展平到 [(T * N), C, H, W]
                size=(128, 128),  # 目标大小
                mode='bilinear',  # 使用双线性插值
                align_corners=False  # 避免插值边界问题
            )
            image = image.reshape(T, N, C, 128, 128)

            output = model(image, data_time, output_time=data_time, method=args.method,
                                 options=options)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = target.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=T, split_by='number')
    dataset_train, dataset_test = split_to_train_test_set(0.9, origin_set, 10)
    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, sampler_train, sampler_val


def apply_epoch_schedule_policy(args):
    """Scale the 100-epoch recipe to the requested training horizon."""
    args.beta_lr_base = float(args.beta_lr)
    args.beta_final_freeze_epoch = -1
    if args.schedule_policy == 'legacy':
        # Preserve the historical behavior that was previously hard-coded below.
        if args.mixup_off_epoch <= 0:
            args.mixup_off_epoch = 75
        return

    epochs = max(1, int(args.epochs))
    reference_epochs = max(1, int(args.reference_epochs))
    ratio = epochs / float(reference_epochs)
    args.warmup_epochs = max(1, round(0.10 * epochs))
    args.beta_freeze_epochs = max(0, round(0.05 * epochs))
    args.mixup_off_epoch = min(epochs, max(1, round(0.75 * epochs)))

    if args.schedule_policy == 'proportional':
        args.beta_lr = args.beta_lr_base / ratio
    elif args.schedule_policy == 'stabilized':
        args.beta_lr = args.beta_lr_base / math.sqrt(ratio)
        train_fraction = 1.0 - float(args.beta_final_freeze_ratio)
        args.beta_final_freeze_epoch = min(
            epochs, max(args.beta_freeze_epochs, round(train_fraction * epochs)))
    elif args.schedule_policy == 'fixed_beta_budget':
        args.beta_final_freeze_epoch = min(
            epochs, args.beta_freeze_epochs + max(0, int(args.beta_train_budget)))


def main(args):

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    rank = utils.get_rank()
    apply_epoch_schedule_policy(args)
    if args.device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
    seed_everything(args.seed + rank)
    print(args)

    if isinstance(alpha, (list, tuple)):
        alpha_tag = "-".join(f"{a:.3f}" for a in alpha)
    else:
        alpha_tag = f"{alpha:.3f}"
    output_dir = os.path.join(
        args.output_dir,
        f'{args.model}_b{args.batch_size}_T{args.T}_e{args.embed_dims}_d{args.depths}_method{args.method}_alpha{alpha_tag}_constraint{args.alpha_constraint}_step{args.step_size}'
    )

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    if args.opt == 'adamw':
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if args.cupy:
        output_dir += '_cupy'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)
    with open(os.path.join(output_dir, 'args.json'), 'w', encoding='utf-8') as args_json:
        json.dump(vars(args), args_json, indent=2, sort_keys=True)

    device = torch.device(args.device)
    
    dataset_train, dataset_test = build_datasets(args)
    assert len(dataset_test) == 1742 and args.batch_size == 32
    assert args.split_seed == 42
    print('ABLATION_TEST full=1742 evaluated=1742 drop_last=False split_seed=42', flush=True)


    sampler_train = None
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train, seed=args.seed)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False, seed=args.seed)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_test)
    train_loader_gen = torch.Generator().manual_seed(args.seed + rank)
    test_loader_gen = torch.Generator().manual_seed(args.seed + rank + 1)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=train_loader_gen,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        sampler=sampler_val,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=test_loader_gen,
    )

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
        integrator=args.integrator,
        alpha=alpha,
        multi_coefficient=multi_coefficient,
        learn_coefficient=args.learn_coefficient,
        coefficient_constraint=args.coefficient_constraint,
        coefficient_update_interval=args.coefficient_update_interval,
        glmulti_history_grad=args.glmulti_history_grad,
        history_grad_mode=args.history_grad_mode,
        learn_alpha=args.learn_alpha,
        alpha_mode=args.alpha_mode,
        alpha_constraint=args.alpha_constraint,
        method=getattr(args, 'method'),
        step_size=getattr(args, 'step_size'),
        time_interval=getattr(args, 'time_interval'),
        memory=getattr(args, 'memory'),
        connect_f=args.connect_f,
        embed_dims=args.embed_dims,
        depths=args.depths,
        device=args.device,
        batch_size=args.batch_size,
        T=args.T,
        backend='cupy' if args.cupy else 'torch',
        beta_mode=args.beta_mode,
        beta_init=args.beta_init,
        beta_lr=args.beta_lr,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_anchor=args.beta_anchor,
        beta_residual_scale=args.beta_residual_scale,
        beta_residual_init=args.beta_residual_init,
        beta_anchor_weight=args.beta_anchor_weight,
        beta_smooth_weight=args.beta_smooth_weight,
        vo_solver=args.vo_solver,
        vo_reset_in_drift=args.vo_reset_in_drift,
        vo_residual_mix=args.vo_residual_mix,
    )
    if args.cupy:
        if device.type != 'cuda':
            raise RuntimeError("CuPy backend requires CUDA device.")
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            raise ImportError("CuPy backend requested but CuPy is not installed.") from exc
        model.fde_backend = 'cupy'
        print(f"CuPy backend enabled (version {cupy.__version__}).")
    else:
        model.fde_backend = 'torch'
    print("Creating model")
    print(model)  # 添加这行来打印模型结构
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    model_num_classes = get_model_head_num_classes(model)
    print(f"requested num_classes: {args.num_classes}, model head num_classes: {model_num_classes}")
    if model_num_classes is not None and model_num_classes != args.num_classes:
        raise ValueError(
            f"Model head num_classes ({model_num_classes}) does not match "
            f"--num-classes ({args.num_classes})."
        )
    model.to(device)
    if args.finetune:
        try:
            finetune_checkpoint = torch.load(
                args.finetune, map_location='cpu', weights_only=False)
        except TypeError:
            finetune_checkpoint = torch.load(args.finetune, map_location='cpu')
        finetune_state = finetune_checkpoint.get('model', finetune_checkpoint)
        model.load_state_dict(finetune_state, strict=True)
        print(
            f"Loaded finetune weights from {args.finetune}; "
            "optimizer, scheduler, epoch, and best accuracy are reset."
        )
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    uses_soft_targets = (
        args.mixup > 0 or args.cutmix > 0
        or args.cutmix_minmax is not None
    )
    criterion_train = (
        SoftTargetCrossEntropy() if uses_soft_targets
        else nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    beta_params = []
    shared_beta = getattr(model, 'beta_raw', None)
    if isinstance(shared_beta, nn.Parameter):
        beta_params.append(shared_beta)
    beta_net = getattr(model, 'beta_net', None)
    if isinstance(beta_net, nn.Module):
        beta_params.extend(list(beta_net.parameters()))
    if beta_params:
        beta_ids = {id(p) for p in beta_params}
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if id(p) not in beta_ids]
        optimizer.add_param_group({
            'params': beta_params,
            'lr': args.beta_lr,
            'weight_decay': 0.0,
            'shared_beta': True,
        })
        print(
            f"beta optimizer group: lr={args.beta_lr}, weight_decay=0, "
            f"parameters={sum(p.numel() for p in beta_params)}"
        )
    coefficient_params = []
    coefficient_param_ids = set()
    for name, parameter in model.named_parameters():
        if 'multi_coefficient' in name and id(parameter) not in coefficient_param_ids:
            coefficient_params.append(parameter)
            coefficient_param_ids.add(id(parameter))
    if coefficient_params and args.coefficient_lr is not None:
        for group in optimizer.param_groups:
            group['params'] = [
                parameter for parameter in group['params']
                if id(parameter) not in coefficient_param_ids
            ]
        optimizer.add_param_group({
            'params': coefficient_params,
            'lr': args.coefficient_lr,
            'weight_decay': args.coefficient_weight_decay,
            'shared_coefficient': True,
        })
        print(
            f"coefficient optimizer group: lr={args.coefficient_lr}, "
            f"weight_decay={args.coefficient_weight_decay}, "
            f"parameters={sum(p.numel() for p in coefficient_params)}"
        )
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    assert num_epochs == args.epochs == 1000, num_epochs
    print(num_epochs)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        # 修改后
        try:
            # 尝试使用 weights_only=False 加载
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        except:
            # 如果失败，回退到旧方法
            checkpoint = torch.load(args.resume, map_location='cpu')

        load_model(model_without_ddp, checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'], train_loader_gen, test_loader_gen)
        print('ABLATION_RESUME next_epoch=', args.start_epoch + 1, flush=True)

    if args.test_only:

        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        tb_root = output_dir + '_logs'
        utils.mkdir(tb_root)
        utils.mkdir(os.path.join(tb_root, 'train'))
        utils.mkdir(os.path.join(tb_root, 'te'))
        train_tb_writer = SummaryWriter(os.path.join(tb_root, 'train'), purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(os.path.join(tb_root, 'te'), purge_step=purge_step_te)
        with open(os.path.join(tb_root, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')


    train_snn_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5)
                    ])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, num_epochs):
        save_max = False
        beta_trainable = (
            epoch >= args.beta_freeze_epochs
            and (args.beta_final_freeze_epoch < 0
                 or epoch < args.beta_final_freeze_epoch)
        )
        for name, parameter in model_without_ddp.named_parameters():
            if name == 'beta_raw' or name.startswith('beta_net.'):
                parameter.requires_grad_(beta_trainable)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if (mixup_fn is not None and args.mixup_off_epoch > 0
                and epoch >= args.mixup_off_epoch):
            mixup_fn.mixup_enabled = False
        train_start_time = time.time()
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion_train, optimizer, data_loader, device, epoch,
            args.print_freq, scaler, args.T_train,
            train_snn_aug, train_trivalaug, mixup_fn)
        train_time = time.time() - train_start_time
        train_samples = len(data_loader) * args.batch_size
        train_speed = train_samples / train_time if train_time > 0 else 0.0
        print(f"Epoch {epoch}: train_acc1={train_acc1:.4f}, train_acc5={train_acc5:.4f}")
        if utils.is_main_process() and train_tb_writer is not None:
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step(epoch + 1)

        test_start_time = time.time()
        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        test_time = time.time() - test_start_time
        test_samples = len(dataset_test)
        test_speed = test_samples / test_time if test_time > 0 else 0.0
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)

        if utils.is_main_process():
            multi_coefficient_values = None
            multi_coefficient_grad_norm = None
            multi_coeff_str = None
            direct_lif = getattr(model_without_ddp, 'lif1', None)
            if direct_lif is None or not hasattr(direct_lif, 'get_multi_coefficient'):
                direct_lif = next(
                    (
                        module for module in model_without_ddp.modules()
                        if hasattr(module, 'get_multi_coefficient')
                    ),
                    None,
                )
            if direct_lif is not None and hasattr(direct_lif, 'get_multi_coefficient'):
                effective_coefficient = direct_lif.get_multi_coefficient()
                if effective_coefficient is not None:
                    multi_coefficient_values = effective_coefficient.detach().flatten().cpu().tolist()
                    coeff_list = ", ".join(f"{c:.6f}" for c in multi_coefficient_values)
                    multi_coeff_str = f"multi_coefficient: {coeff_list}"
                    raw_coefficient = getattr(direct_lif, 'multi_coefficient_raw', None)
                    if raw_coefficient is not None and raw_coefficient.grad is not None:
                        multi_coefficient_grad_norm = float(raw_coefficient.grad.detach().norm().cpu())
            if multi_coeff_str is None:
                if hasattr(model_without_ddp, "multi_coefficient") and model_without_ddp.multi_coefficient is not None:
                    coeff_tensor = model_without_ddp.multi_coefficient.detach().flatten().cpu()
                    multi_coefficient_values = coeff_tensor.tolist()
                    if hasattr(model_without_ddp, 'parametrizations'):
                        raw = model_without_ddp.parametrizations.multi_coefficient.original
                        if raw.grad is not None:
                            multi_coefficient_grad_norm = float(raw.grad.detach().norm().cpu())
                    coeff_list = ", ".join(f"{c:.6f}" for c in multi_coefficient_values)
                    multi_coeff_str = f"multi_coefficient: {coeff_list}"
                elif args.multi_coefficient is not None:
                    coeff_list = ", ".join(f"{c:.4f}" for c in args.multi_coefficient)
                    multi_coeff_str = f"multi_coefficient(init): {coeff_list}"
                else:
                    multi_coeff_str = "multi_coefficient: None"
            print(multi_coeff_str)
            if hasattr(model_without_ddp, "get_alpha_value"):
                alpha_values = model_without_ddp.get_alpha_value()
                if isinstance(alpha_values, list):
                    alpha_tensor = torch.tensor(alpha_values)
                    alpha_grad = getattr(model_without_ddp, "alpha_time_raw", None)
                    if alpha_grad is None:
                        alpha_grad = getattr(model_without_ddp, "alpha_time", None)
                    if alpha_grad is None:
                        alpha_grad = getattr(model_without_ddp, "beta_raw", None)
                    grad_norm = None
                    if alpha_grad is not None and alpha_grad.grad is not None:
                        grad_norm = float(alpha_grad.grad.detach().norm().cpu())
                    print(
                        "alpha_time: " + ", ".join(f"{value:.6f}" for value in alpha_values)
                    )
                    print(
                        f"alpha_time_stats: mean={alpha_tensor.mean():.6f}, "
                        f"min={alpha_tensor.min():.6f}, max={alpha_tensor.max():.6f}, "
                        f"grad_norm={grad_norm}"
                    )


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True


        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
                'scaler': scaler.state_dict() if scaler is not None else None,
                'rng': rng_state(train_loader_gen, test_loader_gen),
                'effective_coefficients': effective_masses(model_without_ddp).detach().cpu().tolist(),
                'implementation': args.implementation,
            }

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_best.pth'))

            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint_latest.pth'))
            if epoch + 1 >= args.epochs:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_final.pth'))

            assert multi_coefficient_values is not None
            assert all(value > 0 and math.isfinite(value) for value in multi_coefficient_values)
            if not args.learn_coefficient:
                assert np.allclose(multi_coefficient_values, args.multi_coefficient, atol=1e-7)
            metrics_path = os.path.join(output_dir, 'metrics.jsonl')
            alpha_values = model_without_ddp.get_alpha_value() if hasattr(model_without_ddp, 'get_alpha_value') else None
            beta_grad_norm = None
            beta_grads = [p.grad.detach().flatten() for name, p in model_without_ddp.named_parameters()
                          if (name == 'beta_raw' or name.startswith('beta_net.')) and p.grad is not None]
            if beta_grads:
                beta_grad_norm = float(torch.cat(beta_grads).norm().cpu())
            peak_allocated_mib = None
            peak_reserved_mib = None
            cupy_pool_mib = None
            if device.type == 'cuda':
                peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                if args.cupy:
                    import cupy
                    cupy_pool_mib = cupy.get_default_memory_pool().total_bytes() / (1024 ** 2)
            with open(metrics_path, 'a', encoding='utf-8') as metrics_file:
                metrics_file.write(json.dumps({
                    'epoch': epoch,
                    'epoch_1based': epoch + 1,
                    'implementation': args.implementation,
                    'test_samples': test_samples,
                    'train_loss': train_loss,
                    'train_acc1': train_acc1,
                    'test_acc1': test_acc1,
                    'test_acc5': test_acc5,
                    'max_test_acc1': max_test_acc1,
                    'alpha_time': alpha_values if isinstance(alpha_values, list) else None,
                    'beta_grad_norm': beta_grad_norm,
                    'multi_coefficient': multi_coefficient_values,
                    'multi_coefficient_grad_norm': multi_coefficient_grad_norm,
                    'train_images_per_second': train_speed,
                    'test_images_per_second': test_speed,
                    'train_epoch_seconds': train_time,
                    'test_epoch_seconds': test_time,
                    'peak_allocated_mib': peak_allocated_mib,
                    'peak_reserved_mib': peak_reserved_mib,
                    'cupy_pool_mib': cupy_pool_mib,
                }) + '\n')
                
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print(f'avg train speed = {train_speed:.4f} images/s, avg test speed = {test_speed:.4f} images/s')
        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        print(output_dir)
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, 'checkpoint_final.pth'))

    return max_test_acc1

if __name__ == "__main__":
    args = parse_args()
    alpha, multi_coefficient = process_alpha_args(args)
    if isinstance(alpha, (list, tuple)) and args.method != "glmulti":
        raise ValueError(
            f"Multi-term alpha requires method 'glmulti', got '{args.method}'. "
            "Please set --method glmulti."
        )
    args.alpha = alpha
    if args.method == 'vopred':
        if args.integrator != 'fdeint':
            raise ValueError("method='vopred' requires --integrator fdeint")
        if args.alpha_mode != 'per_time':
            raise ValueError("method='vopred' requires --alpha-mode per_time")
        if not args.learn_alpha:
            raise ValueError("method='vopred' training requires --learn-alpha")
        if isinstance(alpha, (list, tuple)):
            raise ValueError("method='vopred' requires one scalar --alpha initialization")
    main(args)
