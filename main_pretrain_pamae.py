# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# SatMAE: https://github.com/sustainlab-group/SatMAE
# --------------------------------------------------------

"""
PAMAE-ViT pre-training entry point.
 
Fork of SatMAE's main_pretrain.py, modified to:
  1. Use PAMAEDataset instead of SentinelIndividualImageDataset.
  2. Unpack (image, slope, precip) from the dataloader.
  3. Pass only the image tensor to the MAE model (slope & precip will be
     used in Phase 2 when the physics-aware Coherence Loss is added).
  4. Add --data_dir and --ids_file arguments.

Phase 2 update:
Uses the PAMAEViT model with Pedological Coherence
Loss. The training loop passes slope and precipitation tensors to the
model, and logs three separate losses:
  - loss_total:   L_recon + λ * L_coherence
  - loss_recon:   standard MAE reconstruction MSE
  - loss_physics: pedological coherence penalty

"""

import argparse
import datetime
import json
import math
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
 
import timm
# assert timm.__version__ == "0.3.2"  # relaxed for compatibility
import timm.optim.optim_factory as optim_factory
from timm.optim import param_groups_weight_decay
 
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets_pamae import build_pamae_dataset
 
import models_mae
import models_mae_pamae
import models_mae_group_channels
 
import util.lr_sched as lr_sched

# ──────────────────────────────────────────────────────────
#  Training loop (replaces engine_pretrain.train_one_epoch)
#  Key change: unpack (images, slope, precip) instead of
#  (samples, _).
# ──────────────────────────────────────────────────────────


def train_one_epoch_pamae(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_recon', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_physics', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")
 
    for data_iter_step, (images, slope, precip) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # Per-iteration LR schedule
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )
 
        images = images.to(device, non_blocking=True)
        
    
        slope  = slope.to(device, non_blocking=True)
        precip = precip.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss_total, loss_recon,loss_physics,_, _ = model(
                images, slope=slope, precip=precip, mask_ratio=args.mask_ratio)
 
        loss_value = loss_total.item()
        loss_recon_value = loss_recon.item()
        loss_physics_value = loss_physics.item()


        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(f"loss_recon: {loss_recon_value}, loss_physics: {loss_physics_value}")
            raise ValueError(f"Loss is {loss_value}, stopping training")
 
        loss_total /= accum_iter
        loss_scaler(
            loss_total,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
 
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_recon=loss_recon_value)
        metric_logger.update(loss_physics=loss_physics_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
 
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_recon_reduce = misc.all_reduce_mean(loss_recon_value)
        loss_physics_reduce = misc.all_reduce_mean(loss_physics_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            log_writer.add_scalar("train_loss_total", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("train_loss_recon", loss_recon_reduce, epoch_1000x)
            log_writer.add_scalar("train_loss_physics", loss_physics_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
 
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ──────────────────────────────────────────────────────────
#  Argument parser
# ──────────────────────────────────────────────────────────

def get_args_parser():
    parser = argparse.ArgumentParser("PAMAE-ViT pre-training", add_help=False)
 
    # Training hyper-parameters
    parser.add_argument("--batch_size", default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument("--epochs", default=400, type=int,
                        help='Number of training epochs')
    parser.add_argument("--accum_iter", default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
 
    # Model parameters
    parser.add_argument(
        "--model_type", default="vanilla",
        choices=["group_c", "vanilla"], help = "Use channel model"
    )
    parser.add_argument(
       "--model", default="pamae_vit_base_patch16", type=str,
        choices=["pamae_vit_base_patch16", "pamae_vit_large_patch16"],
    )

    # change input size to 96 for PAMAE (instead of 224 for ImageNet)
    parser.add_argument("--input_size", default=96, type=int,
                        help="Patch spatial size (96 for PAMAE)")
    parser.add_argument("--patch_size", default=8, type=int,
                        help="ViT patch size (96/8 = 12×12 = 144 patches)")
    
    parser.add_argument("--mask_ratio", default=0.75, type=float, 
                        help="Masking ratio (percentage of removed patches).")
    parser.add_argument("--spatial_mask", action="store_true", default=False,
                        help='Whether to mask all channels of a spatial location. Only for indp c model')
    
    parser.add_argument("--norm_pix_loss", action="store_true",
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)


    # Physics loss parameters
    parser.add_argument("--lambda_physics", type=float, default=1.0,
                        help="Weight for the physics-based coherence loss term (L_total = L_recon + lambda * L_coherence)")
    
    parser.add_argument("--slope_threshold", type=float, default=0.33,
                        help="Normalized slope above which terrain is steep (~30 deg)")
    parser.add_argument("--precip_threshold", type=float, default=0.3,
                        help="Normalized precip below which climate is dry (~750mm/yr)")


    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help='Weight decay default=0.05')
    
    parser.add_argument("--lr", type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    
    parser.add_argument("--blr", type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    
    parser.add_argument("--min_lr", type=float, default=0.0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                    help='epochs to warmup LR')
 
    # ── PAMAE dataset parameters ──
    parser.add_argument(
        "--data_dir", required=True, type=str,
        help="Root directory of the PAMAE dataset (contains sentinel2/, slope/, etc.)",
    )
    parser.add_argument(
        "--ids_file", default=None, type=str,
        help="Path to text file listing point IDs for pre-training "
             "(defaults to <data_dir>/pretrain.txt)",
    )
    parser.add_argument(
        "--grouped_bands", type=int, nargs="+", action="append",
        default=[],
        help="Bands to group for GroupC MAE variant",
    )
 
    # Output / logging
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default="./output_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true",
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
 
    # Distributed
    parser.add_argument("--world_size", default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument(
        "--local_rank", default=int(os.getenv("LOCAL_RANK", 0)), type=int,
    )
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://",
                        help='url used to set up distributed training')
 
    return parser


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main(args):
    misc.init_distributed_mode(args)

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_pamae_dataset(args, mode="pretrain")
    print(dataset_train)

    #if True:  # distributed path
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    #    print(f"Sampler_train = {sampler_train}")
    #else:
    #    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    #if args.model_type == "group_c":
    #    if len(args.grouped_bands) == 0:
    #        # Default grouping for 11 Sentinel-2 bands:
    #        # Visible [B2,B3,B4], Red-Edge [B5,B6,B7],
    #        # NIR [B8,B8A], SWIR+WV [B9,B11,B12]
    #        args.grouped_bands = [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9, 10]]
    #    print(f"Grouping bands {args.grouped_bands}")#

    #    model = models_mae_group_channels.__dict__[args.model](
    #        img_size=args.input_size,
    #        patch_size=args.patch_size,
    #        in_chans=dataset_train.in_c,
    #        channel_groups=args.grouped_bands,
    #        spatial_mask=args.spatial_mask,
    #        norm_pix_loss=args.norm_pix_loss,
    #    )
    #else:
    #    model = models_mae.__dict__[args.model](
    #        img_size=args.input_size,
    #        patch_size=args.patch_size,
    #        in_chans=dataset_train.in_c,
    #        norm_pix_loss=args.norm_pix_loss,
    #    )
    #else:
    #    model = models_mae.__dict__[args.model](
    #        img_size=args.input_size,

    

    # # ── Build PAMAE model with Coherence Loss ──
    model = models_mae_pamae.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        in_chans=dataset_train.in_c,
        norm_pix_loss=args.norm_pix_loss,
        lambda_physics=args.lambda_physics,
        slope_threshold=args.slope_threshold,
        precip_threshold=args.precip_threshold,
    )

    model.to(device)

    model_without_ddp = model

    print(f"Model = {model_without_ddp}")
    print(f"Physics: lambda={args.lambda_physics}, "
          f"slope_thresh={args.slope_threshold}, "
          f"precip_thresh={args.precip_threshold}")
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()


    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    
    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    param_groups = param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch_pamae(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

