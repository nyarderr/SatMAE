"""
PAMAE-ViT fine-tuning for SOC regression.

Phase 3 of PAMAE-ViT: Discard the MAE decoder, attach a regression head,
and fine-tune on LUCAS SOC labels using the pre-trained encoder from
checkpoint-100.pth.

Key differences from SatMAE's main_finetune.py:
1. Regression (MSELoss) instead of classification (CrossEntropyLoss)
2. Uses PAMAEDataset in finetune mode → returns (image, slope, precip, soc)
3. Evaluation metrics: R², RMSE, CCC, Physical Violation Rate
4. Layer-wise lr decay applied to encoder blocks
5. Cosine learning rate schedule with warmup
6. No mixup/cutmix/label smoothing (regression task)
7. Global average pooling over patch tokens (excl. CLS) instead of CLS token
   for regression stability (Zhai et al., 2022)
8. Drop path (stochastic depth) added during fine-tuning for regularization
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

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_decay as lrd
import util.lr_sched as lr_sched
from util.datasets_pamae import PAMAEDataset

from models_vit_regression import (
    vit_base_patch8_regression,
    load_pretrained_encoder,
)


# ══════════════════════════════════════════════════════════════
#  Evaluation metrics
# ══════════════════════════════════════════════════════════════

def compute_r2(y_true, y_pred):
    """Coefficient of determination (R²)."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - (ss_res / (ss_tot + 1e-8))


def compute_rmse(y_true, y_pred):
    """Root Mean Square Error."""
    return torch.sqrt(((y_true - y_pred) ** 2).mean())


def compute_ccc(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient.

    Measures agreement between predicted and observed values,
    capturing both precision (correlation) and accuracy (bias).
    CCC = 1 means perfect agreement; CCC = 0 means no agreement.
    """
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var(correction=0)
    var_pred = y_pred.var(correction=0)
    cov = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc


def compute_physical_violation_rate(soc_pred, slope_values, soc_threshold, slope_threshold=0.33):
    """
    Physical Violation Rate: fraction of steep-slope samples
    where the model predicts high SOC.

    Based on CLORPT Relief factor: steep slopes have high erosion,
    so high SOC predictions there are physically implausible.

    Parameters
    ----------
    soc_pred : Tensor (N,)
        Predicted log1p(SOC) values.
    slope_values : Tensor (N,)
        Normalized slope values in [0, 1] (divided by 90°).
    soc_threshold : float
        SOC value above which a prediction is "high SOC"
        (computed as 75th percentile of training SOC distribution).
    slope_threshold : float
        Normalized slope above which terrain is "steep" (default 0.33,
        matching pre-training Coherence Loss threshold).

    Returns
    -------
    violation_rate : float
        Fraction of steep-slope predictions that have high SOC.
    n_violations : int
        Number of violations.
    n_steep : int
        Total number of steep-slope samples.
    """
    steep_mask = slope_values > slope_threshold
    n_steep = steep_mask.sum().item()

    if n_steep == 0:
        return 0.0, 0, 0

    high_soc_mask = soc_pred > soc_threshold
    violations = (steep_mask & high_soc_mask).sum().item()

    return violations / n_steep, violations, n_steep


# ══════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, criterion, data_loader, optimizer, device,
                    epoch, loss_scaler, max_norm=None, log_writer=None, args=None):
    """
    Train for one epoch.

    The data loader yields (image, slope, precip, soc) in finetune mode.
    Only image and soc are used here; slope and precip are loaded but
    not passed to the model (they were used during pre-training).
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (images, slope, precip, soc) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # Per-iteration cosine LR schedule
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        images = images.to(device, non_blocking=True)
        soc = soc.to(device, non_blocking=True)

        # Forward pass
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            soc_pred = model(images)
            loss = criterion(soc_pred, soc)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Backward pass with gradient scaling
        loss_scaler(
            loss, optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        # Logging
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train/loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("train/lr", max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ══════════════════════════════════════════════════════════════
#  Evaluation loop
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(data_loader, model, device, soc_threshold_75):
    """
    Evaluate on the test set.

    Returns a dict with: loss, r2, rmse, ccc, violation_rate.

    Parameters
    ----------
    data_loader : DataLoader
        Test data loader (finetune mode → image, slope, precip, soc).
    model : nn.Module
        The ViTForSOCRegression model.
    device : torch.device
        Device to run evaluation on.
    soc_threshold_75 : float
        75th percentile of log1p(SOC) from the training set,
        used for the Physical Violation Rate metric.
    """
    criterion = torch.nn.MSELoss()
    model.eval()

    all_preds = []
    all_targets = []
    all_slopes = []

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, slope, precip, soc = batch

        images = images.to(device, non_blocking=True)
        soc = soc.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            soc_pred = model(images)
            loss = criterion(soc_pred, soc)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

        # Collect predictions for global metrics
        all_preds.append(soc_pred.cpu())
        all_targets.append(soc.cpu())

        # Slope: extract mean slope per sample from (B, 1, 96, 96) tensor
        # This gives a single scalar per sample for the violation rate
        slope_mean = slope.squeeze(1).mean(dim=(1, 2))  # (B,)
        all_slopes.append(slope_mean)

    metric_logger.synchronize_between_processes()

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_slopes = torch.cat(all_slopes, dim=0)

    # Compute metrics
    r2 = compute_r2(all_targets, all_preds).item()
    rmse = compute_rmse(all_targets, all_preds).item()
    ccc = compute_ccc(all_targets, all_preds).item()
    viol_rate, n_viol, n_steep = compute_physical_violation_rate(
        all_preds, all_slopes, soc_threshold_75
    )

    print(f"* R² = {r2:.4f}  RMSE = {rmse:.4f}  CCC = {ccc:.4f}  "
          f"ViolRate = {viol_rate:.4f} ({n_viol}/{n_steep} steep)")

    results = {
        "loss": metric_logger.loss.global_avg,
        "r2": r2,
        "rmse": rmse,
        "ccc": ccc,
        "violation_rate": viol_rate,
        "n_violations": n_viol,
        "n_steep": n_steep,
    }
    return results


# ══════════════════════════════════════════════════════════════
#  Argument parser
# ══════════════════════════════════════════════════════════════

def get_args_parser():
    parser = argparse.ArgumentParser(
        "PAMAE-ViT fine-tuning for SOC regression", add_help=False
    )

    # Training
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--accum_iter", default=1, type=int,
                        help="Gradient accumulation steps")

    # Model
    parser.add_argument("--input_size", default=96, type=int)
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--drop_path", default=0.2, type=float,
                        help="Stochastic depth rate")

    # Optimizer
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="Gradient clipping norm")
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--lr", default=None, type=float,
                        help="Absolute learning rate")
    parser.add_argument("--blr", default=2e-4, type=float,
                        help="Base learning rate (absolute_lr = blr * batch/256)")
    parser.add_argument("--layer_decay", default=0.75, type=float,
                        help="Layer-wise lr decay factor")
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--warmup_epochs", default=3, type=int)

    # Pre-trained checkpoint
    parser.add_argument("--finetune", default="", type=str,
                        help="Path to pre-trained checkpoint (empty = train from scratch)")

    # Dataset
    parser.add_argument("--data_dir", required=True, type=str,
                        help="Root directory of PAMAE dataset")

    # Output
    parser.add_argument("--output_dir", default="./output_finetune", type=str)
    parser.add_argument("--log_dir", default="", type=str,
                        help="TensorBoard log dir (default: output_dir)")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_every", default=5, type=int)

    # Runtime
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed (kept for compatibility with misc.py)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main(args):
    # ── Setup ────────────────────────────────────────────────
    misc.init_distributed_mode(args)

    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Args: {args}")

    # Device selection: support CUDA, MPS (Apple Silicon), or CPU
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # ── Datasets ─────────────────────────────────────────────
    # Construct PAMAEDataset directly (not via build_pamae_dataset)
    # because finetune needs separate train/test ID files.
    data_dir = args.data_dir
    dataset_train = PAMAEDataset(
        ids_file=os.path.join(data_dir, "finetune_train.txt"),
        metadata_csv=os.path.join(data_dir, "lucas_metadata_final.csv"),
        sentinel2_dir=os.path.join(data_dir, "sentinel2"),
        slope_dir=os.path.join(data_dir, "slope"),
        mode="finetune",
    )
    dataset_test = PAMAEDataset(
        ids_file=os.path.join(data_dir, "finetune_test.txt"),
        metadata_csv=os.path.join(data_dir, "lucas_metadata_final.csv"),
        sentinel2_dir=os.path.join(data_dir, "sentinel2"),
        slope_dir=os.path.join(data_dir, "slope"),
        mode="finetune",
    )

    print(f"Train: {len(dataset_train)} samples")
    print(f"Test:  {len(dataset_test)} samples")

    # Compute 75th percentile of training SOC for Physical Violation Rate
    # Read SOC values from metadata (faster than iterating the dataset)
    import pandas as pd
    meta_path = os.path.join(args.data_dir, "lucas_metadata_final.csv")
    train_ids_path = os.path.join(args.data_dir, "finetune_train.txt")
    with open(train_ids_path) as f:
        train_ids = [line.strip() for line in f if line.strip()]
    meta = pd.read_csv(meta_path)
    meta["point_id"] = meta["point_id"].astype(str)
    train_soc = meta[meta["point_id"].isin(train_ids)]["soc_gkg"].values
    soc_threshold_75 = float(np.log1p(np.percentile(train_soc, 75)))
    print(f"SOC 75th percentile (log1p): {soc_threshold_75:.4f} "
          f"(raw: {np.percentile(train_soc, 75):.2f} g/kg)")

    # Samplers
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_test = DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # ── Model ────────────────────────────────────────────────
    model = vit_base_patch8_regression(drop_path_rate=args.drop_path)

    # Load pre-trained encoder weights (if provided)
    if args.finetune:
        load_pretrained_encoder(model, args.finetune, device="cpu")

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ViTForSOCRegression")
    print(f"Number of params (M): {n_parameters / 1e6:.2f}")

    # ── Optimizer with layer-wise lr decay ───────────────────
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual lr: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")

    # Layer-wise lr decay: encoder blocks get lower lr, head gets full lr
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # MSE loss for regression
    criterion = torch.nn.MSELoss()
    print(f"Criterion: {criterion}")

    # ── Logging ──────────────────────────────────────────────
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    log_dir = args.log_dir if args.log_dir else args.output_dir
    log_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        if misc.is_main_process():
            log_writer = SummaryWriter(log_dir=log_dir)
    except ImportError:
        print("TensorBoard not available, skipping log_writer")

    # ── Training loop ────────────────────────────────────────
    print(f"\nStart training for {args.epochs} epochs")
    start_time = time.time()
    best_r2 = -float("inf")

    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # Train
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device,
            epoch, loss_scaler,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            args=args,
        )

        # Save checkpoint
        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        # Evaluate
        test_stats = evaluate(
            data_loader_test, model, device, soc_threshold_75
        )

        # Track best R²
        if test_stats["r2"] > best_r2:
            best_r2 = test_stats["r2"]
            # Save best model
            if args.output_dir:
                torch.save(
                    {"model": model_without_ddp.state_dict(), "epoch": epoch},
                    os.path.join(args.output_dir, "best_model.pth"),
                )
        print(f"Best R² so far: {best_r2:.4f}")

        # Log
        if log_writer is not None:
            log_writer.add_scalar("test/r2", test_stats["r2"], epoch)
            log_writer.add_scalar("test/rmse", test_stats["rmse"], epoch)
            log_writer.add_scalar("test/ccc", test_stats["ccc"], epoch)
            log_writer.add_scalar("test/loss", test_stats["loss"], epoch)
            log_writer.add_scalar("test/violation_rate",
                                  test_stats["violation_rate"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "best_r2": best_r2,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"),
                      mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # ── Final summary ────────────────────────────────────────
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\nTraining complete in {total_time_str}")
    print(f"Best test R²: {best_r2:.4f}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)