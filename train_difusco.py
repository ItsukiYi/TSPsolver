"""Train DIFUSCO on TSP data with single GPU.

Simplified training script adapted from DIFUSCO's train.py.
Uses our generated Christofides+2opt labels for supervised training.

Usage:
    python train_difusco.py --n 50 --epochs 50 --batch-size 64
"""

import os
import sys

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DIFUSCO-main"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DIFUSCO-main", "difusco"))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser

from pl_tsp_model import TSPModel
from utils.diffusion_schedulers import InferenceSchedule


def parse_args():
    parser = ArgumentParser(description="Train DIFUSCO TSP model")
    # Data
    parser.add_argument("--n", type=int, default=50, help="TSP size")
    parser.add_argument("--data-dir", type=str, default="data/tsp_problems")
    parser.add_argument("--storage-path", type=str, default="outputs/difusco_training")
    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine-decay")
    parser.add_argument("--num-workers", type=int, default=0)
    # Model
    parser.add_argument("--diffusion-type", type=str, default="categorical")
    parser.add_argument("--diffusion-schedule", type=str, default="cosine")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--inference-schedule", type=str, default="cosine")
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--sparse-factor", type=int, default=-1)
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--use-activation-checkpoint", action="store_true", default=False)
    parser.add_argument("--inference-trick", type=str, default="ddim")
    parser.add_argument("--save-numpy-heatmap", action="store_true", default=False)
    # Experiment tracking
    parser.add_argument("--project-name", type=str, default="cs240_tsp_diffusion")
    parser.add_argument("--exp-name", type=str, default=None)
    # Test
    parser.add_argument("--do-test-only", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default=None)
    # Inference
    parser.add_argument("--two-opt-iterations", type=int, default=1000)
    parser.add_argument("--parallel-sampling", type=int, default=1)
    parser.add_argument("--sequential-sampling", type=int, default=1)
    parser.add_argument("--validation-examples", type=int, default=16)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Configure data paths
    args.training_split = os.path.join(args.data_dir, f"tsp{args.n}_train.txt")
    args.validation_split = os.path.join(args.data_dir, f"tsp{args.n}_val.txt")
    args.test_split = os.path.join(args.data_dir, f"tsp{args.n}_test.txt")

    # Map to DIFUSCO's expected arg names and fix path handling
    # DIFUSCO joins storage_path + training_split, so set storage_path to cwd
    args.storage_path = os.getcwd()
    args.learning_rate = args.lr
    args.inference_diffusion_steps = args.inference_steps

    # Check data exists
    for split_name, split_path in [
        ("Training", args.training_split),
        ("Validation", args.validation_split),
        ("Test", args.test_split),
    ]:
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"{split_name} data not found at {split_path}. "
                f"Run 'python generate_training_data.py --n {args.n}' first."
            )

    if args.exp_name is None:
        args.exp_name = f"tsp{args.n}_{args.diffusion_type}"

    # Storage
    os.makedirs(args.storage_path, exist_ok=True)
    exp_dir = os.path.join(args.storage_path, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"DIFUSCO Training: TSP-{args.n} ({args.diffusion_type} diffusion)")
    print(f"{'=' * 60}")
    print(f"  Training data:   {args.training_split}")
    print(f"  Validation data: {args.validation_split}")
    print(f"  Test data:       {args.test_split}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  GPU:             {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Experiment dir:  {exp_dir}")
    print(f"{'=' * 60}")

    # Create model
    model = TSPModel(param_args=args)

    # Logger
    logger = TensorBoardLogger(
        save_dir=args.storage_path,
        name=args.exp_name,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/solved_cost",
        mode="min",
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(exp_dir, "checkpoints"),
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    # Trainer (single GPU)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
            lr_callback,
        ],
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"{'=' * 60}")

    if args.do_test_only:
        if args.ckpt_path is None:
            args.ckpt_path = os.path.join(exp_dir, "checkpoints", "last.ckpt")
        print(f"Testing model from {args.ckpt_path}")
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        # Train
        print("Starting training...")
        trainer.fit(model)

        # Test with best checkpoint
        print("Testing with best checkpoint...")
        trainer.test(ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
