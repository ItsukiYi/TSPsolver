"""Generate TSP training data for DIFUSCO using Christofides+2opt as labels.

Usage:
    python generate_training_data.py --n 50 --num-train 1000 --num-val 100 --num-test 100
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from src.utils import generate_random_tsp_instance
from src.algorithms import christofides_with_2opt


def generate_dataset(
    n: int,
    num_samples: int,
    output_file: str,
    seed: int = 0,
    max_2opt_iter: int = 5000,
):
    """Generate a dataset file in DIFUSCO format.

    Format: x1 y1 x2 y2 ... output t1 t2 ... tn t1 (1-indexed)

    Args:
        n: number of nodes
        num_samples: number of instances
        output_file: path to save the data
        seed: random seed
        max_2opt_iter: max 2-opt iterations for label quality
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    start_time = time.time()
    with open(output_file, "w") as f:
        for i in range(num_samples):
            points = generate_random_tsp_instance(n, seed=seed + i)
            tour, meta = christofides_with_2opt(
                points, max_2opt_iterations=max_2opt_iter
            )

            # Write in DIFUSCO format (1-indexed)
            coord_str = " ".join(f"{x} {y}" for x, y in points)
            tour_1idx = [t + 1 for t in tour]  # 0-indexed → 1-indexed
            tour_str = " ".join(str(t) for t in tour_1idx)
            f.write(f"{coord_str} output {tour_str}\n")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (num_samples - i - 1) / rate
                print(
                    f"  Generated {i + 1}/{num_samples} instances "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                )

    elapsed = time.time() - start_time
    print(f"Saved {num_samples} TSP-{n} instances to {output_file} ({elapsed:.0f}s total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TSP training data for DIFUSCO")
    parser.add_argument("--n", type=int, default=50, help="Number of nodes")
    parser.add_argument("--num-train", type=int, default=1000, help="Training samples")
    parser.add_argument("--num-val", type=int, default=100, help="Validation samples")
    parser.add_argument("--num-test", type=int, default=100, help="Test samples")
    parser.add_argument("--output-dir", type=str, default="data/tsp_problems")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-2opt-iter", type=int, default=5000)

    args = parser.parse_args()

    print(f"Generating TSP-{args.n} datasets...")
    print(f"  Training: {args.num_train} instances")
    print(f"  Validation: {args.num_val} instances")
    print(f"  Test: {args.num_test} instances")
    print(f"  Label solver: Christofides + 2-opt ({args.max_2opt_iter} iterations)")
    print()

    # Training set
    print("[1/3] Training set...")
    generate_dataset(
        n=args.n,
        num_samples=args.num_train,
        output_file=os.path.join(args.output_dir, f"tsp{args.n}_train.txt"),
        seed=args.seed,
        max_2opt_iter=args.max_2opt_iter,
    )

    # Validation set
    print("[2/3] Validation set...")
    generate_dataset(
        n=args.n,
        num_samples=args.num_val,
        output_file=os.path.join(args.output_dir, f"tsp{args.n}_val.txt"),
        seed=args.seed + args.num_train,
        max_2opt_iter=args.max_2opt_iter,
    )

    # Test set
    print("[3/3] Test set...")
    generate_dataset(
        n=args.n,
        num_samples=args.num_test,
        output_file=os.path.join(args.output_dir, f"tsp{args.n}_test.txt"),
        seed=args.seed + args.num_train + args.num_val,
        max_2opt_iter=args.max_2opt_iter,
    )

    print("\nDone! Files saved to", args.output_dir)
