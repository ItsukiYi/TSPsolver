"""Minimal wrapper for DIFUSCO inference and comparison with classic algorithms.

This module provides a simplified interface to:
1. Run the DIFUSCO diffusion model for TSP inference
2. Compare DIFUSCO results with classic algorithm baselines
3. Handle the data format conversion between DIFUSCO and our code

Note: This module requires the full DIFUSCO codebase to be present at
../DIFUSCO-main/ and all dependencies installed.
"""

import os
import sys
import warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

# Add DIFUSCO to path
DIFUSCO_PATH = os.path.join(os.path.dirname(__file__), "..", "DIFUSCO-main")
if os.path.isdir(DIFUSCO_PATH):
    sys.path.insert(0, DIFUSCO_PATH)
    sys.path.insert(0, os.path.join(DIFUSCO_PATH, "difusco"))


def check_difusco_available() -> bool:
    """Check if DIFUSCO codebase and dependencies are available."""
    try:
        import pytorch_lightning
        import torch_geometric
        return os.path.isdir(DIFUSCO_PATH)
    except ImportError as e:
        print(f"DIFUSCO dependencies not available: {e}")
        return False


def load_tsp_data_from_file(filepath: str):
    """Load TSP data in DIFUSCO format.

    Format: x1 y1 x2 y2 ... output t1 t2 ... tn t1

    Args:
        filepath: path to the data file

    Returns:
        list of (points, tour) tuples
    """
    instances = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" output ")
            coords = parts[0].split(" ")
            points = np.array([
                [float(coords[i]), float(coords[i + 1])]
                for i in range(0, len(coords), 2)
            ])
            tour = np.array([int(t) - 1 for t in parts[1].split(" ")])  # 0-indexed
            instances.append((points, tour))
    return instances


def convert_to_difusco_format(points: np.ndarray, tour: List[int], filepath: str):
    """Save a TSP instance in DIFUSCO's data format.

    Args:
        points: (n, 2) array
        tour: list of vertex indices (0-indexed)
        filepath: output file path
    """
    with open(filepath, "a") as f:
        coord_str = " ".join(f"{x} {y}" for x, y in points)
        tour_str = " ".join(str(t + 1) for t in tour[:-1])  # 1-indexed for DIFUSCO
        # Add return to start
        start = tour[0] + 1
        f.write(f"{coord_str} output {tour_str} {start}\n")


def generate_difusco_dataset(
    instances: List[Tuple[np.ndarray, List[int]]],
    output_file: str,
):
    """Generate a dataset file in DIFUSCO format from a list of (points, tour) pairs.

    Args:
        instances: list of (points, tour) tuples
        output_file: path to save the dataset
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Overwrite file
    if os.path.exists(output_file):
        os.remove(output_file)
    for points, tour in instances:
        convert_to_difusco_format(points, tour, output_file)
    print(f"Saved {len(instances)} instances to {output_file}")


class DIFUSCOInference:
    """Lightweight wrapper for running DIFUSCO inference.

    This wraps the TSPModel from DIFUSCO and provides a simple interface
    for running inference on TSP instances.
    """

    def __init__(
        self,
        checkpoint_path: str,
        diffusion_type: str = "categorical",
        n_layers: int = 12,
        hidden_dim: int = 256,
        sparse_factor: int = -1,
        device: str = "cuda",
    ):
        """Initialize DIFUSCO inference wrapper.

        Args:
            checkpoint_path: path to pretrained model checkpoint
            diffusion_type: "categorical" or "gaussian"
            n_layers: number of GNN layers
            hidden_dim: hidden dimension
            sparse_factor: -1 for dense, >0 for sparse (k-NN)
            device: "cuda" or "cpu"
        """
        if not check_difusco_available():
            raise RuntimeError(
                "DIFUSCO not available. Please ensure:\n"
                "1. DIFUSCO codebase is at ../DIFUSCO-main/\n"
                "2. All dependencies are installed"
            )

        from pl_tsp_model import TSPModel
        from argparse import Namespace

        self.device = device

        # Create args matching the checkpoint's configuration
        self.args = Namespace(
            diffusion_type=diffusion_type,
            diffusion_schedule="cosine",
            diffusion_steps=1000,
            inference_diffusion_steps=50,
            inference_schedule="cosine",
            inference_trick="ddim",
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            sparse_factor=sparse_factor,
            aggregation="sum",
            two_opt_iterations=1000,
            parallel_sampling=1,
            sequential_sampling=1,
            save_numpy_heatmap=False,
            storage_path=".",
            training_split="dummy",
            validation_split="dummy",
            test_split="dummy",
            batch_size=1,
            learning_rate=0.0002,
            weight_decay=0.0001,
            lr_scheduler="cosine-decay",
            num_epochs=50,
            num_workers=0,
            validation_examples=8,
            use_activation_checkpoint=False,
            fp16=False,
            project_name="tsp_diffusion",
            wandb_logger_name=None,
            wandb_entity=None,
            resume_id=None,
            ckpt_path=None,
            do_train=False,
            do_test=True,
            do_valid_only=False,
            resume_weight_only=True,
        )

        # Load model from checkpoint
        print(f"Loading DIFUSCO checkpoint from {checkpoint_path}...")
        self.model = TSPModel.load_from_checkpoint(
            checkpoint_path,
            param_args=self.args,
            strict=False,
        )
        self.model = self.model.to(device)
        self.model.eval()
        print("Model loaded successfully.")

    def solve(
        self,
        points: np.ndarray,
        two_opt_iterations: int = 1000,
    ) -> Tuple[List[int], Dict]:
        """Solve a TSP instance using DIFUSCO.

        Args:
            points: (n, 2) array of point coordinates
            two_opt_iterations: number of 2-opt refinement iterations

        Returns:
            (tour, metadata) tuple
        """
        from pl_tsp_model import TSPModel
        from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours
        from co_datasets.tsp_graph_dataset import TSPGraphDataset
        from utils.diffusion_schedulers import InferenceSchedule

        n = len(points)

        # Build adjacency matrix space
        adj_matrix = np.zeros((1, n, n))
        points_tensor = torch.from_numpy(points).float().unsqueeze(0)  # (1, n, 2)

        # Run diffusion inference
        with torch.no_grad():
            device = torch.device(self.device)

            # Initialize random noise
            xt = torch.randn_like(torch.from_numpy(adj_matrix).float())
            xt = xt.to(device)
            points_tensor = points_tensor.to(device)

            if self.args.diffusion_type == "categorical":
                xt = (xt > 0).long()

            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=self.args.inference_schedule,
                T=self.model.diffusion.T,
                inference_T=steps,
            )

            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)

                if self.args.diffusion_type == "gaussian":
                    xt = self.model.gaussian_denoise_step(
                        points_tensor, xt, t1, device, None, target_t=t2,
                    )
                else:
                    xt = self.model.categorical_denoise_step(
                        points_tensor, xt, t1, device, None, target_t=t2,
                    )

            # Get adjacency matrix from diffusion output
            if self.args.diffusion_type == "gaussian":
                adj_mat = xt.cpu().numpy() * 0.5 + 0.5
            else:
                adj_mat = xt.float().cpu().numpy() + 1e-6

        # Extract tour from heatmap using greedy merge
        np_points = points.astype(np.float64)
        tours, merge_iters = merge_tours(
            adj_mat, np_points, None,
            sparse_graph=False,
            parallel_sampling=1,
        )

        # Refine with 2-opt
        solved_tours, ns = batched_two_opt_torch(
            np_points,
            np.array(tours).astype("int64"),
            max_iterations=two_opt_iterations,
            device=device,
        )

        best_tour = solved_tours[0].tolist()

        metadata = {
            "algorithm": f"DIFUSCO ({self.args.diffusion_type})",
            "merge_iterations": merge_iters,
            "two_opt_iterations": ns,
        }
        return best_tour, metadata
