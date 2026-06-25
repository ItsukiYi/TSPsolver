# TSP Solver: Hybrid Diffusion & Neural Refinement Pipeline

A hybrid Traveling Salesman Problem solver combining DIFUSCO (diffusion-based generation, NeurIPS 2023) and DualOpt (divide-and-conquer neural refinement, AAAI 2025) into a unified pipeline. Course project for **CS240: Algorithm Design and Analysis**, Spring 2026, ShanghaiTech University.

## Repository Structure

```
├── src/                          # Core algorithm implementations
│   ├── algorithms.py             # NN, Christofides (Blossom MWPM), 2-opt
│   ├── utils.py                  # Distance matrix, visualization, experiments
│   ├── experiment.py             # Batch evaluation framework
│   ├── difusco_wrapper.py        # DIFUSCO inference integration
│   └── tsplib_loader.py          # TSPLIB parser (60+ known optimal values)
│
├── DIFUSCO-main/                 # Original DIFUSCO codebase (NeurIPS 2023)
│   └── difusco/                  #   with compatibility patches applied
│
├── DualOpt-main/                 # Original DualOpt codebase (AAAI 2025)
│   └── pretrained/               #   with compatibility patches applied
│
├── DualOpt-improved/             # Our improved DualOpt with 5 strategies
│   └── utils/
│       ├── difusco_pipeline.py   #   #2 DIFUSCO→DualOpt pipeline
│       ├── heatmap_guide.py      #   #1 Heatmap-guided reviser
│       ├── adaptive_reviser.py   #   #3 Adaptive window sizing
│       ├── freeze_reviser.py     #   #4 Fragment freezing
│       └── destroy_repair.py     #   #5 Destroy-and-repair
│
├── data/
│   ├── tsp_problems/             # Training/val/test splits (TSP-50/100/200/500)
│   └── tsplib/                   # 8 TSPLIB benchmark instances
│
├── main.py                       # CLI entry point (demo/experiment/difusco modes)
├── train_difusco.py              # Single-GPU DIFUSCO training script
├── generate_training_data.py     # C+2opt label generation for DIFUSCO
├── evaluate_tsplib.py            # Unified TSPLIB benchmark
├── final_comparison.py           # Full method comparison on TSP-50
│
├── run_ablation.py               # DIFUSCO ablation: steps × 2-opt
├── run_dualopt.py                # DualOpt evaluation
├── run_scalability.py            # Scalability benchmark (n=100–1000)
├── run_lkh_baseline.py           # LKH3 gold-standard baseline
│
├── compare_*.py                  # A/B tests for improvements #1–#5
├── test_*.py                     # Unit/smoke/integration tests
│
├── visualize.py                  # Classic algorithm step-by-step visualization
├── visualize_ml.py               # Neural method concept visualization
├── calibration_study.py          # DIFUSCO heatmap calibration analysis
├── pareto_frontier.py            # Speed-quality Pareto frontier
│
├── city_delivery_scenario.py     # Real-world city delivery (500 nodes)
├── clustered_benchmark.py        # Clustered delivery benchmark
├── real_world_scenario.py        # Campus food delivery (31 nodes)
│
└── arch_diagrams.py              # Architecture diagram generation
```

## Setup

### Requirements

- Python 3.12+
- PyTorch 2.12+ with CUDA 12.6
- PyTorch Geometric 2.7+
- PyTorch Lightning 2.6+
- NetworkX, SciPy, NumPy
- LKH3 solver (optional, for gold-standard baseline)
- `tsplib95` and `lkh` Python packages (optional, for TSPLIB+LKH3 evaluation)

### Installation

```bash
# Create virtual environment
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install torch torch-geometric pytorch-lightning networkx scipy numpy tsplib95 lkh
```

### Download DIFUSCO Checkpoints

DIFUSCO pretrained models are not included in this repo. You can either:
- Train from scratch: `python train_difusco.py --problem tsp50`
- Download official checkpoints from the [DIFUSCO repository](https://github.com/Edward-Sun/DIFUSCO)

DualOpt pretrained revisers are included in `DualOpt-main/pretrained/`.

## Usage

### Quick Demo

```bash
# Run all algorithms on a single TSP-20 instance
python main.py --mode demo

# Full benchmark across all scales
python main.py --mode experiment

# DIFUSCO inference with pretrained checkpoint
python main.py --mode difusco --ckpt checkpoints/tsp50_categorical.ckpt

# Generate training data
python main.py --mode generate-data --num-samples 1000
```

### Run Our Hybrid Pipeline

```bash
# DIFUSCO → DualOpt pipeline on TSP-50 test set
python compare_pipeline.py

# Cross-scale pipeline tests
python test_tsp100_pipeline.py
python test_tsp200_v2.py
python test_tsp500_pipeline.py
```

### Evaluate Improvements

```bash
python compare_improvements.py    # #1 Heatmap-guided vs original
python compare_adaptive.py        # #3 Adaptive window sizing
python compare_freeze.py          # #4 Fragment freezing
python compare_destroy_repair.py  # #5 Destroy-and-repair
```

### TSPLIB Benchmark

```bash
python evaluate_tsplib.py         # Classic + DIFUSCO + DualOpt on all 8 instances
python run_lkh_baseline.py        # LKH3 gold standard baseline
```

### Real-World Scenarios

```bash
python city_delivery_scenario.py  # 500-node city-wide delivery
python clustered_benchmark.py     # Clustered delivery (50/100/200 nodes)
python real_world_scenario.py     # 31-node campus food delivery
```

## References

- Sun & Yang. **DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization.** *NeurIPS*, 2023.
- Zhou et al. **DualOpt: A Dual Divide-and-Optimize Algorithm for the Large-Scale TSP.** *AAAI*, 2025.
- Christofides. Worst-case analysis of a new heuristic for the travelling salesman problem. Technical Report 388, GSIA, CMU, 1976.
- Helsgaun. An Extension of the Lin-Kernighan-Helsgaun TSP Solver. Technical Report, Roskilde University, 2017.
- Kool, van Hoof & Welling. Attention, learn to solve routing problems! *ICLR*, 2019.
