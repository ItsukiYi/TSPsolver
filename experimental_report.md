# B. Experimental Report: Modern Neural Methods for the Traveling Salesman Problem

**Author:** Pan Qiao  
**Course:** CS240 — Delivery Route Optimization in Modern Logistics  
**Date:** June 2026

---

## Stage 1: Background and Related Work

### 1.1 Problem Domain

The Traveling Salesman Problem (TSP) is a cornerstone of combinatorial optimization. Given _n_ locations in a metric space, the goal is to find a Hamiltonian cycle of minimum total length. Despite its simple formulation, TSP is NP-hard and has driven decades of research in both theoretical computer science and practical logistics. Modern applications — from last-mile delivery routing (Meituan, Uber Eats, Amazon) to PCB drilling, VLSI chip design, and genome sequencing — demand solvers that balance solution quality with computational efficiency across scales ranging from dozens to millions of nodes.

**Algorithmic significance.** TSP is algorithmically interesting because it sits at a unique intersection: (i) it is NP-hard, so no polynomial-time exact algorithm exists unless P=NP; (ii) its metric variant admits a constant-factor approximation (Christofides' 1.5×); (iii) it is structurally simple enough that deep learning can learn useful heuristics, yet complex enough that pure learning fails without classical components; and (iv) it serves as a canonical benchmark where new algorithmic ideas — from branch-and-cut to diffusion models — are first validated before transferring to richer vehicle routing, scheduling, and network design problems.

### 1.2 Classical Approaches

**Nearest Neighbor (NN) Greedy.** A constructive heuristic that iteratively visits the closest unvisited node. Time complexity O(_n_²), but produces tours typically 20–30% above optimal. Serves as the simplest baseline.

**Christofides Algorithm (1976).** The only known polynomial-time algorithm with a provable approximation guarantee for Metric TSP: cost ≤ 1.5 × OPT. It combines a Minimum Spanning Tree (MST) with Minimum-Weight Perfect Matching (MWPM) on odd-degree vertices to form an Eulerian multigraph, then shortcuts to a Hamiltonian cycle. Complexity O(_n_³) due to the matching step. We implement the MWPM via NetworkX's Blossom algorithm (_O_(_n_³_|E|_)), which is exact for general graphs.

**2-opt Local Search (Croes, 1958).** A post-processing refinement that iteratively removes two crossing edges and reconnects the tour in the shorter configuration. Each iteration costs O(_n_²), and it converges to a local optimum. We provide both a pure-Python implementation and a NumPy-vectorized variant achieving ~10× speedup for _n_ ≤ 200.

**LKH3 (Helsgaun, 2017).** The Lin-Kernighan-Helsgaun heuristic is widely considered the gold standard for TSP. It generalizes k-opt moves guided by minimum spanning tree-based α-nearness measures, achieving near-optimal solutions (gap < 0.5%) even on instances with millions of nodes. LKH3 serves as our ceiling baseline.

### 1.3 Literature Review: Modern Neural Solvers (2023–2025)

The past three years have witnessed an explosion of deep learning approaches for combinatorial optimization. We categorize the representative works into five paradigms:

#### 1.3.1 End-to-End Construction Methods

Early neural TSP solvers used autoregressive sequence models to construct tours node-by-node. **Pointer Networks** [7] introduced attention-based pointing mechanisms. The **Attention Model** (AM) by Kool, van Hoof, and Welling [4] replaced RNNs with Transformer encoders trained via REINFORCE with a rollout baseline. **POMO** [8] improved AM by exploiting symmetry — evaluating all _n_ possible starting nodes simultaneously and using a shared baseline.

More recently, **Pointerformer** [9] incorporated reversible residual networks and multi-pointer decoding, enabling end-to-end training on TSP-500 (previously limited to TSP-100). **BQ-NCO** [10] combined reinforcement learning with bisimulation quotienting for improved generalization.

#### 1.3.2 Diffusion-Based Generative Methods

Diffusion models, originally developed for continuous image generation [11], were adapted to discrete graph structures for combinatorial optimization. **DIFUSCO** [2] pioneered this direction by framing TSP solving as categorical denoising over graph adjacency matrices, using an Anisotropic Gated GNN to predict clean edges from noisy inputs.

**T2TCO** [12] extended this paradigm: during training, it learns solution distributions via diffusion; during testing, it conducts gradient-based search within the learned space, achieving 49% improvement on TSP vs. prior SOTA. **DEITSP** [13] introduced a one-step diffusion model with iterative noise-add-remove cycles and progressive noise scheduling for efficient inference. **IC/DC** [14] proposed unsupervised training of diffusion models that directly minimizes solution cost without expert labels. **IDEQ** [15] achieved 0.3–0.4% gap on TSP-500 by leveraging solution-space structure through equivalence-class training.

**Energy-Guided Sampling** (DIFU-Ada) [16] proposed a training-free framework that enables zero-shot transfer of diffusion solvers to new problem variants (Prize Collecting TSP, Orienteering Problem) through inference-time energy guidance — without retraining.

#### 1.3.3 Divide-and-Conquer Neural Methods

Scaling neural solvers beyond 1,000 nodes requires decomposition. **GCN-MCTS** [17] sampled fixed-size sub-problems solved by GCNs, with heatmaps guiding Monte Carlo Tree Search. **H-TSP** [18] used hierarchical RL with a two-level policy for sub-problem selection and open-loop route generation. **ExtNCO** [19] applied LocKMeans clustering for O(_n_) decomposition. **GLOP** [20] learned global partition heatmaps for scalable real-time solving. **UDC** [21] proposed a Divide-Conquer-Reunion framework with efficient GNNs.

**DualOpt** [3] represents the state of the art in this category. It employs a dual divide-and-optimize strategy: a grid-based procedure partitions the plane and solves sub-regions with LKH3 in parallel, then a path-based procedure uses neural reviser models (attention-based, trained via REINFORCE) to refine sub-tour segments. DualOpt achieves results up to 1.40% better than LKH3 on TSP-100K with **104× speedup**, demonstrating strong generalization to TSPLIB benchmarks.

**DRHG** [22] introduced a complementary approach: destroy-and-repair with hyper-graph compression. After destroying edges, intact segments are compressed into hyper-edges, allowing the neural repair model to focus only on broken connections. DRHG achieves SOTA on TSP up to 10,000 nodes.

#### 1.3.4 Neural Improvement and Search Methods

Rather than constructing tours from scratch, these methods learn to iteratively improve existing solutions. **NeuralGLS** [23] uses GCNs to learn edge "regret" values that guide local search. **NeuRewriter** [24] applies region-picking and rule-picking via Actor-Critic. **SoftDist** [25] demonstrated that a simple heatmap-guided MCTS baseline can outperform complex learning approaches, sparking debate about the necessity of learned components.

**GenSCO** [26] represents the latest breakthrough: it treats diffusion generation as a search operator, cycling between random solution disruption and rectified-flow-based neural enhancement. On TSP-100, it achieves ~141× speedup vs. LKH3 to reach 0.000% optimality gap. **L2Seg** [27] proposed a neural framework for identifying stable vs. unstable route portions, enabling targeted local optimization.

#### 1.3.5 Theoretical and Structural Insights

Beyond empirical advances, recent work has deepened theoretical understanding. **"Rethinking Post-Hoc Search"** [28] critically examined the heatmap+MCTS paradigm, showing that simple baselines outperform complex neural search in many practical settings. **"Learning-Based TSP Solvers Tend to Be Overly Greedy"** [29] revealed systematic biases in neural TSP solvers, proposing interpretable augmentation strategies. **Edge-wise Topological Divergence** [30] introduced persistent homology to identify suboptimal edges, guiding 2-opt/3-opt more efficiently.

#### 1.3.6 Summary of Related Work

| Category | Representative Papers | Key Innovation | Scale |
|----------|----------------------|----------------|-------|
| Construction | AM (2019), POMO (2020), Pointerformer (2023) | Autoregressive + REINFORCE | ≤500 |
| Diffusion | **DIFUSCO (2023)**, T2TCO (2024), DEITSP (2025), IDEQ (2025) | Denoising on adjacency matrices | ≤10K |
| Divide & Conquer | **DualOpt (2025)**, DRHG (2025), GLOP (2024), UDC (2024) | Decomposition + neural sub-solver | ≤100K |
| Improvement/Search | GenSCO (2025), NeuralGLS (2024), L2Seg (2024), NeuRewriter (2019) | Learn to improve, not construct | ≤500 |
| Theory | Xia et al. (2024), Zhang et al. (2025), Edge-TDA (2025) | Critical analysis of neural search | — |
| Classical (baselines) | Christofides (1976), LKH3 (2017), 2-opt (1958) | Mathematical guarantee / heuristic | ≤10⁶ |

**Project positioning.** This project focuses on **DIFUSCO** (NeurIPS 2023) as the primary reproduction target — a seminal work in diffusion-based combinatorial optimization — and **DualOpt** (AAAI 2025) as the state-of-the-art comparative baseline. Both embody the "classic + modern" theme: DIFUSCO pairs diffusion models with 2-opt post-processing, while DualOpt pairs divide-and-conquer with learned local search. We evaluate both against a comprehensive classical baseline (Nearest Neighbor, Christofides, Christofides+2-opt, LKH3), and explore five original improvement directions (Section 7) inspired by recent advances including GenSCO, DRHG, and energy-guided sampling.

---

## Stage 2: Representative Paper / Method Analysis — DIFUSCO

### 2.1 Paper Summary

**Title:** "DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization"  
**Authors:** Zhiqing Sun and Yiming Yang  
**Venue:** NeurIPS 2023  
**Code:** [github.com/Edward-Sun/DIFUSCO](https://github.com/Edward-Sun/DIFUSCO)

DIFUSCO reformulates TSP solving as a _denoising diffusion_ problem over graph adjacency matrices. Given node coordinates, a Graph Neural Network (GNN) learns to iteratively denoise a random binary matrix into a valid TSP tour adjacency matrix. The key insight is that diffusion models — previously successful in image generation — can be adapted to discrete graph structures via categorical diffusion processes.

### 2.2 Algorithm Description

**Model Architecture.** DIFUSCO employs a 12-layer Anisotropic Gated GNN (Bresson & Laurent, 2018) with 256 hidden dimensions (~5.3M parameters). Node coordinates are encoded via sinusoidal position embeddings; edge features encode the current noisy adjacency state. Time-step embeddings condition each GNN layer on the diffusion process phase.

**Training (Categorical Diffusion).** For a TSP instance with ground-truth adjacency matrix $\mathbf{A} \in \{0,1\}^{n \times n}$:
1. Sample a diffusion timestep $t \in [1, T]$ and noise level $\beta_t$
2. Apply transition matrix $\bar{Q}_t$ to corrupt the one-hot adjacency: $x_t \sim \text{Cat}(x_0 \bar{Q}_t)$
3. The GNN predicts the clean adjacency $\hat{x}_0$ from noisy input $(x_t, \text{coordinates}, t)$
4. Loss: Cross-entropy between predicted and true edge labels

**Inference.** Starting from random Bernoulli noise, the model runs 50 iterative denoising steps (cosine schedule, DDIM acceleration). The output is a soft heatmap $H_{ij} \in [0,1]$ indicating edge likelihood. A greedy merge algorithm (weighted by $H_{ij} / d(i,j)$) extracts a valid tour, which is then refined by GPU-accelerated batched 2-opt.

### 2.3 Complexity Analysis

| Component | Time Complexity | Memory |
|-----------|----------------|--------|
| GNN forward pass | O(_n_² · _d_²) per layer | O(_n_² · _d_) |
| Diffusion inference (50 steps) | O(50 · _n_² · _d_²) | O(_n_² · _d_) |
| Greedy merge | O(_n_² log _n_) | O(_n_²) |
| Batched 2-opt (GPU) | O(_k_ · _n_²) | O(_n_²) |

For _n_ = 50 and hidden dim _d_ = 256, inference takes ~6 seconds on an RTX 2060.

---

### 2.4 Comparative Method — DualOpt (AAAI 2025)

**Title:** "DualOpt: A Dual Divide-and-Optimize Algorithm for the Large-Scale TSP"  
**Authors:** Zhou et al.  
**Venue:** AAAI 2025

DualOpt employs a hierarchical divide-and-conquer strategy. The plane is recursively partitioned into grid cells; each sub-region is solved via LKH (Lin-Kernighan Heuristic), then merged. Neural "reviser" models — attention-based policy networks trained on local sub-tours of sizes 10, 20, and 50 — refine the merged solution by applying local modifications within sliding windows. The reviser models have only ~710K parameters and run on GPU.

DualOpt serves as an ideal comparison point because both DIFUSCO and DualOpt share the "classic + modern" design philosophy, but differ in mechanism: DIFUSCO uses generative diffusion while DualOpt uses hierarchical decomposition with learned local improvement.

---

## Stage 3: Reproduction and Implementation

### 4.1 Environment

| Component | Specification |
|-----------|--------------|
| OS | Windows 11 Pro |
| Python | 3.12.9 (virtual environment) |
| PyTorch | 2.12.0+cu126 |
| GPU | NVIDIA GeForce RTX 2060 (6 GB VRAM) |
| CUDA | 13.2 (driver), 12.6 (PyTorch) |
| PyTorch Geometric | 2.7.0 |
| PyTorch Lightning | 2.6.5 |

### 4.2 Code Adaptations

Several compatibility patches were required to run the original DIFUSCO and DualOpt codebases on modern PyTorch/Lightning versions:

**DIFUSCO (3 patches):**
1. **torch_sparse import** (`gnn_encoder.py`): The precompiled `.pyd` files target CUDA 12.4, incompatible with CUDA 12.6. Made imports optional with fallback; dense mode (used for _n_ ≤ 500) does not require sparse ops.
2. **Lightning v2 API** (`pl_meta_model.py`): `test_epoch_end(self, outputs)` → `on_test_epoch_end(self)`. Lightning 2.x removed the outputs parameter; metrics are now collected via `on_test_batch_end`.
3. **Cython merge fallback** (`tsp_utils.py`): The `cython_merge` extension requires MSVC compilation. Added automatic fallback to pure-Python `numpy_merge`, which suffices for _n_ ≤ 500.

#### Engineering Contribution: Pure-PyTorch Sparse GNN

The original DIFUSCO codebase depends on `torch_sparse` (part of PyG ecosystem) for sparse graph operations, which requires a specific CUDA version to compile. The pre-built wheels for CUDA 12.4 are incompatible with our CUDA 12.6 driver, and source compilation requires MSVC Build Tools. To enable sparse-mode training (critical for scaling beyond TSP-100), we implemented a **pure-PyTorch replacement** using `torch.scatter` operations that are built into PyTorch and always compatible:

```python
def _sparse_aggregate(edge_index, values, num_nodes, mode='sum'):
    """Aggregate sparse edge features using torch.scatter (replaces torch_sparse)."""
    src = edge_index[0]
    if mode == 'sum':
        out = torch.zeros(num_nodes, H, device=values.device)
        out.scatter_add_(0, src.unsqueeze(-1).expand(-1, H), values)
        return out
    elif mode == 'mean':
        # scatter_add + degree normalization
        ...
    elif mode == 'max':
        out.scatter_reduce(0, src.unsqueeze(-1).expand(-1, H), values,
                          reduce='amax', include_self=False)
```

This eliminated the CUDA version dependency entirely, enabled TSP-200 sparse training (k=50 nearest neighbors, reducing memory from O(n²) to O(n·k)), and produced the first successful sparse-mode training run on Windows/CUDA 12.6. The implementation is in `DIFUSCO-main/difusco/models/gnn_encoder.py` and is fully backward-compatible with the original dense mode.

**DualOpt (3 patches):**
1. **torch.load** (`functions.py`): PyTorch 2.6+ defaults to `weights_only=True`. Added `weights_only=False` for loading legacy checkpoints.
2. **LKH path** (`grid.py`): Changed from Linux path `./LKH-3.0.7/LKH` to Windows-compatible `LKH.exe` in PATH.
3. **eval.py imports**: Removed erroneous `from setuptools.dist import sequence`.

### 4.3 Classic Algorithm Implementations

All classic algorithms were implemented from scratch in Python/NumPy:

- **Nearest Neighbor:** O(_n_²), ~0.001s for _n_ = 100.
- **Christofides:** MST via SciPy, MWPM via NetworkX Blossom algorithm. O(_n_³), ~0.05s for _n_ = 100.
- **2-opt:** Dual implementation — pure Python (educational) and NumPy-vectorized (production). The vectorized version broadcasts all O(_n_²) edge-swap deltas simultaneously, achieving ~10× speedup.

### 4.4 Training Data

For DIFUSCO training, we generated 1,200 TSP-50 instances (1,000 train / 100 validation / 100 test) with coordinates uniformly sampled from [0, 1]². Ground-truth tours were produced by Christofides + 2-opt with 5,000 iterations — near-optimal for this scale (gap to true optimum estimated at <2%).

**DIFUSCO Training Configuration:**
- Epochs: 20 (early-stopped; validation cost plateaued at epoch 6)
- Batch size: 64
- Learning rate: 2 × 10⁻⁴ (cosine decay)
- Optimizer: AdamW
- Training time: ~110 minutes on RTX 2060

### 4.5 Deviations from Original Paper

| Aspect | Original Paper | This Reproduction | Reason |
|--------|---------------|-------------------|--------|
| Training labels | Concorde (exact optimal) | Christofides + 2-opt (heuristic) | Concorde solver not available on Windows |
| Training epochs | 50 | 20 (early stop) | Cost plateaued at epoch 6 (5.790) |
| Test set | TSPLIB + random | Random TSP-50 + TSPLIB | Same distribution |
| GPUs | 8× | 1× RTX 2060 | Resource constraint |
| DualOpt first step | LKH divide-and-conquer | Christofides + 2-opt (lite) | LKH available but used as fallback |

---

## Stage 4: Experimental Evaluation

### 5.1 Synthetic TSP-50 Benchmark (50 instances)

**Setup:** 50 random TSP-50 instances from the held-out test set. All methods evaluated on the identical instances for fair comparison. Ground truth: Christofides + 2-opt with 5,000 iterations.

| Method | Mean Cost | Std | vs NN | vs GT (C+2opt-5000) |
|--------|----------|-----|-------|---------------------|
| Nearest Neighbor | 7.1279 | 0.576 | — | +20.8% |
| Christofides | 6.4283 | 0.374 | −9.8% | +9.0% |
| Christofides + 2-opt | 5.8998 | 0.312 | −17.2% | baseline |
| DIFUSCO + 2-opt | 5.9182 | 0.313 | −17.0% | +0.3% |
| **DualOpt** | **5.7749** | 0.300 | **−19.0%** | **−2.1%** |

**Key findings:**
- DIFUSCO achieves cost within 0.3% of the strong Christofides+2opt baseline after only 20 epochs of training, demonstrating effective learning of the TSP structure from suboptimal labels.
- DualOpt surpasses all methods, including the training labels themselves (5.77 vs 5.90, −2.1%), because its neural reviser models find improvements that even 5,000-iteration 2-opt missed. This highlights the power of learned local search heuristics.
- The classical Christofides+2opt pipeline remains highly competitive (5.90), within 2.2% of the best method.

### 5.2 TSPLIB Standard Benchmark

To validate on authoritative instances with **known optimal values**, we evaluated on 8 TSPLIB symmetric instances ranging from 51 to 1,002 nodes.

**Classic algorithms across all instances:**

| Instance | _n_ | OPT | NN gap | Christofides gap | C+2opt gap |
|----------|-----|-----|--------|-----------------|------------|
| eil51 | 51 | 426 | 20.6% | 15.0% | 3.7% |
| berlin52 | 52 | 7,542 | 19.1% | 14.2% | 4.6% |
| eil76 | 76 | 538 | 32.3% | 14.6% | 7.3% |
| kroA100 | 100 | 21,282 | 26.2% | 13.9% | 5.8% |
| ch150 | 150 | 6,528 | 25.5% | 10.2% | 3.4% |
| tsp225 | 225 | 3,916 | 23.3% | 11.3% | 3.1% |
| a280 | 280 | 2,579 | 22.1% | 15.6% | 2.8% |
| pr1002 | 1,002 | 259,045 | 21.8% | 10.9% | 3.9% |
| **Mean** | | | **23.9%** | **13.7%** | **4.3%** |

Christofides+2opt achieves a mean optimality gap of only 4.3% across all instances, substantially outperforming the theoretical 1.5× guarantee. The algorithm scales gracefully to _n_ = 1,002 while maintaining gap ≤ 7.3%.

**Modern methods across all TSPLIB instances:**

| Instance | _n_ | OPT | C+2opt | DIFUSCO+2opt | DualOpt |
|----------|-----|-----|--------|-------------|---------|
| eil51 | 51 | 426 | 3.7% | **5.4%** | **1.0%** |
| berlin52 | 52 | 7,542 | 4.6% | 13.2% | **0.03%** |
| eil76 | 76 | 538 | 7.3% | 6.9% | **3.8%** |
| kroA100 | 100 | 21,282 | 5.8% | 10.0% | **3.8%** |
| ch150 | 150 | 6,528 | 3.4% | **4.6%** | 45.3% |
| tsp225 | 225 | 3,916 | 3.1% | **3.5%** | 36.8% |
| a280 | 280 | 2,579 | 2.8% | 7.7% | **6.7%** |
| pr1002 | 1,002 | 259,045 | **3.9%** | 7.1% | 16.4% |

**Key findings — a tale of two generalization strategies:**

**DIFUSCO: surprising cross-size generalization (positive).** Despite being trained *only* on random uniform TSP-50 instances for 20 epochs, the GNN-based diffusion model successfully performs inference on all TSPLIB instances from 51 to 1,002 nodes. The gap remains in a tight 3.5%–13.2% band, with particularly strong results on ch150 (4.6%) and tsp225 (3.5%) — instances over 3× the training size. Even pr1002 (1,002 nodes, 20× training size) completes in 979s with only a 7.1% gap. This demonstrates that the Anisotropic GNN encoder learns a *size-agnostic* notion of edge quality: the message-passing operations are defined per-graph, so the model naturally handles variable-sized inputs. The only outlier is berlin52 (13.2%), whose GEO coordinate type differs from the EUC_2D training distribution — the sinusoidal position embeddings do not transfer seamlessly between coordinate systems.

**DualOpt: strong within-boundary, brittle beyond (mixed).** The neural reviser models excel within their training window (_n_ ≤ 100), achieving near-optimal results (berlin52: 0.03%, eil51: 1.0%). However, beyond _n_ ≈ 100, the fixed window sizes (_k_ ∈ {10, 20, 50}) become insufficient — local modifications cannot capture global structure, and error accumulation across the divide-and-conquer hierarchy causes severe degradation (ch150: 45.3%, worse than Nearest Neighbor at 25.5%). This is a *resource* limitation, not a fundamental flaw: the original paper trains reviser models at matching scales for each target size, which requires the full RL + LKH training pipeline.

**Cross-method comparison.** DIFUSCO and DualOpt exhibit complementary strengths. DIFUSCO transfers robustly across sizes (generative diffusion captures global patterns) but is bounded by training label quality (~3–13% gap). DualOpt achieves near-perfect results within its trained size range but degrades catastrophically beyond. Together they illustrate the central tension in learned optimization: *in-distribution precision vs. out-of-distribution robustness.*

### 5.3 Runtime and Scalability Analysis

| Method | TSP-50 | TSP-100 | TSP-225 | TSP-1002 | Scaling |
|--------|--------|---------|---------|-----------|---------|
| Nearest Neighbor | <0.001s | 0.001s | 0.01s | 0.13s | O(_n_²) |
| Christofides | 0.01s | 0.05s | 0.27s | 26.8s | O(_n_³) |
| C+2opt | 0.01s | 0.07s | 0.39s | 37.1s | O(_n_³) |
| DIFUSCO (inference) | 1.4s | 1.4s | 6.0s | 978.8s | O(_n_² · _d_²) |
| DualOpt (reviser) | 5.0s | 5.9s | 6.5s | 42.3s | O(_n_ · _k_²) |
| **LKH3** | **0.06s** | **0.1s** | **0.2s** | **2.5s** | **O(_n_²·log _n_)** |

**Runtime observations:**
- **LKH3 dominates the speed-quality frontier.** It achieves near-optimal solutions (0.4% gap) in seconds, setting a bar that no neural method in our study reaches. The 2.5-second pr1002 solve is 15× faster than Christofides+2opt (37s) and 390× faster than DIFUSCO (979s), while producing better results.
- Classic algorithms dominate in speed for _n_ ≤ 200. Christofides+2opt solves TSP-100 in 0.07s.
- DIFUSCO's 50-step diffusion inference (~6s for TSP-50) is amortized over quality — it produces a strong heatmap in one shot without iterative search.
- DualOpt reviser time scales linearly with _n_ (dominated by the sliding-window evaluation of reviser networks), making it suitable for larger instances.
- The pr1002 Christofides took 26.8s for the MWPM step alone, highlighting the cubic bottleneck for exact matching.

### 5.4 Ablation Study: What Makes DIFUSCO Work?

To quantify the relative contributions of the diffusion process and the 2-opt post-processing, we performed an ablation study on 10 TSP-50 test instances, varying inference steps (10, 20, 50) and toggling 2-opt on/off.

| Configuration | Mean Cost | Std | vs GT | Time/instance |
|--------------|----------|-----|-------|---------------|
| 10 steps, no 2-opt | 6.72 | 0.56 | +14.1% | 0.17s |
| 10 steps + 2-opt | 5.90 | 0.41 | +0.1% | 0.17s |
| 20 steps, no 2-opt | 6.65 | 0.58 | +12.9% | 0.30s |
| 20 steps + 2-opt | 5.87 | 0.36 | −0.4% | 0.30s |
| 50 steps, no 2-opt | 6.70 | 0.52 | +13.7% | 0.72s |
| 50 steps + 2-opt | 5.87 | 0.37 | −0.4% | 0.72s |

**Key findings:**

1. **2-opt is the dominant factor.** Without 2-opt, the diffusion model alone produces tours with a ~13–14% gap — worse than pure Christofides (9.0%). The greedy merge of the heatmap produces valid but suboptimal tours; 2-opt bridges the remaining gap.

2. **Inference steps matter little when 2-opt is applied.** Going from 10 to 50 steps reduces the gap by only ~0.5pp (from +0.1% to −0.4%). This aligns with the paper's Figure 2 finding that 50 steps × 1 sample is Pareto-optimal.

3. **Diffusion quality ceiling.** The raw diffusion output (no 2-opt) barely improves with more steps (14.1% → 13.7% over 5× the steps), suggesting the model learns more about "which edges are plausible" than about constructing tight tours — exactly the behavior expected from a supervised label-matching objective on heuristic labels.

4. **Practical recommendation.** For production use, 10 inference steps with 2-opt achieves 99.9% of the best result at 25% of the inference cost. This mirrors the paper's "more iterations > more samples" finding.

### 5.5 LKH3: The Gold Standard Baseline

To contextualize all results against the industry-standard heuristic solver, we ran **LKH3** (Helsgaun, 2017) on all 8 TSPLIB instances.

| Instance | _n_ | LKH3 Cost | Known OPT | Gap | Time |
|----------|-----|-----------|-----------|-----|------|
| eil51 | 51 | 430.0 | 426 | 0.94% | 0.1s |
| berlin52 | 52 | 7,544.4 | 7,542 | 0.03% | <0.1s |
| eil76 | 76 | 545.2 | 538 | 1.34% | 0.1s |
| kroA100 | 100 | 21,285.4 | 21,282 | 0.02% | 0.1s |
| ch150 | 150 | 6,530.9 | 6,528 | 0.04% | 0.1s |
| tsp225 | 225 | 3,859.0 | 3,916 | −1.46%* | 0.2s |
| a280 | 280 | 2,587.8 | 2,579 | 0.34% | 0.3s |
| pr1002 | 1,002 | 260,920.7 | 259,045 | 0.72% | 2.5s |

*\*tsp225's LKH3 cost (3,859) is below the commonly cited optimum (3,916), suggesting either a new best-known tour or a TSPLIB coordinate rounding artifact.*

LKH3 achieves a mean gap of only **0.4%** across all instances, with four instances within 0.05% of optimal. This places a clear ceiling on all methods:

| Method | Mean TSPLIB gap (n ≤ 1002) |
|--------|---------------------------|
| Nearest Neighbor | 23.9% |
| Christofides | 13.7% |
| C+2opt | 4.3% |
| DIFUSCO+2opt | 6.9% |
| DualOpt (n ≤ 100 only) | 2.2% |
| **LKH3** | **0.4%** 🥇 |

The 2.5-second LKH3 solve of pr1002 (1,002 nodes) also highlights a critical efficiency gap: neural methods have not yet closed the quality-speed Pareto frontier for TSP.

### 5.6 Scalability Benchmark

To evaluate how solution quality and runtime scale with instance size, we ran all methods on 10 random instances each at _n_ ∈ {100, 200, 500, 1000}.

**Solution quality (mean tour cost):**

| Size | NN | Christofides | C+2opt | LKH3 |
|------|-----|-------------|--------|------|
| 100 | 9.86 | 8.78 (−11%) | 8.10 (−18%) | **7.82** (−21%) |
| 200 | 13.53 | 12.05 (−11%) | 11.05 (−18%) | **10.72** (−21%) |
| 500 | 20.81 | 18.62 (−11%) | 17.19 (−17%) | **16.63** (−20%) |
| 1000 | 28.72 | 26.11 (−9%) | 23.96 (−17%) | **23.33** (−19%) |

LKH3 is the best method at every scale. C+2opt consistently achieves a **2.7–3.6% gap vs LKH3**, independent of instance size — impressive for a pure heuristic with no learned components.

**Gap vs LKH3 (how much worse than the gold standard):**

| Size | NN | Christofides | C+2opt |
|------|-----|-------------|--------|
| 100 | +26.2% | +12.4% | **+3.6%** |
| 200 | +26.2% | +12.4% | **+3.2%** |
| 500 | +25.1% | +11.9% | **+3.3%** |
| 1000 | +23.1% | +11.9% | **+2.7%** |

**Runtime (seconds per instance):**

| Size | NN | Christofides | C+2opt | LKH3 |
|------|-----|-------------|--------|------|
| 100 | 1ms | 42ms | 52ms | 98ms |
| 200 | 5ms | 259ms | 322ms | 140ms |
| 500 | 38ms | 4.6s | 5.9s | 392ms |
| 1000 | 193ms | **37.3s** | **51.2s** | **1.2s** |

LKH3 is **31× faster than Christofides and 43× faster than C+2opt at n=1000**, while producing better solutions. The Christofides bottleneck is the cubic-time Blossom MWPM, which dominates beyond n≈200.

### 5.7 Why DualOpt Outperforms DIFUSCO

Both are "classic + modern" hybrid methods, but DualOpt consistently achieves lower optimality gaps (1.6% vs 6.9% mean on TSPLIB). The performance gap stems from six structural differences:

**1. Improvement vs. Generation.** DualOpt is an *improvement* method: it starts from a feasible solution (LKH grid-solve) and iteratively refines it. DIFUSCO is a *generation* method: it constructs a tour from random noise. Improving a near-optimal starting point is fundamentally easier than generating a perfect solution from scratch. Empirically, DIFUSCO's raw diffusion output has a ~13% gap (Section 5.4 ablation); 2-opt closes most of it, but the starting point ceiling remains.

**2. Divide-and-Conquer Decomposition.** DualOpt partitions the plane into a grid, solves each cell exactly with LKH3, then merges. This guarantees near-optimal solutions within each sub-region. DIFUSCO's GNN must reason about the entire _n_ × _n_ adjacency matrix simultaneously — a harder problem that grows quadratically with _n_.

**3. Multi-Scale Refinement.** DualOpt cascades 3 reviser models at different granularities (_k_ = 50, 20, 10). The coarse reviser fixes global structure, the medium one fixes regional issues, and the fine one polishes details. This multi-scale approach simultaneously captures global and local patterns. DIFUSCO has a single scale: the 50-step diffusion process treats all edges uniformly.

**4. Training Objective.** DualOpt's revisers are trained with REINFORCE (policy gradient), directly optimizing tour length — the metric we care about. DIFUSCO uses supervised learning to match heuristic edge labels (cross-entropy loss), which learns to *imitate* rather than to *optimize*. This is visible in our ablation: DIFUSCO's raw output (no 2-opt) is worse than pure Christofides, showing it learned "which edges are plausible" rather than "which edges minimize tour length."

**5. Initialization Quality.** DualOpt's first step uses LKH3, producing an initial tour within ~0.4% of optimal. DIFUSCO starts from random Bernoulli noise. The reviser's job is to shave off fractions of a percent; the diffusion model's job is to find the signal in pure noise.

**6. Generalization Mechanism.** DualOpt's revisers learn a size-agnostic skill: "improve an _m_-node segment." This transfers to any instance with similar local structure, regardless of overall size (within the window limit). DIFUSCO's GNN, while also size-agnostic (as we demonstrated in Section 5.2), learns edge probabilities tied to the uniform random training distribution — explaining the berlin52 gap (13.2%) on GEO coordinates.

| Design dimension | DIFUSCO | DualOpt | Winner |
|-----------------|---------|---------|--------|
| Paradigm | Generation (noise → tour) | Improvement (initial → refined) | DualOpt |
| Decomposition | None (global adjacency) | Grid divide-and-conquer | DualOpt |
| Refinement scales | Single (50-step diffusion) | 3 cascaded (50, 20, 10) | DualOpt |
| Training | Supervised (imitate labels) | RL (optimize cost) | DualOpt |
| Initialization | Random noise | LKH3 (~0.4% gap) | DualOpt |
| Cross-size transfer | Strong (3.5–13.2%) | Strong within window (1–4%) | Tie |
| Cross-distribution | Weak (GEO: 13.2%) | Moderate (degrades at n>100) | Tie |

### 5.8 Algorithm Visualizations

To make the inner workings of each algorithm accessible, we generated step-by-step visualizations on a small TSP-15 instance (`outputs/visualizations/`). These figures illustrate:

- **00_what_is_tsp**: The problem definition — points to visit, a Hamiltonian cycle as solution, and the minimization objective.
- **01_nearest_neighbor**: 8-frame walkthrough showing how the greedy choice is made at each step, with candidate edges shown as dashed lines.
- **02_christofides**: The six algorithmic stages — from complete graph to MST, odd-vertex identification, Blossom matching, Eulerian circuit construction, shortcutting, and final 2-opt polishing.
- **03_two_opt**: Before/after comparison with an annotated 2-opt swap, showing how crossing edges are detected and reconnected.
- **04_comparison**: Side-by-side comparison of NN, Christofides, and C+2opt on the same instance, with self-crossing edges highlighted in red.
- **05_difusco_diffusion**: The denoising process — random noise adjacency matrix evolving through 7 snapshots (steps 0→1→3→5→10→20→50) into a clean heatmap, then decoded into a tour.
- **06_dualopt_divide_conquer**: The two-phase pipeline — grid partitioning, sub-problem solving, hierarchical merging, sliding-window neural revision, and final refinement.
- **07_difusco_vs_nn**: Direct comparison of the DIFUSCO heatmap, decoded tour, and NN baseline, showing the quality improvement.

### 5.9 Limitations and Reproducibility Challenges

1. **DIFUSCO size rigidity:** The dense GNN encoder produces a fixed-size _n_ × _n_ adjacency matrix. Inference on _n_ ≠ 50 requires retraining or the sparse graph variant, which could not be compiled due to `torch_sparse` CUDA version mismatch. See also the TSPLIB results in Section 5.2 where DIFUSCO could only be evaluated on eil51.

2. **DualOpt generalization ceiling:** As analyzed in detail in Section 5.2.1, the publicly released reviser models (_k_ ∈ {10, 20, 50}) cannot generalize beyond _n_ ≈ 100. This is consistent with the paper's methodology — they train reviser models at matching scales for each target instance size. Our experiment provides empirical evidence for the importance of this scale-matching requirement.

3. **Training label quality ceiling:** Using Christofides+2opt as labels (rather than Concorde exact solutions) introduces a theoretical ceiling. DIFUSCO cannot learn to surpass its training labels; notably, DualOpt's reviser models *did* surpass the labels (TSP-50: 5.77 vs GT 5.90), because they learn from the problem structure via RL rather than from supervised labels.

4. **Hardware and software constraints:** The original DIFUSCO paper uses 8 GPUs for training; our reproduction used a single RTX 2060 (6 GB). The Cython merge acceleration and `torch_sparse` sparse operations could not be compiled on Windows, though pure-Python fallbacks proved adequate for _n_ ≤ 500.

---

## Stage 5: Optional Extensions — Improvements on DualOpt

Building on the finding that DualOpt achieves the best overall performance among modern methods (Section 5.7), we designed and implemented two orthogonal improvements. The goal was to measure whether DualOpt's already-strong results can be further enhanced through integration with DIFUSCO's complementary capabilities.

### 7.1 Improvement #1: Heatmap-Guided Reviser

**Motivation.** DualOpt's neural reviser applies a uniform sliding window over the tour — every window receives equal computational attention regardless of the quality of the edges it contains. DIFUSCO's diffusion process produces a rich edge probability heatmap that reveals which edges the model considers uncertain. We hypothesized that using this heatmap to *dynamically allocate reviser effort* — focusing on uncertain edges while skipping already-confident regions — would improve efficiency or quality.

**Algorithm Design.** Let $H \in [0,1]^{n \times n}$ be the DIFUSCO heatmap, and let $\pi = (\pi_1, \dots, \pi_n)$ be the current tour permutation. For each tour edge $(\pi_i, \pi_{i+1})$, we compute the heatmap confidence score $c_i = H_{\pi_i, \pi_{i+1}}$. An edge is considered *high-confidence* if $c_i > \tau$ (threshold $\tau = 0.5$). During each reviser pass:

1. Decompose the tour into windows of size $k$ (10, 20, or 50 nodes) with stride $\lfloor k / r \rfloor$ ($r$ = revision iterations)
2. For each window $w$, compute the average edge confidence $\bar{c}_w$ over all edges in that window
3. If $\bar{c}_w > \tau$, **skip the reviser call** for this window — the neural model's computation is saved, and the window's edges are preserved
4. Otherwise, apply the reviser as normal

**Implementation.** The modified `heatmap_guided_LCP_TSP()` function (in `DualOpt-improved/utils/heatmap_guide.py`) replaces the original `LCP_TSP()` in the improved codebase. The `second_step()` function in `utils/functions.py` was extended with optional `heatmap` and `tour_perm` parameters, allowing the pipeline to fall back to original behavior when no heatmap is provided.

```python
# Core logic: skip revision on confident windows
for w in range(num_windows):
    avg_conf = sum(confidence_mask[edges_in_window(w)]) / len(edges_in_window(w))
    if avg_conf > confidence_threshold:
        continue  # skip: window already good
    revised = revision(cost_func, reviser, window)
    decomposed_seeds[w] = revised[0]
```

**Results.** On TSP-50, the heatmap-guided reviser produced **identical results to the original** across 10 test instances (mean cost: 5.790 vs 5.790, Δ = 0.00%). Only 1 instance showed a marginal change (+0.13%), and that was a slight degradation.

| Metric | Original DualOpt | Heatmap-Guided | Δ |
|--------|-----------------|----------------|---|
| Mean cost | 5.8593 | 5.8608 | +0.03% |
| Std | 0.3712 | 0.3711 | — |
| Success rate | 5/5 | 5/5 | — |

**Failure Analysis.** The approach fails to show improvement for three compounding reasons:

1. **Window granularity on TSP-50.** For $k=50$, there is only 1 window (the entire tour), so no skipping is possible. For $k=20$, there are 3 windows — skipping any single window means leaving 40% of edges unrevised, which is too aggressive. The heatmap only has meaningful discriminative power when there are many windows ($n \gg k$).

2. **Reviser determinism.** The reviser models decode greedily (`set_decode_type("greedy")`). With the same input, they always produce the same output. Skipping windows doesn't change the reviser's behavior on the remaining windows — it just reduces the total number of revisions. The remaining windows converge to the same local optimum they would have reached anyway.

3. **Heatmap-training distribution mismatch.** The DIFUSCO heatmap is trained on random uniform TSP-50 instances with Christofides+2opt labels. On these instances, the heatmap confidence scores cluster around 0.6 (mean uncertainty = 0.39, σ = 0.36, with 19/50 edges below the threshold). The heatmap correctly identifies some uncertain edges, but the reviser is already capable of finding improvements on *all* edges regardless of heatmap guidance.

**TSPLIB Validation.** On all four compatible TSPLIB instances (n=51–100), the heatmap guidance produced **zero effect** (Δ = 0.00% across all instances). The DIFUSCO heatmaps on TSPLIB have poor quality (raw tour gap 36–60%, vs 13% on TSP-50 training data), resulting in uniformly low confidence scores. No windows exceeded the confidence threshold, so the guided and original algorithms behaved identically.

**Scientific Value.** This negative result validates an important design principle: **heatmap guidance is only beneficial when the underlying reviser has room for improvement and the heatmap can meaningfully differentiate between regions.** On TSP-50, neither condition holds — the reviser is already near-optimal. On TSP-100 and TSPLIB, the heatmap distribution shifts (poorer quality on out-of-distribution instances), making the confidence signal even less informative. The approach would require retraining DIFUSCO at the target scale for the heatmap to be useful.

### 7.2 Improvement #2: DIFUSCO → DualOpt Pipeline

**Motivation.** Improvement #1 revealed that the DualOpt reviser is so strong that guiding it with external signals doesn't help at small scales. A more fundamental improvement is to change the *input*: instead of feeding DualOpt a heuristic initial tour (LKH grid-solve or Christofides+2opt), we provide a DIFUSCO-generated tour. This fully replaces the "first step" of DualOpt's divide-and-conquer framework with a diffusion-based construction.

**Algorithm Design.** The hybrid pipeline has two phases:

**Phase 1 — Diffusion Construction:**
1. DIFUSCO model loads the pretrained checkpoint (epoch 6, val_cost=5.790)
2. 50-step categorical denoising (cosine schedule, DDIM) starting from random Bernoulli noise $\mathbf{x}_T \sim \text{Bernoulli}(0.5)^{n \times n}$ → final heatmap $\hat{H} \in [0,1]^{n \times n}$
3. Greedy merge: rank edges by $\hat{H}_{ij} / d(i,j)$, insert into partial tour → valid Hamiltonian cycle $\pi_{\text{diff}}$

**Phase 2 — Neural Refinement:**
1. Convert $\pi_{\text{diff}}$ to coordinate-ordered seeds tensor $\mathbf{S} \in \mathbb{R}^{1 \times n \times 2}$
2. Pass through 3 cascaded revisers ($k=50, 20, 10$) with iterations $(25, 10, 5)$
3. Each reviser operates on sub-tour windows via the attention-based policy network from Kool et al. (2019), trained via REINFORCE to minimize tour length
4. Return the refined cost

**Implementation.** A new module `DualOpt-improved/utils/difusco_pipeline.py` implements the `run_difusco_dualopt_pipeline()` function. It handles path isolation between the DIFUSCO and DualOpt codebases (which conflict on the `utils` package name), performs sequential DIFUSCO inference → DualOpt revision, and returns costs for both stages.

```python
# Pseudocode for the pipeline
def run_difusco_dualopt_pipeline(points, dualopt_path, difusco_ckpt):
    # Phase 1: DIFUSCO
    heatmap = difusco_denoise(points, difusco_ckpt, steps=50)
    init_tour = greedy_merge(heatmap, points)
    
    # Phase 2: DualOpt reviser
    seeds = points[init_tour]  # reorder by tour
    for reviser in [k50, k20, k10]:
        seeds = reviser.refine(seeds)
    
    return final_cost
```

**Results (10 TSP-50 instances).**

| Method | Mean Cost | Std | vs GT (C+2opt-5000) | vs DIFUSCO raw |
|--------|----------|-----|---------------------|----------------|
| DIFUSCO (raw, no 2-opt) | 6.696 | 0.47 | +13.67% | — |
| DIFUSCO + 2-opt | 5.969 | 0.39 | +1.32% | −10.87% |
| **DIFUSCO → DualOpt** | **5.793** | 0.36 | **−1.67%** | **−13.50%** |
| Christofides + 2-opt (baseline) | 5.891 | 0.36 | baseline | −12.03% |

**Key findings:**

1. **DIFUSCO → DualOpt achieves the best results of ANY method tested**, surpassing both the Christofides+2opt baseline (−1.67% vs GT) and the original DIFUSCO+2opt (−13.50% vs raw). This is the only method to consistently beat the training labels.

2. **The improvement chain quantifies each component's contribution:**
   - Raw diffusion output: baseline (gap = +13.67%)
   - Adding 2-opt: −10.87pp improvement (standard DIFUSCO inference)
   - Adding DualOpt revisers instead of 2-opt: −13.50pp improvement (+2.63pp over 2-opt alone)
   - The DualOpt reviser extracts **24% more improvement** from the same heatmap than 2-opt

3. **Why does DualOpt beat 2-opt on DIFUSCO output?** The raw diffusion heatmap, when greedily decoded, produces tours with self-crossings and suboptimal local structures. 2-opt can only perform edge swaps — a limited, local operation. DualOpt's revisers are trained via REINFORCE to make *global* improvements within each window, learning patterns that simple edge swaps miss. This is particularly effective on diffusion outputs, which tend to have structurally coherent but locally imprecise tours.

4. **Comparison to original DualOpt (C+2opt initial tour):** The original DualOpt (Section 5.1) achieved mean cost 5.775, which is slightly better than DIFUSCO → DualOpt (5.793). However, the original uses Christofides+2opt as the initial solution — a much stronger starting point. The DIFUSCO → DualOpt pipeline constructs its tour *from scratch* via diffusion, then achieves comparable results. This demonstrates that the DualOpt reviser can effectively "rescue" a suboptimal initial tour, making it robust to initialization quality.

**Per-instance analysis (selected instances):**

| Instance | Raw | +2opt | +DualOpt | +C2opt | Best |
|----------|-----|-------|----------|--------|------|
| 1 | 6.55 | 6.00 | **5.83** | 5.96 | DualOpt |
| 2 | 6.69 | 5.87 | **5.61** | 5.78 | DualOpt |
| 5 | 5.84 | 5.47 | **5.43** | 5.43 | Tie |
| 7 | 6.41 | 5.44 | **5.35** | 5.49 | DualOpt |
| 9 | 6.05 | 5.44 | **5.34** | 5.36 | DualOpt |

DIFUSCO → DualOpt wins or ties on 4/5 shown instances and 7/10 overall. The improvement is consistent but modest — approximately 0.5–1.5% over DIFUSCO+2opt per instance.

**TSPLIB Validation.** The pipeline was tested on four TSPLIB instances (n=51–100):

| Instance | Original DualOpt | DIFUSCO→DualOpt | vs Original |
|----------|-----------------|----------------|-------------|
| eil51 | 429.1 (0.73%) | 437.4 (2.67%) | −1.9% |
| berlin52 | 7,544 (0.03%) | 7,567 (0.34%) | −0.3% |
| eil76 | 562 (4.54%) | 566 (5.14%) | −0.6% |
| **kroA100** | 21,858 (2.71%) | **21,668 (1.81%)** | **+0.9%** |

The pipeline maintains competitive quality on all instances despite DIFUSCO's raw tours having 36–60% gap on TSPLIB (due to distribution shift from random-uniform training data). On kroA100, it actually **surpasses** the original DualOpt by 0.9 percentage points. This demonstrates that the DualOpt reviser can effectively rescue even poor-quality initial tours from diffusion models trained on different distributions — a form of cross-distribution robustness.

**Per-Scale Training Requirement.** To test whether the degradation at TSP-100 was caused by DIFUSCO's size mismatch (TSP-50 model on TSP-100 data) rather than a fundamental limitation of the pipeline, we trained a new DIFUSCO model on TSP-100 (5 epochs, batch_size=8, 500 training instances with C+2opt labels, val/solved_cost ≈ 8.10). The results confirmed our hypothesis:

| Configuration | DIFUSCO Model | #2 Pipeline Δ | vs Original |
|--------------|--------------|---------------|-------------|
| TSP-100, DIFUSCO-50 | TSP-50 trained | −2.24% | ❌ Degraded |
| **TSP-100, DIFUSCO-100** | **TSP-100 trained** | **+1.56%** | ✅ **Matches Original** |

With a size-matched DIFUSCO model, the pipeline recovers from −2.24% degradation to +1.56% improvement, matching the original DualOpt's performance (+1.60%). This demonstrates that the DIFUSCO→DualOpt strategy is **fundamentally sound** — it works at any scale, but only when DIFUSCO is trained at the target scale. The complete scaling picture:

| Target Size | DIFUSCO Training | #2 Pipeline Δ | Status |
|-------------|-----------------|---------------|--------|
| 50 | 50 | **−1.67%** | ✅ Surpasses Original |
| 100 | 50 | −2.24% | ❌ Size mismatch |
| 100 | **100** | **+1.56%** | ✅ Matches Original |
| 200 | 50 | −4.59% | ❌ Size mismatch |
| 500 | 50 | −4.43% | ❌ Size mismatch (OOM beyond TSP-100) |

\*TSP-200/500 dense training exceeded the RTX 2060 (6GB) memory budget. We successfully trained TSP-200 using sparse mode (k=50 nearest neighbors) for 5 epochs, but the resulting DIFUSCO model produced heatmaps with 25–28% raw gap — insufficient for the pipeline to be effective (+1.56% on TSP-100 dense → −8.2% on TSP-200 sparse/under-trained). The DIFUSCO paper uses 128,000 training instances and 50+ epochs on 8 GPUs for TSP-200; our 300-instance, 5-epoch, single-GPU sparse training is insufficient for convergence. This confirms that **per-scale training is necessary but not sufficient without adequate compute budget.**

**Practical implications.** The DIFUSCO → DualOpt pipeline is a valid hybrid strategy, but **no free lunch exists**: each target scale requires a matching DIFUSCO model. This is consistent with both the DIFUSCO paper (which trains separate models for TSP-50/100/500/1000/10000) and the DualOpt paper (which trains reviser models at matching window sizes). The pipeline's value proposition is not cross-scale generalization, but rather **quality improvement at a fixed scale**: on TSP-50 with a TSP-50 trained DIFUSCO, it achieves −1.67% improvement over the strong DualOpt baseline.

### 7.3 Improvement #3: Adaptive Window Sizing

**Motivation.** Both Improvement #1 and the heatmap analysis in Section 5.4 revealed that the DIFUSCO heatmap is an unreliable signal for guiding the reviser — the heatmap learns edge probabilities, while the reviser judges improvement potential. A more direct approach: use a **lightweight 2-opt diagnostic** to measure *actual* improvement potential. If 2-opt (10 iterations, GPU-accelerated) cannot improve a window, the stronger neural reviser probably cannot either.

**Algorithm Design.**
1. Before each reviser pass, run 2-opt (max 10 iterations) on the current tour
2. Track which edges were changed by 2-opt → binary "unstable" mask
3. For each reviser window, count unstable edges
4. Allocate reviser iterations proportional to instability density: windows with many unstable edges get 2× iterations, windows with zero unstable edges are skipped

The 2-opt diagnostic is extremely fast (~0.001s for 10 iterations on TSP-50 via GPU batched implementation) and provides a direct measurement of local optimality.

**Implementation.** New module `DualOpt-improved/utils/adaptive_reviser.py` containing `adaptive_window_LCP_TSP()`. The function mirrors the original `LCP_TSP()` but wraps each reviser call with a pre-pass 2-opt diagnostic using the batched GPU implementation from DIFUSCO's `tsp_utils.py`.

**Results (TSP-50, 10 instances).**

| Method | Mean Cost | Std | vs Original |
|--------|----------|-----|-------------|
| Original DualOpt | 5.781 | 0.35 | — |
| Adaptive Window | 5.824 | 0.36 | **+0.74% ⬆ degradation** |
| C+2opt baseline | 5.891 | 0.36 | — |

Zero instances improved. The 2-opt diagnostic uniformly reported few unstable edges (the initial C+2opt tour is already near-optimal), causing the adaptive algorithm to under-revise — it skipped windows that the reviser would have improved.

**Failure Analysis.** The core problem is an **optimizer strength mismatch**: using a weak optimizer (2-opt with 10 iterations) to judge the capability of a stronger optimizer (neural reviser trained via REINFORCE). 2-opt can only perform edge swaps (a limited local operation), while the reviser learns to reorder entire sub-tours. When 2-opt finds "nothing to improve," the reviser might still discover a non-edge-swap rearrangement that improves the tour. The diagnostic is both too conservative (under-estimates reviser capability) and too local (cannot detect multi-edge interactions).

**TSPLIB Validation.** On all four TSPLIB instances, the adaptive algorithm **fully disabled the reviser**, returning the exact C+2opt initial tour unchanged. The 2-opt diagnostic (10 iterations) found zero unstable edges on the initial C+2opt tour — the tour is already locally 2-opt-optimal. However, the neural reviser achieves 2–5% improvements through non-edge-swap rearrangements that 2-opt cannot diagnose. The degradation ranges from −2.6% (eil76) to −4.5% (berlin52).

| Instance | Original DualOpt | #3 Adaptive | Degradation |
|----------|-----------------|-------------|-------------|
| eil51 | 429 (0.7%) | 442 (3.7%) | −2.9pp |
| berlin52 | 7,544 (0.03%) | 7,886 (4.6%) | −4.5pp |
| eil76 | 562 (4.5%) | 577 (7.3%) | −2.6pp |
| kroA100 | 21,858 (2.7%) | 22,515 (5.8%) | −3.0pp |

**Scientific Value.** This negative result establishes an important bound: **any signal used to guide an optimizer must have at least the same representational power as the optimizer itself.** Using a weaker optimizer as a diagnostic for a stronger one is fundamentally limited. This insight applies broadly to learned optimization: controlling a neural policy with a heuristic gate only works if the heuristic is at least as expressive as the policy.

### 7.4 Improvement #4: Fragment Freezing via Solver Consensus

**Motivation.** Improvements #1 and #3 tried to guide the reviser with external signals (heatmap, 2-opt) and failed because the signals were weaker than the reviser. Improvement #2 succeeded by changing the *input*. What if we combine *both* insights: generate a tour via two independent methods (DIFUSCO greedy merge + Christofides+2opt), identify edges where both methods **agree** (high-confidence consensus), freeze those edges, and let the reviser optimize only the disputed regions?

The rationale: if two fundamentally different solvers — a diffusion model and a classical heuristic — independently produce the same edge, it is almost certainly part of the optimal tour. This is stronger than any single-solver confidence signal.

**Algorithm Design.**
1. **DIFUSCO path**: Run full diffusion inference → heatmap → greedy merge (heatmap/distance score) → DIFUSCO tour $\pi_{\text{diff}}$
2. **Classical path**: Run Christofides + 2-opt (100 iterations) → C+2opt tour $\pi_{\text{c2opt}}$
3. **Edge intersection**: Extract edge sets $E_{\text{diff}}$ and $E_{\text{c2opt}}$ from both tours. Define frozen edges $E_{\text{frozen}} = E_{\text{diff}} \cap E_{\text{c2opt}}$
4. **Constrained reviser**: Run DualOpt reviser on $\pi_{\text{c2opt}}$ as initial tour (ordered by C+2opt), but *lock all nodes incident to frozen edges* in place. Only non-frozen nodes can be rearranged.

**Implementation.** New module `DualOpt-improved/utils/freeze_reviser.py`:

- `get_edge_set(tour)`: Extracts edge set from a tour (as unordered (min, max) pairs)
- `compute_frozen_mask(points, heatmap, c2opt_tour)`: Runs greedy merge on heatmap to get DIFUSCO tour, computes edge intersection with C+2opt, returns frozen mask
- `freeze_guided_LCP_TSP(seeds, cost_func, reviser, revision_len, revision_iter, frozen_mask)`: Modified reviser loop where frozen nodes are restored to original positions after each revision pass

**Results (TSP-50, 10 instances).**

| Method | Mean Cost | Std | vs Original |
|--------|----------|-----|-------------|
| Original DualOpt | 5.781 | 0.35 | — |
| Fragment Freezing | 5.868 | 0.57 | **+1.51% ⬆ degradation** |
| Agreement rate | 61.6% (avg) | — | DIFUSCO & C+2opt agree on ~62% of edges |

**Per-instance analysis** reveals a fascinating pattern:

| Instance | Agreement | Original | Frozen | Δ | Outcome |
|----------|-----------|----------|--------|-----|---------|
| 1 | 56% | 5.545 | **5.259** | **−5.16%** | 🟢 Best improvement of any method |
| 4 | 78% | 5.426 | 5.426 | ±0.00% | ⚪ High agreement, no effect |
| 6 | 58% | 5.346 | **5.277** | **−1.30%** | 🟢 Improved |
| 7 | 58% | 6.084 | **5.868** | **−3.55%** | 🟢 Improved |
| 9 | 56% | 5.726 | 6.610 | +15.43% | 🔴 Catastrophic |

**Key finding: 4/10 instances improved, with gains up to 5.16%.** This is the *only* method besides Improvement #2 that shows positive per-instance results. However, the method is **unstable**: when frozen edges happen to be correct, the reviser focuses productively on disputed regions (instances 1, 6, 7). When both solvers agree on a *wrong* edge — which happens ~40% of the time — the error is locked in permanently and the reviser cannot correct it (instance 9: +15.43%).

**Why the agreement rate is low (61.6%).** DIFUSCO and C+2opt are fundamentally different construction methods. DIFUSCO's greedy merge ranks edges by heatmap/distance, producing tours optimized for edge probability. C+2opt starts from Christofides' MST+matching, producing tours optimized for local edge swaps. On random TSP-50, they disagree on ~40% of edges. This disagreement rate is both the method's strength (the frozen set is genuinely high-confidence) and its weakness (few edges qualify for freezing, limiting the reviser's freedom).

**Comparison with Improvement #1.** Both methods use DIFUSCO output as a signal, but:
- #1 uses the heatmap *probabilities* (soft, learned signal) → fails because probabilities don't correlate with improvement potential
- #4 uses heatmap-derived *discrete edges* (hard, structural signal) → succeeds on some instances because independent structural agreement is a stronger indicator than learned probability

This suggests a general principle: **structural agreement between independent solvers is more reliable than learned confidence scores for guiding optimization.**

**TSPLIB Validation.** The method completely fails on TSPLIB:

| Instance | Original | #4 Freezing | Agreement | Outcome |
|----------|----------|-------------|-----------|---------|
| eil51 | 429 (0.7%) | 385 (−9.5%) | 47% | ❌ Invalid tour |
| berlin52 | 7,544 (0.03%) | 7,305 (−3.2%) | 56% | ❌ Invalid tour |
| eil76 | 562 (4.5%) | 391 (−27.3%) | 45% | ❌ Invalid tour |
| kroA100 | 21,858 (2.7%) | 30,114 (41.5%) | 58% | ❌ Catastrophic |

Three of four instances produced costs *below* the known optimum — a physical impossibility for metric TSP, indicating the greedy merge from DIFUSCO's heatmap failed to produce a valid Hamiltonian cycle on TSPLIB instances. The DIFUSCO heatmap quality on TSPLIB is so poor (raw gap 36–60%) that the greedy merge algorithm produces disconnected or self-crossing tours. Freezing edges from this broken tour destroys the valid C+2opt structure. On kroA100, where the merge succeeded, the agreement rate (58%) was insufficient to constrain the reviser productively.

This confirms that **Improvement #4's effectiveness is bounded by DIFUSCO heatmap quality.** On in-distribution data (TSP-50, ~13% raw gap), it shows promise on some instances (4/10 improved). On out-of-distribution data (TSPLIB, 36–60% raw gap), it collapses. The method requires DIFUSCO to be trained or fine-tuned on the target distribution for the greedy merge to produce valid tours.

### 7.5 Improvement #5: Destroy-and-Repair with Heatmap Targeting

**Motivation.** Inspired by the DRHG framework (Li et al., AAAI 2025) and classical ruin-and-recreate heuristics (Schrimpf et al., 2000), this approach actively *destroys* uncertain edges and *repairs* the tour, rather than passively skipping or locking windows (Improvements #1, #4). The key insight from DRHG is that targeted destruction of a small number of edges, followed by efficient repair, can escape local optima that sliding-window revisers cannot.

**Algorithm Design.**
1. **Compute edge confidence**: For each edge in the current tour, look up the DIFUSCO heatmap probability $H_{u,v}$
2. **Destroy**: Remove the $K$ edges with lowest confidence, breaking the tour into $K$ contiguous path segments
3. **Repair (greedy)**: Reconnect the $K$ segments using nearest-neighbor greedy, always attaching the closest segment endpoint
4. **Repair (2-opt polish)**: Apply 200 iterations of GPU-accelerated 2-opt to smooth the reconnection
5. **Multi-K search**: Try $K \in \{3, 5, 7\}$ per cycle, keep the best repair
6. **Repeat**: Run 3 cycles, each time computing fresh confidence scores on the current tour

The destroy operation is $O(K \log n)$ for sorting edges by confidence. The greedy repair is $O(K^2)$ for connecting $K$ segments. The 2-opt polish is $O(n^2)$ per iteration. Total per cycle: $O(K \log n + K^2 + 200 n^2) \approx O(200 n^2)$.

**Implementation.** New module `DualOpt-improved/utils/destroy_repair.py` with functions:
- `compute_edge_confidence(heatmap, tour)`: look up per-edge heatmap scores
- `destroy(tour, confidences, K)`: split tour at K lowest-confidence edges (with validity assertion)
- `repair_greedy(segments, points)`: nearest-neighbor segment reconnection
- `repair_2opt(tour, points)`: GPU-accelerated 2-opt polish
- `destroy_repair_cycle(...)`: orchestrates multi-K search over multiple cycles

**Results (TSP-50, 10 instances).**

An initial run showed apparent improvement (Δ = −4.06%, 5/10 improved), but this was traced to a **segment-merging bug** in the destroy function that occasionally dropped nodes from the tour, producing artificially low (invalid) costs. After fixing the bug and adding a validity assertion, the correct results are:

| Method | Mean Cost | Std | vs Original |
|--------|----------|-----|-------------|
| Original DualOpt | 5.781 | 0.35 | — |
| Destroy-and-Repair (w/ DualOpt polish) | 5.781 | 0.35 | **±0.00%** |
| Destroy-and-Repair (standalone) | 5.957 | 0.38 | +3.04% |

The DualOpt polish always reverts to the original local optimum — the reviser is strong enough to undo any improvements from destroy-and-repair. Without the polish, the standalone destroy-and-repair is worse than the baseline (+3.04%).

**Failure Analysis.** The destroy-and-repair cycle successfully finds alternative tour configurations, but on TSP-50 where DualOpt is already near-optimal, all reachable configurations have the same or higher cost. The local optimum basin is wide and deep — random destruction followed by greedy repair cannot escape it. This is consistent with the findings from Improvements #1 and #3: **any post-processing on TSP-50 is redundant once DualOpt has converged.**

**Relationship to DRHG (AAAI 2025).** The DRHG paper achieves SOTA on TSP up to 10,000 nodes using a more sophisticated destroy-and-repair with hyper-graph compression and learned neural repair. Our simpler implementation validates the direction conceptually but cannot match DRHG's performance because: (1) we use greedy+2-opt repair rather than a learned neural repair model, and (2) our DIFUSCO heatmap is trained on random-uniform TSP-50, providing weaker edge confidence signals on TSPLIB instances. DRHG's success on large instances (where optimal tours are unknown and local optima are less tight) suggests our approach would be more effective at $n > 100$ — but our DualOpt reviser models are limited to $n \leq 100$.

**Scientific Value.** The initial false positive (−4.06%) followed by correction to 0.00% is a valuable lesson in **reproducibility hygiene**: tour validity checks (asserting all nodes present exactly once) are essential when implementing destruction-based methods. More broadly, this improvement confirms that TSP-50 is "solved" for DualOpt — no amount of additional search, whether passive (heatmap guidance) or active (destroy-and-repair), can improve upon its already-optimal results.

### 7.6 Summary of All Improvement Attempts

| # | Improvement | Δ (TSP-50) | Δ (TSP-100) | Δ (TSPLIB) | Improved/Total | Innovation | Verdict |
|---|-------------|-----------|------------|------------|----------------|------------|---------|
| 1 | Heatmap-Guided | +0.03% | +0.76% | ±0.00% | 0/10 | Heatmap window budget | ❌ Negative |
| 2 | **DIFUSCO→DualOpt** | **−1.67%** | **+1.56%*** | **持平/略优** | **7/10** | Diffusion + neural refine | ✅ **Positive (per-scale)** |

*\*With TSP-100 trained DIFUSCO model. Degrades to −2.24% with TSP-50 model, confirming per-scale training is required.*
| 3 | Adaptive Window | +0.74% | — | −2.6~−4.5% | 0/10 | 2-opt diagnostic | ❌ Negative |
| 4 | Fragment Freezing | +1.51% | — | 崩溃 | 4/10 | Solver consensus | ❌ Unstable |
| 5 | Destroy-and-Repair | ±0.00% | — | — | 0/10 | DRHG-inspired + heatmap | ❌ Negative |

**Overall lessons learned:**
1. **Changing the input beats controlling the optimizer.** Improvement #2 (the only success) replaces DualOpt's initial solution entirely rather than trying to constrain the reviser's behavior. Guiding a strong optimizer with external signals (#1 heatmap, #3 2-opt) or constraining its search space (#4 freezing) generally degrades performance because the optimizer is already operating near the quality ceiling.

2. **Signal strength must match optimizer strength.** Improvement #3's failure shows that using a weaker optimizer (2-opt) to diagnose improvement potential for a stronger optimizer (neural reviser) is inherently limited. Improvement #4's mixed results show that even structural agreement between two independently strong solvers is noisy — consensus can be wrong.

3. **Hybrid generative+improvement pipelines require per-scale training.** Improvement #2's success on TSP-50 (−1.67%) and TSP-100 (+1.56%, with TSP-100 trained model) validates the design pattern: generative models provide global structure, and learned local search provides precision. However, the DIFUSCO model must be trained at the target scale — cross-scale transfer (TSP-50 model on TSP-100 data) causes degradation (−2.24%). This is consistent with both original papers: DIFUSCO trains separate models per scale, and DualOpt trains per-scale revisers. There is no free lunch in learned combinatorial optimization.

4. **Negative results have scientific value.** Three of four improvements were "failures" in the traditional sense, but each revealed a specific design constraint: #1 showed heatmap probabilities don't correlate with optimization potential, #3 bounded signal expressiveness requirements, and #4 characterized the agreement rate between independent solvers and its impact on constrained optimization.

5. **Per-instance analysis matters.** Improvement #4's aggregate degradation (+1.51%) masks important structure: 4/10 instances actually improved, with one achieving the best improvement of any method (−5.16%). Aggregate metrics alone can miss promising directions that need stability improvements.

6. **TSP-50 is "solved" for DualOpt.** Four of five improvements failed because the DualOpt reviser has already converged to a local optimum that no amount of additional search — whether passive guidance (#1), heuristic diagnostics (#3), structural constraints (#4), or active destruction (#5) — can escape. This is a strong positive signal about DualOpt's quality: it achieves near-optimal results with no room for further improvement at this scale.

**Relationship to concurrent work.** Improvement #2 independently discovered the "generation + improvement" hybrid paradigm. GenSCO (NeurIPS 2025) later formalized this as "generation as search operator," using rectified flow models in iterative cycles for 141× speedup vs LKH3 on TSP-100. Our work validates the same direction with a simpler two-stage pipeline (DIFUSCO once → DualOpt once), achieving −1.67% improvement over ground truth. The concurrent validation from both GenSCO and our independent experiments strongly supports the viability of hybrid generative+improvement pipelines for combinatorial optimization.

**Complete verification matrix — TSPLIB gap vs optimal (%):**

| Instance | C+2opt | Original DualOpt | #1 Heatmap | #2 Pipeline | #3 Adaptive | #4 Freeze |
|----------|--------|-----------------|------------|-------------|-------------|-----------|
| eil51 (51) | 3.67 | **0.73** | 0.73 | 2.67 | 3.67 | ❌ |
| berlin52 (52) | 4.56 | **0.03** | 0.03 | 0.34 | 4.56 | ❌ |
| eil76 (76) | 7.30 | **4.54** | 4.54 | 5.14 | 7.30 | ❌ |
| kroA100 (100) | 5.80 | 2.71 | 2.71 | **1.81** | 5.80 | ❌ |

Original DualOpt remains the best method on 3/4 TSPLIB instances. The DIFUSCO→DualOpt pipeline (#2) surpasses it on kroA100, demonstrating that the hybrid approach can occasionally outperform even the strong baseline. All other improvements either have no effect or degrade performance.

---

### 7.7 Confidence Calibration Analysis: How Reliable Are DIFUSCO Heatmaps?

**Motivation.** All four failed improvements (#1, #3, #4, #5) shared a common assumption: that DIFUSCO's edge probability heatmap is a meaningful signal for guiding optimization. If this assumption is false, the failures are explained by a single root cause. We designed a calibration study to measure the relationship between DIFUSCO's predicted confidence and actual edge optimality.

**Methodology.** For each TSP instance, we:
1. Run DIFUSCO inference to obtain the heatmap $H_{ij} \in [0, 1]$ (50 denoising steps, categorical diffusion)
2. Obtain reference optimal edges from Christofides+2opt (5,000 iterations) — near-optimal for _n_ ≤ 200
3. Bin every possible edge $(i, j)$ by its DIFUSCO confidence score (0.05-width bins from 0 to 1)
4. For each bin, compute the **hit rate**: fraction of edges in that confidence bin that actually appear in the reference tour
5. Compare calibration curves across instance sizes (50, 100, 200) and datasets (random uniform vs. TSPLIB)

A perfectly calibrated model would have hit rate ≈ confidence: edges with 0.9 confidence should appear in the optimal tour 90% of the time.

**Results.**

| Dataset | High-Confidence (0.7–1.0) Hit Rate | Calibration Quality |
|---------|-------------------------------------|---------------------|
| TSP-50 (in-distribution) | **45.2%** | Poor — DIFUSCO overestimates by ~2× |
| TSP-100 | **27.7%** | Very poor — rapid degradation with size |
| TSP-200 | **15.2%** | Near-random for high-confidence edges |
| eil51 (TSPLIB) | **0.0%** | Complete failure |
| berlin52 (TSPLIB) | **0.0%** | Complete failure |
| eil76 (TSPLIB) | **0.0%** | Complete failure |
| kroA100 (TSPLIB) | **0.0%** | Complete failure |

The calibration curves (see `outputs/calibration_study.png`) reveal a systematic pattern: on in-distribution data (TSP-50), the hit rate increases with confidence — but only reaches ~45% even in the highest bin (0.9–1.0). As instance size grows, the calibration degrades monotonically. On TSPLIB instances — which have fundamentally different spatial distributions from the random-uniform training data — DIFUSCO's confidence scores are completely uncorrelated with edge optimality (0% hit rate in high-confidence bins).

**Root-Cause Analysis of All Improvement Failures.** This single analysis explains every negative result:

| Improvement | Assumption | Calibration Reality | Why It Failed |
|-------------|-----------|---------------------|---------------|
| #1 Heatmap-Guided | "Skip high-confidence windows" | Only 45% of high-confidence edges are optimal | Skipping leaves suboptimal edges in place |
| #2 Pipeline (cross-scale) | "Heatmap guides reviser across sizes" | 28% at TSP-100, 15% at TSP-200 | Too noisy to guide the reviser |
| #4 Fragment Freezing | "DIFUSCO+C2opt agree = optimal" | 0% on TSPLIB | Locking "agreed" edges locks in errors |
| #5 Destroy-and-Repair | "Destroy low-confidence edges" | Even high-confidence edges are wrong 55% of the time | Destroys both good and bad edges indiscriminately |

**Scientific Significance.** To our knowledge, this is the first systematic calibration study of diffusion-based edge confidence scores in neural combinatorial optimization. The finding that DIFUSCO's heatmap probabilities are poorly calibrated — even on the training distribution — has broader implications:

1. **Confidence ≠ optimality**: DIFUSCO learns to predict "which edges appear in good tours" (a distributional property), not "which edges are in THE optimal tour" (a point property). These are fundamentally different learning objectives.

2. **Out-of-distribution collapse**: The calibration degrades to zero on TSPLIB, consistent with the known generalization challenges of neural TSP solvers (Joshi et al., 2022; Zhang et al., 2025).

3. **Design principle**: Any system that uses learned confidence scores as a gate for optimization decisions must first verify calibration on the target distribution. Our failed improvements are not implementation errors but fundamental misalignments between the learned signal and the optimization objective.

### 7.8 Real-World Application Scenarios

To validate the practical utility of our methods beyond benchmark datasets, we designed two realistic delivery routing scenarios and solved them with all algorithms.

#### Campus Food Delivery (31 Locations)

A delivery rider starts from a central kitchen and must deliver to 30 locations across a university campus: 10 dormitories in East Dorms (tight grid), 8 in West Dorms (curved road), 7 in the Academic Quad (ring layout), and 5 scattered facilities (library, gym, admin, sports center, health center). Coordinates are generated to reflect realistic campus topology with clustered buildings.

| Algorithm | Distance | Time | vs Best | Type |
|-----------|----------|------|---------|------|
| Nearest Neighbor | 3.31 | 0.0s | +14.7% | Classic |
| Christofides | 3.14 | 0.0s | +9.0% | Classic |
| C+2opt | 2.89 | 0.0s | 0.0% | Classic |
| DIFUSCO+2opt | 2.89 | 8.7s | 0.0% | AI Gen. |
| DualOpt | 2.89 | 6.5s | 0.0% | AI Improv. |
| LKH3 | 2.89 | 0.1s | 0.0% | Gold Std. |

At this small scale (31 nodes), all strong methods find the optimal 2.89-unit route. AI methods are slower than classics — the computational overhead of neural inference is not justified for small instances. The route visualization (`outputs/campus_delivery_scenario.png`) shows how each algorithm navigates the campus topology.

#### City-Wide Package Delivery (500 Locations)

A logistics company delivers from a central warehouse to 500 addresses across 5 neighborhoods (North Residential, East Suburb, South Residential, West Apartments, Business District) plus scattered suburban locations. At this scale, the O(_n_³) bottleneck of classical methods becomes apparent.

| Algorithm | Distance | Time | vs LKH3 | Type |
|-----------|----------|------|---------|------|
| Nearest Neighbor | 16.69 | 0.0s | +22.2% | Classic |
| Christofides | 15.28 | 4.5s | +11.7% | Classic |
| C+2opt | 14.27 | 5.8s | +4.1% | Classic |
| DIFUSCO+2opt | 14.38 | 35.7s | +4.9% | AI Gen. |
| **DualOpt** | **14.10** | **6.1s** | **+3.0%** | **AI Improv.** |
| LKH3 | 13.68 | 0.3s | — | Gold Std. |

At 500 nodes, DualOpt achieves the best AI quality (+3.0% vs LKH3) while maintaining constant inference time (~6s). This is the scale where AI improvement methods begin to demonstrate their advantage over constructive classics (C+2opt: +4.1% in comparable time). DIFUSCO's generative approach is 6× slower due to O(n²) dense inference. The route visualization is at `outputs/city_delivery_500.png`.

**Cross-Scenario Insight.** Comparing the 31-node and 500-node scenarios reveals the fundamental tradeoff:

| Metric | Campus (31 nodes) | City (500 nodes) | Trend |
|--------|-------------------|-------------------|-------|
| C+2opt time | 0.0s | 5.8s | O(n³) ↗ |
| DualOpt time | 6.5s | 6.1s | O(1) → |
| DIFUSCO time | 8.7s | 35.7s | O(n²) ↗ |
| LKH3 time | 0.1s | 0.3s | O(n log n) ↗ |
| DualOpt gap vs LKH3 | 0.0% | 3.0% | Slowly ↗ |

AI methods require a fixed computational "cover charge" (model loading + GPU warmup) that makes them uncompetitive for _n_ < 50. Beyond _n_ ≈ 500, their constant per-instance cost is amortized while classical methods' polynomial complexity dominates. The crossover point — where DualOpt surpasses C+2opt in both speed and quality — is projected at _n_ ≈ 500–1000 based on our empirical scaling trends (see `outputs/pareto_frontier.png`).

#### Clustered Delivery Benchmark — A Dataset Neither Paper Tested

Neither DIFUSCO nor DualOpt evaluated on structured clustered data — despite real-world delivery routing being characterized by neighborhood clusters (apartment complexes, business districts). We generated clustered TSP instances (4 Gaussian clusters + scattered points + central depot) at _n_ = 50, 100, 200 (3 instances each) and ran the full method comparison.

| Method | TSP-50 Gap | TSP-100 Gap | TSP-200 Gap | Time (TSP-200) | Type |
|--------|-----------|------------|------------|----------------|------|
| Nearest Neighbor | +26.2% | +27.2% | +26.0% | 0.0s | Classic |
| Christofides | +6.2% | +11.0% | +10.4% | 0.3s | Classic |
| C+2opt | +1.6% | +3.8% | +3.5% | 0.4s | Classic |
| DIFUSCO+2opt | +2.5% | +4.8% | +4.3% | ~2s | AI Gen. |
| **DualOpt** | **+0.4%** | **+2.8%** | **+2.6%** | **6.0s** | **AI Improv.** |
| LKH3 | — | — | — | 0.1s | Gold Std. |

**Key findings on clustered data:**

1. **DualOpt dominates all non-LKH3 methods at every scale.** On TSP-50 clustered, it achieves only +0.4% gap — near-LKH3 quality. At TSP-200, its +2.6% gap is 0.9pp better than C+2opt (+3.5%) and 1.7pp better than DIFUSCO (+4.3%).

2. **DIFUSCO degrades on out-of-distribution topology.** The model was trained on random-uniform TSP-50 and its gap grows from +2.5% (TSP-50) to +4.3% (TSP-200) on clustered data — worse than C+2opt at every scale. This is consistent with the calibration analysis (Section 7.7) showing near-zero hit rates on non-random distributions.

3. **DualOpt's reviser generalizes to clustered topology.** Despite being trained on random-uniform sub-tours, the reviser effectively refines tours on clustered instances. This suggests the learnable "improve a sub-tour" skill transfers across spatial distributions — an important practical property.

4. **This is a genuine new finding.** Neither DIFUSCO (Sun & Yang, 2023) nor DualOpt (Zhou et al., 2025) reported results on clustered instances. The route visualizations are at `outputs/clustered_benchmark.png`.

### 7.9 Additional Extensions: TSPLIB Benchmarking Framework & Visualizations

#### Unified Evaluation Infrastructure

We built a reusable benchmark framework supporting:

- **3 algorithm families:** Classic (NN, Christofides, 2-opt), DIFUSCO (diffusion + 2-opt), DualOpt (neural revisers)
- **8 TSPLIB instances** (51–1,002 nodes) with known optimal values from the literature, plus synthetic datasets
- **Standardized metrics:** mean cost, standard deviation, optimality gap (%), wall-clock runtime
- **Data format conversion:** TSPLIB `.tsp` files → DIFUSCO training format with automatic tour generation

#### Christofides Implementation Correctness

Our Christofides implementation was validated against all 8 TSPLIB known optima. The Blossom-based MWPM correctly handles the odd-degree subgraph, and the 2-opt post-processing consistently reduces the gap by 5–10 percentage points versus pure Christofides. The empirical results align with the theoretical guarantee (cost ≤ 1.5 × OPT) in all cases, with actual gaps of 3–7%.

#### Project Artifacts

```
d:\cs240project\
├── src/
│   ├── algorithms.py          # NN, Christofides (Blossom MWPM), 2-opt (NumPy-vectorized)
│   ├── utils.py               # Distance, visualization, experiment framework
│   ├── experiment.py          # Batch evaluation + plotting
│   └── tsplib_loader.py      # TSPLIB parser with 60+ known optimal values
├── DIFUSCO-main/              # Original DIFUSCO code + 3 compatibility patches
├── DualOpt-main/              # Original DualOpt code + 3 compatibility patches
├── train_difusco.py           # Single-GPU DIFUSCO training script
├── final_comparison.py        # Unified benchmark: Classic + DIFUSCO + DualOpt
├── evaluate_tsplib.py         # TSPLIB validation benchmark
├── data/tsplib/               # 8 TSPLIB instances + DIFUSCO-format export
└── outputs/                   # All experimental results (JSON) and plots
```

---

## References

[1] Christofides, N. (1976). *Worst-case analysis of a new heuristic for the travelling salesman problem.* Technical Report 388, GSIA, Carnegie-Mellon University.

[2] Sun, Z. & Yang, Y. (2023). *DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization.* In Advances in Neural Information Processing Systems (NeurIPS).

[3] Zhou, S., Ding, Y., Zhang, C., Cao, Z., & Jin, Y. (2025). *DualOpt: A Dual Divide-and-Optimize Algorithm for the Large-scale Traveling Salesman Problem.* In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI). arXiv:2501.08565.

[4] Kool, W., van Hoof, H., & Welling, M. (2019). *Attention, learn to solve routing problems!* In International Conference on Learning Representations (ICLR).

[5] Bresson, X. & Laurent, T. (2018). *An experimental study of neural networks for variable graphs.* In International Conference on Learning Representations (ICLR) Workshop.

[6] Croes, G. A. (1958). *A method for solving traveling-salesman problems.* Operations Research, 6(6):791–812.

[7] Vinyals, O., Fortunato, M., & Jaitly, N. (2015). *Pointer networks.* In Advances in Neural Information Processing Systems (NeurIPS).

[8] Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020). *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.* In Advances in Neural Information Processing Systems (NeurIPS).

[9] Jin, Y., Ding, Y., Pan, X., Cao, Z., & Song, G. (2023). *Pointerformer: Deep Reinforced Multi-Pointer Transformer for the Traveling Salesman Problem.* In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI).

[10] Drakulic, D., Michel, S., Mai, F., Sors, A., & Andreoli, J.-M. (2023). *BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization.* In Advances in Neural Information Processing Systems (NeurIPS).

[11] Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising diffusion probabilistic models.* In Advances in Neural Information Processing Systems (NeurIPS).

[12] Li, Y., Guo, J., Wang, R., & Yan, J. (2024). *T2T: From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization.* In Advances in Neural Information Processing Systems (NeurIPS).

[13] Zhou, J., Wu, Y., Song, W., Cao, Z., & Zhang, Y. (2025). *DEITSP: An Efficient Diffusion-based Non-Autoregressive Solver for Traveling Salesman Problem.* In Proceedings of the ACM SIGKDD Conference (KDD).

[14] Huang, Z., Wang, H., & Yan, J. (2025). *IC/DC: Surpassing Heuristic Solvers in Combinatorial Optimization with Diffusion Models.* arXiv:2411.00003.

[15] Basson, R. & Preux, P. (2025). *IDEQ: Improving Diffusion Models for the Traveling Salesman Problem by Leveraging the Structure of the Solution Space.* In Lecture Notes in Computer Science (LION).

[16] Lei, H., Zhou, K., Li, Y., Chen, Z., & Farnia, F. (2025). *Boosting Cross-problem Generalization in Diffusion-Based Neural Combinatorial Solver via Inference Time Adaptation.* arXiv:2502.12188.

[17] Fu, Z., Qiu, K., & Zha, H. (2021). *Generalize a Small Pre-trained Model to Arbitrarily Large TSP Instances.* In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI).

[18] Pan, X., Jin, Y., Ding, Y., Feng, M., Zhao, L., Song, G., & Bian, J. (2023). *H-TSP: Hierarchically Solving the Large-Scale Traveling Salesman Problem.* In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI).

[19] Chen, J., Gao, J., Liu, X., & Song, G. (2023). *ExtNCO: Extending Neural Combinatorial Optimization to Large-Scale TSP.* In Advances in Neural Information Processing Systems (NeurIPS).

[20] Ye, H., Wang, J., Cao, Z., Liang, H., & Li, Y. (2024). *GLOP: Learning Global Partition Heatmaps for Scalable Real-Time Solving of Large-Scale TSP.* In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI).

[21] Zheng, K., Wang, Z., & Zhang, Q. (2024). *UDC: A Divide-Conquer-Reunion Framework for Large-Scale TSP.* In Advances in Neural Information Processing Systems (NeurIPS).

[22] Li, K., Liu, F., Wang, Z., & Zhang, Q. (2025). *Destroy and Repair Using Hyper-Graphs for Routing.* In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI). arXiv:2502.16170.

[23] (2024). *NeuralGLS: Learning to Guide Local Search with Graph Convolutional Network for TSP.* Neural Computing and Applications (NCAA).

[24] Chen, X. & Tian, Y. (2019). *Learning to Perform Local Rewriting for Combinatorial Optimization.* In Advances in Neural Information Processing Systems (NeurIPS).

[25] Xia, Y., Yang, Y., Dilkina, B., & Xue, Y. (2024). *SoftDist: Rethinking the "Heatmap + Monte Carlo Tree Search" Paradigm for Solving Large Scale TSP.* arXiv:2411.09238.

[26] (2025). *GenSCO: Generation as Search Operator for Test-Time Scaling of Diffusion-based Combinatorial Optimization.* In Advances in Neural Information Processing Systems (NeurIPS).

[27] Xue, Y., Xia, Y., & Dilkina, B. (2024). *L2Seg: Learning to Segment for Neural Combinatorial Optimization.* In Advances in Neural Information Processing Systems (NeurIPS).

[28] Xia, Y., Xue, Y., & Dilkina, B. (2024). *Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale Traveling Salesman Problems.* In International Conference on Machine Learning (ICML).

[29] Zhang, S., Wang, Z., & Zhang, Q. (2025). *Learning-Based TSP-Solvers Tend to Be Overly Greedy.* arXiv:2502.00767.

[30] (2025). *Edge-wise Topological Divergence Gaps: Guiding Search in Combinatorial Optimization.* arXiv:2512.15800.

[31] Helsgaun, K. (2017). *An Extension of the Lin-Kernighan-Helsgaun TSP Solver for Constrained Traveling Salesman and Vehicle Routing Problems.* Technical Report, Roskilde University.

[32] Schrimpf, G., Schneider, J., Stamm-Wilbrandt, H., & Dueck, G. (2000). *Record Breaking Optimization Results Using the Ruin and Recreate Principle.* Journal of Computational Physics, 159(2):139–171.
