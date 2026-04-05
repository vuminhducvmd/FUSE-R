# FUSE — Fused Unified centrality Score Estimation (R version)

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2511.22959"><img src="https://img.shields.io/badge/arXiv-2511.22959-AD1C18" style="margin-right: 5px;"></a>
</div>

An R implementation of the **Fused Unified centrality Score Estimation (FUSE)**, combining **global data centrality learning** and **local denoising score matching** through a shared encoder with two scalar heads.

---

## Algorithm (high-level)

<p align="center">
  <img src="https://raw.githubusercontent.com/vuminhducvmd/FUSE/main/docs/assets/fuse_overview.png" width="100%">
</p>

FUSE maps inputs through a **shared encoder** followed by two scalar heads:
a *global head* trained via anchor-based comparisons, and a *local head* trained
via denoising score matching.

At inference time, a homotopy function \( f(x, t) \) interpolates between the two,
producing an integrated centrality score.

---

## Examples

The example script [`examples/example.R`](examples/example.R) demonstrates FUSE on both
synthetic and real datasets.

<p align="center">
  <img src="https://raw.githubusercontent.com/vuminhducvmd/FUSE/main/docs/assets/gaussian_mixture.png" width="100%">
</p>

On synthetic distributions (e.g. Gaussian mixtures), FUSE visualizes how centrality
evolves from **global** to **local** as the interpolation parameter \( t \) varies.

---

## Repository structure

```
FUSE/
├── R/
│   ├── FUSE.R          # Main FUSE model and training logic
│   └── utils.R         # Visualization utilities
├── examples/
│   └── example.R       # Runnable demonstrations
├── DESCRIPTION         # Package metadata
├── NAMESPACE           # Exported functions
└── README.md           # This file
```

---

## Installation

### From source

Clone the repository:

```bash
git clone https://github.com/yourusername/FUSE-R.git
cd FUSE-R
```

Install the package in development mode:

```r
devtools::install()
```

Or install normally:

```r
install.packages(".", repos = NULL, type = "source")
```

### Dependencies

Make sure you have the required R packages:

```r
install.packages(c("torch", "ggplot2", "umap", "MASS", "patchwork"))
```

For torch, you may need to install it separately:

```r
install.packages("torch")
torch::install_torch()
```

---

## Usage

```r
library(FUSE)
library(torch)

# Create model
model <- FUSE(input_dim = 2, hidden = c(32, 32))

# Generate data
X <- torch_randn(1000, 2)

# Compute dissimilarities (optional)
D <- torch_cdist(X, X, p = 2)

# Fit model
history <- model$fit(X, dissimilarity_matrix = D)

# Inference
scores <- model$inference(X, t = 0.5)  # t between 0 (global) and 1 (local)
```

---

## Citation

If you use FUSE in your research, please cite:

```bibtex
@article{vu2025trainable,
  title={A Trainable Centrality Framework for Modern Data},
  author={Vu, Minh Duc and Liu, Mingshuo and Zhou, Doudou},
  journal={arXiv preprint arXiv:2511.22959},
  year={2025}
}
```