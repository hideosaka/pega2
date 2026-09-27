# Rank-Adaptive Local Empirical Processes in Low-Rank Attention

This repository contains the numerical simulation code accompanying the paper **"Rank-Adaptive Local Empirical Processes in Low-Rank Attention"** (Submitted for Double-Blind Review).

---

## Overview

This repository reproduces the key theoretical curves and capacity contraction bounds presented in the paper:
1. **Figure 1**: Effective dimension $p_{\mathrm{eff}}(s) = s(d_{\mathrm{out}} + d_{\mathrm{in}} - s)$ and capacity reduction percentage as attention weight matrix rank $s$ collapses ($d_{\mathrm{in}} = d_{\mathrm{out}} = 4096$).
2. **Figure 2**: Accelerated contraction rate of the local Rademacher complexity $\mathfrak{R}_n(\mathcal{F}(r, s))$ as excess risk $r \to 0$ across various rank constraints $s \in \{1, 4, 16, 64, 4096\}$.

---

## Getting Started

### Prerequisites

The code requires standard Python packages for scientific computing:
```bash
pip install numpy matplotlib seaborn
```

### Running in Google Colab / Local Machine

Simply run the standalone simulation script:

```bash
python colab_rank_adaptive_simulation.py
```

Upon completion, the script generates two high-resolution PNG figures in the working directory.
- `fig1_rank_dimension_reduction.png`
- `fig2_rademacher_shrinkage.png`

---

## Repository Structure

```text
.
├── colab_rank_adaptive_simulation.py  # Self-contained Python reproduction script
├── requirements.txt                   # Required Python libraries
└── README.md                          # Repository documentation
```

---

## Anonymity Notice
To preserve the double-blind review process, this repository contains no author names, institutional affiliations, or identifiable personal metadata. An anonymous link to this repository is generated via [Anonymous GitHub](https://anonymous.4open.science/).

