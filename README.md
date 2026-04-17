# Reasoning for Hierarchical Classification (RHC)

This repository contains the code and data for the paper:

> **Reasoning for Hierarchical Text Classification: The Case of Patents**
> *ACL 2026 Findings* · [arXiv:2510.07167](https://arxiv.org/abs/2510.07167)

## Overview

RHC applies chain-of-thought reasoning to hierarchical text classification, using patent classification under the International Patent Classification (IPC) scheme as the primary benchmark. The model predicts IPC codes in a step-by-step manner — first the Section, then the Class, then the Subclass — and is trained with Group Relative Policy Optimisation (GRPO) using a reward signal that combines classification accuracy and output length shaping.

**IPC hierarchy:**

```
Section   (e.g. A)
  └─ Class      (e.g. A61)
       └─ Subclass   (e.g. A61K)
```

## Repository Structure

```
RHC/
├── data/
│   ├── train_us/      # USPTO training split (HuggingFace Arrow format)
│   ├── test_us/       # USPTO test split
│   └── test_eu/       # EPO test split (cross-jurisdictional evaluation)
├── reward.py          # GRPO reward function (accuracy + length shaping)
├── data.py            # Dataset loading example
└── README.md
```

## Data

The dataset is available on [HuggingFace Hub](https://huggingface.co/datasets/lj408/RHC-patents). Download and save it to the data folder as listed above or use the following code. 

```python
from datasets import load_dataset

ds = load_dataset("lj408/RHC-patents")
```

Each split is stored with the following fields:

| Field | Type | Description |
|---|---|---|
| `title` | string | Patent title |
| `abstract` | string | Patent abstract |
| `claims` | string | Patent claims |
| `main_ipcr_label` | string | Ground-truth IPC label |
| `filing_date` | string | Filing date |
| `year` | int | Filing year |
| `__uid` | string | Unique patent identifier |

**Splits:**
- `train` — USPTO patents (training)
- `test_us` — USPTO patents (in-distribution test)
- `test_eu` — EPO patents (out-of-distribution, cross-jurisdictional test)

## Reward Function

`reward.py` implements the GRPO reward used during training. It expects the model to produce a structured response with three hierarchical `\box{...}` decisions.

**Expected output format:**

```
Step 1 — Section ...
Brief Justification: ...
Decision: \box{A}

Step 2 — Class ...
Brief Justification: ...
Decision: \box{A61}

Step 3 — Subclass ...
Brief Justification: ...
Decision: \box{A61K}
```

## Citation

```bibtex
@article{jiang2025reasoning,
  title={Reasoning for Hierarchical Text Classification: The Case of Patents},
  author={Jiang, Lekang and Sun, Wenjun and Goetz, Stephan},
  journal={arXiv preprint arXiv:2510.07167},
  year={2025}
}
```

## License

The code is released under the CC-BY-NC-4.0 License. 
