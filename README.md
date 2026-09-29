<h1 align="center">What Do Current Systematic Generalization Tasks Miss? A Reasoning-Centered Analysis</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.19212"><img src="https://img.shields.io/badge/arXiv-2609.19212-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://github.com/BlueWhaleLab/TranSGrid"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/datasets/BlueWhaleLab/TranSGrid"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface" alt="Hugging Face Dataset"></a>
  <a href="https://huggingface.co/BlueWhaleLab/TranSGrid"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface" alt="Hugging Face Models"></a>
  <a href="https://huggingface.co/spaces/BlueWhaleLab/TranSGrid"><img src="https://img.shields.io/badge/TranSGrid-Demo-27BBF5?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAlCAMAAAAzx5qdAAAAwFBMVEUAAABEerMA//9VVao%2Bfbn///9VqqoAAP8Af3%2BX2%2Bt5tdd5t9lqorh0tdxvb2%2BStOV8uNlzrNVxrtV8w%2Bx2u%2BKByvR4vON5veNtr9l4u%2BGDyOuF0fqAudiIxuN7wOZ9wud%2BxuNLhLh9wehGgbuByO0zZswhPXE/v78lQngUH3VFaKI0Yp46bKgvYGtUirp/v/88cKtVqv9///8xWpJDeKw5ZJl/f/93t9dJeKhEebBEe7RIhMJDe7ZGgLxDe7RDd61p4GW7AAAAQHRSTlMAkgEDBgEDAQIXTjIVogIaaxlX/fv/zpDxc5HvY0zRrR%2BUcNWgBVMERB0MpPcLVgTnAwJdLRsCDhBss//7/NBOM2IPSQAAAuJJREFUeNqNk4mymkoQhn84LLK5DaKeNcldcxajosJAz8z7v9XtAa1ojLn5Cqih%2BL%2Bpnq4GPydEDHxjAhcWf2NfkMa4QYRsHh2XLtwROqqXDDf4huwhz5bL6acJcODoZLpcZvnDEjcYIRa5mM1E/jz/hGqeCyH4RSS4geuOcpEzbBXVYpZ3iMXEv32I8THFojitYqS4RTDM2LhgliHATXy/YuMyX2193MZHOhddMYuuHDEfgvO/NBAXgvOtzRcxXBe/JtxguOTW0ELkS2CD/2fjIxGLdV4k8AP8FhXGLS0SRPgZrn9Vpbc1pDLUuGab%2BtZJRxfSJtRE6T8BXO%2ByJn8LoKomx/k8UUMqRR/oOVy0cDIv8rwo5tMUCNDh/gVDbWNko7V2DHD3PZ9kQthJ4wEtvqSIgEHgAUYpKf9oO5Te9YYbTLwk%2B5JUk0kynRc8xs9x3/Taack4HCXdaFJKGTb4uuTpRQixxD41DnG%2B4Z1NDXhGK1I7YMBLpznjrXxi475ct0Rd3uG4OwCGjVINAEnHEhXf9kFv6YOY3ZfvSnV5wAvBD3eoiQxkSz/Srvd/svH3u%2BT6HQQDdHgwRA7Yujbe/70Xs5evUrXNfh%2BiJ0RNSkO3dI1i4%2BF5pVpdllz%2BERdaEaafNVF7bZT3K1JUlvuvOBNaQhQl08xR7OiLqt4%2BSClZ7hECJ15tSYiTOIpGq6aR53nH8B6yLGGPcHboBoc0tkSj8nthrZbSSuWajO0nLHd39ggGIaq4V1YN9WjJ/ae2QUmKnD1w8DxvAHDh%2BgNwT8bjaLrqhTUHlYM77LVSWtaw7BpFtOv7i2Q87pSnKfUobTCwRkNKUeM4jibGHCfPR8SKlZ4%2BH4t6tZ/sbfRpdOwmAXqGQJXE48fH8VjZQ0u4Nt8N3YdpNDGNrM8ne4v%2BFx2lZJUaA/TYTFDvdl63PufAjmWnFXfIO%2BBEEPTfvRA/EvpDP8Cr1KTkxXaD4eCU/g%2BfF5GlDnTZ4wAAAABJRU5ErkJggg==" alt="Hugging Face Space"></a>
</p>

Code for **TranSGrid**, introduced in [What Do Current Systematic Generalization Tasks Miss? A Reasoning-Centered Analysis](https://arxiv.org/abs/2609.19212).

## Overview

> To quickly understand the TranSGrid task and what models are asked to do, try our interactive demo on [Hugging Face Spaces](https://huggingface.co/spaces/BlueWhaleLab/TranSGrid).

TranSGrid is a grid-transformation testbed for studying systematic generalization. Given an initial board and a target board, a model must generate a sequence of actions that transforms one into the other. The action space contains ten operation types on rows, columns, and local 2 × 2 blocks.

The task brings together three reasoning demands:

- **Deduction:** Track how each action changes the board.
- **Induction:** Discover and reuse patterns in how actions interact.
- **Abduction:** Infer an action sequence from its desired outcome.

A prediction is correct if executing it on the initial board produces the target board. It does not need to be the shortest solution or match the reference sequence.

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Generate Datasets (Optional)](#generate-datasets-optional)
  - [Generate Train/Dev/Test Splits](#generate-traindevtest-splits)
  - [Generate TranSGrid Splits](#generate-transgrid-splits)
- [Train a Model (Optional)](#train-a-model-optional)
- [Evaluate](#evaluate)
- [Controlled Variants](#controlled-variants)
  - [TranSGrid (Decoupled)](#transgrid-decoupled)
  - [TranSGrid (SCAN)](#transgrid-scan)
- [Experimental Results](#experimental-results)
- [Citation](#citation)

## Installation

Note: This code requires `Python 3.10` or newer.

```bash
git clone https://github.com/BlueWhaleLab/TranSGrid.git
cd TranSGrid

conda create -p ./.conda python=3.12
conda activate ./.conda
pip install -r requirements.txt
```

## Data

All experiments in our paper use 6 × 6 boards whose cells contain digits from 0 to 9. Each JSONL row contains an initial board, a target board, and a reference action sequence. TranSGrid instances also include deductive, inductive, and abductive metadata.

| Path | Contents |
| --- | --- |
| `data/transgrid.jsonl` | 4,800 TranSGrid instances; 400 at each reference length from 1 to 12 |
| `data/transgrid_decoupled.jsonl` | Decoupled variant; lengths 1–9 are rewritten and lengths 10–12 are retained from the source |

The train, dev, and held-out Test splits are hosted on the [Hugging Face Dataset](https://huggingface.co/datasets/BlueWhaleLab/TranSGrid). To download all three splits into `data/`, run the following from the code repository's root directory:

```bash
hf download BlueWhaleLab/TranSGrid \
  train.jsonl dev.jsonl test.jsonl \
  --repo-type dataset \
  --local-dir data
```

The details of each split are summarized in the following table:

| Path after download | Instances | Reference sequence lengths | Distribution | Size |
| --- | ---: | --- | --- | --- |
| `data/train.jsonl` | 30,000,000 | 1–9 | Weighted sampling | 12.8 GB |
| `data/dev.jsonl` | 2,700 | 1–9 | 300 per length | 1.08 MB |
| `data/test.jsonl` | 4,800 | 1–12 | 400 per length | 2.03 MB |

## Generate Datasets (Optional)

The complete datasets are available on [Hugging Face](https://huggingface.co/datasets/BlueWhaleLab/TranSGrid). This section is for users who want to create their own datasets or generate additional examples; readers using the released data can skip it.

### Generate Train/Dev/Test Splits

Run these commands to generate `data/train.jsonl` (30 M instances), `data/dev.jsonl` (2,700 instances), and `data/test.jsonl` (4,800 instances):

```bash
python utils/data.py
python gen_test_split.py \
  --output data/test.jsonl \
  --per-length 400 \
  --lengths 1-12
```

### Generate TranSGrid Splits

Run these commands to regenerate `data/transgrid.jsonl` and `data/transgrid_decoupled.jsonl` (4,800 instances each):

```bash
python gen_transgrid.py \
  --balanced-count 4800 \
  --output data/transgrid.jsonl \
  --overwrite

python ablation/decoupling/make_decoupled.py \
  --data data/transgrid.jsonl \
  --output data/transgrid_decoupled.jsonl \
  --lengths 1-9 \
  --overwrite
```

## Train a Model (Optional)

The paper evaluates seven encoder–decoder Transformers. The following command trains the largest model using the GRID+PAIR encoding:

```bash
python train.py \
  --level level7 \
  --train data/train.jsonl \
  --eval data/dev.jsonl \
  --src-pos grid \
  --src-enc pair \
  --epochs 8 \
  --name level7_grid_pair
```

Change `--level` to `level1` through `level7` to select another model size. `best.pt` is selected using dev set solve rate. Add `--wandb` if you want to use Weights & Biases.


## Evaluate

Our trained checkpoints are available on [Hugging Face](https://huggingface.co/BlueWhaleLab/TranSGrid). The commands below use the GRID+PAIR Level 7 checkpoint. Download it from the repository root:

```bash
hf download BlueWhaleLab/TranSGrid grid_pair/level7.pt --local-dir checkpoints
```

If you trained your own model, change `--ckpt` in the commands below to your checkpoint path, such as `runs/level7_grid_pair/best.pt`.

Evaluate a checkpoint on TranSGrid with **greedy decoding**:

```bash
python eval.py \
  --ckpt checkpoints/grid_pair/level7.pt \
  --data data/transgrid.jsonl \
  --beams 1 \
  --batch-size 128 \
  --save results/TranSGrid/level7_grid_pair-greedy-bs128.jsonl \
  --group-by deductive_score inductive_score abductive_level stop_reason
```

For **Top@8**, use `--beams 8` and a different output filename:

```bash
python eval.py \
  --ckpt checkpoints/grid_pair/level7.pt \
  --data data/transgrid.jsonl \
  --beams 8 \
  --batch-size 128 \
  --save results/TranSGrid/level7_grid_pair-beam8-bs128.jsonl \
  --group-by deductive_score inductive_score abductive_level stop_reason
```

Top@8 counts an instance as solved if any of the eight candidates reaches the target board. Each saved JSONL row contains one selected prediction, its solve status, and the rank of the first successful candidate; it does not contain all eight candidates.

**To print saved results without decoding again:**

```bash
python report.py results/TranSGrid \
  --axes deductive_score inductive_score abductive_level stop_reason
```

Use `data/test.jsonl` with `eval.py` to evaluate on the held-out Test set.

## Controlled Variants

### TranSGrid (Decoupled)

Evaluate the decoupled dataset (Top@8):

```bash
python eval.py \
  --ckpt checkpoints/grid_pair/level7.pt \
  --data data/transgrid_decoupled.jsonl \
  --beams 8 \
  --batch-size 128 \
  --save results/ablation/decoupling/level7_grid_pair-decoupled-beam8-bs128.jsonl
```

### TranSGrid (SCAN)

Provide the first `L−1` reference actions as a decoder prefix and predict the final action (Top@8):

```bash
python ablation/action_explicit_goal/eval_prefix.py \
  --ckpt checkpoints/grid_pair/level7.pt \
  --data data/transgrid.jsonl \
  --beams 8 \
  --batch-size 128 \
  --save results/ablation/action_explicit_goal/level7_grid_pair-scan-beam8-bs128.jsonl
```
---


## Experimental Results

Solve rates (%) computed from the JSONL files in `results/Test/` and `results/TranSGrid/`. Each set contains 4,800 instances.

| Model | Parameters | Test Greedy | Test Top@8 | TranSGrid Greedy | TranSGrid Top@8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Level 1 | 0.96M | 25.81 | 36.38 | 16.10 | 20.25 |
| Level 2 | 1.89M | 35.79 | 51.54 | 18.10 | 23.77 |
| Level 3 | 7.45M | 61.85 | 64.12 | 26.83 | 33.02 |
| Level 4 | 11.13M | 71.27 | 72.50 | 32.06 | 40.85 |
| Level 5 | 44.29M | 73.90 | 75.15 | 37.73 | 47.90 |
| Level 6 | 59.00M | 78.33 | 78.90 | 50.40 | 58.56 |
| Level 7 | 88.43M | 79.63 | 80.17 | 55.31 | 61.60 |

## Citation

If you use TranSGrid, please cite:

```bibtex
@article{qi2026current,
  title={What Do Current Systematic Generalization Tasks Miss? A Reasoning-Centered Analysis},
  author={Qi, Chengwen and Ye, Deheng and Bian, Yatao},
  journal={arXiv preprint arXiv:2609.19212},
  year={2026}
}
```