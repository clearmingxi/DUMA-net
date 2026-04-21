# UBP-NCT Public Release

Public code release for EEG-to-vision retrieval experiments built around:

- a legacy training entry point: `EEG_retrieval.py`
- a config-driven wrapper: `run_experiment.py`
- experiment scripts for P1, P2, and P3
- result collection utilities

This release is intentionally cleaned for GitHub:

- no local dataset paths
- no CLIP checkpoints
- no generated outputs or logs
- no cached feature tensors

## What is included

- core training code
- dataset loading code
- experiment configs
- P1 / P2 / P3 launcher scripts
- result collection scripts
- a template `data_config.json`

## What you need to provide

Before running experiments, you need local access to:

1. the EEG dataset
2. the training/test image directories
3. local CLIP checkpoints for the backbones you want to use

Edit `data_config.json` and `configs/base.yaml` to point to your local paths.

## Recommended environment

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Depending on your CUDA / PyTorch setup, you may want to install PyTorch separately first.

## Minimal setup

1. Edit `data_config.json`
2. Check `configs/base.yaml`
3. Run a dry-run:

```bash
python run_experiment.py --config configs/within_subject_baseline.yaml --dry-run
```

## Example commands

P1 minimal matrix:

```bash
bash scripts/run_p1_minimal_matrix.sh --gpu cuda:0 --epochs 100
```

P2 backbone matrix:

```bash
bash scripts/run_p2_backbone_matrix.sh --mode both --gpu cuda:0 --epochs 100 --seeds "3047"
```

P3 encoder-backbone matrix:

```bash
bash scripts/run_p3_encoder_backbone_matrix.sh --gpu cuda:0 --epochs 100
```

## Notes

- The repository keeps the original training logic largely unchanged.
- `run_experiment.py` is the preferred entry point for reproducible runs.
- `tools/collect_results.py` and the table builders are used for normalized result summaries.

## Repository layout

```text
.
├── base/
├── configs/
├── scripts/
├── subject_layers/
├── tools/
├── utils/
├── EEG_retrieval.py
├── eegdatasets.py
├── efficient_encoder.py
├── experiment_utils.py
├── run_experiment.py
└── data_config.json
```

## Citation

This repository includes a `CITATION.cff` file for GitHub citation metadata.

## License

This release is distributed under the MIT License. See `LICENSE`.
