# resonance-probe

Standalone wav-native byte-level and geometry probe for Roland Sharp's `resonance` models.

## What it does

- loads `oscillator.py` or `causal.py` directly from a local `resonance` checkout
- optionally loads a checkpoint
- optionally evaluates on a real WAV file
- otherwise runs a self-contained synthetic-vowel sanity check
- quantizes predictions to 8-bit mu-law bytes
- reports byte-style metrics and oscillator-state geometry metrics
- writes a machine-readable JSON artifact and an optional Markdown report

## Install

```bash
cd /path/to/resonance-probe
python -m pip install -e .
```

## Usage

```bash
resonance-probe \
  --resonance-repo /path/to/resonance \
  --model-type oscillator \
  --quick-train-steps 80 \
  --output results/oscillator_probe.json \
  --report-md results/oscillator_probe.md
```

Checkpoint-backed run:

```bash
resonance-probe \
  --resonance-repo /path/to/resonance \
  --model-type causal \
  --checkpoint /path/to/checkpoint.pt \
  --wav /path/to/input.wav \
  --output results/causal_probe.json
```

## Outputs

The JSON artifact includes:

- model metadata
- reconstruction metrics such as `mse` and `snr_db`
- byte metrics such as `byte_accuracy`, `topk_accuracy`, and confusion summaries
- geometry metrics such as intrinsic dimension, isotropy, curvature, and effective rank

## Notes

- WAV input currently expects 16-bit PCM.
- The intrinsic-dimension estimate can become unstable on highly degenerate representations.
- This repo intentionally avoids any private registry or experiment-tracking dependency so it is easy to open-source.

## Before publishing

Licensed under Apache-2.0. See [LICENSE](LICENSE).
