# resonance-probe

Standalone wav-native byte-level and geometry probe for Roland Sharp's `resonance` models.

`resonance-probe` is a small standalone CLI for evaluating waveform-native Resonance models with:

- byte-style waveform metrics via 8-bit mu-law quantization
- oscillator-state geometry diagnostics
- JSON output for automation
- optional Markdown reporting for sharing results

## What it does

- loads `oscillator.py` or `causal.py` directly from a local `resonance` checkout
- optionally loads a checkpoint
- optionally evaluates on a real WAV file or a separate input/target WAV pair
- optionally evaluates an entire directory of WAV files
- otherwise runs a self-contained synthetic-vowel sanity check
- quantizes predictions to 8-bit mu-law bytes
- reports byte-style metrics and oscillator-state geometry metrics
- writes a machine-readable JSON artifact and an optional Markdown report
- can save input, target, and predicted audio for manual inspection

## Install

```bash
cd /path/to/resonance-probe
python -m pip install -e .
```

## Requirements

- Python 3.11+
- PyTorch
- a local checkout of the `resonance` repo

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

Separate input and target WAVs:

```bash
resonance-probe \
  --resonance-repo /path/to/resonance \
  --model-type oscillator \
  --checkpoint /path/to/checkpoint.pt \
  --input-wav /path/to/noisy.wav \
  --target-wav /path/to/clean.wav \
  --save-audio-dir results/audio_debug \
  --output results/paired_probe.json
```

Batch a whole directory:

```bash
resonance-probe \
  --resonance-repo /path/to/resonance \
  --model-type oscillator \
  --checkpoint /path/to/checkpoint.pt \
  --wav-dir /path/to/wavs \
  --output results/batch_probe.json \
  --report-md results/batch_probe.md
```

## Example output

```json
{
  "tool": "resonance-probe",
  "version": "0.1.0",
  "model_type": "oscillator",
  "params": 70432,
  "mse": 0.2651,
  "snr_db": 0.0,
  "byte_probe": {
    "byte_accuracy": 0.0,
    "topk_accuracy": {
      "top_5": 0.00033
    }
  },
  "geometry_probe": {
    "processed_isotropy": 0.00019
  }
}
```

## Outputs

The JSON artifact includes:

- model metadata
- reconstruction metrics such as `mse` and `snr_db`
- byte metrics such as `byte_accuracy`, `topk_accuracy`, and confusion summaries
- geometry metrics such as intrinsic dimension, isotropy, curvature, and effective rank
- batch aggregate summaries when `--wav-dir` is used

## Notes

- WAV input currently expects 16-bit PCM.
- `--wav` treats one WAV as both input and target. Use `--input-wav` and `--target-wav` when those should differ.
- The intrinsic-dimension estimate can become unstable on highly degenerate representations.
- This repo intentionally avoids any private registry or experiment-tracking dependency so it is easy to open-source.

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
