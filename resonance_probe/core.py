from __future__ import annotations

import importlib.util
import json
import logging
import math
import struct
import wave
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def save_wav(path: Path, signal: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = signal.detach().float().clamp(-1.0, 1.0).cpu()
    pcm = (signal * 32767.0).round().short().tolist()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_resonance_model(
    repo_path: Path,
    model_type: str,
    checkpoint: Path | None,
    device: torch.device,
    n_oscillators: int,
    n_layers: int,
    n_heads: int,
    sample_rate: int,
    hop: int,
):
    if model_type == "causal":
        mod = load_module("resonance_causal", repo_path / "causal.py")
        model_cls = mod.CausalOscillatorNetwork
        model = model_cls(
            n_oscillators=n_oscillators,
            n_layers=n_layers,
            n_heads=n_heads,
            sample_rate=sample_rate,
            hop=hop,
        )
    else:
        mod = load_module("resonance_oscillator", repo_path / "oscillator.py")
        model_cls = mod.OscillatorNetwork
        model = model_cls(
            n_oscillators=n_oscillators,
            n_layers=n_layers,
            n_heads=n_heads,
            sample_rate=sample_rate,
            hop=hop,
            causal=False,
        )

    if checkpoint and checkpoint.exists():
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
        logger.info("Loaded checkpoint: %s", checkpoint)

    return model.to(device).eval()


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        n = f.getnframes()
        sampwidth = f.getsampwidth()
        channels = f.getnchannels()
        raw = f.readframes(n)

    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV is currently supported")

    data = torch.tensor(struct.unpack(f"<{n * channels}h", raw), dtype=torch.float32)
    data = data.view(n, channels).mean(dim=1) / 32768.0
    data = data / (data.abs().max() + 1e-8)
    return data, sr


def resample_linear(signal: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return signal
    x = signal.view(1, 1, -1)
    target_len = int(signal.numel() * dst_sr / src_sr)
    y = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return y.view(-1)


def make_vowel(batch: int, length: int, sample_rate: int, device: torch.device):
    t = torch.arange(length, device=device, dtype=torch.float32) / sample_rate
    t = t.unsqueeze(0).expand(batch, -1)
    f0 = 80 + 220 * torch.rand(batch, 1, device=device)
    formants = torch.cat(
        [
            300 + 500 * torch.rand(batch, 1, device=device),
            800 + 1700 * torch.rand(batch, 1, device=device),
            2000 + 1500 * torch.rand(batch, 1, device=device),
        ],
        dim=1,
    )
    bandwidths = 50 + 100 * torch.rand(batch, 3, device=device)

    signal = torch.zeros(batch, length, device=device)
    for h in range(1, 25):
        freq = f0 * h
        amp = torch.ones(batch, 1, device=device)
        for i in range(3):
            fc = formants[:, i : i + 1]
            bw = bandwidths[:, i : i + 1]
            amp = amp / (1.0 + ((freq - fc) / bw) ** 2)
        phase = 2 * math.pi * torch.rand(batch, 1, device=device)
        signal = signal + amp * torch.sin(2 * math.pi * freq * t + phase)

    signal = signal / (signal.abs().max(dim=-1, keepdim=True).values + 1e-8)
    noisy = signal + 0.2 * torch.randn_like(signal)
    return signal, noisy


def mu_law_encode(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    x = x.clamp(-1.0, 1.0)
    fx = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)
    code = ((fx + 1.0) * 0.5 * mu).round().long()
    return code.clamp(0, mu)


def mu_law_decode(code: torch.Tensor, mu: int = 255) -> torch.Tensor:
    y = code.float() / mu * 2.0 - 1.0
    x = torch.sign(y) * (torch.expm1(y.abs() * math.log1p(mu)) / mu)
    return x


def mu_law_codebook(device: torch.device) -> torch.Tensor:
    return mu_law_decode(torch.arange(256, device=device))


def audio_byte_class(byte_val: int, codebook: torch.Tensor) -> str:
    amp = float(codebook[byte_val].item())
    if abs(amp) < 0.05:
        return "silence"
    if amp < -0.5:
        return "neg_high"
    if amp < -0.1:
        return "neg_mid"
    if amp <= 0.1:
        return "center"
    if amp <= 0.5:
        return "pos_mid"
    return "pos_high"


def pseudo_logits_from_wave(pred_wave: torch.Tensor, codebook: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return -((pred_wave.unsqueeze(-1) - codebook.view(1, 1, -1)) ** 2) / max(2 * sigma * sigma, 1e-6)


def intrinsic_dimensionality_mle(X: torch.Tensor, k: int = 10) -> float:
    if X.ndim != 2 or X.shape[0] < max(k + 1, 4):
        return float(X.shape[-1]) if X.ndim == 2 else 0.0
    dists = torch.cdist(X, X)
    dists.fill_diagonal_(float("inf"))
    knn, _ = dists.topk(min(k, X.shape[0] - 1), largest=False)
    knn = knn.clamp_min(1e-10)
    lr = torch.log(knn[:, -1:] / knn[:, :-1]).clamp_min(1e-10)
    d_hat = (knn.shape[1] - 1) / lr.mean(dim=-1)
    finite = d_hat[torch.isfinite(d_hat)]
    if finite.numel() == 0:
        return float("inf")
    return float(finite.median().item())


def isotropy(X: torch.Tensor) -> float:
    if X.ndim != 2 or X.shape[0] < 2:
        return 0.0
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.T @ X) / max(X.shape[0], 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    if eigvals.numel() < 2:
        return 0.0
    return float((eigvals.min() / eigvals.max()).item())


def representation_curvature(X: torch.Tensor, n_samples: int = 256) -> float:
    if X.ndim != 2 or X.shape[0] < 16:
        return 0.0
    idx = torch.randint(0, X.shape[0], (n_samples, 3), device=X.device)
    a, b, c = X[idx[:, 0]], X[idx[:, 1]], X[idx[:, 2]]
    midpoint = (a + c) / 2.0
    deviation = (midpoint - b).norm(dim=-1)
    baseline = (a - c).norm(dim=-1).clamp_min(1e-8)
    return float((deviation / baseline).mean().item())


def effective_rank(W: torch.Tensor) -> float:
    if W.ndim != 2:
        return 0.0
    sv = torch.linalg.svdvals(W.float())
    sv = sv[sv > 1e-10]
    if sv.numel() == 0:
        return 0.0
    p = sv / sv.sum()
    return float((-(p * p.log()).sum()).exp().item())


def oscillator_states(model, waveform: torch.Tensor, model_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if model_type == "causal":
            B, T = waveform.shape
            hop = model.hop
            n_frames = T // hop
            F_all = model.input_drive(waveform.unsqueeze(-1))
            F_input = F_all[:, : n_frames * hop].view(B, n_frames, hop, model.n_osc).mean(dim=2)
            coeffs = model.osc.precompute()
            x = model.x0.unsqueeze(0).expand(B, -1)
            v = model.v0.unsqueeze(0).expand(B, -1)
            states = []
            for i in range(n_frames):
                x, v = model.osc.step(x, v, F_input[:, i], coeffs)
                states.append(torch.cat([x, v], dim=-1))
            raw = torch.stack(states, dim=1)
            proc = raw
            for layer in model.process:
                proc = layer(proc)
            return raw[0].detach().float(), proc[0].detach().float()

        position, velocity = model.analyze(waveform)
        raw = torch.cat([position, velocity], dim=-1)[:, :: model.hop, :]
        proc = raw
        for layer in model.process:
            proc = layer(proc)
        return raw[0].detach().float(), proc[0].detach().float()


def evaluate_waveform_bytes(
    input_wave: torch.Tensor,
    target_wave: torch.Tensor,
    pred_wave: torch.Tensor,
    *,
    codebook: torch.Tensor,
    seq_len: int,
    n_seqs: int,
) -> dict[str, Any]:
    max_start = max(target_wave.numel() - seq_len - 1, 1)
    if n_seqs <= 0:
        starts = [0]
    else:
        stride = max(1, max_start // n_seqs)
        starts = list(range(0, min(max_start, stride * n_seqs), stride))
        if not starts:
            starts = [0]

    logits_all = []
    targets_all = []
    inputs_all = []

    for start in starts:
        inp = input_wave[start : start + seq_len]
        tgt = target_wave[start : start + seq_len]
        pred = pred_wave[start : start + seq_len]
        logits_all.append(pseudo_logits_from_wave(pred.view(1, -1), codebook))
        targets_all.append(mu_law_encode(tgt).view(1, -1))
        inputs_all.append(mu_law_encode(inp).view(1, -1))

    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    inputs = torch.cat(inputs_all, dim=0)
    preds = logits.argmax(dim=-1)

    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    byte_accuracy = (preds == targets).float().mean().item()

    topk_accuracy = {}
    for k in [1, 3, 5, 10, 20]:
        tk = logits.topk(k, dim=-1).indices
        topk_accuracy[f"top_{k}"] = (tk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

    entropy_by_context: dict[str, list[float]] = {}
    confusion: dict[str, dict[str, int]] = {}
    correct = within = cross = 0
    for b in range(inputs.shape[0]):
        for t in range(1, inputs.shape[1]):
            prev_cls = audio_byte_class(int(inputs[b, t - 1].item()), codebook)
            entropy_by_context.setdefault(f"after_{prev_cls}", []).append(float(entropy[b, t - 1].item()))

            true_cls = audio_byte_class(int(targets[b, t].item()), codebook)
            pred_cls = audio_byte_class(int(preds[b, t].item()), codebook)
            confusion.setdefault(true_cls, {})
            confusion[true_cls][pred_cls] = confusion[true_cls].get(pred_cls, 0) + 1

            tb = int(targets[b, t].item())
            pb = int(preds[b, t].item())
            if tb == pb:
                correct += 1
            elif true_cls == pred_cls:
                within += 1
            else:
                cross += 1

    confusion_pct = {}
    for true_cls, row in confusion.items():
        total = max(sum(row.values()), 1)
        confusion_pct[true_cls] = {pred_cls: count / total for pred_cls, count in row.items()}

    total = max(correct + within + cross, 1)
    return {
        "byte_accuracy": byte_accuracy,
        "bits_per_byte": float(math.log2(max(1e-8, 1.0 / max(byte_accuracy, 1e-8)))),
        "topk_accuracy": topk_accuracy,
        "entropy_by_context": {
            key: {"mean": sum(vals) / len(vals), "n": len(vals)}
            for key, vals in sorted(entropy_by_context.items())
            if len(vals) >= 4
        },
        "confusion": confusion_pct,
        "error_analysis": {
            "correct": correct / total,
            "within_class_error": within / total,
            "cross_class_error": cross / total,
            "within_to_cross_ratio": within / max(cross, 1),
        },
        "logit_geometry": {
            "logit_spread": float(logits.std().item()),
            "entropy_mean": float(entropy.mean().item()),
            "top1_confidence": float(probs.max(dim=-1).values.mean().item()),
            "effective_vocab": float(entropy.mean().exp().item()),
        },
    }


def geometry_summary(model, raw_states: torch.Tensor, proc_states: torch.Tensor) -> dict[str, Any]:
    weight_ranks = []
    for _, param in model.named_parameters():
        if param.ndim == 2 and min(param.shape) >= 8:
            weight_ranks.append(effective_rank(param.detach().float().cpu()))

    return {
        "raw_intrinsic_dim": intrinsic_dimensionality_mle(raw_states),
        "raw_isotropy": isotropy(raw_states),
        "raw_curvature": representation_curvature(raw_states),
        "processed_intrinsic_dim": intrinsic_dimensionality_mle(proc_states),
        "processed_isotropy": isotropy(proc_states),
        "processed_curvature": representation_curvature(proc_states),
        "state_delta_norm": float((proc_states - raw_states).norm(dim=-1).mean().item()),
        "mean_weight_effective_rank": float(sum(weight_ranks) / max(len(weight_ranks), 1)),
    }


def quick_train(
    model,
    model_type: str,
    *,
    steps: int,
    batch_size: int,
    length: int,
    sample_rate: int,
    device: torch.device,
) -> dict[str, float]:
    if steps <= 0:
        return {}
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    last_loss = 0.0
    for step in range(steps):
        clean, noisy = make_vowel(batch_size, length, sample_rate, device)
        pred = model(clean if model_type == "causal" else noisy)
        target = clean
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())
        if step == 0 or (step + 1) % max(steps // 5, 1) == 0:
            logger.info("quick-train step %d/%d loss=%.6f", step + 1, steps, last_loss)
    model.eval()
    return {"quick_train_final_mse": last_loss}


def auto_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": results["params"],
        "mse": results["mse"],
        "snr_db": results["snr_db"],
        "byte_accuracy": results["byte_probe"]["byte_accuracy"],
        "top_5_accuracy": results["byte_probe"]["topk_accuracy"]["top_5"],
        "within_to_cross_ratio": results["byte_probe"]["error_analysis"]["within_to_cross_ratio"],
        "processed_intrinsic_dim": results["geometry_probe"]["processed_intrinsic_dim"],
        "processed_isotropy": results["geometry_probe"]["processed_isotropy"],
        "processed_curvature": results["geometry_probe"]["processed_curvature"],
    }


def mean_metric(items: list[dict[str, Any]], key: str) -> float:
    vals = [float(item[key]) for item in items]
    return sum(vals) / max(len(vals), 1)


def aggregate_batch_results(per_file: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [summarize_results(item) for item in per_file]
    return {
        "num_files": len(per_file),
        "mean_mse": mean_metric(summaries, "mse"),
        "mean_snr_db": mean_metric(summaries, "snr_db"),
        "mean_byte_accuracy": mean_metric(summaries, "byte_accuracy"),
        "mean_top_5_accuracy": mean_metric(summaries, "top_5_accuracy"),
        "mean_within_to_cross_ratio": mean_metric(summaries, "within_to_cross_ratio"),
        "mean_processed_isotropy": mean_metric(summaries, "processed_isotropy"),
        "mean_processed_curvature": mean_metric(summaries, "processed_curvature"),
    }


def run_single_probe(args, model, device: torch.device, sample_name: str = "sample") -> dict[str, Any]:
    if getattr(args, "_skip_training", False):
        train_metrics = {}
    else:
        train_metrics = quick_train(
            model,
            args.model_type,
            steps=args.quick_train_steps,
            batch_size=args.train_batch_size,
            length=args.train_length,
            sample_rate=args.sample_rate,
            device=device,
        )

    if args.input_wav or args.target_wav or args.wav:
        input_path = args.input_wav or args.wav
        target_path = args.target_wav or args.wav or input_path
        if input_path is None or target_path is None:
            raise ValueError("WAV evaluation needs either --wav or both --input-wav and --target-wav")
        input_wave, input_sr = load_wav(input_path)
        target_wave, target_sr = load_wav(target_path)
        input_wave = resample_linear(input_wave, input_sr, args.sample_rate)[: args.eval_length]
        target_wave = resample_linear(target_wave, target_sr, args.sample_rate)[: args.eval_length]
        min_len = min(input_wave.numel(), target_wave.numel())
        input_wave = input_wave[:min_len]
        target_wave = target_wave[:min_len]
        eval_source = "wav_pair" if args.input_wav or args.target_wav else "wav"
    else:
        clean, noisy = make_vowel(1, args.eval_length, args.sample_rate, device)
        target_wave = clean[0].detach().cpu()
        input_wave = noisy[0].detach().cpu() if args.model_type == "oscillator" else clean[0].detach().cpu()
        eval_source = "synthetic_vowel"

    with torch.no_grad():
        model_in = input_wave.to(device).unsqueeze(0)
        pred_wave = model(model_in).squeeze(0).detach().cpu()

    codebook = mu_law_codebook(torch.device("cpu"))
    byte_results = evaluate_waveform_bytes(
        input_wave=input_wave,
        target_wave=target_wave,
        pred_wave=pred_wave,
        codebook=codebook,
        seq_len=args.seq_len,
        n_seqs=args.n_seqs,
    )

    raw_states, proc_states = oscillator_states(model, model_in, args.model_type)
    geom = geometry_summary(model, raw_states.cpu(), proc_states.cpu())

    mse = float(F.mse_loss(pred_wave, target_wave).item())
    snr_num = target_wave.pow(2).mean().item()
    snr_den = (pred_wave - target_wave).pow(2).mean().item() + 1e-10
    snr_db = float(10.0 * math.log10(max(snr_num, 1e-10) / snr_den))
    params = int(sum(p.numel() for p in model.parameters()))

    if args.save_audio_dir:
        sample_dir = args.save_audio_dir / sample_name
        save_wav(sample_dir / "input.wav", input_wave, args.sample_rate)
        save_wav(sample_dir / "target.wav", target_wave, args.sample_rate)
        save_wav(sample_dir / "prediction.wav", pred_wave, args.sample_rate)

    return {
        "tool": "resonance-probe",
        "version": "0.1.0",
        "sample_name": sample_name,
        "model_type": args.model_type,
        "resonance_repo": str(args.resonance_repo),
        "checkpoint": str(args.checkpoint) if args.checkpoint else "",
        "wav": str(args.wav) if args.wav else "",
        "input_wav": str(args.input_wav) if args.input_wav else "",
        "target_wav": str(args.target_wav) if args.target_wav else "",
        "eval_source": eval_source,
        "params": params,
        "sample_rate": args.sample_rate,
        "hop": args.hop,
        "quick_train_steps": args.quick_train_steps,
        "mse": mse,
        "snr_db": snr_db,
        "byte_probe": byte_results,
        "geometry_probe": geom,
        "train_metrics": train_metrics,
    }


def run_probe(args) -> dict[str, Any]:
    device = auto_device(args.device)
    torch.manual_seed(args.seed)
    logger.info("Device=%s model_type=%s repo=%s", device, args.model_type, args.resonance_repo)
    model = load_resonance_model(
        repo_path=args.resonance_repo,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=device,
        n_oscillators=args.n_oscillators,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        sample_rate=args.sample_rate,
        hop=args.hop,
    )

    if args.wav_dir:
        wav_paths = sorted(path for path in args.wav_dir.iterdir() if path.suffix.lower() == ".wav")
        if not wav_paths:
            raise ValueError(f"No .wav files found in {args.wav_dir}")

        per_file = []
        for idx, wav_path in enumerate(wav_paths):
            logger.info("batch item %d/%d: %s", idx + 1, len(wav_paths), wav_path.name)
            args.wav = wav_path
            args.input_wav = None
            args.target_wav = None
            args._skip_training = idx > 0
            result = run_single_probe(args, model, device, sample_name=wav_path.stem)
            per_file.append(result)

        return {
            "tool": "resonance-probe",
            "version": "0.1.0",
            "mode": "batch",
            "model_type": args.model_type,
            "resonance_repo": str(args.resonance_repo),
            "checkpoint": str(args.checkpoint) if args.checkpoint else "",
            "wav_dir": str(args.wav_dir),
            "sample_rate": args.sample_rate,
            "hop": args.hop,
            "quick_train_steps": args.quick_train_steps,
            "aggregate": aggregate_batch_results(per_file),
            "per_file": per_file,
        }

    args._skip_training = False
    return run_single_probe(args, model, device)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
