from __future__ import annotations

from pathlib import Path


def build_markdown_report(results: dict) -> str:
    if results.get("mode") == "batch":
        agg = results["aggregate"]
        lines = [
            "# Resonance probe batch report",
            "",
            "## Run",
            "",
            f"- tool: `{results['tool']}` `{results['version']}`",
            f"- model type: `{results['model_type']}`",
            f"- resonance repo: `{results['resonance_repo']}`",
            f"- checkpoint: `{results['checkpoint'] or 'none'}`",
            f"- wav dir: `{results['wav_dir']}`",
            f"- files: `{agg['num_files']}`",
            "",
            "## Aggregate metrics",
            "",
            f"- mean MSE: `{agg['mean_mse']:.6f}`",
            f"- mean SNR (dB): `{agg['mean_snr_db']:.6f}`",
            f"- mean byte accuracy: `{agg['mean_byte_accuracy']:.6f}`",
            f"- mean top-5 byte accuracy: `{agg['mean_top_5_accuracy']:.6f}`",
            f"- mean within/cross error ratio: `{agg['mean_within_to_cross_ratio']:.6f}`",
            f"- mean processed isotropy: `{agg['mean_processed_isotropy']:.6f}`",
            f"- mean processed curvature: `{agg['mean_processed_curvature']:.6f}`",
            "",
            "## Interpretation",
            "",
            "This is an aggregate summary over a directory of WAV files.",
            "Per-file metrics remain available in the JSON artifact under `per_file`.",
        ]
        return "\n".join(lines) + "\n"

    byte_probe = results["byte_probe"]
    geom = results["geometry_probe"]

    lines = [
        "# Resonance probe report",
        "",
        "## Run",
        "",
        f"- tool: `{results['tool']}` `{results['version']}`",
        f"- model type: `{results['model_type']}`",
        f"- resonance repo: `{results['resonance_repo']}`",
        f"- checkpoint: `{results['checkpoint'] or 'none'}`",
        f"- wav: `{results['wav'] or 'none'}`",
        f"- eval source: `{results['eval_source']}`",
        f"- params: `{results['params']}`",
        "",
        "## Metrics",
        "",
        f"- MSE: `{results['mse']:.6f}`",
        f"- SNR (dB): `{results['snr_db']:.6f}`",
        f"- byte accuracy: `{byte_probe['byte_accuracy']:.6f}`",
        f"- top-5 byte accuracy: `{byte_probe['topk_accuracy']['top_5']:.6f}`",
        f"- within/cross error ratio: `{byte_probe['error_analysis']['within_to_cross_ratio']:.6f}`",
        f"- processed isotropy: `{geom['processed_isotropy']:.6f}`",
        f"- processed curvature: `{geom['processed_curvature']:.6f}`",
        f"- processed intrinsic dim: `{geom['processed_intrinsic_dim']}`",
        "",
        "## Interpretation",
        "",
        "This report is best read as a probe artifact, not as a claim of model quality in isolation.",
        "Byte metrics summarize mu-law quantized waveform prediction behavior, while geometry metrics summarize oscillator-state structure before and after processing layers.",
    ]
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown_report(results))
