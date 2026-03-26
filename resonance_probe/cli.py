from __future__ import annotations

import argparse
import logging
from pathlib import Path

from resonance_probe.core import run_probe, write_json
from resonance_probe.report import write_markdown_report


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone byte-level and geometry probe for Resonance models")
    parser.add_argument("--resonance-repo", type=Path, required=True)
    parser.add_argument("--model-type", choices=["oscillator", "causal"], default="oscillator")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-oscillators", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--hop", type=int, default=16)
    parser.add_argument("--quick-train-steps", type=int, default=80)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-length", type=int, default=1600)
    parser.add_argument("--eval-length", type=int, default=3200)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n-seqs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/resonance_probe.json"))
    parser.add_argument("--report-md", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_probe(args)
    write_json(args.output, results)
    logger.info("Saved JSON results to %s", args.output)

    if args.report_md:
        write_markdown_report(args.report_md, results)
        logger.info("Saved Markdown report to %s", args.report_md)

    logger.info(
        "summary: snr=%.4f dB byte_acc=%.6f top5=%.6f proc_iso=%.6f",
        results["snr_db"],
        results["byte_probe"]["byte_accuracy"],
        results["byte_probe"]["topk_accuracy"]["top_5"],
        results["geometry_probe"]["processed_isotropy"],
    )


if __name__ == "__main__":
    main()
