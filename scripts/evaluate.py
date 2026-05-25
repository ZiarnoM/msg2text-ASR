"""Batch evaluation: compare ASR models on a voice message dataset."""

import argparse
import sys

# Allow running from project root
sys.path.insert(0, ".")

from src.data.custom_dataset import CustomVoiceDataset
from src.evaluation.runner import evaluate_all_models


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR models on a voice message dataset.")
    parser.add_argument("manifest", help="Path to manifest.json")
    parser.add_argument("--models", default="distil-whisper-pl", help="Comma-separated model names")
    parser.add_argument("--output-dir", default=None, help="Directory for per-model JSON results")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    dataset = CustomVoiceDataset(args.manifest)

    print(f"Evaluating {len(dataset)} samples with {len(model_names)} model(s)...\n")

    summaries = evaluate_all_models(model_names, dataset, args.output_dir)

    header = f"{'Model':<28} {'WER raw':>8} {'CER raw':>8} {'WER norm':>8} {'CER norm':>8} {'Time':>7} {'Conf':>6}"
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s['model']:<28} "
            f"{s['avg_wer_raw']:>7.2%} "
            f"{s['avg_cer_raw']:>7.2%} "
            f"{s['avg_wer_normalized']:>8.2%} "
            f"{s['avg_cer_normalized']:>8.2%} "
            f"{s['avg_inference_time_s']:>5.1f}s "
            f"{s['avg_confidence']:>5.3f}"
        )

    best = min(summaries, key=lambda s: s["avg_wer_normalized"])
    print(f"\nBest WER (normalized): {best['model']} ({best['avg_wer_normalized']:.2%})")


if __name__ == "__main__":
    main()
