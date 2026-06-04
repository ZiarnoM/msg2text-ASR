#!/usr/bin/env python3
"""Demo script: multi-model ASR comparison for live presentation."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.models  # noqa: F401 — registers models
from src.asr_interface import get_model, list_models
from src.preprocessing import get_duration, load_audio

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "signal_voice", "results")
SEPARATOR = "=" * 70
SEP_THIN = "-" * 70


def print_banner():
    print()
    print(SEPARATOR)
    print("  POLISH VOICE MESSAGE ASR  --  Model Comparison Demo")
    print("  NLP Laboratory · Semester 6 · June 2026")
    print(SEPARATOR)


def show_summary():
    """Print pre-computed summary from all evaluated models."""
    print("\nRESULTS SUMMARY (6 voice messages, MacBook M1 CPU)\n")
    print(f"  {'Model':<30} {'WER norm':>10} {'CER norm':>10} {'Time':>8}")
    print(f"  {SEP_THIN:<30} {SEP_THIN:<10} {SEP_THIN:<10} {SEP_THIN:<8}")

    models = ["whisper-large-v3", "whisper-medium", "whisper-small", "distil-whisper-pl", "wav2vec2-xlsr"]
    results = []

    for name in models:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        s = data["summary"]
        results.append((name, s))

    results.sort(key=lambda r: r[1]["avg_wer_normalized"])
    best_wer = results[0][1]["avg_wer_normalized"]

    for name, s in results:
        marker = ">" if s["avg_wer_normalized"] == best_wer else " "
        print(
            f"{marker} {name:<28} "
            f"{s['avg_wer_normalized']:>8.1%} "
            f"{s['avg_cer_normalized']:>10.1%} "
            f"{s['avg_inference_time_s']:>6.1f}s"
        )

    print(f"\n  Best: {results[0][0]} (WER norm {results[0][1]['avg_wer_normalized']:.1%})")


def show_per_file():
    """Print per-file WER comparison table."""
    print("\nPER-FILE BREAKDOWN (WER normalized)\n")

    model_names = ["whisper-large-v3", "whisper-medium", "whisper-small", "distil-whisper-pl", "wav2vec2-xlsr"]
    model_data = {}

    for name in model_names:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            model_data[name] = json.load(f)

    file_labels = {
        0: "bede_po_12 (8s)",
        1: "jutro_plany (10s)",
        2: "paulina_o_ataku (30s)",
        3: "paulina_o_siatkówce (18s)",
        4: "paulina_wroclawska (3s)",
        5: "wczoraj_wieczorem (10s)",
    }

    header = f"  {'File':<28}"
    for name in model_data:
        header += f" {name:<14}"
    print(header)
    print(f"  {SEP_THIN:<28}" + SEP_THIN * len(model_data))

    for idx in range(6):
        row = f"  {file_labels[idx]:<28}"
        for name in model_data:
            wer = model_data[name]["samples"][idx]["wer_normalized"]
            tag = "GOOD" if wer < 0.15 else ("OK  " if wer < 0.35 else "BAD ")
            row += f" {wer:>5.1%} {tag} "
        print(row)


def live_demo(audio_path, models_to_run):
    """Run live transcription with selected models."""
    if not os.path.exists(audio_path):
        print(f"\nERROR: File not found: {audio_path}")
        return

    duration = get_duration(audio_path)
    audio = load_audio(audio_path, sr=16000)

    print(f"\nLIVE TRANSCRIPTION\n")
    print(f"  File:     {os.path.basename(audio_path)}")
    print(f"  Duration: {duration:.1f}s\n")

    for i, model_name in enumerate(models_to_run):
        print(f"  [{i+1}/{len(models_to_run)}] Loading {model_name}...", end=" ", flush=True)
        model = get_model(model_name)
        print("transcribing...", end=" ", flush=True)
        result = model.transcribe(audio, 16000)
        print(f"done ({result.inference_time_s:.1f}s)")
        print(f"  {SEP_THIN}")
        print(f"  {result.text}")
        print(f"  Confidence: {result.confidence:.3f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="ASR Demo — presentation script")
    parser.add_argument("--live", help="Path to an audio file for live demo")
    parser.add_argument("--models", default="distil-whisper-pl,whisper-small",
                        help="Models for live demo (comma-separated)")
    parser.add_argument("--all", action="store_true", help="Show full per-file breakdown")
    args = parser.parse_args()

    print_banner()

    if args.live:
        models = [m.strip() for m in args.models.split(",")]
        live_demo(args.live, models)
    else:
        show_summary()

        if args.all:
            show_per_file()

        print(f"\n  Live demo:  python scripts/demo.py --live data/signal_voice/raw/<file>")
        print(f"  Full view:  python scripts/demo.py --all")
        print()


if __name__ == "__main__":
    main()
