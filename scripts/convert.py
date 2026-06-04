"""Transcribe a single voice message to text."""

import argparse
import json
import sys

sys.path.insert(0, ".")

import src.models  # noqa: F401 — registers models

from src.asr_interface import get_model, list_models
from src.preprocessing import detect_format, get_duration, load_audio


def main():
    parser = argparse.ArgumentParser(description="Convert a voice message to text.")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--model", default="distil-whisper-pl", help="ASR model to use")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Show per-segment details")
    parser.add_argument("--list-models", action="store_true", help="Show available models and exit")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:", ", ".join(list_models()))
        return

    fmt = detect_format(args.audio_file)
    duration = get_duration(args.audio_file)
    audio = load_audio(args.audio_file)
    sr = 16000

    model = get_model(args.model)

    print(f"File:   {args.audio_file}")
    print(f"Format: {fmt}")
    print(f"Model:  {args.model}\n")

    # Model handles chunking internally (faster-whisper VAD, Wav2Vec2 sliding window)
    result = model.transcribe(audio, sr)

    if args.output == "json":
        out = {
            "file": args.audio_file,
            "model": args.model,
            "duration_s": round(duration, 1),
            "inference_time_s": result.inference_time_s,
            "confidence": result.confidence,
            "text": result.text,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in result.segments],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(result.text)

    if args.verbose and result.segments:
        print(f"\nSegments ({len(result.segments)}):")
        for seg in result.segments:
            print(f"  [{seg.start:.1f}s — {seg.end:.1f}s] {seg.text}")

    print(f"\nInference: {result.inference_time_s:.1f}s | Confidence: {result.confidence:.3f}")


if __name__ == "__main__":
    main()
