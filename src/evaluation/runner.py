import json
import os
import time

from ..asr_interface import get_model
from .metrics import compute_all_metrics


def evaluate_model(model_name: str, dataset, output_dir: str | None = None) -> dict:
    model = get_model(model_name)
    results = []

    for i in range(len(dataset)):
        audio, sr, reference = dataset[i]

        t0 = time.perf_counter()
        result = model.transcribe(audio, sr)
        elapsed = time.perf_counter() - t0

        metrics = compute_all_metrics(reference, result.text)

        results.append({
            "index": i,
            "reference": reference,
            "hypothesis": result.text,
            "inference_time_s": round(elapsed, 2),
            "confidence": result.confidence,
            **metrics,
        })

    summary = _aggregate(results, model_name)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{model_name}.json")
        with open(path, "w") as f:
            json.dump({"summary": summary, "samples": results}, f, indent=2, ensure_ascii=False)

    return summary


def evaluate_all_models(model_names: list[str], dataset, output_dir: str | None = None) -> list[dict]:
    summaries = []
    for name in model_names:
        summary = evaluate_model(name, dataset, output_dir)
        summaries.append(summary)
    return summaries


def _aggregate(results: list[dict], model_name: str) -> dict:
    n = len(results)
    return {
        "model": model_name,
        "samples": n,
        "avg_wer_raw": round(sum(r["wer_raw"] for r in results) / n, 4),
        "avg_cer_raw": round(sum(r["cer_raw"] for r in results) / n, 4),
        "avg_wer_normalized": round(sum(r["wer_normalized"] for r in results) / n, 4),
        "avg_cer_normalized": round(sum(r["cer_normalized"] for r in results) / n, 4),
        "avg_inference_time_s": round(sum(r["inference_time_s"] for r in results) / n, 2),
        "avg_confidence": round(sum(r["confidence"] for r in results) / n, 4) if results[0]["confidence"] else 0,
    }
