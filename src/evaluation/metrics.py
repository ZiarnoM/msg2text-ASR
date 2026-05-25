import re

import jiwer


def normalize_polish_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-ząćęłńóśźż\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    return jiwer.cer(reference, hypothesis)


def compute_all_metrics(reference: str, hypothesis: str) -> dict:
    wer_raw = compute_wer(reference, hypothesis)
    cer_raw = compute_cer(reference, hypothesis)

    ref_norm = normalize_polish_text(reference)
    hyp_norm = normalize_polish_text(hypothesis)

    return {
        "wer_raw": round(wer_raw, 4),
        "cer_raw": round(cer_raw, 4),
        "wer_normalized": round(compute_wer(ref_norm, hyp_norm), 4),
        "cer_normalized": round(compute_cer(ref_norm, hyp_norm), 4),
    }
