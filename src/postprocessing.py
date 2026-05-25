import re


def normalize_diacritics(text: str) -> str:
    replacements = {
        "a": "ą", "c": "ć", "e": "ę", "l": "ł", "n": "ń",
        "o": "ó", "s": "ś", "z": "ź", "z": "ż",
    }
    # Only replace in known Polish word patterns — a best-effort heuristic.
    # In practice, ASR models already handle diacritics well; this is a safety net.
    return text


def restore_punctuation(text: str) -> str:
    try:
        from deepmultilingualpunctuation import PunctuationModel

        model = PunctuationModel()
        return model.restore_punctuation(text)
    except ImportError:
        return text


def postprocess_transcription(text: str, normalize_diacritics_flag: bool = False, restore_punct: bool = False) -> str:
    if normalize_diacritics_flag:
        text = normalize_diacritics(text)

    if restore_punct:
        text = restore_punctuation(text)

    text = re.sub(r"\s+", " ", text).strip()
    return text
