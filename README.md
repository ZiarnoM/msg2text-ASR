# Polish Voice Message to Text (ASR)

Converts single-speaker Polish voice messages (Messenger, Signal, WhatsApp) to text. Compares multiple open-source ASR models on Polish spontaneous speech using WER/CER metrics.

**Course:** NLP Laboratory, semester 6 — final project

## Setup

```bash
# Clone and install
git clone <repo-url> && cd msg2text-ASR
pip install -r requirements.txt

# System dependency (for audio conversion)
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Linux
```

First model download happens automatically on first use — expect ~1.5–3 GB depending on the model.

## Quick Start

```bash
# Transcribe a single voice message
python scripts/convert.py data/signal_voice/raw/msg_001.m4a --model distil-whisper-pl

# JSON output with segments
python scripts/convert.py audio.m4a --model distil-whisper-pl --output json --verbose

# List available models
python scripts/convert.py --list-models
```

## Available Models

| Model | ID | Size | Speed | Notes |
|---|---|---|---|---|
| Distil-Whisper PL (CT2) | `distil-whisper-pl` | ~1.5 GB | Fast (CPU-friendly) | Polish fine-tuned, **recommended default** |
| Whisper Large V3 | `whisper-large-v3` | ~3 GB | Slow (GPU recommended) | Best overall accuracy |
| Whisper Medium | `whisper-medium` | ~1.5 GB | Medium | Size/accuracy trade-off |
| Whisper Small | `whisper-small` | ~500 MB | Fast | Lightweight, lower accuracy |
| Wav2Vec2 XLS-R | `wav2vec2-xlsr` | ~1.2 GB | Medium | Encoder-only architecture |

## Batch Evaluation

```bash
# Compare models on a custom voice message dataset
python scripts/evaluate.py data/signal_voice/manifest.json \
  --models distil-whisper-pl,whisper-small \
  --output-dir data/signal_voice/results/

# Download BIGOS V2 benchmark dataset
python scripts/download_bigos.py
```

### Manifest Format

Create `data/signal_voice/manifest.json`:

```json
[
  {
    "file": "normalized/msg_001.wav",
    "duration_s": 12.4,
    "speaker": "friend",
    "reference": "Cześć, co robisz jutro wieczorem?"
  }
]
```

## Project Structure

```
src/
├── preprocessing.py      # Audio loading, ffmpeg conversion, 16kHz resampling
├── chunking.py           # Silence-aware splitting, VAD, 30s chunking
├── asr_interface.py      # Abstract base class + model registry
├── postprocessing.py     # Diacritics normalization, punctuation restoration
├── models/
│   ├── whisper_large.py        # Whisper Large V3 via faster-whisper
│   ├── distil_whisper_pl.py    # Distil-Whisper PL via CTranslate2
│   ├── whisper_small_medium.py # Whisper Small & Medium (ablation study)
│   └── wav2vec2.py             # Wav2Vec2 XLS-R (encoder-only comparison)
├── data/
│   ├── bigos.py           # BIGOS V2 benchmark dataset loader
│   └── custom_dataset.py  # Custom voice message manifest loader
└── evaluation/
    ├── metrics.py         # WER, CER, Polish text normalization
    └── runner.py          # Batch evaluation across models
scripts/
├── convert.py             # Single-file transcription CLI
├── evaluate.py            # Multi-model comparison CLI
└── download_bigos.py      # BIGOS V2 dataset downloader
data/
└── signal_voice/
    ├── raw/                # Original .m4a files
    ├── normalized/         # Converted 16kHz .wav files
    ├── manifest.json       # File list + reference transcripts
    └── results/            # Per-model evaluation JSON + comparison CSV
```

## Running on a MacBook (No GPU)

- **Development:** use `distil-whisper-pl` — CTranslate2 runs well on CPU with int8 quantization
- **Light evaluation:** `whisper-small` is usable (~3–5× slower than realtime on short audio)
- **Full evaluation:** run on [Google Colab](https://colab.research.google.com) with a free T4 GPU — clone the repo, install dependencies, upload your `data/signal_voice/`, and run `scripts/evaluate.py` with all models

## Benchmarks

*[To be filled after running full evaluation]*

| Model | WER (raw) | WER (norm) | CER (norm) | Avg Time | Confidence |
|---|---|---|---|---|---|
| distil-whisper-pl | — | — | — | — | — |
| whisper-large-v3 | — | — | — | — | — |
| whisper-medium | — | — | — | — | — |
| whisper-small | — | — | — | — | — |
| wav2vec2-xlsr | — | — | — | — | — |
