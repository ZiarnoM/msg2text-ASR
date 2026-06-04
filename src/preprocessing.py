import subprocess
import os

import librosa
import numpy as np
import soundfile as sf


def detect_format(path):
    ext = os.path.splitext(path)[1].lower()
    return ext.lstrip(".")


def convert_to_wav(input_path, output_path, sr=16000):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sr),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def load_audio(path, sr=16000):
    """Load any audio file as 16kHz mono float32 numpy array.

    Uses soundfile for WAV/FLAC, ffmpeg pipe for everything else (M4A, AAC, MP3, OGG).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".flac"):
        audio, actual_sr = sf.read(path, dtype="float32")
        if actual_sr != sr:
            audio = librosa.resample(audio, orig_sr=actual_sr, target_sr=sr)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio

    # ffmpeg pipe for all other formats
    cmd = [
        "ffmpeg", "-i", path,
        "-ar", str(sr), "-ac", "1",
        "-f", "f32le", "pipe:1",
        "-loglevel", "error",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError(f"ffmpeg produced empty output for: {path}")
    return audio


def get_duration(path):
    """Get audio duration in seconds using librosa."""
    return librosa.get_duration(path=path)
