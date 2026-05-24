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
    # print(f"Converted {input_path} -> {output_path}")


def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def get_duration(path):
    return librosa.get_duration(path=path)
