import numpy as np
import torch


def split_audio(audio, sr, chunk_duration_s=30):
    chunk_len = chunk_duration_s * sr
    chunks = []
    for start in range(0, len(audio), chunk_len):
        chunk = audio[start:start + chunk_len]
        chunks.append(chunk)
    return chunks


def detect_speech_segments(audio, sr):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    get_speech_timestamps, _, _, _, _ = utils

    audio_tensor = torch.from_numpy(audio).float()

    timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=sr)
    segments = [(t['start'] / sr, t['end'] / sr) for t in timestamps]
    return segments


def chunk_on_silence(audio, sr):
    segments = detect_speech_segments(audio, sr)
    chunks = []
    for start_s, end_s in segments:
        start_sample = int(start_s * sr)
        end_sample = int(end_s * sr)
        chunks.append(audio[start_sample:end_sample])
    return chunks
