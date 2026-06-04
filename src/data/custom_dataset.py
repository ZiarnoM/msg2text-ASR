import json
import os

from .. import preprocessing


def load_manifest(manifest_path: str):
    with open(manifest_path) as f:
        return json.load(f)


class CustomVoiceDataset:
    def __init__(self, manifest_path: str):
        self.base_dir = os.path.dirname(manifest_path)
        self.entries = load_manifest(manifest_path)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio_path = os.path.join(self.base_dir, entry["file"])
        audio = preprocessing.load_audio(audio_path)
        return audio, 16000, entry["reference"]
