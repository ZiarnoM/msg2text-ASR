from datasets import load_dataset

BIGOS_DATASET_ID = "amu-cai/pl-asr-bigos-v2"


def load_bigos_subset(subset: str, split: str = "test"):
    return load_dataset(BIGOS_DATASET_ID, subset, split=split)


class BIGOSDataset:
    def __init__(self, subset: str, split: str = "test"):
        self.dataset = load_bigos_subset(subset, split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        reference = sample["sentence"]
        return audio, sr, reference
