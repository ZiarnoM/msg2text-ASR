"""Download and cache BIGOS V2 subsets."""

from datasets import load_dataset

BIGOS_ID = "amu-cai/pl-asr-bigos-v2"
SUBSETS = [
    "pelcra_pl_asr_pelcra_for_bigos",
    "common_voice_13_0_pl",
    "fleurs_pl",
    "voxpopuli_pl",
]


def download_all():
    for subset in SUBSETS:
        print(f"Downloading {subset}...")
        try:
            load_dataset(BIGOS_ID, subset, split="test")
            print(f"  -> OK")
        except Exception as e:
            print(f"  -> FAILED: {e}")


if __name__ == "__main__":
    download_all()
