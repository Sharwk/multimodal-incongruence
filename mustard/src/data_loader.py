"""Load MUStARD labels, audio features, and video paths into one structure."""
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MUSTARD = PROJECT_ROOT / "MUStARD"
LABELS_PATH = MUSTARD / "data" / "sarcasm_data.json"
AUDIO_PATH = MUSTARD / "data" / "audio_features.p"
VIDEO_DIR = MUSTARD / "videos" / "utterances_final"


def load_dataset():
    """Returns a list of dicts, one per clip."""
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(AUDIO_PATH, "rb") as f:
        audio = pickle.load(f, encoding="latin1")

    dataset = []
    missing_video = 0
    missing_audio = 0
    for clip_id, info in labels.items():
        video_path = VIDEO_DIR / f"{clip_id}.mp4"
        if not video_path.exists():
            missing_video += 1
            continue
        if clip_id not in audio:
            missing_audio += 1
            continue

        # Pool audio features across time -> single vector
        audio_vec = audio[clip_id].mean(axis=0)
        # Pad to consistent width (some clips have 11-18 features)
        if audio_vec.shape[0] < 18:
            audio_vec = np.pad(audio_vec, (0, 18 - audio_vec.shape[0]), mode="constant")
        elif audio_vec.shape[0] > 18:
            audio_vec = audio_vec[:18]

        dataset.append({
            "clip_id": clip_id,
            "utterance": info["utterance"],
            "speaker": info["speaker"],
            "show": info["show"],
            "label": int(info["sarcasm"]),  # 0 or 1
            "audio": audio_vec,
            "video_path": str(video_path),
        })

    print(f"Loaded {len(dataset)} clips")
    print(f"  missing video: {missing_video}")
    print(f"  missing audio: {missing_audio}")
    print(f"  sarcastic: {sum(d['label'] for d in dataset)}")
    print(f"  non-sarcastic: {sum(1 - d['label'] for d in dataset)}")
    return dataset


def split_dataset(dataset, test_size=0.2, seed=42):
    """Stratified train/test split by label."""
    labels = [d["label"] for d in dataset]
    train, test = train_test_split(
        dataset, test_size=test_size, stratify=labels, random_state=seed
    )
    print(f"Train: {len(train)} | Test: {len(test)}")
    return train, test


if __name__ == "__main__":
    dataset = load_dataset()
    train, test = split_dataset(dataset)

