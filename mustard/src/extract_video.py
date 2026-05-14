"""Extract video features by sampling frames and encoding with CLIP."""
import pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2

from data_loader import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)
OUT_PATH = FEATURES_DIR / "video_features.pkl"

NUM_FRAMES = 8  # frames sampled per clip
MODEL_NAME = "openai/clip-vit-base-patch32"

# Use MPS on Apple Silicon, fall back to CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def sample_frames(video_path, num_frames=NUM_FRAMES):
    """Sample evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # cv2 returns BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def main():
    dataset = load_dataset()
    print(f"Loading CLIP on device: {DEVICE}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    features = {}
    failed = []
    for i, d in enumerate(dataset):
        clip_id = d["clip_id"]
        frames = sample_frames(d["video_path"])
        if not frames:
            failed.append(clip_id)
            continue
        with torch.no_grad():
            inputs = processor(images=frames, return_tensors="pt").to(DEVICE)
            embs = model.get_image_features(**inputs)
            # Handle both tensor and wrapper-object returns across transformers versions
            if hasattr(embs, "image_embeds"):
                embs = embs.image_embeds
            elif hasattr(embs, "last_hidden_state"):
                embs = embs.last_hidden_state.mean(dim=1)
            pooled = embs.mean(dim=0).cpu().numpy()  # (512,)
        features[clip_id] = pooled
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(dataset)} processed")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(features, f)
    print(f"\nSaved {len(features)} video features to {OUT_PATH}")
    if failed:
        print(f"Failed to process {len(failed)} clips: {failed[:5]}...")
    print(f"Each feature shape: {next(iter(features.values())).shape}")


if __name__ == "__main__":
    main()