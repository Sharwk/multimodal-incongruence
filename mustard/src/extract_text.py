"""Extract text embeddings for each clip's utterance using sentence-transformers."""
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from data_loader import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)
OUT_PATH = FEATURES_DIR / "text_features.pkl"


def main():
    dataset = load_dataset()
    print("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, 384-dim

    utterances = [d["utterance"] for d in dataset]
    clip_ids = [d["clip_id"] for d in dataset]

    print(f"Encoding {len(utterances)} utterances...")
    embeddings = model.encode(utterances, show_progress_bar=True, batch_size=32)

    features = {cid: emb for cid, emb in zip(clip_ids, embeddings)}

    with open(OUT_PATH, "wb") as f:
        pickle.dump(features, f)
    print(f"Saved {len(features)} text features to {OUT_PATH}")
    print(f"Each feature shape: {embeddings[0].shape}")


if __name__ == "__main__":
    main()