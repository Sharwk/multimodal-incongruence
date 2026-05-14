"""Train per-modality classifiers and a late-fusion model on MUStARD."""
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from data_loader import load_dataset, split_dataset

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_features():
    with open(FEATURES_DIR / "text_features.pkl", "rb") as f:
        text = pickle.load(f)
    with open(FEATURES_DIR / "video_features.pkl", "rb") as f:
        video = pickle.load(f)
    return text, video


def build_matrix(items, text_feats, video_feats):
    """Stack features for a list of dataset items."""
    X_text, X_audio, X_video, y = [], [], [], []
    for d in items:
        cid = d["clip_id"]
        if cid not in text_feats or cid not in video_feats:
            continue
        X_text.append(text_feats[cid])
        X_audio.append(d["audio"])
        X_video.append(video_feats[cid])
        y.append(d["label"])
    return (
        np.stack(X_text), np.stack(X_audio), np.stack(X_video), np.array(y)
    )


def train_one(name, X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["sincere", "sarcasm"]))
    return clf, probs, acc


def main():
    dataset = load_dataset()
    train_items, test_items = split_dataset(dataset)
    text_feats, video_feats = load_features()

    Xt_tr, Xa_tr, Xv_tr, y_tr = build_matrix(train_items, text_feats, video_feats)
    Xt_te, Xa_te, Xv_te, y_te = build_matrix(test_items, text_feats, video_feats)

    print(f"\nTrain shapes: text {Xt_tr.shape}, audio {Xa_tr.shape}, video {Xv_tr.shape}")
    print(f"Test shapes:  text {Xt_te.shape}, audio {Xa_te.shape}, video {Xv_te.shape}")

    clf_text, probs_text, acc_text = train_one("TEXT only", Xt_tr, y_tr, Xt_te, y_te)
    clf_audio, probs_audio, acc_audio = train_one("AUDIO only", Xa_tr, y_tr, Xa_te, y_te)
    clf_video, probs_video, acc_video = train_one("VIDEO only", Xv_tr, y_tr, Xv_te, y_te)

    # Late fusion: equal-weighted average
    fused_probs = (probs_text + probs_audio + probs_video) / 3
    fused_preds = fused_probs.argmax(axis=1)
    acc_fused = accuracy_score(y_te, fused_preds)
    print(f"\n--- LATE FUSION (equal weights) ---")
    print(f"Accuracy: {acc_fused:.3f}")
    print(classification_report(y_te, fused_preds, target_names=["sincere", "sarcasm"]))

    # Weighted fusion: weight by per-modality accuracy
    w_text, w_audio, w_video = acc_text, acc_audio, acc_video
    total = w_text + w_audio + w_video
    weighted_probs = (
        probs_text * w_text + probs_audio * w_audio + probs_video * w_video
    ) / total
    weighted_preds = weighted_probs.argmax(axis=1)
    acc_weighted = accuracy_score(y_te, weighted_preds)
    print(f"\n--- LATE FUSION (accuracy-weighted) ---")
    print(f"Accuracy: {acc_weighted:.3f}")
    print(classification_report(y_te, weighted_preds, target_names=["sincere", "sarcasm"]))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_te, weighted_preds))

    # Video + text only (drops the weakest modality)
    vt_probs = (probs_video + probs_text) / 2
    vt_preds = vt_probs.argmax(axis=1)
    acc_vt = accuracy_score(y_te, vt_preds)
    print(f"\n--- VIDEO + TEXT only ---")
    print(f"Accuracy: {acc_vt:.3f}")

    print("\n=== SUMMARY ===")
    print(f"Text only:           {acc_text:.3f}")
    print(f"Audio only:          {acc_audio:.3f}")
    print(f"Video only:          {acc_video:.3f}")
    print(f"Fusion (equal):      {acc_fused:.3f}")
    print(f"Fusion (weighted):   {acc_weighted:.3f}")
    print(f"Video + Text only:   {acc_vt:.3f}")

    # Save classifiers for the Gradio app
    joblib.dump(clf_text, MODELS_DIR / "clf_text.joblib")
    joblib.dump(clf_audio, MODELS_DIR / "clf_audio.joblib")
    joblib.dump(clf_video, MODELS_DIR / "clf_video.joblib")
    print(f"\nSaved classifiers to {MODELS_DIR}")


if __name__ == "__main__":
    main()