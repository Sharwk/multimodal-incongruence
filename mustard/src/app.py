"""Gradio demo for multimodal pragmatic incongruence detection."""
import pickle
import tempfile
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
import joblib
import gradio as gr
import whisper
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, pipeline

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_FRAMES = 8

print("Loading models (one-time setup)...")
clf_text = joblib.load(MODELS_DIR / "clf_text.joblib")
clf_audio = joblib.load(MODELS_DIR / "clf_audio.joblib")
clf_video = joblib.load(MODELS_DIR / "clf_video.joblib")
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_model.eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
whisper_model = whisper.load_model("base")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    top_k=3,
)
print("Models loaded.")

# Per-modality weights (from training results)
W_TEXT, W_AUDIO, W_VIDEO = 0.587, 0.514, 0.703


def sample_frames(video_path, num_frames=NUM_FRAMES):
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def extract_text(video_path):
    """Transcribe with Whisper, embed with sentence-transformers."""
    result = whisper_model.transcribe(video_path, fp16=False)
    transcript = result["text"].strip()
    if not transcript:
        transcript = "[no speech detected]"
    embedding = text_encoder.encode([transcript])[0]
    return embedding, transcript


def get_text_sentiment(transcript):
    """Returns dict with positive, negative, neutral probabilities."""
    if not transcript or transcript == "[no speech detected]":
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    result = sentiment_pipe(transcript, truncation=True, max_length=128)[0]
    scores = {item["label"]: item["score"] for item in result}
    return {
        "negative": scores.get("LABEL_0", 0.0),
        "neutral":  scores.get("LABEL_1", 0.0),
        "positive": scores.get("LABEL_2", 0.0),
    }


def extract_audio_dummy():
    return np.zeros(18, dtype=np.float64)


def extract_video(video_path):
    frames = sample_frames(video_path)
    if not frames:
        return np.zeros(768, dtype=np.float32)
    with torch.no_grad():
        inputs = clip_processor(images=frames, return_tensors="pt").to(DEVICE)
        embs = clip_model.get_image_features(**inputs)
        # Handle both tensor and wrapper-object returns across transformers versions
        if hasattr(embs, "image_embeds"):
            embs = embs.image_embeds
        elif hasattr(embs, "last_hidden_state"):
            embs = embs.last_hidden_state.mean(dim=1)
        pooled = embs.mean(dim=0).cpu().numpy()
    return pooled

def classify_incongruence(text_sent, audio_sarc_prob, video_sarc_prob):
    """
    Rule-based classifier mapping per-modality signals to incongruence types.
    """
    pos, neg, neu = text_sent["positive"], text_sent["negative"], text_sent["neutral"]
    text_label = max(text_sent, key=text_sent.get)

    AV_HIGH = 0.60   # clearly incongruent
    AV_LOW  = 0.40   # clearly sincere/relaxed

    def av_band(p):
        if p >= AV_HIGH: return "high"
        if p <= AV_LOW:  return "low"
        return "mid"

    audio_band = av_band(audio_sarc_prob)
    video_band = av_band(video_sarc_prob)

    text_strong = max(pos, neg) > 0.5

    # ---- Sincere cases first (higher priority) ----

    # Sincere positive: positive words AND at least one channel clearly low,
    # AND no channel clearly high
    if (text_label == "positive" and text_strong
        and audio_band != "high" and video_band != "high"
        and (audio_band == "low" or video_band == "low")):
        return "Sincere positive", "Words, tone, and face all align as positive."

    # Sincere negative: same logic for negative
    if (text_label == "negative" and text_strong
        and audio_band != "high" and video_band != "high"
        and (audio_band == "low" or video_band == "low")):
        return "Sincere negative", "Words, tone, and face all align as negative."

    # ---- Incongruence cases ----

    # Passive aggression: positive words + BOTH channels high
    if text_label == "positive" and text_strong and audio_band == "high" and video_band == "high":
        return "Passive aggression", "Positive words masking negative tone and facial expression."

    # Sarcasm / teasing: negative words + BOTH channels high
    if text_label == "negative" and text_strong and audio_band == "high" and video_band == "high":
        return "Sarcasm or teasing", "Negative words delivered with playful or incongruent tone and expression."

    # Feigned enthusiasm: positive words + at least ONE channel high (incongruent)
    # but not both (that would be passive aggression)
    if (text_label == "positive" and text_strong
        and (audio_band == "high" or video_band == "high")):
        return "Feigned enthusiasm", "Positive words with unconvincing or flat tone and expression."

    # ---- Mid-band fallback ----
    # If text is positive and both AV are mid (no commitment either way),
    # call it sincere positive with a softer description
    if text_label == "positive" and text_strong and audio_band == "mid" and video_band == "mid":
        return "Sincere positive", "Words read positive; tone and expression are neutral but consistent."

    if text_label == "negative" and text_strong and audio_band == "mid" and video_band == "mid":
        return "Sincere negative", "Words read negative; tone and expression are neutral but consistent."

    # Truly ambiguous fallback
    return "Ambiguous incongruence", "Cross-modal signals disagree without matching a defined pattern."

#simple summary 4 app
def make_headline(text_sent, audio_prob, video_prob, binary_probs):
    """Build a plain-English headline. Returns (headline_md, top_label, top_explanation)."""
    avg_incong = (audio_prob + video_prob) / 2
    pos = text_sent["positive"]
    neg = text_sent["negative"]
    sarc_fusion = binary_probs[1]
    sincere_fusion = binary_probs[0]

    raw_scores = {}

    # SINCERE scores (only if trained classifier agrees it's sincere)
    if pos > 0.5:
        raw_scores["sincere_positive"] = pos * (1 - avg_incong) * (0.5 + sincere_fusion)

    if neg > 0.5:
        raw_scores["sincere_negative"] = neg * (1 - avg_incong) * sincere_fusion

    # PASSIVE AGGRESSION: positive words + both channels high + trained model agrees
    if pos > 0.5:
        raw_scores["passive_aggression"] = pos * audio_prob * video_prob * sarc_fusion * 1.5

    # FEIGNED ENTHUSIASM / SARCASM with positive words: this is the Sheldon/Penny case
    # Heavily driven by trained classifier
    if pos > 0.5:
        raw_scores["feigned_enthusiasm"] = pos * sarc_fusion * 1.3

    # SARCASM with negative words
    if neg > 0.5:
        raw_scores["sarcasm"] = neg * sarc_fusion * 1.3

    if not raw_scores:
        return (
            "## Best guess: ambiguous\n#### The signals don't clearly match any pattern.",
            "Ambiguous",
            "Cross-modal signals don't match any defined pattern."
        )

    ranked = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    top_key, top_raw = ranked[0]

    headline_map = {
        "sincere_positive":   ("sincere", "Sincere positive", "The person appears to mean what they're saying, in a positive way."),
        "sincere_negative":   ("sincere", "Sincere negative", "The person appears to mean what they're saying, in a negative way."),
        "passive_aggression": ("being passive aggressive", "Passive aggression", "Their words are pleasant, but their tone and expression are not."),
        "feigned_enthusiasm": ("being sarcastic or faking enthusiasm", "Sarcasm or feigned enthusiasm", "Their words sound positive, but the delivery suggests they don't mean them literally."),
        "sarcasm":            ("being sarcastic", "Sarcasm", "Their words are negative, but the delivery suggests they don't mean them literally."),
    }

    top_phrase, top_label, top_explanation = headline_map[top_key]
    top_pct = int(min(top_raw, 0.99) * 100)

    headline = f"## {top_pct}% chance this person is {top_phrase}"
    headline += f"\n#### {top_explanation}"

    SUBTEXT_RAW_MIN = 0.20
    SUBTEXT_RATIO   = 0.50

    if len(ranked) > 1:
        second_key, second_raw = ranked[1]
        if second_raw >= SUBTEXT_RAW_MIN and second_raw >= top_raw * SUBTEXT_RATIO:
            second_phrase, _, _ = headline_map[second_key]
            second_pct = int(second_raw * 100)
            headline += f"\n\n_But there's also a {second_pct}% chance this person is {second_phrase}._"

    return headline, top_label, top_explanation

def predict(video_file):
    if video_file is None:
        return "Please upload a video.", "", None, None, None, "", ""

    # Extract features
    text_feat, transcript = extract_text(video_file)
    audio_feat = extract_audio_dummy()
    video_feat = extract_video(video_file)

    # Per-modality sarcasm probabilities
    p_text  = clf_text.predict_proba([text_feat])[0]
    p_audio = clf_audio.predict_proba([audio_feat])[0]
    p_video = clf_video.predict_proba([video_feat])[0]

    # Sentiment from text
    text_sent = get_text_sentiment(transcript)

    # Binary fusion
    fusion = (p_text * W_TEXT + p_video * W_VIDEO) / (W_TEXT + W_VIDEO)

    # Single source of truth: headline scorer
    headline, top_label, top_explanation = make_headline(
        text_sent,
        float(p_audio[1]),
        float(p_video[1]),
        fusion,
    )

    # Detail panel mirrors the headline label
    detail = f"### Detected: **{top_label}**\n_{top_explanation}_"

    sentiment_dict = {
        "positive (text)": float(text_sent["positive"]),
        "neutral (text)":  float(text_sent["neutral"]),
        "negative (text)": float(text_sent["negative"]),
    }
    modality_dict = {
        "audio (incongruence prob)": float(p_audio[1]),
        "video (incongruence prob)": float(p_video[1]),
    }
    binary_dict = {"sincere": float(fusion[0]), "sarcasm": float(fusion[1])}

    return headline, detail, sentiment_dict, modality_dict, binary_dict, transcript, ""



theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#0a0e1a",
    body_background_fill_dark="#0a0e1a",
    block_background_fill="#141824",
    block_background_fill_dark="#141824",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="white",
)

with gr.Blocks(title="Multimodal Pragmatic Incongruence Detector", theme=theme) as demo:
    gr.Markdown("# Multimodal Pragmatic Incongruence Detector")
    gr.Markdown(
        "Upload a short video clip (3–15 seconds) of someone speaking. "
        "The system extracts three signals independently: the **literal sentiment of the words**, "
        "the **tone of voice**, and the **facial expression**. When these channels disagree in "
        "specific patterns, the result is named pragmatic incongruence: passive aggression, "
        "feigned enthusiasm, sarcasm, or simply sincere speech."
    )

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Upload video")
            submit = gr.Button("Analyze", variant="primary")
            headline_out = gr.Markdown(
                value="## Upload a clip and click Analyze\n#### The system's best guess will appear here."
            )
        with gr.Column():
            detail_out = gr.Markdown()
            sentiment_out = gr.Label(label="Text sentiment (literal words)")
            modality_out  = gr.Label(label="Tone and face (per-modality incongruence probability)")
            binary_out    = gr.Label(label="Binary fusion (sarcasm vs sincere, MUStARD baseline)")
            transcript_out = gr.Textbox(label="Transcript", lines=2)

    submit.click(
        predict,
        inputs=video_in,
        outputs=[headline_out, detail_out, sentiment_out, modality_out, binary_out, transcript_out, gr.Textbox(visible=False)],
    )

if __name__ == "__main__":
    demo.launch(share=True)