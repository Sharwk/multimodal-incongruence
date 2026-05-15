# When Modalities Disagree

### Detecting Multimodal Sentiment Incongruence

**CS466: Multimodal Interaction and Learning — Sean Bell, Colby College, Spring 2026**

This project asks a question most multimodal sentiment systems are designed to avoid: what happens when the words, the tone, and the face of an utterance disagree with each other? Most fusion architectures treat that disagreement as noise to suppress. This project treats it as the signal worth measuring.

## Project Arc

The work was carried out in two deliberate stages.

**Stage 1 — CMU-MOSI feasibility study.** Trained per-modality classifiers on the MOSI sentiment dataset (text, audio, video) and computed a variance-based cross-modal disagreement score per sample. The goal was to test whether cross-modal disagreement was a structured, learnable signal before committing to a full system that depends on it. Found that high-disagreement utterances correlated with the cases sentiment models tend to fail on — sarcasm, hesitation, mixed feelings. This validated the premise.

**Stage 2 — MUStARD with named pragmatic categories.** Applied the validated framework to the MUStARD sarcasm dataset, then layered a rule-based interpretive module on top of the trained binary classifier. The rule layer maps per-modality signal patterns to specific pragmatic categories: sincere positive, sincere negative, passive aggression, feigned enthusiasm, and sarcasm. Wrapped the system in a Gradio app for live demos.

## Repository Structure

```
multimodal-incongruence/
├── README.md
├── requirements.txt
├── 01_preprocessing.ipynb         # Stage 1: MOSI preprocessing and baselines
├── notebooks/
│   └── 01_preprocessing.ipynb     # (mirror copy for organization)
├── app/                           # (placeholder, superseded by mustard/src/app.py)
└── mustard/                       # Stage 2: MUStARD pipeline and Gradio app
    ├── src/
    │   ├── data_loader.py
    │   ├── extract_text.py
    │   ├── extract_video.py
    │   ├── train.py
    │   └── app.py
    ├── features/                  # Pre-extracted text and video features
    └── models/                    # Trained per-modality classifiers
```

## Results

### Stage 1: CMU-MOSI

| Model                  | Test Accuracy |
|------------------------|---------------|
| Audio only             | 50.2%         |
| Video only             | 48.5%         |
| Text only (RoBERTa)    | 82.3%         |
| Late fusion            | 79.0%         |
| Fusion + disagreement  | 76.2%         |

A 3-class disagreement-aware model achieved 0.90 precision when flagging utterances as ambiguous.

### Stage 2: MUStARD

| Model                     | Test Accuracy |
|---------------------------|---------------|
| Text only                 | 58.0%         |
| Audio only                | 54.3%         |
| Video only                | 69.6%         |
| Equal-weighted fusion     | 70.3%         |
| Accuracy-weighted fusion  | 71.7%         |
| Video + Text only         | 72.5%         |

The strongest configuration drops audio entirely — a real finding that argues against naive "more modalities = better" assumptions.

## Setup

```bash
git clone https://github.com/Sharwk/multimodal-incongruence
cd multimodal-incongruence
pip install -r requirements.txt
```

### To run the MUStARD Gradio demo

```bash
cd mustard

# Download MUStARD video clips (1.4 GB, gitignored)
git clone https://github.com/soujanyaporia/MUStARD.git MUStARD-data
cd MUStARD-data && mkdir -p videos && cd videos
curl -L -o mmsd_raw_data.zip "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"
unzip -q mmsd_raw_data.zip "utterances_final/*" -x "__MACOSX/*"
cd ../..

# Launch the app
python3 src/app.py
```

The app loads at `http://127.0.0.1:7860`.

### To run the MOSI exploration

Open `notebooks/01_preprocessing.ipynb` in Google Colab or Jupyter and run all cells in order.

The MOSI pickle file can be downloaded directly in Colab:
```python
!wget "https://www.dropbox.com/s/sv94igp7zi3rsj1/mosi.pkl?dl=1" -O mosi.pkl
```

## Acknowledgments

- CMU MultiComp Lab for CMU-MOSI and the Multimodal SDK
- Castro et al. for MUStARD
- OpenAI for Whisper and CLIP
- Anthropic Claude for pair-programming assistance during development
- Dr. Chowdhury and the CS466 course staff