# When Modalities Disagree

### Detecting Multimodal Sentiment Incongruence

**CS466: Multimodal Interaction and Learning — Sean Bell, Colby College, Spring 2026**

This project asks a question most multimodal sentiment systems are designed to avoid: what happens when the words, the tone, and the face of an utterance disagree with each other? Most fusion architectures treat that disagreement as noise to suppress. This project treats it as the signal worth measuring.

## Project Arc

The work was carried out in two stages:

**Stage 1 — CMU-MOSI exploration.** Trained per-modality classifiers on the MOSI sentiment dataset (text, audio, video) and computed a variance-based cross-modal disagreement score per sample. Found that high-disagreement utterances correlated with the cases sentiment models tend to fail on — sarcasm, hesitation, mixed feelings. This motivated the second stage.

**Stage 2 — MUStARD with named pragmatic categories.** Applied the same multimodal framework to the MUStARD sarcasm dataset, then layered a rule-based interpretive module on top of the trained binary classifier. The rule layer maps per-modality signal patterns to specific pragmatic categories: sincere positive, sincere negative, passive aggression, feigned enthusiasm, and sarcasm. Wrapped the system in a Gradio app for live demos.

## Repository Structure
multimodal-incongruence/
notebooks/
01_preprocessing.ipynb   # Data loading, preprocessing, baselines
app/
demo.py                  # Gradio live demo (coming)
README.md
requirements.txt 

## Setup

### 1. Clone the repo
git clone https://github.com/Sharwk/multimodal-incongruence
cd multimodal-incongruence

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download the dataset
Do not commit the dataset to this repo. Download it directly in Colab:
!wget "https://www.dropbox.com/s/sv94igp7zi3rsj1/mosi.pkl?dl=1" -O mosi.pkl

### 4. Run the notebook
Open notebooks/01_preprocessing.ipynb in Google Colab or Jupyter and run all cells in order.

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

| Model                  | Test Accuracy |
|------------------------|---------------|
| Text only              | 58.7%         |
| Audio only             | 51.4%         |
| Video only             | 70.3%         |
| Equal-weighted fusion  | 70.3%         |
| Accuracy-weighted fusion | 71.7%       |
| Video + Text only      | 73.2%         |

The strongest configuration drops audio entirely — a real finding that argues against naive "more modalities = better" assumptions.

## Setup

```bash
git clone https://github.com/Sharwk/multimodal-incongruence
cd multimodal-incongruence
pip install -r requirements.txt
```

To run the MUStARD Gradio demo:

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

To run the MOSI exploration, open `notebooks/01_preprocessing.ipynb` in Colab or Jupyter.

## Acknowledgments

- CMU MultiComp Lab for CMU-MOSI and the Multimodal SDK
- Castro et al. for MUStARD
- OpenAI for Whisper and CLIP
- Anthropic Claude for pair-programming assistance during development
- Dr. Chowdhury and the CS466 course staff