# When Modalities Disagree
### Detecting Multimodal Incongruence in Human Sentiment Expression
**CS466: Multimodal Interaction and Learning — Sean Bell**

## Project Overview
This project investigates cross-modal disagreement in human sentiment 
expression using the CMU-MOSI dataset. Rather than fusing modalities 
to maximize accuracy, I explicitly measure disagreement across text, 
audio, and visual signals and use that as an analytical and predictive feature.

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

| Model | Test Accuracy |
|---|---|
| Audio only | 50.2% |
| Video only | 48.5% |
| Text only (RoBERTa) | 82.3% |
| Fusion | 79.0% |
| Fusion + Disagreement | 76.2% |

3-class ambiguity model: 0.90 precision on Ambiguous class

## Requirements
torch, transformers, scikit-learn, numpy, matplotlib, librosa, gdown
