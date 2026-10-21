# Steering Non-Positional Sparse Autoencoder Features in a Language Model

Research project completed independently. The notebooks in this repository contain the experiments.

## Research Question
Can modifying the activations of selected sparse autoencoder (SAE) non-positional features systematically change the behavior of a large language model?

## Background
A language model's internal activations are hard to interpret directly. A sparse autoencoder decomposes those activations into a larger set of sparse features, some of which correspond to recognizable concepts or patterns. This project focuses on **non-positional** features, meaning features that respond to content rather than to a token's position in the text. The goal is to test whether adjusting these features can control what the model produces.

## Method
- **Model:** Gemma 2B
- **Tools:** Python and PyTorch
- **Feature selection:** Found top activation differences from positive-negative sentence pairs to select SAE features
- **Example pairs:** Used positive and negative example pairs to identify which features represent a target concept
- **Interventions:** Ran steering (increasing or changing selected feature activations) and ablation (removing them), then compared model outputs against a baseline
- **Evaluation:** used an LLM-as-a-judge to evaluate the prevalence of non-positional features in generated sentences

## Results
- Evaluated 468 steering interventions (3 concepts × 26 layers × 3 top features × 2 activation methods, last-token and mean).
- Steering raised average judge-rated concept prevalence for most anger features using last-token activations (67 of 78), but effects were small overall (best case: anger 0.031 → 0.080 at layer 4, feature 11253).
- Most individual effects were not statistically significant (only 3 of 78 anger last-token features at p < 0.05, none after correcting for multiple comparisons).
- Takeaway: single-feature steering gave weak, inconsistent control of these concepts in Gemma 2B.

## Repository Contents
- `data_to_activations.ipynb`: each of the three data to activations files take 1000 sentence pairs which showcase the three different features (Shakespearean, anger, politeness) and record the SAE feature activations each sentence induces.
- `top_features.ipynb`: each top features takes the raw data activation files created by the previous file and finds 3 features with most differences. Generates 100 sentences per layer for each feature.
- `SentenceScoring.ipynb`: Uses GP4o mini to grade each generated sentence (baseline, top feature 1, top feature 2, top feature 3), and calculates the average steering score effect for each layer.

## How to Run
Install the dependencies from requirements.txt, then open the notebooks in Jupyter or Google Colab and run the cells in order. The data to activations file will need a strong GPU and VRAM to run, you can use Runpod or an equivalent service.
