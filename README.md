# Spot the Bot: Semantic Trajectories of Natural Language

Bachelor's thesis project at HSE University (2023) on distinguishing human- and machine-generated text through structural representations of language.

Instead of relying only on surface-level text features, the project represents text as trajectories through a semantic space and studies graph-based properties of those trajectories.

## Approach

The research pipeline includes:

1. preprocessing corpora in Russian and English;
2. TF-IDF vectorization;
3. dimensionality reduction with SVD;
4. representation of words as vectors and construction of text sequences;
5. semantic trajectories through text;
6. data-driven graph construction;
7. graph metrics and text-classification experiments for separating human and generated text.

## Tech

Python · NumPy · pandas · SciPy · NetworkX · Ray · Jupyter

## Repository structure

- `EnPreprocessing/` — English-language preprocessing, TF-IDF/SVD, and vectorization;
- `Russian/` — Russian-language preprocessing and feature construction;
- `Graph/` — graph construction, graph metrics, trajectory analysis, and classification experiments;
- `TextGenBot/` — tooling used to generate machine-written text for the research pipeline;
- `EpubToTXT.ipynb` — utility for converting source texts into a usable text format.

## Research context

This repository preserves the original thesis workflow rather than a production package. Some notebooks were run in a local research environment and therefore contain historical absolute paths and cached outputs.
