# Generative AI Ensemble: Research Paper Title Generator

An end-to-end Natural Language Processing (NLP) pipeline designed to automatically generate highly accurate, context-aware titles for complex academic research papers. 

This system fine-tunes three state-of-the-art transformer models and utilizes a custom Post-Hoc Selection algorithm to evaluate and output the best generative summary.

---

## Project Overview

Manually extracting the core thesis of dense academic literature is time-consuming. This project automates title generation by leveraging advanced abstractive text summarization. 

**Key Features:**
* **Multi-Model Architecture:** Fine-tuned `facebook/bart-large`, `t5-base`, and `google/pegasus-xsum` on a massive dataset of arXiv abstracts.
* **Hardware-Optimized Training:** Utilized Gradient Checkpointing, Gradient Accumulation, and Automatic Mixed Precision (AMP) to train gigabyte-scale models on constrained GPU cloud environments.
* **Algorithmic Ensemble Judge:** Built a custom post-hoc selection engine that mathematically evaluates generated outputs from all three models against the source abstract to autonomously select the most contextually accurate title.
* **Interactive UI:** A fully containerized Streamlit web application for real-time inference and side-by-side model comparison.

---

## Technology Stack

* **Machine Learning & NLP:** PyTorch, Hugging Face Transformers, Datasets, Evaluate (ROUGE)
* **Data Processing:** Pandas, NumPy
* **Cloud Infrastructure:** Kaggle GPUs (T4 x2 / P100)
* **Frontend:** Streamlit

---

## Model Evaluation (ROUGE Metrics)

The models were evaluated against an unseen test set of 100 research abstracts using the industry-standard ROUGE metric. 

* **BART:** Excels at factual consistency and grammar.
* **T5:** Provides highly structured, balanced summaries.
* **PEGASUS:** Consistently generates the most abstractive and human-like academic titles due to its pre-training objective (Gap Sentences Generation).

*(Note: See `results/model_comparison_chart.png` for visual metric breakdowns).*

---

## Installation & Local Setup

To run the ensemble generator locally, you will need the trained weights for BART, T5, and PEGASUS.


# Developer Information
* **Developer**: Muhammad Subhan Shahid
* **Affiliation**: National University of Computer and Emerging Sciences (FAST-NU)