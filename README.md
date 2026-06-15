# Generative AI Ensemble: Research Paper Title Generator
### Abstractive Text Summarization | Hugging Face Transformers | Streamlit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-F9AB00?logo=huggingface&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

An end-to-end Natural Language Processing (NLP) pipeline designed to automatically generate highly accurate, context-aware titles for complex academic research papers. 

Manually extracting the core thesis of dense academic literature is time-consuming. This system automates title generation by leveraging advanced abstractive text summarization. It fine-tunes three state-of-the-art transformer models and utilizes a custom Post-Hoc Selection algorithm to evaluate and output the best generative summary.

---

## Key Features

* **Multi-Model Architecture:** Fine-tuned `facebook/bart-large`, `t5-base`, and `google/pegasus-xsum` on a massive dataset of arXiv abstracts.
* **Hardware-Optimized Training:** Utilized Gradient Checkpointing, Gradient Accumulation, and Automatic Mixed Precision (AMP) to train gigabyte-scale models on constrained GPU cloud environments.
* **Algorithmic Ensemble Judge:** Built a custom post-hoc selection engine that mathematically evaluates generated outputs from all three models against the source abstract to autonomously select the most contextually accurate title.
* **Interactive UI:** A fully containerized Streamlit web application for real-time inference and side-by-side model comparison.

---

## System Architecture & Workflow

```text
[ Raw Academic Abstract ] 
       │
       ├───> Model 1: BART (Factual Consistency)
       ├───> Model 2: T5 (Structured Balance)
       └───> Model 3: PEGASUS (Highly Abstractive)
       │
[ Custom Ensemble Judge ] ──(Calculates Keyword Density & Context)──> [ 🏆 Final Optimal Title ]
```

## Dataset & Data Engineering

The models were fine-tuned using a comprehensive dataset of academic research papers. To ensure full reproducibility and open-source collaboration, the processed dataset has been hosted on the Hugging Face Hub.

* **Dataset Hub:** [View Dataset on Hugging Face](https://huggingface.co/datasets/subhan1501/arxiv-abstractive-summarization)
* **Domain:** Computer Science, Physics, and Mathematics (arXiv)
* **Structure:** The dataset consists of complex, multi-paragraph `abstracts` mapped to their corresponding highly condensed academic `titles`.

## Model Evaluation (ROUGE Metrics)
The models were evaluated against an unseen test set of research abstracts using the industry-standard ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.

* **BART:** Excels at factual consistency and grammatical structuring.

* **T5:** Provides highly structured, balanced summaries.

* **PEGASUS:** Consistently generates the most abstractive and human-like academic titles due to its pre-training objective (Gap Sentences Generation).

## Sample Output
### Input Abstract:

"We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers..."

### Transformer Ensemble Outputs:

* **BART:** BERT: Pre-training of Deep Bidirectional Transformers

* **T5:** A Study on Bidirectional Encoder Representations

* **PEGASUS:** BERT: Deep Bidirectional Transformers for Language Understanding

* **Ensemble Selected Title:** BERT: Deep Bidirectional Transformers for Language Understanding (Selected by PEGASUS)

## Installation & Usage
This project utilizes cloud-hosted model weights. You do not need to manually download the gigabyte-scale transformer models; the Streamlit application will automatically pull the fine-tuned weights from the Hugging Face Hub upon first execution.

1. Clone the repository:

```Bash
git clone [https://github.com/your-username/Arxiv-Title-Generator-Ensemble.git](https://github.com/your-username/Arxiv-Title-Generator-Ensemble.git)
cd Arxiv-Title-Generator-Ensemble
```
2. Create a Virtual Environment & Install Dependencies:

```Bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
3. Launch the Application:

```Bash
streamlit run app.py
```
(Note: The first run may take 1-2 minutes as the system caches the subhan1501 Hugging Face models).

## Technology Stack
* **Machine Learning & NLP:** PyTorch, Hugging Face transformers, datasets, evaluate (ROUGE)

* **Data Processing:** Pandas, NumPy

* **Cloud Infrastructure:** Kaggle GPUs (T4 x2 / P100), Hugging Face Model Hub

* **Frontend:** Streamlit

## Developer Information
* **Developer:** Muhammad Subhan Shahid

* **Affiliation:** National University of Computer and Emerging Sciences (FAST-NU)

* **Program:** BS Artificial Intelligence (BSAI)*