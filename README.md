# Textbook Tutor: Interactive Textbook Learning with Retrieval-Augmented Generation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sravani919/AI_Tutor_Interactive_learning/blob/main/latestcoding.ipynb)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Research](https://img.shields.io/badge/Research-EMNLP%202026-orange)

---

## Companion Repository

Research repository accompanying the paper:


> **Interactive Textbook Learning with Retrieval-Augmented Generation: The Role of Instructional Structure**

**Authors**
- Sravani Pati
- Carlos Toxtli Hernandez

*(Under review at EMNLP 2026)*

---

# Overview

This repository contains the implementation of **Textbook Tutor**, an interactive AI tutoring system that transforms static textbook chapters into multiple instructional learning formats through a reproducible **Textbook-to-Interaction (T2I)** pipeline.

The system automatically converts textbook chapters into:

- Chapter summaries
- Chapter-specific question-answer pairs
- Storytelling modules
- Business case simulations
- Interactive challenges
- Chapter-aware tutoring

The accompanying paper investigates how **instructional structure**, **retrieval grounding**, and **lightweight deployment** influence interactive textbook learning under resource-constrained settings.

The experiments and user study reported in the paper were conducted using the **Google Colab prototype**. The included Streamlit application represents a lightweight deployment of the same T2I pipeline.

---

# Repository Contents

This repository includes:

- Source code
- Google Colab prototype
- Streamlit application
- DeepSeek preprocessing pipeline
- Generated instructional dataset
- Evaluation notebooks
- User study analysis
- Participant questionnaires
- Documentation

---

# Repository Structure

```
AI_Tutor_Interactive_learning/
│
├── app.py
├── requirements.txt
├── README.md
│
├── Merged_Chapter_Dataset.csv
│
├── latestcoding.ipynb
├── AITUTORFINALVERSION.ipynb
├── AI_TUTOR_METRICS.ipynb
├── Updated_AI_Tutor_metrics.ipynb
├── AI_tutor_Analysis.ipynb
│
├── initialDeepSeek for Question & Summary Generation.ipynb
├── deepseekgenerating questionand asnwer.ipynb
│
├── AI Tutor Experience Feedback.csv
├── Pre-Assessment AI tutor.csv
│
├── AI TUTOR MINI GUIDE.pdf
└── Updatedguide.pdf
```

---

# Textbook-to-Interaction (T2I) Pipeline

The Textbook Tutor pipeline consists of four stages.

## Stage 1 — Offline Content Preparation

- Extract textbook chapters using PyMuPDF.
- Generate chapter summaries using DeepSeek.
- Generate up to five chapter-specific question-answer pairs.
- Construct the instructional dataset.

Output:

```
Merged_Chapter_Dataset.csv
```

---

## Stage 2 — Instructional Transformation

The generated instructional dataset is transformed into multiple instructional formats, including:

- Storytelling
- Business case simulations
- Flashcards
- Multiple-choice quizzes
- Fill-in-the-blank exercises
- Matching activities
- Timed questions
- Scenario-based reasoning

---

## Stage 3 — Retrieval-Grounded Tutoring

The tutoring system retrieves chapter summaries and generated question-answer pairs before generating responses.

The experiments reported in the accompanying paper evaluate lightweight lexical retrieval methods:

- TF-IDF
- BM25
- Hybrid lexical retrieval

Earlier development notebooks also include experiments using semantic retrieval (MiniLM + FAISS). These notebooks are retained for completeness but are **not** the retrieval configuration evaluated in the paper.

---

## Stage 4 — Interactive Tutoring

The generated instructional dataset supports:

- Storytelling
- Business case simulations
- Interactive quizzes
- Chapter-aware tutoring
- Progress tracking
- XP-based learning analytics

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Language Models | DeepSeek, Mistral-7B, Falcon-RW-1B |
| Retrieval | TF-IDF, BM25 |
| Framework | Streamlit |
| Libraries | Transformers, PyMuPDF, Pandas |
| Development | Google Colab |
| Deployment | Streamlit, FastAPI |

---

# Dataset Generation

The instructional dataset is automatically generated from the OpenStax **Workplace Software and Skills** textbook.

Pipeline:

1. Extract textbook chapters.
2. Generate chapter summaries.
3. Generate chapter-specific question-answer pairs.
4. Export the instructional dataset.

Output:

```
Merged_Chapter_Dataset.csv
```

The generated dataset supports:

- chapter-aware tutoring
- storytelling
- business scenarios
- interactive quizzes
- concept assessment
- retrieval-grounded learning

---

# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Launch the Streamlit Application

```bash
streamlit run app.py
```

---

## Run the Google Colab Prototype

Launch directly from Colab:

https://colab.research.google.com/github/sravani919/AI_Tutor_Interactive_learning/blob/main/latestcoding.ipynb

---

# Reproducing the Paper

To reproduce the experiments described in the paper:

1. Install the required Python packages.

2. Download the OpenStax *Workplace Software and Skills* textbook.

3. Run the preprocessing notebooks to:

   - extract textbook chapters
   - generate chapter summaries
   - generate question-answer pairs

4. Construct the instructional dataset.

5. Launch the tutoring application.

6. Run the evaluation notebooks to reproduce retrieval and user-study analyses.

---

# Research Artifacts

This repository contains the primary research artifacts used in the paper.

Included artifacts:

- Source code
- Streamlit application
- Google Colab prototype
- Generated instructional dataset
- DeepSeek preprocessing notebooks
- Evaluation notebooks
- User-study analysis notebooks
- Participant questionnaires
- Documentation
This repository contains the code, generated instructional dataset, preprocessing pipeline, and evaluation notebooks necessary to reproduce the experiments described in the accompanying paper.
---

# Data

The instructional dataset (`Merged_Chapter_Dataset.csv`) was generated from the OpenStax **Workplace Software and Skills** textbook using the preprocessing pipeline provided in this repository.

The original textbook is distributed by OpenStax under its applicable license and should be obtained directly from the OpenStax website.

---

# Limitations

The experiments reported in the accompanying paper were conducted using the Google Colab prototype.

The repository additionally includes a lightweight Streamlit deployment implementing the same Textbook-to-Interaction pipeline.

Some larger language models (e.g., Mistral-7B) were originally executed on Clemson University's Palmetto HPC cluster during development.

---

# Future Work

Future work includes:

- adaptive instructional strategies
- longitudinal classroom studies
- larger educational datasets
- expanded retrieval methods
- multimodal instructional content

---


# Citation

Citation information will be updated after publication.

If you use this repository before publication, please cite the associated manuscript.


---

# License

This project is released under the **MIT License**.

---

# Contact

**Sravani Pati**

GitHub:
https://github.com/sravani919

Email:
*(spati@clemson.edu)*
