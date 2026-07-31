# Textbook Tutor: Interactive Textbook Learning with Retrieval-Augmented Generation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sravani919/AI_Tutor_Interactive_learning/blob/main/latestcoding.ipynb)

Companion repository for the paper:

**Interactive Textbook Learning with Retrieval-Augmented Generation: The Role of Instructional Structure**

(Currently under review)

> **Interactive Textbook Learning with Retrieval-Augmented Generation: The Role of Instructional Structure**

---

# Overview

This repository contains the implementation of **Textbook Tutor**, an interactive AI tutoring system that transforms static textbook chapters into multiple instructional learning formats using a reproducible **Textbook-to-Interaction (T2I)** pipeline.

The system converts textbook chapters into:

- Chapter summaries
- Question-answer pairs
- Storytelling modules
- Business case simulations
- Interactive challenges
- Chapter-aware tutoring

The project investigates how instructional structure and retrieval-augmented generation support interactive textbook learning under lightweight computational settings.

---

# Repository Contents

This repository contains:

- Source code for the tutoring system
- Streamlit application
- Google Colab prototype
- DeepSeek dataset generation notebooks
- Generated instructional dataset
- Retrieval evaluation notebooks
- User study analysis notebooks
- Participant questionnaires
- Project documentation

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
├── AI_tutor_Analysis.ipynb
├── Updated_AI_Tutor_metrics.ipynb
│
├── initialDeepSeek for Question & Summary Generation.ipynb
├── deepseekgenerating questionand asnwer.ipynb
│
├── AI Tutor Experience Feedback.csv
├── Pre-Assessment AI tutor.csv
│
└── AI TUTOR MINI GUIDE.pdf
```

---

# Textbook-to-Interaction Pipeline

The pipeline consists of four stages.

## Stage 1 — Offline Content Preparation

- Extract textbook chapters using PyMuPDF.
- Generate chapter summaries.
- Generate chapter-specific question-answer pairs using DeepSeek.
- Construct the instructional dataset.

Output:

```
Merged_Chapter_Dataset.csv
```

---

## Stage 2 — Instructional Transformation

The generated instructional dataset is transformed into multiple instructional formats:

- Storytelling
- Business case simulations
- Flashcards
- Multiple-choice quizzes
- Fill-in-the-blank activities
- Matching challenges
- Timed quizzes
- Scenario reasoning

---

## Stage 3 — Retrieval-Grounded Tutoring

The tutoring system retrieves chapter summaries and generated question-answer pairs using lightweight retrieval before producing responses.

The paper evaluates:

- TF-IDF
- BM25
- Hybrid lexical retrieval

Earlier development notebooks also include experiments using semantic retrieval (MiniLM + FAISS). The experiments reported in the paper correspond to the retrieval configurations described in the manuscript.

---

## Stage 4 — Interactive Tutoring

The generated instructional dataset powers:

- Storytelling
- Business scenarios
- Interactive quizzes
- Chapter-aware tutoring
- Progress tracking
- XP system

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Language Models | DeepSeek, Mistral-7B, Falcon-RW-1B |
| Retrieval | TF-IDF, BM25 |
| Framework | Streamlit |
| Libraries | Transformers, Pandas, PyMuPDF |
| Development | Google Colab |
| Deployment | Streamlit, FastAPI |

---

# Dataset Generation

The instructional dataset is automatically generated from the OpenStax *Workplace Software and Skills* textbook.

Pipeline:

1. Extract textbook chapters.
2. Generate chapter summaries using DeepSeek.
3. Generate up to five chapter-specific question-answer pairs.
4. Export to:

```
Merged_Chapter_Dataset.csv
```

The generated dataset supports:

- tutoring
- storytelling
- interactive quizzes
- concept assessment
- chapter-aware retrieval

---

# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Launch Streamlit

```bash
streamlit run app.py
```

---

## Google Colab

The notebook can also be executed directly:

https://colab.research.google.com/github/sravani919/AI_Tutor_Interactive_learning/blob/main/latestcoding.ipynb

---

# Reproducing the Paper

To reproduce the experiments reported in the paper:

1. Install the required packages.
2. Download the OpenStax Workplace Software and Skills textbook.
3. Run the DeepSeek preprocessing notebooks.
4. Generate:

   - chapter summaries
   - question-answer pairs

5. Launch the tutoring application.
6. Run the evaluation notebooks.

---

# Research Artifacts

This repository includes:

- Source code
- Generated instructional dataset
- Prompt generation notebooks
- Evaluation notebooks
- User-study analysis
- Participant questionnaires
- Streamlit application

---

# Limitations

Some advanced language models (e.g., Mistral-7B) were originally executed on Clemson University's Palmetto HPC cluster.

The public repository provides the complete preprocessing pipeline together with lightweight deployment suitable for reproduction and experimentation.

---

# Future Work

Future work includes:

- adaptive instructional strategies
- longitudinal classroom studies
- larger educational datasets
- additional retrieval methods
- multimodal instructional content

---

# Citation

If you use this repository, please cite:

```bibtex
@inproceedings{pati2026textbook,
  title={Interactive Textbook Learning with Retrieval-Augmented Generation: The Role of Instructional Structure},
  author={Pati, Sravani and Hernandez, Carlos Toxtli},
  booktitle={Proceedings of EMNLP},
  year={2026}
}
```

---

# License

This repository is released under the MIT License.

---

# Contact

Sravani Pati

GitHub:
https://github.com/sravani919
