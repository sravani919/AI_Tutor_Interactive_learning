# AI_Tutor_Interactive_learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sravani919/AI_Tutor_Interactive_learning/blob/main/latestcoding.ipynb)






# AI Tutor – Interactive Learning System

**Transforming traditional textbook content into adaptive, engaging, and intelligent learning experiences**

---

## 📘 Project Overview

This AI-powered tutor transforms the *Workplace Software and Skills* textbook into a fully interactive, chapter-wise learning experience. Instead of passive reading, learners can engage in storytelling, business scenarios, gamified quizzes, and more—powered entirely by AI models.

This project demonstrates how AI can revolutionize education by offering dynamic learning paths, real-time feedback, and cognitive engagement using large-scale models, without requiring manual curation of content.

---

## 🚀 Live Demo

> ✅ [Launch on Google Colab (Public)](https://colab.research.google.com/github/sravani919/AI_Tutor_Interactive_learning/blob/main/latestcoding.ipynb)

**Note:**  
Deployment is in progress. Advanced models like Mistral-7B currently run locally on high-performance compute clusters. The public version on Colab is fully functional with most learning modes available for exploration.

---

## ⚙️ Technology Stack

| Layer            | Technology Used                                              |
|------------------|--------------------------------------------------------------|
| Language Models  | `Mistral-7B`, `DeepSeek Coder-6.7B`, HuggingFace Pipelines   |
| Retrieval        | `FAISS` (semantic vector store)                              |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2`                     |
| Frameworks       | `LangChain`, `Transformers`, `ipywidgets`, `Matplotlib`      |
| APIs             | Google Custom Search API                                     |
| Interface        | Google Colab UI (Widgets, HTML, Charts)                      |
| Deployment       | Colab (Public), Clemson Palmetto HPC (Private), Cloud-ready |

---

## 📚 Dataset Generation Pipeline

To support intelligent interactivity, a custom dataset was automatically generated from a 1000+ page PDF textbook:

1. **PDF Text Extraction:**  
   Used `PyMuPDF` to extract raw text and split into chapters.

2. **LLM-Based Question and Summary Generation:**  
   Applied `DeepSeek Coder 6.7B` locally to generate:
   - 5+ questions per chapter
   - Chapter summaries  
   Token usage was optimized with timeout handling and max-length constraints.

3. **Exporting Dataset:**  
   Final outputs were saved as JSON and consolidated into a `Merged_Chapter_Dataset.csv` for use by the AI tutor.

---

## 🧠 Learning Modes Available

| Mode                      | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| Business Case Generator   | Real-world company scenarios from each chapter               |
| AI-Powered Storytelling   | Contextual business stories with characters and challenges    |
| Flashcards                | Flip-based practice for quick recall                          |
| Multiple Choice Quizzes   | Auto-generated MCQs with instant feedback                     |
| Fill in the Blank         | Gamified sentence completion with distractors                |
| Matching Challenge        | Match questions to correct answers                            |
| Timed Questions           | Countdown-based recall drills                                 |
| Scenario Reasoning        | Workplace decision simulations with hints                     |
| AI Chat Support           | Conversation with an AI tutor about any chapter               |
| XP Dashboard              | Tracks XP, level, progress, badges, and performance           |

---

## 🎯 Key Features

- Adaptive, multi-modal learning system
- Zero manual content creation – fully LLM-driven
- XP system with levels, badges, history, and analytics
- Designed to scale to any subject or organization
- Supports real-time AI search, summaries, and guidance
- Personalized learner profile and chapter tracking

---

## 💡 Infrastructure & Performance

- **Locally**, the full version runs on **Clemson Palmetto Supercomputing Cluster** for real-time AI inference using Mistral-7B and DeepSeek.
- **Public Colab version** uses smaller models and APIs for broader accessibility.
- The system can be deployed in cloud environments such as **AWS**, **Azure**, or **Google Cloud** for enterprise use cases.

---

## ✅ Use Cases

| Application Area        | Benefit                                                 |
|-------------------------|----------------------------------------------------------|
| Schools and Universities | Convert any textbook into interactive AI content         |
| Corporate Training      | Scenario-based onboarding, quizzes, and automation        |
| Ed-Tech Platforms       | Scalable, gamified content generation                     |
| Self-Learning Platforms | 24/7 AI tutor with contextual understanding               |

---

## 🧪 Future Work

- Fine-tune domain-specific LLMs for even more contextual accuracy
- Build a dedicated web app for broader adoption
- Integrate with LMS platforms (e.g., Moodle, Canvas)
- Extend support for video, diagram, and audio generation

---

## 📝 Author Notes

This system was created using freely available tools and open-source models, showcasing what is possible even without enterprise-level funding. It demonstrates complex architectural thinking, end-to-end data generation, and interactive deployment—all adaptable to any scale or domain.

The project represents a blueprint that can be deployed across industries, making learning more accessible and intelligent.

---

## 📂 Repository Structure

