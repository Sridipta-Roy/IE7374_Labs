# 🚀 MLflow + GenAI Experiment Demo

This repository demonstrates how to integrate **MLflow** with **Groq’s LLM API** for tracking and evaluating generative AI experiments.  
It includes two example scripts showing progressively advanced use cases—from a single text generation to prompt optimization with evaluation.

---

## 📂 Repository Structure

```
.
├── mlflow-1.py   # Basic MLflow tracking for a single LLM generation
├── mlflow-2.py   # Advanced MLflow experiment for prompt optimization & evaluation
├── .env          # Contains GROQ_API_KEY
├── .gitignore    # Ensures ml_venv and artifacts are excluded
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sridipta-Roy/IE7374_Labs.git
   cd IE7374_Labs\MLFlow_Labs\Lab1
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv ml_venv
   source ml_venv/bin/activate    # (Linux/Mac)
   ml_venv\Scripts\activate       # (Windows)
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(You can also install key packages manually:)*
   ```bash
   pip install mlflow python-dotenv groq
   ```

4. **Add your API key**  
   Create a `.env` file and add:
   ```bash
   GROQ_API_KEY=your_api_key_here
   ```

---

## 🧠 Scripts Overview

### `mlflow-1.py`
A simple demonstration that:
- Uses Groq’s **Llama-3.3-70B** model to answer a prompt  
- Logs experiment parameters, generated text, and metrics in MLflow  
- Stores outputs (`generated_text.txt` and `chat_completion.json`) as artifacts  

### `mlflow-2.py`
An extended experiment that:
- Tests **multiple summarization prompts** on a sample article  
- Uses MLflow to track each run (prompt name, length, temperature, etc.)  
- Evaluates results via a **mock LLM judge function** scoring conciseness, relevance, and coherence  
- Logs detailed metrics and artifacts for each prompt to enable prompt optimization analysis  

---

## 📊 Viewing MLflow UI

Launch the MLflow tracking UI locally:
```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) to explore:
- Parameters and metrics for each run  
- Artifacts (generated text and model outputs)  
- Comparison of runs for prompt optimization  

---

## 🧩 Example Output
Each run produces:
- `generated_text.txt` or `generated_summary_*.txt`  
- `chat_completion.json` (raw LLM response)  
- Metrics like text length, coherence, and conciseness  

---

## 🏁 Summary
This project demonstrates:
- **End-to-end experiment tracking** for Generative AI tasks  
- **Prompt engineering evaluation** using MLflow  
- **Integration of LLM outputs** into a reproducible workflow  

---
