# MLflow LLMOps: LangChain RAG Pipeline Demo

A complete demonstration of MLflow for LLMOps using LangChain RAG pipeline with GROQ models, evaluated on ArXiv paper.

## 🎯 Project Overview

This project showcases:
- **RAG Pipeline**: Building a Retrieval-Augmented Generation system with LangChain
- **MLflow Integration**: Experiment tracking, model logging, and evaluation with built-in scorers
- **Cost-Effective**: Using free GROQ API (Llama 3.1 models)
- **ArXiv Papers**: Automatic loading and evaluation of research papers, "Attention Is All You Need" (ArXiv ID: 1706.03762)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create project directory
mkdir mlflow-langchain-rag
cd mlflow-langchain-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key_here
MLFLOW_TRACKING_URI=./mlruns
```

### 3. Run the Project

```bash
# Run with default ArXiv paper (Attention Is All You Need)
python src/main.py

# Start MLflow UI to view results
mlflow ui --port 5000
```

## 📁 Project Structure

```
mlflow-langchain-rag/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── data/
│   └── vector_store/          # Vector Store
├── src/
│   ├── config.py             # Configuration management
│   ├── data_loader.py        # Document loading utilities
│   ├── rag_pipeline.py       # RAG chain implementation
│   ├── evaluator.py          # MLflow evaluation logic
│   └── main.py               # Main execution script
└── mlruns/                   # MLflow artifacts (auto-generated)
```


## 📊 Evaluation Metrics

The pipeline uses **MLflow's built-in genai scorers** for evaluation:

1. **RelevanceToQuery**: Measures how relevant the response is to the query
2. **Correctness**: Evaluates factual accuracy of the response
3. **ExpectationsGuidelines**: Checks if response meets predefined expectations:
   - Key concepts mentioned
   - Expected facts included
   - Guidelines followed (factual, concise, etc.)

Each question is evaluated with specific expectations tailored to the paper content.

## 🎓 Learning Objectives

This project highlights:
- Building RAG pipelines with LangChain
- Tracking ML experiments with MLflow
- Using MLflow's genai evaluation scorers
- Evaluating LLM applications with structured expectations

## 📈 MLflow UI Overview

After running the pipeline:

1. **Navigate to**: http://localhost:5000
2. **View Experiments**: Compare different configurations
3. **Check Metrics**: Review evaluation scores
4. **Inspect Traces**: Debug retrieval and generation steps
5. **Export Models**: Save best-performing versions


## 📝 License

MIT License - feel free to use for learning and demonstration.

**Happy Learning! 🚀**
