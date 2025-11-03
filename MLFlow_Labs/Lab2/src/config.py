import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline with MLflow tracking"""
    
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    )
    mlflow_experiment_name: str = "LangChain-RAG-GROQ"
    
    # LLM Configuration     
    llm_model: str = "llama-3.1-8b-instant"  
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    
    # Using local sentence-transformers 
    #embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384 
    embeddings_model: str = "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions    
   
    chunk_size: int = 500
    chunk_overlap: int = 100    
    retriever_k: int = 3  # Number of documents to retrieve
    retriever_type: str = "similarity"  # or "mmr" for diversity
    
    #  Prompt Configuration 
    system_prompt: str = """You are a helpful AI assistant. Use the following context to answer the question.    
    Rules:
    - Provide concise answers (2-3 sentences maximum)
    - Base your answer ONLY on the provided context
    - If the answer is not in the context, say "I don't have enough information to answer that."
    - Be factual and avoid speculation

    Context:
    {context}
    """
    
    human_prompt: str = "Question: {question}\n\nAnswer:"
    
    # Vector Store Configuration 
    vector_store_type: str = "faiss"  
    vector_store_path: str = "./data/vectorstore"
    
    # Evaluation Configuration 
    eval_metrics: list = field(default_factory=lambda: [
        "relevance",
        "correctness", 
        "faithfulness",
        "answer_similarity"
    ])
    
    # ============ Data Paths ============
    data_dir: str = "./data/sample_docs"
    output_dir: str = "./outputs"
        
    def to_dict(self) -> dict:
        """Convert config to dictionary for MLflow logging"""
        return {
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "embeddings_model": self.embeddings_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retriever_k": self.retriever_k,
            "retriever_type": self.retriever_type,
            "vector_store_type": self.vector_store_type,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGConfig":
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


# Predefined configurations for different use cases
CONFIGS = {    
    "fast": RAGConfig(
        llm_model="llama-3.1-8b-instant",
        chunk_size=300,
        retriever_k=2,
    ),
    
    "balanced": RAGConfig(
        llm_model="llama-3.1-8b-instant",
        chunk_size=500,
        retriever_k=3,
    ),
    
    "quality": RAGConfig(
        llm_model="llama-3.1-70b-versatile",
        chunk_size=600,
        retriever_k=5,
        embeddings_model="sentence-transformers/all-mpnet-base-v2",
    ),
}


def get_config(config_name: str = "quality") -> RAGConfig:
    """Get predefined configuration by name"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]


if __name__ == "__main__":
    # Test configuration
    config = RAGConfig()    
    print(f"\nConfig dict: {config.to_dict()}")