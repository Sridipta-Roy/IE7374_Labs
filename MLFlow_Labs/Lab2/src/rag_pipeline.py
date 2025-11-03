import mlflow
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore

from src.config import RAGConfig
from src.config import get_config
from src.data_loader import DocumentLoader, join_chunks


class RAGPipeline:
    
    def __init__(self, config: RAGConfig):        
        self.config = config
        self.documents: Optional[List[Document]] = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # Initialize components
        self._initialize_llm()
        self._initialize_embeddings()
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)
        mlflow.langchain.autolog()
        
        print(f"RAG Pipeline initialized")
        print(f"\nLLM: {config.llm_model}")
        print(f"\nEmbeddings: {config.embeddings_model}")
        print(f"\nMLflow Experiment: {config.mlflow_experiment_name}")
    
    def _initialize_llm(self):
        """Initialize GROQ LLM"""
        self.llm = ChatGroq(
            groq_api_key=self.config.groq_api_key,
            model_name=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        print(f"LLM initialized: {self.config.llm_model}")
    
    def _initialize_embeddings(self):
        """Initialize local embeddings model"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"Embeddings initialized: {self.config.embeddings_model}")
    
    def load_documents(self, source: str, source_type: str = "arxiv") -> List[Document]:        
        loader = DocumentLoader(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        if source_type == "arxiv":           
            self.documents = loader.load_arxiv(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        print(f"Loaded {len(self.documents)} document chunks")
        return self.documents
    
    def create_vectorstore(self, persist: bool = False) -> None:        
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        print(f"Creating vector store with {len(self.documents)} chunks...")
        
        if self.config.vector_store_type == "faiss":
            self.vectorstore = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embeddings
            )
            if persist:
                self.vectorstore.save_local(self.config.vector_store_path)
                print(f"Vector store persisted to {self.config.vector_store_path}")
        
        elif self.config.vector_store_type == "inmemory":
            self.vectorstore = InMemoryVectorStore.from_documents(
                documents=self.documents,
                embedding=self.embeddings
            )
        
        else:
            raise ValueError(f"Unsupported vector store: {self.config.vector_store_type}")
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type=self.config.retriever_type,
            search_kwargs={"k": self.config.retriever_k}
        )
        
        print(f"\nVector store created ({self.config.vector_store_type})")
        print(f"\nRetriever configured (k={self.config.retriever_k})")
    
    def load_vectorstore(self, path: Optional[str] = None) -> None:
        """Load persisted vector store"""
        if path is None:
            path = self.config.vector_store_path
        
        if self.config.vector_store_type == "faiss":
            self.vectorstore = FAISS.load_local(
                path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type=self.config.retriever_type,
                search_kwargs={"k": self.config.retriever_k}
            )
            print(f"Vector store loaded from {path}")
        else:
            raise ValueError("Only FAISS vector store supports persistence")
    
    def build_chain(self) -> None:
        """Build the RAG chain"""
        if not self.retriever:
            raise ValueError("Vector store not created. Call create_vectorstore() first.")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.human_prompt)
        ])
        
        # Build the RAG chain
        self.rag_chain = (
            {
                "context": self.retriever | RunnableLambda(join_chunks),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("RAG chain built successfully")
    
    @mlflow.trace
    def query(self, question: str, trace: bool = True) -> str:
        """
        Query the RAG pipeline
        
        Args:
            question: User question
            trace: Whether to log trace to MLflow
        
        Returns:
            Generated answer
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not built. Call build_chain() first.")
        
        response = self.rag_chain.invoke(question)
        return response
    
    def batch_query(self, questions: List[str]) -> List[str]:
        """Process multiple queries"""
        if not self.rag_chain:
            raise ValueError("RAG chain not built. Call build_chain() first.")
        
        responses = []
        for question in questions:
            response = self.query(question, trace=False)
            responses.append(response)
        
        return responses
    
    def get_relevant_documents(self, question: str) -> List[Document]:
        """Retrieve relevant documents for a question"""
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        
        return self.retriever.invoke(question)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline configuration and statistics"""
        return {
            "config": self.config.to_dict(),
            "num_documents": len(self.documents) if self.documents else 0,
            "vectorstore_type": self.config.vector_store_type,
            "retriever_configured": self.retriever is not None,
            "chain_built": self.rag_chain is not None,
        }


def create_pipeline_from_config(config_name: str = "quality") -> RAGPipeline:       
    config = get_config(config_name)
    return RAGPipeline(config)


if __name__ == "__main__":
    # Test RAG pipeline

    config = RAGConfig()
    pipeline = RAGPipeline(config)
        
    # Load and process documents
    pipeline.load_documents("1706.03762", source_type="arxiv")
    pipeline.create_vectorstore()
    pipeline.build_chain()
    
    # Test query
    test_question = "What is the main idea of the paper?"
    print(f"\n{'='*60}")
    print(f"Question: {test_question}")
    print(f"{'='*60}")
    
    response = pipeline.query(test_question)
    print(f"\nAnswer:\n{response}")
    
    # Show retrieved context
    print(f"\n{'='*60}")
    print("Retrieved Documents:")
    print(f"{'='*60}")
    docs = pipeline.get_relevant_documents(test_question)
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content[:200] + "...")