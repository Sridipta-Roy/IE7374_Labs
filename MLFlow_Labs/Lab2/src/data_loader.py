import os
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


class DocumentLoader:
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """ Initialize document loader """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    
    def load_arxiv(self, query: str, max_docs: int = 1) -> List[Document]:
        """ Load papers from ArXiv """
        loader = ArxivLoader(query=query, load_max_docs=max_docs)
        documents = loader.load()
        return self.splitter.split_documents(documents)        
    

def join_chunks(chunks: List[Document]) -> str:
    """Join document chunks into a single string"""
    return "\n\n".join([chunk.page_content for chunk in chunks])


if __name__ == "__main__":
    # Test document loader
    loader = DocumentLoader(chunk_size=500, chunk_overlap=100)
    
    # Load a paper from ArXiv (Attention Is All You Need)
    docs = loader.load_arxiv(query="1706.03762", max_docs=1)
    
    print(f"\nTotal chunks: {len(docs)}")
    print(f"\nFirst chunk preview:\n{docs[0].page_content[:200]}...")