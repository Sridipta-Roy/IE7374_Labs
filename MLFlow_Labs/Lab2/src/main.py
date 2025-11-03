import sys
import argparse
from pathlib import Path

from src.config import RAGConfig, get_config
from src.rag_pipeline import RAGPipeline
from src.evaluator import RAGEvaluator


def initialize_pipeline(config_name: str = "balanced") -> RAGPipeline:
    """Initialize RAG pipeline"""
    print("\n" + "="*60)
    print("Step 1: Initializing RAG Pipeline")
    print("="*60)
    
    config = get_config(config_name)
    pipeline = RAGPipeline(config)
    
    return pipeline


def load_arxiv_paper(pipeline: RAGPipeline, arxiv_id: str = "1706.03762"):
    """Load ArXiv paper"""
    print("\n" + "="*60)
    print("Step 2: Loading ArXiv Paper")
    print("="*60)
    
    print(f"ArXiv ID: {arxiv_id}")
    print("Downloading and processing paper...")
    
    pipeline.load_documents(arxiv_id, source_type="arxiv")
    pipeline.create_vectorstore(persist=True)
    pipeline.build_chain()
    
    print("Paper loaded and processed")


def interactive_query_demo(pipeline: RAGPipeline):
    """Run interactive query demonstration"""
    print("\n" + "="*60)
    print("Interactive Query Demo")
    print("="*60)
    
    sample_questions = [
        "What is the main idea of the paper?",
        "What does the attention mechanism do?",
        "What's the difference between a transformer and a recurrent neural network?",
    ]
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n[Query {i}] {question}")
        print("-" * 60)
        
        # Get answer
        answer = pipeline.query(question)
        print(f"Answer: {answer}")
        
        # Show retrieved documents
        docs = pipeline.get_relevant_documents(question)
        print(f"\nRetrieved {len(docs)} relevant chunks")


def run_evaluation(pipeline: RAGPipeline, arxiv_id: str = "1706.03762"):
    """Run MLflow evaluation"""
    print("\n" + "="*60)
    print("Running MLflow Evaluation")
    print("="*60)
    
    evaluator = RAGEvaluator(pipeline)
    
    # Create evaluation dataset
    eval_dataset = evaluator.create_arxiv_eval_dataset(arxiv_id)
    
    # Run evaluation
    results = evaluator.run_evaluation(eval_dataset, run_name=f"arxiv_{arxiv_id}_evaluation")
    
    return results


def compare_configs(pipeline: RAGPipeline, arxiv_id: str = "1706.03762"):
    """Compare different configurations"""
    print("\n" + "="*60)
    print("Step 5: Comparing Configurations")
    print("="*60)
    
    evaluator = RAGEvaluator(pipeline)
    eval_dataset = evaluator.create_arxiv_eval_dataset(arxiv_id)
    
    # Compare configurations
    configs_to_compare = ["fast", "balanced"]
    comparison_df = evaluator.compare_configurations(configs_to_compare, eval_dataset)
    
    return comparison_df


def main(args):
    """Main execution flow"""
    try:
        print("\n" + "="*60)
        print("MLflow LLMOps: ArXiv Paper RAG Evaluation")
        print("="*60)
        print(f"Configuration: {args.config}")
        print(f"ArXiv ID: {args.arxiv_id}")
        print(f"Skip Evaluation: {args.skip_eval}")
        print(f"Compare Configs: {args.compare}")
        
        # Step 1: Initialize pipeline
        pipeline = initialize_pipeline(args.config)
        
        # Step 2: Load ArXiv paper
        load_arxiv_paper(pipeline, args.arxiv_id)
        
        # Step 3: Interactive query demo
        if not args.skip_demo:
            interactive_query_demo(pipeline)
        
        # Step 4: Run evaluation
        if not args.skip_eval:
            results = run_evaluation(pipeline, args.arxiv_id)
        
        # Step 5: Compare configurations (optional)
        if args.compare:
            comparison = compare_configs(pipeline, args.arxiv_id)        
                
        return 0
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLflow RAG Pipeline Demo for ArXiv Papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate "Attention Is All You Need" paper
  python src/main.py
  
  # Evaluate different paper with fast config
  python src/main.py --arxiv-id 2103.14030 --config fast
  
  # Skip demo, only evaluate
  python src/main.py --skip-demo
  
  # Compare configurations
  python src/main.py --compare
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "quality"],
        help="Configuration preset to use"
    )
    
    parser.add_argument(
        "--arxiv-id",
        type=str,
        default="1706.03762",
        help="ArXiv paper ID (default: Attention Is All You Need)"
    )
    
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip interactive query demonstration"
    )
    
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation step"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple configurations"
    )
    
    args = parser.parse_args()
    sys.exit(main(args))