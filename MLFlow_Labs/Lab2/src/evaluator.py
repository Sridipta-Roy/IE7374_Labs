import mlflow
import pandas as pd
from typing import List, Dict, Any
from mlflow.genai.scorers import RelevanceToQuery, Correctness, ExpectationsGuidelines

from src.config import RAGConfig, get_config
from src.rag_pipeline import RAGPipeline


class RAGEvaluator:    
    def __init__(self, pipeline: RAGPipeline):       
        self.pipeline = pipeline
        self.config = pipeline.config
    
    def create_arxiv_eval_dataset(self, arxiv_id: str = "1706.03762") -> List[Dict]:        
        # Default evaluation for "Attention Is All You Need" paper
        if arxiv_id == "1706.03762":
            return [
                {
                    "inputs": {"inputs": "What is the main idea of the paper?"},
                    "expectations": {
                        "key_concepts": ["attention mechanism", "transformer", "neural network"],
                        "expected_facts": ["attention mechanism is a key component of the transformer model"],
                        "guidelines": ["The response must be factual and concise"],
                    }
                },
                {
                    "inputs": {"inputs": "What's the difference between a transformer and a recurrent neural network?"},
                    "expectations": {
                        "key_concepts": ["sequential", "attention mechanism", "hidden state"],
                        "expected_facts": ["transformer processes data in parallel while RNN processes data sequentially"],
                        "guidelines": ["The response must be factual and focus on the difference between the two models"],
                    }
                },
                {
                    "inputs": {"inputs": "What does the attention mechanism do?"},
                    "expectations": {
                        "key_concepts": ["query", "key", "value", "relationship", "similarity"],
                        "expected_facts": ["attention allows the model to weigh the importance of different parts of the input sequence when processing it"],
                        "guidelines": ["The response must be factual and explain the concept of attention"],
                    }
                },
                {
                    "inputs": {"inputs": "What are the main components of the Transformer architecture?"},
                    "expectations": {
                        "key_concepts": ["encoder", "decoder", "attention", "feedforward"],
                        "expected_facts": ["the transformer consists of encoder and decoder stacks with self-attention and feedforward layers"],
                        "guidelines": ["The response must describe the architectural components"],
                    }
                },
                {
                    "inputs": {"inputs": "Why is the paper called 'Attention Is All You Need'?"},
                    "expectations": {
                        "key_concepts": ["attention", "recurrence", "convolution"],
                        "expected_facts": ["the paper shows that attention mechanisms alone are sufficient without recurrence or convolutions"],
                        "guidelines": ["The response must explain the significance of the title"],
                    }
                },
            ]
        else:
            # Generic evaluation questions for any paper
            return [
                {
                    "inputs": {"inputs": "What is the main contribution of this paper?"},
                    "expectations": {
                        "key_concepts": ["research", "method", "contribution"],
                        "expected_facts": ["the paper presents a novel approach or finding"],
                        "guidelines": ["The response must be factual and concise"],
                    }
                },
                {
                    "inputs": {"inputs": "What problem does this paper address?"},
                    "expectations": {
                        "key_concepts": ["problem", "challenge", "limitation"],
                        "expected_facts": ["the paper identifies and addresses a specific problem"],
                        "guidelines": ["The response must clearly state the problem"],
                    }
                },
                {
                    "inputs": {"inputs": "What are the key results or findings?"},
                    "expectations": {
                        "key_concepts": ["results", "performance", "findings"],
                        "expected_facts": ["the paper reports experimental results or theoretical findings"],
                        "guidelines": ["The response must summarize the main results"],
                    }
                },
            ]
    
    @mlflow.trace
    def predict_fn(self, inputs: str) -> str:        
        return self.pipeline.query(inputs, trace=False)
    
    def run_evaluation(self, eval_dataset: List[Dict] = None, run_name: str = "rag_evaluation") -> pd.DataFrame:        
        if eval_dataset is None:
            eval_dataset = self.create_arxiv_eval_dataset()
        
        print(f"\n{'='*60}")
        print(f"Running MLflow Evaluation: {run_name}")
        print(f"{'='*60}\n")
        print(f"Questions to evaluate: {len(eval_dataset)}")
        print(f"Using scorers: RelevanceToQuery, Correctness, ExpectationsGuidelines")
        print()
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log configuration parameters
            mlflow.log_params(self.config.to_dict())
            
            # Run MLflow evaluation
            results = mlflow.genai.evaluate(
                data=eval_dataset,
                predict_fn=self.predict_fn,
                scorers=[
                    RelevanceToQuery(),
                    Correctness(),
                    ExpectationsGuidelines()
                ],
            )
            
            print(f"\n{'='*60}")
            print("Evaluation Complete!")
            print(f"{'='*60}")
            print(f"MLflow Run ID: {run.info.run_id}")
            print(f"View results at: {mlflow.get_tracking_uri()}")
            print()
            
            # Display summary metrics
            if hasattr(results, 'metrics'):
                print("Average Scores:")
                print("-" * 60)
                for metric_name, metric_value in results.metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")
            
            return results
    
    def compare_configurations(self, config_names: List[str], eval_dataset: List[Dict] = None) -> pd.DataFrame:        
        if eval_dataset is None:
            eval_dataset = self.create_arxiv_eval_dataset()
        
        comparison_results = []
        
        for config_name in config_names:
            print(f"\n{'='*60}")
            print(f"Evaluating Configuration: {config_name}")
            print(f"{'='*60}")
            
            # Create new pipeline with this config
            config = get_config(config_name)
            pipeline = RAGPipeline(config)
            
            # Load same documents
            pipeline.documents = self.pipeline.documents
            pipeline.create_vectorstore()
            pipeline.build_chain()
            
            # Evaluate
            evaluator = RAGEvaluator(pipeline)
            results = evaluator.run_evaluation(eval_dataset, run_name=f"eval_{config_name}")
            
            # Store summary
            result_dict = {"config": config_name}
            if hasattr(results, 'metrics'):
                result_dict.update(results.metrics)
            comparison_results.append(result_dict)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        print(f"\n{'='*60}")
        print("Configuration Comparison:")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False))
        
        return comparison_df


def evaluate_arxiv_paper(arxiv_id: str = "1706.03762", config_name: str = "balanced", custom_eval_dataset: List[Dict] = None): 
    print(f"\n{'='*60}")
    print(f"Evaluating ArXiv Paper: {arxiv_id}")
    print(f"{'='*60}\n")
    
    # Initialize pipeline
    config = get_config(config_name)
    pipeline = RAGPipeline(config)
    
    # Load ArXiv paper
    print(f"Loading ArXiv paper {arxiv_id}...")
    pipeline.load_documents(arxiv_id, source_type="arxiv")
    pipeline.create_vectorstore()
    pipeline.build_chain()
    
    # Create evaluator
    evaluator = RAGEvaluator(pipeline)
    
    # Get evaluation dataset
    if custom_eval_dataset is None:
        eval_dataset = evaluator.create_arxiv_eval_dataset(arxiv_id)
    else:
        eval_dataset = custom_eval_dataset
    
    # Run evaluation
    results = evaluator.run_evaluation(eval_dataset, run_name=f"arxiv_{arxiv_id}_eval")
    
    return results


if __name__ == "__main__":
    # Test evaluator with ArXiv paper
    print("Testing MLflow Evaluator with ArXiv Paper")
    print("="*60)
    
    # Method 1: Using convenience function
    print("\nMethod 1: Quick evaluation")
    results = evaluate_arxiv_paper(
        arxiv_id="1706.03762",  # Attention Is All You Need
        config_name="balanced"
    )
    
    # Method 2: Manual setup with custom evaluation
    print("\n\nMethod 2: Custom evaluation")
    
    config = RAGConfig()
    pipeline = RAGPipeline(config)
    
    # Load ArXiv paper
    pipeline.load_documents("1706.03762", source_type="arxiv")
    pipeline.create_vectorstore()
    pipeline.build_chain()
    
    # Create evaluator
    evaluator = RAGEvaluator(pipeline)
    
    # Custom evaluation dataset
    custom_eval = [
        {
            "inputs": {"inputs": "What is the Transformer model?"},
            "expectations": {
                "key_concepts": ["architecture", "attention", "encoder-decoder"],
                "expected_facts": ["The Transformer is a model architecture based entirely on attention mechanisms"],
                "guidelines": ["The response must explain the key innovation"],
            }
        }
    ]
    
    # Run evaluation
    results = evaluator.run_evaluation(custom_eval, run_name="custom_eval_test")
    
    print("\nEvaluation test complete!")