import yaml
import time
from pathlib import Path
from pipeline import MLOpsPipeline

def load_experiments_config():
    """Load experiment configurations from YAML"""
    config_path = Path("config/experiments.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['experiments']

def main():
    print("=" * 70)
    print("MLOps Training Pipeline with ELK Stack Monitoring")
    print("=" * 70)
    print("\nWaiting for ELK stack to be ready...")
    time.sleep(45)  # Wait for Logstash and Elasticsearch
    
    print("\nLoading experiment configurations...")
    experiments = load_experiments_config()
    
    print(f"\nFound {len(experiments)} experiments to run\n")
    
    results = []
    
    for idx, experiment in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"Running Experiment {idx}/{len(experiments)}: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"Model: {experiment['model_type']}")
        print(f"{'='*70}\n")
        
        try:
            # Create and run pipeline
            pipeline = MLOpsPipeline(experiment)
            metrics = pipeline.run(dataset='breast_cancer')
            
            results.append({
                'experiment': experiment['name'],
                'model': experiment['model_type'],
                'test_accuracy': metrics['test_accuracy'],
                'f1_score': metrics['f1_score'],
                'status': 'SUCCESS'
            })
            
            print(f"\n✓ Experiment completed successfully!")
            print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            print(f"\n✗ Experiment failed: {str(e)}")
            results.append({
                'experiment': experiment['name'],
                'model': experiment['model_type'],
                'status': 'FAILED',
                'error': str(e)
            })
        
        # Wait between experiments
        if idx < len(experiments):
            print(f"\nWaiting 10 seconds before next experiment...")
            time.sleep(10)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for result in results:
        status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"\n{status_symbol} {result['experiment']} ({result['model']})")
        if result['status'] == 'SUCCESS':
            print(f"  Accuracy: {result['test_accuracy']:.4f} | F1: {result['f1_score']:.4f}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("View results in Kibana: http://localhost:5601")
    print("="*70 + "\n")
    
    # Keep container running to view logs in Kibana
    print("Container will keep running for log analysis...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()