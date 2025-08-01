#!/usr/bin/env python3
"""
Main pipeline orchestrator for customer intent ontology building.

This script orchestrates the complete pipeline:
1. Intent Generation - Create initial intents from conversations
2. Clustering - Group similar intents using clustering algorithms  
3. Ontology Building - Create higher-level intent categories
4. Evaluation - Assess ontology quality and coverage
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import FileManager, DataLoader, DataPreprocessor
from src.utils import LLMClient
from src.ontology import OntologyBuilder
from src.evaluation import OntologyEvaluator
from src.visualization import PCAVisualizer


def main():
    """Run the complete ontology building pipeline."""
    print("=" * 60)
    print("CUSTOMER INTENT ONTOLOGY BUILDING PIPELINE")
    print("=" * 60)
    
    # Initialize core components
    file_manager = FileManager()
    data_loader = DataLoader(file_manager) 
    preprocessor = DataPreprocessor(file_manager)
    llm_client = LLMClient()
    
    # Initialize main orchestrators
    ontology_builder = OntologyBuilder(file_manager)
    evaluator = OntologyEvaluator(file_manager, data_loader, preprocessor, llm_client)
    visualizer = PCAVisualizer(file_manager, data_loader, llm_client)
    
    # Ensure all directories exist
    file_manager.ensure_directories()
    
    # Step 1: Build full ontology
    print("\n" + "=" * 60)
    print("STEP 1: BUILDING ONTOLOGY")  
    print("=" * 60)
    
    ontology_builder.build_full_ontology(
        distance_threshold=0.6,
        batch_size=5,
        max_batches=300  # Limit for development, set to None for full dataset
    )
    
    # Step 2: Evaluate the built ontology
    print("\n" + "=" * 60)
    print("STEP 2: EVALUATING ONTOLOGY")
    print("=" * 60)
    
    evaluation_metrics = evaluator.evaluate_ontology("agglomerative", 0.6)
    
    print(f"\nEVALUATION RESULTS:")
    print(f"  Coverage: {evaluation_metrics.coverage:.2f}%")
    print(f"  Number of clusters: {evaluation_metrics.num_clusters}")
    print(f"  Max similarity: {evaluation_metrics.max_similarity:.3f}")
    print(f"  Passes exclusivity: {'✅' if evaluation_metrics.passes_exclusivity else '❌'}")
    print(f"  Redundant intents: {len(evaluation_metrics.redundant_intents)}")
    
    # Step 3: Create visualizations
    print("\n" + "=" * 60)
    print("STEP 3: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    visualizer.create_clustering_visualization("agglomerative", 0.6)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"✅ Ontology built with {evaluation_metrics.num_clusters} categories")
    print(f"✅ Coverage: {evaluation_metrics.coverage:.2f}%")
    print(f"✅ Visualization saved to outputs/visualizations/")
    print(f"✅ All results saved to organized data structure")


def compare_thresholds():
    """Compare different clustering thresholds."""
    print("=" * 60)
    print("THRESHOLD COMPARISON MODE")
    print("=" * 60)
    
    # Initialize components
    file_manager = FileManager()
    data_loader = DataLoader(file_manager)
    preprocessor = DataPreprocessor(file_manager)
    llm_client = LLMClient()
    
    ontology_builder = OntologyBuilder(file_manager)
    evaluator = OntologyEvaluator(file_manager, data_loader, preprocessor, llm_client)
    
    # Define thresholds to test
    thresholds = [0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64]
    
    # Compare clustering results
    clustering_results = ontology_builder.compare_clustering_thresholds(thresholds)
    
    # Evaluate each threshold
    methods_and_params = [("agglomerative", threshold) for threshold in thresholds]
    comparison_df = evaluator.compare_ontologies(methods_and_params)
    
    print("\n" + "=" * 60)
    print("THRESHOLD COMPARISON COMPLETE!")
    print("=" * 60)
    print("Results saved to data/evaluations/comparison_reports/ontology_comparison_summary.xlsx")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Customer Intent Ontology Building Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["pipeline", "compare"], 
        default="pipeline",
        help="Run mode: 'pipeline' for full pipeline, 'compare' for threshold comparison"
    )
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        main()
    elif args.mode == "compare":
        compare_thresholds()