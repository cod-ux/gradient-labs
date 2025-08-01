#!/usr/bin/env python3
"""
Script for comparing different clustering thresholds and parameters.
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


def compare_agglomerative_thresholds(thresholds=None):
    """Compare different distance thresholds for agglomerative clustering."""
    if thresholds is None:
        thresholds = [0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64]
    
    print("=" * 60)
    print("AGGLOMERATIVE CLUSTERING THRESHOLD COMPARISON")
    print("=" * 60)
    
    # Initialize components
    file_manager = FileManager()
    data_loader = DataLoader(file_manager)
    preprocessor = DataPreprocessor(file_manager)
    llm_client = LLMClient()
    
    ontology_builder = OntologyBuilder(file_manager)
    evaluator = OntologyEvaluator(file_manager, data_loader, preprocessor, llm_client)
    
    # Compare clustering results
    clustering_results = ontology_builder.compare_clustering_thresholds(thresholds)
    
    # Evaluate each threshold
    methods_and_params = [("agglomerative", threshold) for threshold in thresholds]
    comparison_df = evaluator.compare_ontologies(methods_and_params)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)
    print("Results saved to data/evaluations/comparison_reports/ontology_comparison_summary.xlsx")
    
    return comparison_df


def compare_hdbscan_parameters(min_cluster_sizes=None):
    """Compare different min_cluster_size parameters for HDBSCAN clustering."""
    if min_cluster_sizes is None:
        min_cluster_sizes = [2, 3, 4, 5, 6]
    
    print("=" * 60)
    print("HDBSCAN CLUSTERING PARAMETER COMPARISON")
    print("=" * 60)
    
    # Initialize components
    file_manager = FileManager()
    data_loader = DataLoader(file_manager)
    preprocessor = DataPreprocessor(file_manager)
    llm_client = LLMClient()
    
    ontology_builder = OntologyBuilder(file_manager)
    evaluator = OntologyEvaluator(file_manager, data_loader, preprocessor, llm_client)
    
    # Test each parameter
    for min_cluster_size in min_cluster_sizes:
        print(f"\n{'='*50}")
        print(f"Testing HDBSCAN with min_cluster_size: {min_cluster_size}")
        print(f"{'='*50}")
        
        ontology_builder.cluster_with_hdbscan(min_cluster_size)
    
    # Evaluate each parameter
    methods_and_params = [("hdbscan", param) for param in min_cluster_sizes]
    comparison_df = evaluator.compare_ontologies(methods_and_params)
    
    print("\n" + "=" * 60)
    print("HDBSCAN COMPARISON COMPLETE!")
    print("=" * 60)
    print("Results saved to data/evaluations/comparison_reports/ontology_comparison_summary.xlsx")
    
    return comparison_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare clustering parameters")
    parser.add_argument(
        "--method",
        choices=["agglomerative", "hdbscan", "both"],
        default="agglomerative",
        help="Clustering method to compare"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="Distance thresholds for agglomerative clustering"
    )
    parser.add_argument(
        "--min-cluster-sizes",
        nargs="+",
        type=int,
        help="Min cluster sizes for HDBSCAN clustering"
    )
    
    args = parser.parse_args()
    
    if args.method == "agglomerative":
        compare_agglomerative_thresholds(args.thresholds)
    elif args.method == "hdbscan":
        compare_hdbscan_parameters(args.min_cluster_sizes)
    elif args.method == "both":
        compare_agglomerative_thresholds(args.thresholds)
        compare_hdbscan_parameters(args.min_cluster_sizes)