#!/usr/bin/env python3
"""
Script for generating visualizations from existing clustering results.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import FileManager, DataLoader
from src.utils import LLMClient
from src.visualization import PCAVisualizer


def main():
    """Generate visualizations for all available clustering results."""
    print("=" * 60)
    print("VISUALIZATION GENERATION")
    print("=" * 60)
    
    # Initialize components
    file_manager = FileManager()
    data_loader = DataLoader(file_manager)
    llm_client = LLMClient()
    
    visualizer = PCAVisualizer(file_manager, data_loader, llm_client)
    
    # Generate all available visualizations
    visualizer.create_all_clustering_visualizations()
    
    print("\n" + "=" * 60)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("=" * 60)
    print("Check outputs/visualizations/ for HTML files")


def generate_specific_visualization(method, parameter):
    """Generate visualization for a specific clustering result."""
    print(f"Generating visualization for {method} clustering with parameter {parameter}")
    
    # Initialize components
    file_manager = FileManager()
    data_loader = DataLoader(file_manager)
    llm_client = LLMClient()
    
    visualizer = PCAVisualizer(file_manager, data_loader, llm_client)
    
    # Generate specific visualization
    visualizer.create_clustering_visualization(method, parameter)
    
    print(f"Visualization saved to outputs/visualizations/{method}_clustering_pca_visualization.html")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate clustering visualizations")
    parser.add_argument(
        "--method",
        choices=["agglomerative", "hdbscan"],
        help="Clustering method"
    )
    parser.add_argument(
        "--parameter",
        type=float,
        help="Clustering parameter (distance threshold or min_cluster_size)"
    )
    
    args = parser.parse_args()
    
    if args.method and args.parameter:
        generate_specific_visualization(args.method, args.parameter)
    else:
        main()