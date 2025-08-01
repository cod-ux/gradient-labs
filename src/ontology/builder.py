from typing import List
import pandas as pd

from ..intent_generation import IntentGenerator
from ..clustering import AgglomerativeClusterer, HDBSCANClusterer
from ..data import FileManager, DataLoader, DataPreprocessor
from ..utils.llm_client import LLMClient


class OntologyBuilder:
    """Main orchestrator for building customer intent ontology."""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.data_loader = DataLoader(file_manager)
        self.preprocessor = DataPreprocessor(file_manager)
        self.llm_client = LLMClient()
        
        # Initialize components
        self.intent_generator = IntentGenerator(
            file_manager, self.data_loader, self.preprocessor, self.llm_client
        )
        self.agglomerative_clusterer = AgglomerativeClusterer(
            file_manager, self.data_loader, self.preprocessor, self.llm_client
        )
        self.hdbscan_clusterer = HDBSCANClusterer(
            file_manager, self.data_loader, self.preprocessor, self.llm_client
        )
    
    def build_full_ontology(
        self,
        distance_threshold: float = 0.6,
        batch_size: int = 5,
        max_batches: int = None
    ) -> None:
        """Build complete ontology pipeline using entire dataset."""
        print("=== Starting Full Ontology Building Pipeline ===")
        
        # Step 1: Load or download dataset
        try:
            conversations_df = self.data_loader.get_full_dataset()
        except FileNotFoundError:
            print("Dataset not found, downloading...")
            self.data_loader.download_dataset()
            conversations_df = self.data_loader.get_full_dataset()
        
        # Step 2: Generate initial intents from conversations
        print("\n=== Step 1: Generating Initial Intents ===")
        self.intent_generator.generate_intent_list(
            conversations_df, batch_size=batch_size, max_batches=max_batches
        )
        
        # Step 3: Cluster intents to create higher-level categories
        print(f"\n=== Step 2: Clustering Intents (threshold={distance_threshold}) ===")
        self.agglomerative_clusterer.generate_clusters(distance_threshold)
        
        print("=== Ontology Building Pipeline Complete ===")
    
    def cluster_with_agglomerative(self, distance_threshold: float) -> dict:
        """Cluster existing ontology using Agglomerative clustering."""
        return self.agglomerative_clusterer.generate_clusters(distance_threshold)
    
    def cluster_with_hdbscan(self, min_cluster_size: int) -> dict:
        """Cluster existing ontology using HDBSCAN clustering."""
        return self.hdbscan_clusterer.generate_clusters(min_cluster_size)
    
    def compare_clustering_thresholds(self, thresholds: List[float]) -> dict:
        """Compare different distance thresholds for agglomerative clustering."""
        results = {}
        
        for threshold in thresholds:
            print(f"\n{'='*50}")
            print(f"Testing distance threshold: {threshold}")
            print(f"{'='*50}")
            
            # Generate clusters with current threshold
            cluster_results = self.cluster_with_agglomerative(threshold)
            
            # Store basic results
            results[threshold] = {
                'num_clusters': len(cluster_results),
                'cluster_sizes': [
                    cluster_data["cluster_size"] 
                    for cluster_data in cluster_results.values()
                ],
                'avg_cluster_size': sum(
                    cluster_data["cluster_size"] 
                    for cluster_data in cluster_results.values()
                ) / len(cluster_results) if cluster_results else 0
            }
            
            print(f"Results for threshold {threshold}:")
            print(f"  Number of clusters: {results[threshold]['num_clusters']}")
            print(f"  Average cluster size: {results[threshold]['avg_cluster_size']:.2f}")
        
        return results