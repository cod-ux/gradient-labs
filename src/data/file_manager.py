from pathlib import Path
from typing import Optional


class FileManager:
    """Centralized file path management for the ontology building system."""
    
    def __init__(self, base_data_dir: str = "data"):
        self.base_dir = Path(base_data_dir)
        
        # Data directories
        self.raw_dir = self.base_dir / "raw"
        self.initial_intents_dir = self.base_dir / "initial_intents"
        self.clusters_dir = self.base_dir / "clusters"
        self.ontologies_dir = self.base_dir / "ontologies"
        self.evaluations_dir = self.base_dir / "evaluations"
        
        # Output directories (simplified - at root level)
        self.visualizations_dir = Path("visualizations")
    
    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.raw_dir,
            self.initial_intents_dir,
            self.clusters_dir / "agglomerative",
            self.clusters_dir / "hdbscan",
            self.ontologies_dir / "agglomerative",
            self.ontologies_dir / "hdbscan",
            self.evaluations_dir / "classified_conversations",
            self.evaluations_dir / "comparison_reports",
            self.evaluations_dir / "metrics",
            self.visualizations_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # Raw data paths
    def get_raw_conversations_path(self) -> Path:
        """Get path to the raw customer conversations Excel file."""
        return self.raw_dir / "customer_conversations.xlsx"
    
    # Initial intents paths (Stage 1: Raw intents from conversations)
    def get_initial_intents_path(self, version: Optional[str] = None) -> Path:
        """Get path to the initial intents JSON file generated from conversations."""
        if version:
            filename = f"initial_intents_v{version}.json"
        else:
            filename = "initial_intents.json"
        return self.initial_intents_dir / filename
    
    # Cluster paths
    def get_cluster_path(self, method: str, threshold_or_param: float) -> Path:
        """Get path to cluster results file based on method and parameter."""
        method_dir = self.clusters_dir / method.lower()
        
        if method.lower() == "agglomerative":
            filename = f"reduced_cluster_agglomerative_{threshold_or_param}.json"
        elif method.lower() == "hdbscan":
            filename = f"reduced_cluster_hdbscan_{threshold_or_param}.json"
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return method_dir / filename
    
    # Ontology paths (Stage 3: Final clustered categories - the actual ontology)
    def get_ontology_path(
        self, 
        method: str, 
        threshold_or_param: float, 
        merged: bool = False
    ) -> Path:
        """Get path to final ontology file based on method and parameter."""
        method_dir = self.ontologies_dir / method.lower()
        
        if method.lower() == "agglomerative":
            filename = f"ontology_agg_{threshold_or_param}.json"
        elif method.lower() == "hdbscan":
            filename = f"ontology_hdbscan_{threshold_or_param}.json"
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        if merged:
            filename = filename.replace('.json', '_merged.json')
        
        return method_dir / filename
    
    # Evaluation paths
    def get_classified_conversations_path(
        self, 
        method: str, 
        threshold_or_param: float
    ) -> Path:
        """Get path to classified conversations Excel file."""
        classified_dir = self.evaluations_dir / "classified_conversations"
        
        if method.lower() == "agglomerative":
            filename = f"classified_customer_conversations_agg_{threshold_or_param}.xlsx"
        elif method.lower() == "hdbscan":
            filename = f"classified_customer_conversations_hdbscan_{threshold_or_param}.xlsx"
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return classified_dir / filename
    
    def get_comparison_report_path(self, report_name: str) -> Path:
        """Get path to comparison report file."""
        return self.evaluations_dir / "comparison_reports" / f"{report_name}.xlsx"
    
    def get_metrics_path(self, metric_name: str) -> Path:
        """Get path to metrics JSON file."""
        return self.evaluations_dir / "metrics" / f"{metric_name}.json"
    
    # Visualization paths
    def get_visualization_path(self, viz_name: str) -> Path:
        """Get path to visualization HTML file."""
        return self.visualizations_dir / f"{viz_name}.html"
    
    def get_pca_visualization_path(self, method: str) -> Path:
        """Get path to PCA visualization file for a clustering method."""
        filename = f"{method.lower()}_clustering_pca_visualization.html"
        return self.visualizations_dir / filename
    
    # Legacy file support (for migration)
    def get_legacy_file_paths(self) -> dict:
        """Get paths to existing files in the root directory for migration."""
        root_dir = Path(".")
        
        legacy_files = {
            # Initial intents files (previously called "ontology")
            "initial_intents": root_dir / "customer_intent_ontology.json",
            
            # Raw data
            "conversations": root_dir / "customer_conversations.xlsx",
            
            # Cluster files - agglomerative
            "clusters_agg": [
                f for f in root_dir.glob("reduced_cluster_agglomerative_*.json")
            ],
            
            # Cluster files - hdbscan
            "clusters_hdb": [
                f for f in root_dir.glob("reduced_cluster_hdbscan_*.json")
            ],
            
            # Ontology files - agglomerative (previously called "intent_categories")
            "ontologies_agg": [
                f for f in root_dir.glob("clustered_customer_intents_[0-9]*.json")
            ],
            
            # Ontology files - hdbscan (previously called "intent_categories")
            "ontologies_hdb": [
                f for f in root_dir.glob("clustered_customer_intents_hdbscan_*.json")
            ],
            
            # Evaluation files
            "classified": [
                f for f in root_dir.glob("classified_customer_conversations*.xlsx")
            ],
            
            # Comparison reports
            "reports": [
                f for f in root_dir.glob("ontology_comparison_summary*.xlsx")
            ],
            
            # Visualizations
            "visualizations": [
                f for f in root_dir.glob("*_visualization.html")
            ]
        }
        
        return legacy_files