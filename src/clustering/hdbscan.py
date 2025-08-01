import hdbscan
from .base import BaseClustering


class HDBSCANClusterer(BaseClustering):
    """HDBSCAN clustering implementation."""
    
    def get_method_name(self) -> str:
        return "HDBSCAN"
    
    def _create_clustering_model(self, parameter_value: float, **kwargs):
        """Create HDBSCAN model with min_cluster_size parameter."""
        return hdbscan.HDBSCAN(
            min_cluster_size=max(2, int(parameter_value)),
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="leaf",
            cluster_selection_epsilon=0.1
        )