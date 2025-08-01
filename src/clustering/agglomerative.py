from sklearn.cluster import AgglomerativeClustering
from .base import BaseClustering


class AgglomerativeClusterer(BaseClustering):
    """Agglomerative clustering implementation."""
    
    def get_method_name(self) -> str:
        return "Agglomerative"
    
    def _create_clustering_model(self, parameter_value: float, **kwargs):
        """Create AgglomerativeClustering model with distance threshold."""
        return AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=parameter_value,
            metric="cosine",
            linkage="average"
        )