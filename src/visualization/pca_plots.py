import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from typing import List

from ..intent_generation.models import CustomerIntent
from ..data import FileManager, DataLoader
from ..utils.llm_client import LLMClient


class PCAVisualizer:
    """Creates PCA visualizations for clustering results."""
    
    def __init__(
        self,
        file_manager: FileManager,
        data_loader: DataLoader,
        llm_client: LLMClient
    ):
        self.file_manager = file_manager
        self.data_loader = data_loader
        self.llm_client = llm_client
    
    def create_clustering_visualization(
        self,
        method: str,
        threshold_or_param: float
    ) -> None:
        """Create PCA visualization for clustering results."""
        print(f"Creating PCA visualization for {method} clustering...")
        
        # Load initial intents and cluster data
        initial_intents = self.data_loader.load_initial_intents()
        cluster_data = self.data_loader.load_cluster_data(method, threshold_or_param)
        
        # Create embeddings
        embeddings_array = self._create_embeddings(initial_intents)
        
        # Extract cluster labels
        labels = self._extract_cluster_labels(cluster_data, len(initial_intents))
        
        # Create visualization
        output_file = self.file_manager.get_pca_visualization_path(method)
        self._visualize_clustering_pca(
            embeddings_array, labels, initial_intents, output_file, method
        )
    
    def _create_embeddings(self, initial_intents: List[CustomerIntent]) -> np.ndarray:
        """Create embeddings for the initial intents."""
        sentences = [
            f"{intent.name}: {intent.description}. Examples: {', '.join(intent.examples)}" 
            for intent in initial_intents
        ]
        
        embeddings = self.llm_client.create_embeddings(sentences)
        embeddings_array = normalize(np.array(embeddings))
        
        return embeddings_array
    
    def _extract_cluster_labels(self, cluster_data: dict, num_intents: int) -> np.ndarray:
        """Extract cluster labels from cluster data."""
        labels = np.full(num_intents, -1)  # Initialize with -1
        
        for cluster_id, cluster_info in cluster_data.items():
            cluster_id_int = int(cluster_id)
            for intent_info in cluster_info['customer_intents']:
                intent_index = intent_info['index']
                labels[intent_index] = cluster_id_int
        
        return labels
    
    def _visualize_clustering_pca(
        self,
        embeddings_array: np.ndarray,
        labels: np.ndarray,
        initial_intents: List[CustomerIntent],
        output_file,
        clustering_method: str
    ) -> None:
        """Create 3D PCA visualization."""
        # Perform PCA to reduce dimensions to 3 for visualization
        pca = PCA(n_components=3)
        embeddings_pca = pca.fit_transform(embeddings_array)

        # Create a DataFrame for plotting
        df_pca = pd.DataFrame(embeddings_pca, columns=['PCA1', 'PCA2', 'PCA3'])
        df_pca['Cluster'] = labels
        df_pca['Intent Name'] = [initial_intents[i].name for i in range(len(initial_intents))]
        df_pca['Description'] = [initial_intents[i].description for i in range(len(initial_intents))]
        
        # Convert cluster labels to strings for better legend display
        df_pca['Cluster'] = df_pca['Cluster'].astype(str)
        
        # Handle noise points for HDBSCAN (label -1)
        if clustering_method.lower() == 'hdbscan':
            df_pca['Cluster'] = df_pca['Cluster'].replace('-1', 'Noise')

        # Create a 3D scatter plot
        fig = px.scatter_3d(
            df_pca,
            x='PCA1', y='PCA2', z='PCA3',
            color='Cluster',
            hover_data=['Intent Name', 'Description'],
            title=f'Customer Intent Clustering - {clustering_method} (PCA Visualization)',
            labels={'Cluster': f'{clustering_method} Cluster'}
        )

        # Customize hover template
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                          "Description: %{customdata[1]}<br>" +
                          "Cluster: %{marker.color}<br>" +
                          "PCA1: %{x:.3f}<br>" +
                          "PCA2: %{y:.3f}<br>" +
                          "PCA3: %{z:.3f}<br>" +
                          "<extra></extra>"
        )

        # Customize layout
        fig.update_layout(
            scene=dict(
                xaxis_title=f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PCA Component 3 ({pca.explained_variance_ratio_[2]:.1%} variance)',
                aspectmode='cube'
            ),
            title={
                'text': f'Customer Intent Clustering - {clustering_method} (PCA Visualization)',
                'x': 0.5,
                'xanchor': 'center'
            },
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            width=1000,
            height=800
        )

        # Save the plot to a file
        self.file_manager.ensure_directories()
        fig.write_html(output_file)
        print(f"{clustering_method} PCA visualization saved to {output_file}")
    
    def create_all_clustering_visualizations(self) -> None:
        """Create PCA visualizations for all available clustering results."""
        print("Creating 3D PCA visualizations for clustering results...")
        
        # Check for agglomerative clustering files
        agg_files = list(self.file_manager.clusters_dir.glob("agglomerative/*.json"))
        for file_path in agg_files:
            # Extract threshold from filename
            filename = file_path.stem
            if "agglomerative_" in filename:
                threshold_str = filename.split("agglomerative_")[1]
                try:
                    threshold = float(threshold_str)
                    self.create_clustering_visualization("agglomerative", threshold)
                except ValueError:
                    print(f"Could not parse threshold from {filename}")
        
        # Check for HDBSCAN clustering files
        hdb_files = list(self.file_manager.clusters_dir.glob("hdbscan/*.json"))
        for file_path in hdb_files:
            # Extract min_cluster_size from filename
            filename = file_path.stem
            if "hdbscan_" in filename:
                param_str = filename.split("hdbscan_")[1]
                try:
                    param = float(param_str)
                    self.create_clustering_visualization("hdbscan", param)
                except ValueError:
                    print(f"Could not parse parameter from {filename}")
        
        print("Visualization complete!")