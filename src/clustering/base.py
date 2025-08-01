from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.preprocessing import normalize

from ..intent_generation.models import CustomerIntent, ClusterNamingResponse
from ..data import FileManager, DataLoader, DataPreprocessor
from ..utils.llm_client import LLMClient


class BaseClustering(ABC):
    """Abstract base class for clustering algorithms."""
    
    def __init__(
        self,
        file_manager: FileManager,
        data_loader: DataLoader,
        preprocessor: DataPreprocessor,
        llm_client: LLMClient
    ):
        self.file_manager = file_manager
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.llm_client = llm_client
    
    @abstractmethod
    def _create_clustering_model(self, **kwargs):
        """Create the clustering model with specified parameters."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the clustering method."""
        pass
    
    def generate_clusters(self, parameter_value: float, **kwargs) -> Dict[str, Any]:
        """Generate clusters using the specific clustering algorithm."""
        print(f"Starting {self.get_method_name()} clustering with parameter: {parameter_value}")
        
        # Load initial intents
        initial_intents = self.data_loader.load_initial_intents()
        
        # Create embeddings
        embeddings_array = self._create_embeddings(initial_intents)
        
        # Perform clustering
        clustering_model = self._create_clustering_model(parameter_value=parameter_value, **kwargs)
        labels = clustering_model.fit_predict(embeddings_array)
        
        # Print clustering metrics
        self._print_clustering_metrics(labels)
        
        # Create cluster mapping
        cluster_mapping = self._create_cluster_mapping(labels, initial_intents)
        
        # Generate names for clusters
        named_cluster_mapping = self._generate_cluster_names(cluster_mapping)
        
        # Save results
        self._save_clustering_results(named_cluster_mapping, parameter_value)
        
        return named_cluster_mapping
    
    def _create_embeddings(self, initial_intents: List[CustomerIntent]) -> np.ndarray:
        """Create embeddings for the initial intents."""
        sentences = self.preprocessor.create_embeddings_sentences(initial_intents)
        
        # Create embeddings using LLM client
        embeddings = self.llm_client.create_embeddings(sentences)
        
        # Convert to numpy array and normalize
        embeddings_array = normalize(np.array(embeddings))
        
        return embeddings_array
    
    def _print_clustering_metrics(self, labels: np.ndarray) -> None:
        """Print clustering metrics."""
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        
        print(f"{self.get_method_name()} Clustering Labels:", labels)
        print(f"Number of clusters: {num_clusters}")
        
        # Handle noise points for algorithms that support them
        if -1 in labels:
            num_noise = len(labels[labels == -1])
            non_noise_labels = labels[labels >= 0]
            print(f"Number of noise points: {num_noise}")
            if len(non_noise_labels) > 0:
                print(f"Distribution of clusters: {np.bincount(non_noise_labels)}")
        else:
            print(f"Distribution of clusters: {np.bincount(labels)}")
    
    def _create_cluster_mapping(
        self, 
        labels: np.ndarray, 
        initial_intents: List[CustomerIntent]
    ) -> Dict[int, Dict[str, Any]]:
        """Create mapping of cluster IDs to their contents."""
        cluster_mapping = {}
        
        for i, label in enumerate(labels):
            label_key = int(label)
            if label_key not in cluster_mapping:
                cluster_mapping[label_key] = {
                    "cluster_size": 0,
                    "customer_intents": []
                }
            
            cluster_mapping[label_key]["customer_intents"].append({
                "index": i,
                "intent_name": initial_intents[i].name,
                "description": initial_intents[i].description,
                "examples": initial_intents[i].examples
            })
            cluster_mapping[label_key]["cluster_size"] += 1
        
        return cluster_mapping
    
    def _generate_cluster_names(
        self, 
        cluster_mapping: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Generate names for clusters using LLM."""
        named_cluster_mapping = {}
        total_clusters = len(cluster_mapping)
        
        for idx, (cluster_id, cluster_data) in enumerate(cluster_mapping.items()):
            print(f"Processing cluster {idx + 1}/{total_clusters} (ID: {cluster_id})")
            
            cluster_intents = cluster_data["customer_intents"]
            cluster_size = cluster_data["cluster_size"]
            
            # Generate cluster name
            cluster_name_response = self._generate_single_cluster_name(cluster_intents)
            
            cluster_data["common_intent"] = cluster_name_response.cluster_name
            cluster_data["common_intent_description"] = cluster_name_response.description
            named_cluster_mapping[cluster_id] = cluster_data
            
            print(f"  Cluster {cluster_id}: {cluster_size} intents -> '{cluster_name_response.cluster_name}'")
        
        return named_cluster_mapping
    
    def _generate_single_cluster_name(self, cluster_intents: List[Dict[str, Any]]) -> ClusterNamingResponse:
        """Generate name for a single cluster."""
        cluster_naming_prompt = f"""
You are an ontology engineer. You are building an ontology to classify customer conversations with customer support agents based on the intent of the customer. You will be given a list of customer intents that are related to each other because of a common underlying customer intent/need. Your job is to provide a single customer intent category name that can be used to commonly describe the customer intent/need of the majority of the intents given.

List of customer intents:
<Customer Intents>
{cluster_intents}
</Customer Intents>

<Rules>
- Name ≤ 4 words, description ≤ 25 words
- Include 2 example bullet points
- Do not rename/delete existing intents
- Only add genuinely new intents
</Rules>

<Naming Rules>
- If the intent is not covered by the current ontology, then it is a new intent.
- If the intent is a duplicate of an existing intent, then it is not a new intent.
- If the intent is a synonym of an existing intent, then it is not a new intent.
- If the intent is a subset of an existing intent, then it is not a new intent.
- If the intent is a super set of an existing intent, then it is not a new intent.
- If the intent is a related intent of an existing intent, then it is not a new intent.
- The following generic intents are not correct customer intents: "Request Explanation", "Escalation Issues". This is because the customer's intent is not clear and the intent is not specific to a particular need or goal.
- Don't use the phrasing "Escalation" in your intent categories.
- Name intent categories purely based on what the customer's need was and not based on other details of the conversation.
- For new intents, go high level instead of specific details (ex: "Get Refund" instead of "Refund Escalation")
- The customer intent name should be action-oriented, i.e, an action the customer wants to take or a  goal they want to achieve (ex: "Get Refund").
- In your naming try to title the intent category to represent all the similar intents in the cluster. If there are some intents that are completely dissimilar, then you don't have to try to incorporate them in the name.
</Naming Rules>

Here is the exact format of the JSON array of new intents:

    "name": "Customer Intent Name",
    "description": "New Intent Description",

Return the customer intent category name and a short description of the customer intent category.
"""
        
        return self.llm_client.parse_response(
            prompt=cluster_naming_prompt,
            response_model=ClusterNamingResponse,
            model="gpt-4.1",
            temperature=0.0
        )
    
    def _save_clustering_results(
        self, 
        named_cluster_mapping: Dict[int, Dict[str, Any]], 
        parameter_value: float
    ) -> None:
        """Save clustering results to files."""
        method_name = self.get_method_name().lower()
        
        # Save detailed cluster mapping
        self.preprocessor.save_cluster_data(
            named_cluster_mapping, 
            method_name, 
            parameter_value
        )
        
        # Save final ontology (clustered categories)
        ontology_categories = []
        for cluster_id, cluster_data in named_cluster_mapping.items():
            ontology_categories.append({
                "customer_intent": cluster_data["common_intent"],
                "customer_intent_description": cluster_data["common_intent_description"]
            })
        
        self.preprocessor.save_ontology(
            ontology_categories,
            method_name,
            parameter_value
        )
        
        # Print summary
        average_cluster_size = np.mean([
            cluster_data["cluster_size"] 
            for cluster_data in named_cluster_mapping.values()
        ])
        num_clusters = len(named_cluster_mapping)
        
        print("--------------------------------")
        print(f"{self.get_method_name()} Clustering Metrics:")
        print(f"  Average cluster size: {average_cluster_size:.2f}")
        print(f"  Number of clusters: {num_clusters}")
        print("--------------------------------")