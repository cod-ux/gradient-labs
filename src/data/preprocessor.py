import pandas as pd
from typing import List, Dict, Any
import json

from .file_manager import FileManager


class DataPreprocessor:
    """Handles data preprocessing and formatting operations."""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    def format_conversations_for_prompt(
        self, 
        conversations: List[str], 
        start_index: int = 0
    ) -> str:
        """Format a list of conversations for LLM prompts."""
        formatted = "\n".join([
            f"{start_index + idx + 1}. {conv}" 
            for idx, conv in enumerate(conversations)
        ])
        return formatted
    
    def batch_conversations(
        self, 
        df: pd.DataFrame, 
        batch_size: int = 5
    ) -> List[List[str]]:
        """Split conversations into batches."""
        conversations = df['conversation'].tolist()
        batches = []
        
        for i in range(0, len(conversations), batch_size):
            batch = conversations[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def save_initial_intents(self, initial_intents: List[Any], version: str = None) -> None:
        """Save initial intents to JSON file."""
        from ..intent_generation.models import CustomerIntent
        
        # Ensure all items are CustomerIntent objects
        if initial_intents and isinstance(initial_intents[0], CustomerIntent):
            initial_intents_dict = [intent.model_dump() for intent in initial_intents]
        else:
            initial_intents_dict = initial_intents
        
        self.file_manager.ensure_directories()
        initial_intents_path = self.file_manager.get_initial_intents_path(version)
        
        with open(initial_intents_path, 'w') as f:
            json.dump(initial_intents_dict, f, indent=2)
        
        print(f"Initial intents saved to '{initial_intents_path}' with {len(initial_intents_dict)} intents")
    
    def save_cluster_data(
        self, 
        cluster_data: Dict[str, Any], 
        method: str, 
        threshold_or_param: float
    ) -> None:
        """Save cluster data to JSON file."""
        self.file_manager.ensure_directories()
        cluster_path = self.file_manager.get_cluster_path(method, threshold_or_param)
        
        with open(cluster_path, 'w') as f:
            json.dump(cluster_data, f, indent=2)
        
        print(f"Cluster data saved to '{cluster_path}'")
    
    def save_ontology(
        self, 
        ontology: List[Dict[str, str]], 
        method: str, 
        threshold_or_param: float,
        merged: bool = False
    ) -> None:
        """Save final ontology to JSON file."""
        self.file_manager.ensure_directories()
        ontology_path = self.file_manager.get_ontology_path(
            method, threshold_or_param, merged
        )
        
        with open(ontology_path, 'w') as f:
            json.dump(ontology, f, indent=2)
        
        suffix = " (merged)" if merged else ""
        print(f"Ontology{suffix} saved to '{ontology_path}' with {len(ontology)} categories")
    
    def save_classified_conversations(
        self, 
        df: pd.DataFrame, 
        method: str, 
        threshold_or_param: float
    ) -> None:
        """Save classified conversations to Excel file."""
        self.file_manager.ensure_directories()
        classified_path = self.file_manager.get_classified_conversations_path(
            method, threshold_or_param
        )
        
        df.to_excel(classified_path, index=False)
        print(f"Classified conversations saved to '{classified_path}' with {len(df)} rows")
    
    def save_comparison_report(self, df: pd.DataFrame, report_name: str) -> None:
        """Save comparison report to Excel file."""
        self.file_manager.ensure_directories()
        report_path = self.file_manager.get_comparison_report_path(report_name)
        
        df.to_excel(report_path, index=False)
        print(f"Comparison report saved to '{report_path}'")
    
    def save_metrics(self, metrics: Dict[str, Any], metric_name: str) -> None:
        """Save metrics to JSON file."""
        self.file_manager.ensure_directories()
        metrics_path = self.file_manager.get_metrics_path(metric_name)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to '{metrics_path}'")
    
    def extract_intent_names_from_categories(
        self, 
        categories: List[Dict[str, str]]
    ) -> List[str]:
        """Extract intent names from categories list."""
        return [category['customer_intent'] for category in categories]
    
    def create_embeddings_sentences(self, ontology: List[Any]) -> List[str]:
        """Create sentences for embedding generation from ontology."""
        from ..intent_generation.models import CustomerIntent
        
        sentences = []
        for intent in ontology:
            if isinstance(intent, CustomerIntent):
                sentence = f"{intent.name}: {intent.description}. Examples: {', '.join(intent.examples)}"
            else:
                # Assume it's a dict
                sentence = f"{intent['name']}: {intent['description']}. Examples: {', '.join(intent['examples'])}"
            sentences.append(sentence)
        
        return sentences
    
    def validate_data_consistency(self) -> Dict[str, bool]:
        """Validate consistency across data files."""
        validation_results = {}
        
        try:
            # Check if ontology exists
            ontology_path = self.file_manager.get_ontology_path()
            validation_results['ontology_exists'] = ontology_path.exists()
            
            # Check if raw conversations exist
            conversations_path = self.file_manager.get_raw_conversations_path()
            validation_results['conversations_exist'] = conversations_path.exists()
            
            # Check for cluster files
            cluster_files = list(self.file_manager.clusters_dir.rglob("*.json"))
            validation_results['cluster_files_exist'] = len(cluster_files) > 0
            validation_results['cluster_files_count'] = len(cluster_files)
            
            # Check for intent category files
            category_files = list(self.file_manager.intent_categories_dir.rglob("*.json"))
            validation_results['category_files_exist'] = len(category_files) > 0
            validation_results['category_files_count'] = len(category_files)
            
        except Exception as e:
            validation_results['error'] = str(e)
        
        return validation_results