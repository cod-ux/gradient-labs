import pandas as pd
from datasets import load_dataset
from pathlib import Path
from typing import Optional

from .file_manager import FileManager


class DataLoader:
    """Handles loading and initial processing of conversation datasets."""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    def download_dataset(self, dataset_name: str = "NebulaByte/E-Commerce_Customer_Support_Conversations") -> pd.DataFrame:
        """Download dataset from HuggingFace and save to Excel file."""
        print(f"Downloading dataset: {dataset_name}")
        
        # Load dataset from HuggingFace
        ds = load_dataset(dataset_name)
        df = ds['train'].to_pandas()
        
        # Randomly shuffle the dataset
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Ensure raw directory exists
        self.file_manager.ensure_directories()
        
        # Save only the 'conversation' column to Excel file
        conversations_path = self.file_manager.get_raw_conversations_path()
        df[['conversation']].to_excel(conversations_path, index=False)
        
        print(f"Dataset saved to '{conversations_path}' with {len(df)} rows")
        print("Column: conversation")
        
        return df[['conversation']]
    
    def load_conversations(self) -> pd.DataFrame:
        """Load conversations from the saved Excel file."""
        conversations_path = self.file_manager.get_raw_conversations_path()
        
        if not conversations_path.exists():
            raise FileNotFoundError(
                f"Conversations file not found at {conversations_path}. "
                "Please run download_dataset() first."
            )
        
        df = pd.read_excel(conversations_path)
        print(f"Loaded {len(df)} conversations from {conversations_path}")
        
        return df[['conversation']]
    
    def get_full_dataset(self) -> pd.DataFrame:
        """Get the complete dataset for processing."""
        df = self.load_conversations()
        print(f"Full dataset: {len(df)} conversations")
        return df
    
    def load_initial_intents(self, version: Optional[str] = None) -> list:
        """Load initial intents from JSON file."""
        initial_intents_path = self.file_manager.get_initial_intents_path(version)
        
        if not initial_intents_path.exists():
            raise FileNotFoundError(f"Initial intents file not found at {initial_intents_path}")
        
        import json
        with open(initial_intents_path, 'r') as f:
            initial_intents_data = json.load(f)
        
        # Import here to avoid circular imports
        from ..intent_generation.models import CustomerIntent
        initial_intents = [CustomerIntent(**intent_data) for intent_data in initial_intents_data]
        
        print(f"Loaded initial intents with {len(initial_intents)} intents from {initial_intents_path}")
        return initial_intents
    
    def load_cluster_data(self, method: str, threshold_or_param: float) -> dict:
        """Load cluster data from JSON file."""
        cluster_path = self.file_manager.get_cluster_path(method, threshold_or_param)
        
        if not cluster_path.exists():
            raise FileNotFoundError(f"Cluster file not found at {cluster_path}")
        
        import json
        with open(cluster_path, 'r') as f:
            cluster_data = json.load(f)
        
        print(f"Loaded cluster data from {cluster_path}")
        return cluster_data
    
    def load_ontology(
        self, 
        method: str, 
        threshold_or_param: float, 
        merged: bool = False
    ) -> list:
        """Load final ontology from JSON file."""
        ontology_path = self.file_manager.get_ontology_path(
            method, threshold_or_param, merged
        )
        
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found at {ontology_path}")
        
        import json
        with open(ontology_path, 'r') as f:
            ontology_data = json.load(f)
        
        print(f"Loaded ontology with {len(ontology_data)} categories from {ontology_path}")
        return ontology_data
    
    def load_classified_conversations(
        self, 
        method: str, 
        threshold_or_param: float
    ) -> pd.DataFrame:
        """Load classified conversations from Excel file."""
        classified_path = self.file_manager.get_classified_conversations_path(
            method, threshold_or_param
        )
        
        if not classified_path.exists():
            raise FileNotFoundError(f"Classified conversations not found at {classified_path}")
        
        df = pd.read_excel(classified_path)
        print(f"Loaded {len(df)} classified conversations from {classified_path}")
        
        return df