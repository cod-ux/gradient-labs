import pandas as pd
import asyncio
from typing import List, Dict, Any, Tuple
from collections import Counter

from ..intent_generation.models import ClassificationResponse, MutualExclusivityResponse, EvaluationMetrics
from ..data import FileManager, DataLoader, DataPreprocessor
from ..utils.llm_client import LLMClient
from .metrics import MetricsCalculator
from .classifier import ConversationClassifier


class OntologyEvaluator:
    """Main evaluator for ontology quality assessment."""
    
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
        
        self.metrics_calculator = MetricsCalculator(llm_client)
        self.classifier = ConversationClassifier(llm_client)
    
    def evaluate_ontology(
        self, 
        method: str, 
        threshold_or_param: float
    ) -> EvaluationMetrics:
        """Comprehensive evaluation of an ontology."""
        print(f"Starting ontology evaluation for {method} clustering with parameter {threshold_or_param}")
        
        # Load ontology
        ontology_categories = self.data_loader.load_ontology(
            method, threshold_or_param
        )
        
        print(f"Evaluating ontology with {len(ontology_categories)} categories")
        
        # Step 1: Check for duplicate intents
        print("Checking for duplicate customer intents...")
        duplicate_check = self.metrics_calculator.check_for_duplicates(ontology_categories)
        
        print(f"Has duplicate intents: {duplicate_check.has_duplicates}")
        if duplicate_check.duplicate_pairs:
            print(f"Duplicate pairs found: {duplicate_check.duplicate_pairs}")
        print(f"Reason: {duplicate_check.reason}")
        
        # Step 2: Evaluate mutual exclusivity using cosine similarity
        max_similarity, passes_exclusivity = self.metrics_calculator.evaluate_mutual_exclusivity_cosine_similarity(
            ontology_categories
        )
        print(f"Cosine similarity mutual exclusivity score: {max_similarity:.3f}")
        print("Ontology is mutually-exclusive ✅" if passes_exclusivity else "Ontology has overlapping labels ❌")
        
        # Step 3: Evaluate collective exhaustivity
        print("Evaluating collective exhaustivity...")
        unclassified_percentage, redundant_intents = self._evaluate_collective_exhaustivity(
            ontology_categories, method, threshold_or_param
        )
        
        coverage = 100 - unclassified_percentage
        print(f"Coverage: {coverage:.2f}%")
        print(f"Unclassified conversations: {unclassified_percentage:.2f}%")
        
        if redundant_intents:
            print(f"Found {len(redundant_intents)} redundant intents: {redundant_intents}")
        
        # Create evaluation metrics
        metrics = EvaluationMetrics(
            coverage=coverage,
            mutual_exclusivity_score=max_similarity,
            num_clusters=len(ontology_categories),
            max_similarity=max_similarity,
            passes_exclusivity=passes_exclusivity,
            redundant_intents=redundant_intents
        )
        
        return metrics
    
    def _evaluate_collective_exhaustivity(
        self,
        ontology_categories: List[Dict[str, str]],
        method: str,
        threshold_or_param: float
    ) -> Tuple[float, List[str]]:
        """Evaluate how well the ontology covers all conversations."""
        # Load conversations
        conversations_df = self.data_loader.load_conversations()
        
        print(f"Classifying {len(conversations_df)} conversations...")
        
        # Classify conversations
        classified_df = self.classifier.classify_conversations_batch(
            conversations_df, ontology_categories
        )
        
        # Save classified results
        self.preprocessor.save_classified_conversations(
            classified_df, method, threshold_or_param
        )
        
        # Calculate unclassified percentage - look for 'Other' or 'other' after trimming spaces
        unclassified_count = 0
        for classification in classified_df['classification']:
            if pd.notna(classification):
                trimmed_classification = str(classification).strip()
                if trimmed_classification in ['Other', 'other']:
                    unclassified_count += 1
        
        unclassified_percentage = (unclassified_count / len(classified_df)) * 100
        
        # Check for redundant intents (0% usage)
        redundant_intents = self._check_redundant_intents(
            classified_df, ontology_categories
        )
        
        return unclassified_percentage, redundant_intents
    
    def _check_redundant_intents(
        self,
        classified_df: pd.DataFrame,
        ontology_categories: List[Dict[str, str]]
    ) -> List[str]:
        """Check for intents that are never used in classification."""
        # Extract all intent names
        intent_names = [intent['customer_intent'] for intent in ontology_categories]
        
        # Count classifications (handle comma-separated classifications)
        all_classifications = []
        for classification in classified_df['classification']:
            if pd.notna(classification):
                individual_classifications = [cls.strip() for cls in str(classification).split(',')]
                all_classifications.extend(individual_classifications)
        
        classification_counts = Counter(all_classifications)
        
        # Find intents with 0 occurrences
        redundant_intents = []
        for intent_name in intent_names:
            if classification_counts.get(intent_name, 0) == 0:
                redundant_intents.append(intent_name)
        
        return redundant_intents
    
    def compare_ontologies(self, methods_and_params: List[Tuple[str, float]]) -> pd.DataFrame:
        """Compare multiple ontologies and return comparison results."""
        results = []
        
        for method, param in methods_and_params:
            print(f"\n{'='*50}")
            print(f"Evaluating {method} clustering with parameter {param}")
            print(f"{'='*50}")
            
            try:
                metrics = self.evaluate_ontology(method, param)
                
                result = {
                    'Method': method,
                    'Parameter': param,
                    'Coverage': metrics.coverage,
                    'Num_Clusters': metrics.num_clusters,
                    'Max_Similarity': metrics.max_similarity,
                    'Passes_Exclusivity': '✅' if metrics.passes_exclusivity else '❌',
                    'Redundant_Intents_Count': len(metrics.redundant_intents),
                    'Redundant_Intents': ', '.join(metrics.redundant_intents)
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {method} with parameter {param}: {e}")
                continue
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Save comparison report
        self.preprocessor.save_comparison_report(comparison_df, "ontology_comparison_summary")
        
        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")
        print(comparison_df.to_string(index=False))
        
        return comparison_df