import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

from ..intent_generation.models import MutualExclusivityResponse
from ..utils.llm_client import LLMClient


class MetricsCalculator:
    """Calculates various metrics for ontology evaluation."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def check_for_duplicates(
        self, 
        customer_intent_categories: List[Dict[str, str]]
    ) -> MutualExclusivityResponse:
        """Check for duplicate customer intents using LLM."""
        mutual_exclusivity_prompt = f"""
You are an ontology engineer. You are given a list of categories to classify customer conversations based on the customer's intent noticed in the conversation. Your job is to identify if there are any exact duplicates among the customer intent categories.

<Rules>
- Look for pairs of customer intents that are essentially the same thing but just phrased differently
- Similar intents that represent different levels of specificity are acceptable (e.g., "Get Refund" vs "Get Partial Refund")
- Only flag true duplicates where two intents serve the exact same purpose and could be merged without losing meaning
- Return "yes" if there are any exact duplicates, "no" if all intents are unique
</Rules>

List of customer intent categories:
<Customer Intent Categories>
{customer_intent_categories}
</Customer Intent Categories>

Return whether there are any exact duplicate customer intents and list the duplicate pairs if any exist.
"""
        
        return self.llm_client.parse_response(
            prompt=mutual_exclusivity_prompt,
            response_model=MutualExclusivityResponse,
            model="o3"
        )
    
    def evaluate_mutual_exclusivity_cosine_similarity(
        self, 
        customer_intent_categories: List[Dict[str, str]]
    ) -> Tuple[float, bool]:
        """Evaluate mutual exclusivity using cosine similarity between embeddings."""
        print("Starting mutual exclusivity evaluation using cosine similarity...")
        
        # Create embeddings for all customer intents
        embeddings = []
        for intent in customer_intent_categories:
            text = f"{intent['customer_intent']}. {intent['customer_intent_description']}"
            embedding = self.llm_client.create_single_embedding(text)
            embeddings.append(embedding)
        
        # Convert to numpy array and stack
        embeddings_matrix = np.vstack(embeddings)
        
        # Calculate pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Get upper triangular values (excluding diagonal)
        upper_triangular = []
        for i in range(len(customer_intent_categories)):
            for j in range(i + 1, len(customer_intent_categories)):
                upper_triangular.append(similarity_matrix[i, j])
        
        # Check exclusivity with threshold
        max_similarity = max(upper_triangular) if upper_triangular else 0.0
        threshold = 0.85
        passes_exclusivity = max_similarity < threshold
        
        print(f"Max pairwise similarity = {max_similarity:.3f}")
        print(f"Threshold = {threshold}")
        
        return max_similarity, passes_exclusivity