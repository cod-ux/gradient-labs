from typing import List, Dict, Any
import json

from ..intent_generation.models import MergeIntentsResponse, MergedIntent
from ..data import FileManager, DataLoader, DataPreprocessor
from ..utils.llm_client import LLMClient


class IntentMerger:
    """Handles merging of similar customer intents."""
    
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
    
    def merge_intents(
        self, 
        method: str, 
        threshold_or_param: float
    ) -> List[Dict[str, str]]:
        """Merge similar intents from clustered categories."""
        print(f"Starting intent merging for {method} clustering with parameter {threshold_or_param}")
        
        # Load the intent categories
        customer_intent_categories = self.data_loader.load_intent_categories(
            method, threshold_or_param
        )
        
        # Find groups of similar intents
        merge_groups = self._find_similar_intent_groups(customer_intent_categories)
        
        if not merge_groups.groups:
            print("No similar intent groups found for merging")
            return customer_intent_categories
        
        # Merge the intents in each group
        updated_intents = self._merge_intent_groups(
            customer_intent_categories, 
            merge_groups
        )
        
        # Save merged results
        self.preprocessor.save_intent_categories(
            updated_intents,
            method,
            threshold_or_param,
            merged=True
        )
        
        print(f"Intent merging complete:")
        print(f"  Original intents: {len(customer_intent_categories)}")
        print(f"  Final intents: {len(updated_intents)}")
        print(f"  Merged {len(merge_groups.groups)} groups, reducing by {len(customer_intent_categories) - len(updated_intents)} intents")
        
        return updated_intents
    
    def _find_similar_intent_groups(
        self, 
        customer_intent_categories: List[Dict[str, str]]
    ) -> MergeIntentsResponse:
        """Find groups of similar intents that should be merged."""
        prompt = f"""
You are a classification engineer. You are given a list of customer intent categories. Your job is to return groups of customer intent categories that seem too similar to each other. Your groupings of customer intent categories will be used to reduce the number of customer intent categories by merging similar intents into a single intent and make the ontology mutually exclusive.

List of customer intent categories:
<Customer Intent Categories>
{customer_intent_categories}
</Customer Intent Categories>

<RulesForSuccess>
- Each customer intent category can be merged only once. So no two groups should have the same customer intent category in them.
- Each customer intent category should appear in at most one group.
- Use the 'customer_intent' field to identify the intents.
- Return groups as objects with an 'intents' field containing a list of customer_intent values that should be merged together.
- A group can contain 2 or more intents that are similar to each other and should be merged.
- Only group intents that are truly similar and serve the same customer intent.
- If there are no groups of intents that are similar to each other, return an empty list.
</RulesForSuccess>

Return a list of groups of intents that are not mutually exclusive from each other and should be merged.
"""
        
        return self.llm_client.parse_response(
            prompt=prompt,
            response_model=MergeIntentsResponse,
            model="gpt-4.1",
            temperature=0.0
        )
    
    def _merge_intent_groups(
        self,
        customer_intent_categories: List[Dict[str, str]],
        merge_groups: MergeIntentsResponse
    ) -> List[Dict[str, str]]:
        """Merge intents in each group and return updated list."""
        updated_intents = customer_intent_categories.copy()
        intents_to_remove = set()
        
        for group in merge_groups.groups:
            # Find all intents in the group
            group_intent_data = []
            
            for intent_name in group.intents:
                for intent in customer_intent_categories:
                    if intent['customer_intent'] == intent_name:
                        group_intent_data.append(intent)
                        break
            
            if len(group_intent_data) >= 2:  # Only merge if we have at least 2 intents
                # Create merged intent
                merged_intent_data = self._create_merged_intent(group_intent_data)
                
                # Add merged intent to the list
                updated_intents.append(merged_intent_data)
                
                # Mark original intents for removal
                intent_names = [intent['customer_intent'] for intent in group_intent_data]
                intents_to_remove.update(intent_names)
                
                print(f"Merged {len(group_intent_data)} intents: {', '.join(intent_names)} -> '{merged_intent_data['customer_intent']}'")
        
        # Remove the original intents that were merged
        final_intents = [
            intent for intent in updated_intents 
            if intent['customer_intent'] not in intents_to_remove
        ]
        
        return final_intents
    
    def _create_merged_intent(
        self, 
        group_intent_data: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """Create a single merged intent from a group of intents."""
        # Create detailed description of all intents to merge
        intents_description = "\n".join([
            f"- {intent['customer_intent']}: {intent['customer_intent_description']}" 
            for intent in group_intent_data
        ])
        
        merge_prompt = f"""
You are a classification engineer. You are given a group of customer intent categories. Your job is to merge all intents in the group into a single, unified intent.

Intents to merge:
{intents_description}

Here are the best practises for generating customer intent categories:
<Rules>
- One intent = one atomic purpose
- Name ≤ 4 words, description ≤ 25 words
- Include 2 example bullet points
- Do not rename/delete existing intents
- Only add genuinely new intents
</Rules>

<Best Practise for deciding if an intent is genuinely new>
- If the intent is not covered by the current ontology
- If the intent is not a duplicate of an existing intent
- If the intent is not a synonym of an existing intent
- If the intent is not a subset of an existing intent
- If the intent is not a super set of an existing intent
- If the intent is not a related intent of an existing intent
- For new intents, go high level instead of specific details (ex: "Get Refund" instead of "Refund Escalation")
- The customer intent name should be action-oriented, i.e, an action the customer wants to take or a  goal they want to achieve (ex: "Get Refund").
- Customer intent category name should be verb based, like a specific action the customer wants to take or a specific goal they want to achieve (ex: "Get Refund").
- Start the customer intent category name with an infinitive or gerund: "Track Order", "Return Product", "Report Billing Error".
</Best Practise for deciding if an intent is genuinely new>

Return a merged customer intent that combines all {len(group_intent_data)} intents into a single, cohesive category that captures their common purpose.
"""
        
        merge_response = self.llm_client.parse_response(
            prompt=merge_prompt,
            response_model=MergedIntent,
            model="o3"
        )
        
        return {
            "customer_intent": merge_response.customer_intent,
            "customer_intent_description": merge_response.customer_intent_description
        }