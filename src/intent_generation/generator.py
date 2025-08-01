import pandas as pd
from typing import List, Optional

from .models import CustomerIntent, IntentResponse
from ..data import FileManager, DataLoader, DataPreprocessor
from ..utils.llm_client import LLMClient


class IntentGenerator:
    """Generates customer intents from conversation data."""
    
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
    
    def generate_intent_list(
        self, 
        conversations_df: pd.DataFrame,
        batch_size: int = 5,
        max_batches: int = None
    ) -> List[CustomerIntent]:
        """Generate a list of customer intents from conversation data."""
        total_batches = (len(conversations_df) + batch_size - 1) // batch_size
        max_batches = max_batches or total_batches
        
        print(f"Starting intent generation with batch_size={batch_size}")
        print(f"Processing {min(max_batches, total_batches)} batches out of {total_batches} total batches")
        
        # Initialize empty ontology
        current_ontology = []
        
        # Process conversations in batches
        batches_processed = 0
        
        for i in range(0, len(conversations_df), batch_size):
            if batches_processed >= max_batches:
                break
            
            batch_conversations = conversations_df.iloc[i:i+batch_size]['conversation'].tolist()
            
            # Format conversations for prompt
            formatted_conversations = self.preprocessor.format_conversations_for_prompt(
                batch_conversations
            )
            
            # Generate new intents for this batch
            new_intents = self._generate_intents_for_batch(
                formatted_conversations, 
                current_ontology,
                len(batch_conversations)
            )
            
            # Update ontology with new intents
            if new_intents:
                unique_new_intents = self._filter_duplicate_intents(new_intents, current_ontology)
                
                if unique_new_intents:
                    current_ontology.extend(unique_new_intents)
                    print(f"Added {len(unique_new_intents)} new intents from batch {batches_processed + 1}")
                else:
                    print(f"No new unique intents found in batch {batches_processed + 1}")
            else:
                print(f"No new intents found in batch {batches_processed + 1}")
            
            batches_processed += 1
        
        print(f"Processed {batches_processed} batches")
        print(f"Generated {len(current_ontology)} initial customer intents")
        
        # Save the generated initial intents
        self.preprocessor.save_initial_intents(current_ontology)
        
        return current_ontology
    
    def _generate_intents_for_batch(
        self, 
        formatted_conversations: str, 
        current_ontology: List[CustomerIntent],
        batch_size: int
    ) -> List[CustomerIntent]:
        """Generate intents for a single batch of conversations."""
        
        # Create the ontology generation prompt
        ontology_gen_prompt = f"""
You are building an ontology to classify customer conversations with customer support agents based on the intent of the customer. Your job is to identify if the current conversations can be classified into the current ontology and if not, add new intents to the ontology.

Current ontology (JSON):
{[intent.model_dump() for intent in current_ontology]}

New conversations ({batch_size} items):
{formatted_conversations}

TASK:
1. Identify intents in conversations NOT covered by current ontology
2. For each new intent, create an object with: name, description, examples
3. Merge any duplicates/synonyms with existing intents
4. Group related intents under broader parents (max 2 levels)

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
- The customer intent name should be action-oriented, i.e, an action the customer wants to take or a goal they want to achieve (ex: "Get Refund").
</Naming Rules>

<Good Examples of similar intents>
-> Same Intent: Order Status ↔ Order Status Explanation ↔ Order Tracking Issue
-> Order Delivery Delay ↔ Order Not Received
-> Account Creation Help ↔ Technical Signup Issue
</Good Examples>

Here is the exact format of the JSON array of new intents:
For each new intent, create an object with: name, description, examples.
        'name': 'Customer Intent Name',
        'description': 'New Intent Description',
        'examples': ['Example 1', 'Example 2']

Return JSON array of new intents ONLY. If no new intents needed, return an empty []."""
        
        try:
            # Generate intents using LLM
            response = self.llm_client.parse_response(
                prompt=ontology_gen_prompt,
                response_model=IntentResponse,
                model="gpt-4.1",
                temperature=0.0
            )
            
            return response.intents
            
        except Exception as e:
            print(f"Error generating intents for batch: {e}")
            return []
    
    def _filter_duplicate_intents(
        self, 
        new_intents: List[CustomerIntent], 
        current_ontology: List[CustomerIntent]
    ) -> List[CustomerIntent]:
        """Filter out duplicate intents from new intents."""
        existing_intent_names = {intent.name.lower() for intent in current_ontology}
        unique_new_intents = []
        
        for intent in new_intents:
            intent_name = intent.name.lower()
            if intent_name not in existing_intent_names:
                unique_new_intents.append(intent)
                existing_intent_names.add(intent_name)
            else:
                print(f"Skipping duplicate intent: {intent.name}")
        
        return unique_new_intents
    
    def load_existing_initial_intents(self, version: Optional[str] = None) -> List[CustomerIntent]:
        """Load existing initial intents from file."""
        try:
            return self.data_loader.load_initial_intents(version)
        except FileNotFoundError:
            print("No existing initial intents found, starting with empty list")
            return []