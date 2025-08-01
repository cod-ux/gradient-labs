import pandas as pd
import asyncio
from typing import List, Dict

from ..intent_generation.models import ClassificationResponse
from ..utils.llm_client import LLMClient


class ConversationClassifier:
    """Handles classification of conversations into intent categories."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def classify_conversations_batch(
        self,
        conversations_df: pd.DataFrame,
        customer_intent_categories: List[Dict[str, str]],
        batch_size: int = 50
    ) -> pd.DataFrame:
        """Classify conversations in batches for efficiency."""
        # Create a copy to avoid modifying original
        classified_df = conversations_df.copy()
        classified_df['classification'] = ''
        
        # Process in batches
        for batch_start in range(0, len(classified_df), batch_size):
            batch_end = min(batch_start + batch_size, len(classified_df))
            batch_df = classified_df.iloc[batch_start:batch_end]
            
            if batch_start % 100 == 0:
                print(f"  Progress: {batch_start}/{len(classified_df)} conversations classified")
            
            # Process batch concurrently
            batch_results = asyncio.run(self._process_batch(batch_df, customer_intent_categories))
            
            # Update the dataframe with results
            for index, category in batch_results:
                classified_df.at[index, 'classification'] = category
        
        print("Classification complete.")
        return classified_df
    
    async def _process_batch(
        self,
        batch_df: pd.DataFrame,
        customer_intent_categories: List[Dict[str, str]]
    ) -> List[tuple]:
        """Process a batch of conversations concurrently."""
        tasks = []
        for index, row in batch_df.iterrows():
            conversation = row['conversation']
            task = self._classify_single_conversation(index, conversation, customer_intent_categories)
            tasks.append(task)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        return results
    
    async def _classify_single_conversation(
        self,
        index: int,
        conversation: str,
        customer_intent_categories: List[Dict[str, str]]
    ) -> tuple:
        """Classify a single conversation."""
        classification_prompt = f"""
You are a classification engineer. You are given a customer conversation and a list of customer intent categories. Your job is to classify the conversation into one of the given categories based on the customer's intent noticed in the conversation.

If the conversation is not related to any of the given categories, then strictly classify it as the 'Other' category.

List of customer intent categories:
<Customer Intent Categories>
{customer_intent_categories}
</Customer Intent Categories>

Customer conversation:
<Customer Conversation>
{conversation}
</Customer Conversation>

Only return the customer intent category that the conversation is most related to and nothing else, no other explanations.

If multiple labels fit the conversation, then return multiple labels separated by a comma without any explanations. For example: "Refund, Cancel Order"
"""
        
        # Run the synchronous API call in a thread pool to make it concurrent
        response = await asyncio.to_thread(
            self.llm_client.parse_response,
            prompt=classification_prompt,
            response_model=ClassificationResponse,
            model="gpt-4.1-mini",
            temperature=0.0
        )
        
        return index, response.category