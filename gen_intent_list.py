from datasets import load_dataset
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import json
import math

load_dotenv()

ds = load_dataset("NebulaByte/E-Commerce_Customer_Support_Conversations")

client = OpenAI()

class CustomerIntent(BaseModel):
    name: str = Field(description="The name of the customer intent category for classifying conversations")
    description: str = Field(description="A short description of the customer intent category")
    examples: List[str] = Field(description="A list of example conversations that fall under this intent category")

class IntentResponse(BaseModel):
    intents: List[CustomerIntent] = Field(description="List of new customer intent categories")


# Convert the dataset to a pandas DataFrame
def download_dataset():
    df = ds['train'].to_pandas()

    # Randomly shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # Save only the 'conversation' column to Excel file
    df[['conversation']].to_excel('customer_conversations.xlsx', index=False)

    print(f"Dataset saved to 'customer_conversations.xlsx' with {len(df)} rows and 1 column")
    print(f"Column: conversation")
    
    return df[['conversation']]

def load_dataset():
    df = pd.read_excel('customer_conversations.xlsx')
    return df[['conversation']]

def generate_intent_list(train_df):
    # Initialize empty ontology
    current_ontology = []
    
    # Process conversations in batches of 10
    batch_size = 5
    # Temporarily process only 3 batches for testing
    max_batches = 300
    batches_processed = 0
    
    for i in range(0, len(train_df), batch_size):
        if batches_processed >= max_batches:
            break
            
        batch_conversations = train_df.iloc[i:i+batch_size]['conversation'].tolist()
        
        # Format conversations for prompt
        formatted_conversations = "\n".join([f"{idx+1}. {conv}" for idx, conv in enumerate(batch_conversations)])
        
        # Create prompt for ontology building
        ontology_gen_prompt = f"""
        You are building an ontology to classify customer conversations with customer support agents based on the intent of the customer. Your job is to identify if the current conversations can be classified into the current ontology and if not, add new intents to the ontology.

Current ontology (JSON):
{current_ontology}

New conversations ({len(batch_conversations)} items):
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
- The customer intent name should be action-oriented, i.e, an action the customer wants to take or a  goal they want to achieve (ex: "Get Refund").
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
        
        # Generate ontology using GPT-4
        response = client.responses.parse(
            model="gpt-4.1",
            input=ontology_gen_prompt,
            text_format=IntentResponse,
            temperature=0.0
        )
        
        # Parse response and update ontology
        try:
            new_intents = response.output_parsed.intents # this is a list of CustomerIntent objects
            # print(new_intents)
            if new_intents:
                # Check for duplicates before adding
                existing_intent_names = {intent.name.lower() for intent in current_ontology}
                unique_new_intents = []
                
                for intent in new_intents:
                    intent_name = intent.name.lower()
                    if intent_name not in existing_intent_names:
                        unique_new_intents.append(intent)
                        existing_intent_names.add(intent_name)
                    else:
                        print(f"Skipping duplicate intent: {intent.name}")
                
                if unique_new_intents:
                    current_ontology.extend(unique_new_intents)
                    print(f"Added {len(unique_new_intents)} new intents from batch {batches_processed + 1}")
                else:
                    print(f"No new unique intents found in batch {batches_processed + 1}")
            else:
                print(f"No new intents found in batch {batches_processed + 1}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error parsing response for batch {batches_processed + 1}")
        
        batches_processed += 1
    
    print(f"Processed {batches_processed} batches (testing mode)")
    print(f"Final ontology contains {len(current_ontology)} customer intent categories")

    # Convert Pydantic objects to dictionaries for JSON serialization
    ontology_dict = [intent.model_dump() for intent in current_ontology]
    
    # Save ontology to JSON file
    with open('customer_intent_ontology.json', 'w') as f:
        json.dump(ontology_dict, f, indent=2)

    print(f"Ontology saved to 'customer_intent_ontology.json'")

    return current_ontology