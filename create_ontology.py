from datasets import load_dataset
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import json
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import numpy as np
import asyncio
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from gen_intent_list import CustomerIntent, IntentResponse, generate_intent_list

load_dotenv()

client = OpenAI()

class ClusterNamingResponse(BaseModel):
            cluster_name: str = Field(description="The common customer intent category name for the cluster")
            description: str = Field(description="A short description of the customer intent category")

class ClassificationResponse(BaseModel):
    category: str = Field(description="The customer intent category that the conversation is most related to")



def generate_clusters(distance_threshold: float = 0.61):
    """Generate clusters using Agglomerative Clustering."""
    
    ontology_file_path = 'customer_intent_ontology.json'
    clustered_customer_intents_file_path = 'clustered_customer_intents.json'

    # Load the ontology from the JSON file
    with open(ontology_file_path, 'r') as f:
        ontology_data = json.load(f)
    
    # Convert dictionaries back to CustomerIntent objects
    ontology = [CustomerIntent(**intent_data) for intent_data in ontology_data]

    sentences = [f"{intent.name}: {intent.description}. Examples: {', '.join(intent.examples)}" for intent in ontology]

    # using OpenAI-embeeding-3-large for turning sentences to vectors
    
    embeddings = client.embeddings.create(
        input=sentences,
        model="text-embedding-3-large"
    )

    # Convert embeddings to numpy array
    embeddings_array = normalize(np.array([embedding.embedding for embedding in embeddings.data]))

    # Create a clustering model using Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average").fit(embeddings_array)

    # Get the cluster labels
    labels = clustering.labels_

    # Print the cluster labels
    print("Agglomerative Clustering Labels:", labels)

    # Print the number of clusters
    print(f"Number of clusters: {len(np.unique(labels))}")
    # Print the number of unclustered items
    print(f"Number of unclustered items: {len(np.where(labels == -1)[0])}")
    # Print distribution of clusters
    print(f"Distribution of clusters: {np.bincount(labels)}")

    # For Agglomerative Clustering
    cluster_mapping = {}
    for i, label in enumerate(labels):
        label_key = int(label)  # Convert numpy.int64 to regular Python int
        if label_key not in cluster_mapping:
            cluster_mapping[label_key] = {
                "cluster_size": 0,
                "customer_intents": []
            }
        cluster_mapping[label_key]["customer_intents"].append({
            "index": i,
            "intent_name": ontology[i].name,
            "description": ontology[i].description,
            "examples": ontology[i].examples
        })
        cluster_mapping[label_key]["cluster_size"] += 1

    # Calculate cohesiveness for each cluster using LLM to get structured output score out of 5
    named_cluster_mapping = {}
    total_clusters = len(cluster_mapping)
    
    for idx, (cluster_id, cluster_data) in enumerate(cluster_mapping.items()):
        print(f"Processing cluster {idx + 1}/{total_clusters} (ID: {cluster_id})")
        
        cluster_intents = cluster_data["customer_intents"]
        cluster_size = cluster_data["cluster_size"]

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
        
        cluster_naming_response = client.responses.parse(
            model="gpt-4.1",
            input=cluster_naming_prompt,
            text_format=ClusterNamingResponse,
            temperature=0.0
        )

        cluster_data["common_intent"] = cluster_naming_response.output_parsed.cluster_name
        cluster_data["common_intent_description"] = cluster_naming_response.output_parsed.description
        named_cluster_mapping[cluster_id] = cluster_data
        
        print(f"  Cluster {cluster_id}: {cluster_size} intents")

    # Calculate metrics for Agglomerative Clustering
    average_cluster_size_agg = np.mean([cluster_data["cluster_size"] for cluster_data in named_cluster_mapping.values()])
    number_of_clusters_agg = len(named_cluster_mapping)
    unclustered_items_agg = len(np.where(labels == -1)[0])
    
    print("--------------------------------")
    print("Agglomerative Clustering Metrics:")
    print(f"  Average cluster size: {average_cluster_size_agg}")
    print(f"  Number of clusters: {number_of_clusters_agg}")
    print(f"  Number of unclustered items: {unclustered_items_agg}")

    # Save cluster mapping to JSON
    with open(f'reduced_cluster_agglomerative_{distance_threshold}.json', 'w') as f:
        json.dump(named_cluster_mapping, f, indent=2)
    
    # Save list of common_intent and common_intent_description as a separate JSON file
    common_intents = []
    for cluster_id, cluster_data in named_cluster_mapping.items():
        common_intents.append({
            "customer_intent": cluster_data["common_intent"],
            "customer_intent_description": cluster_data["common_intent_description"]
        })

    with open(f'clustered_customer_intents_{distance_threshold}.json', 'w') as f:
        json.dump(common_intents, f, indent=2)

    print(f"Agglomerative clustering results saved to 'reduced_cluster_agglomerative_{distance_threshold}.json'")
    print(f"Common intents saved to 'clustered_customer_intents_{distance_threshold}.json'")

    return named_cluster_mapping

def generate_clusters_with_hdbscan(min_cluster_size: int = 2):
    """Generate clusters using HDBSCAN clustering."""
    
    ontology_file_path = 'customer_intent_ontology.json'

    # Load the ontology from the JSON file
    with open(ontology_file_path, 'r') as f:
        ontology_data = json.load(f)
    
    # Convert dictionaries back to CustomerIntent objects
    ontology = [CustomerIntent(**intent_data) for intent_data in ontology_data]

    sentences = [f"{intent.name}: {intent.description}. Examples: {', '.join(intent.examples)}" for intent in ontology]

    # using OpenAI-embeeding-3-large for turning sentences to vectors
    embeddings = client.embeddings.create(
        input=sentences,
        model="text-embedding-3-large"
    )

    # Convert embeddings to numpy array
    embeddings_array = normalize(np.array([embedding.embedding for embedding in embeddings.data]))

    # Create a clustering model using HDBSCAN with more aggressive parameters
    # Reduce min_cluster_size and min_samples to create more clusters and reduce noise
    clustering = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min_cluster_size), 
        min_samples=1,  # Reduce this to be more inclusive
        metric="euclidean", 
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.1  # Add this to be more permissive
    ).fit(embeddings_array)

    # Get the cluster labels
    labels = clustering.labels_

    # Print the cluster labels
    print("HDBSCAN Clustering Labels:", labels)

    # Print the number of clusters
    print(f"Number of clusters: {len(np.unique(labels))}")
    # Print the number of unclustered items
    print(f"Number of unclustered items: {len(np.where(labels == -1)[0])}")
    # Print distribution of clusters
    # Filter out noise points (-1) for HDBSCAN before bincount
    labels_filtered = labels[labels >= 0]
    noise_count = len(labels[labels == -1])
    if len(labels_filtered) > 0:
        print(f"Distribution of clusters: {np.bincount(labels_filtered)}")
        print(f"Number of noise points (unclustered): {noise_count}")
    else:
        print("Distribution of clusters: No valid clusters (all noise)")
        print(f"Number of noise points (unclustered): {noise_count}")

    # For HDBSCAN Clustering
    cluster_mapping = {}
    for i, label in enumerate(labels):
        label_key = int(label)  # Convert numpy.int64 to regular Python int
        if label_key not in cluster_mapping:
            cluster_mapping[label_key] = {
                "cluster_size": 0,
                "customer_intents": []
            }
        cluster_mapping[label_key]["customer_intents"].append({
            "index": i,
            "intent_name": ontology[i].name,
            "description": ontology[i].description,
            "examples": ontology[i].examples
        })
        cluster_mapping[label_key]["cluster_size"] += 1

    # Generate names for clusters
    named_cluster_mapping = {}
    total_clusters = len(cluster_mapping)
    
    for idx, (cluster_id, cluster_data) in enumerate(cluster_mapping.items()):
        print(f"Processing cluster {idx + 1}/{total_clusters} (ID: {cluster_id})")
        
        cluster_intents = cluster_data["customer_intents"]
        cluster_size = cluster_data["cluster_size"]
        
        # Generate names for clusters
        cluster_naming_prompt = f"""
        You are an ontology engineer. You are given a list of customer intents that are related to a specific customer support topic. Your job is to provide a single customer intent category name that is common to all the intents in the cluster.

        List of customer intents:
        <Customer Intents>
        {cluster_intents}
        </Customer Intents>

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

        Here is the exact format of the JSON array of new intents:
        
            "name": "Customer Intent Name",
            "description": "New Intent Description",
        

        Return the customer intent category name and a short description of the customer intent category.
        """

        cluster_naming_response = client.responses.parse(
            model="gpt-4.1",
            input=cluster_naming_prompt,
            text_format=ClusterNamingResponse,
            temperature=0.0
        )

        cluster_data["common_intent"] = cluster_naming_response.output_parsed.cluster_name
        cluster_data["common_intent_description"] = cluster_naming_response.output_parsed.description
        named_cluster_mapping[cluster_id] = cluster_data

    # Calculate metrics for HDBSCAN Clustering
    average_cluster_size_hdb = np.mean([cluster_data["cluster_size"] for cluster_data in named_cluster_mapping.values()])
    number_of_clusters_hdb = len([k for k in named_cluster_mapping.keys() if k != -1])  # Exclude noise cluster
    unclustered_items_hdb = len(np.where(labels == -1)[0])

    print("--------------------------------")
    print("HDBSCAN Clustering Metrics:")
    print(f"  Average cluster size: {average_cluster_size_hdb}")
    print(f"  Number of clusters: {number_of_clusters_hdb}")
    print(f"  Number of unclustered items: {unclustered_items_hdb}")

    # Save cluster mapping to JSON
    with open(f'reduced_cluster_hdbscan_{min_cluster_size}.json', 'w') as f:
        json.dump(named_cluster_mapping, f, indent=2)
    
    # Save list of common_intent and common_intent_description as a separate JSON file
    common_intents_hdbscan = []
    for cluster_id, cluster_data in named_cluster_mapping.items():
        common_intents_hdbscan.append({
            "customer_intent": cluster_data["common_intent"],
            "customer_intent_description": cluster_data["common_intent_description"]
        })
    with open(f'clustered_customer_intents_hdbscan_{min_cluster_size}.json', 'w') as f: 
        json.dump(common_intents_hdbscan, f, indent=2)

    print(f"HDBSCAN clustering results saved to 'reduced_cluster_hdbscan_{min_cluster_size}.json'")
    print(f"Common intents saved to 'clustered_customer_intents_hdbscan_{min_cluster_size}.json'")

    return named_cluster_mapping

def check_redundant_intents(file_path, customer_intents):
    """Check for redundant customer intents that have 0% distribution"""
    try:
        # Load the classification results from Excel file
        df = pd.read_excel(file_path)
        
        print(f"\n=== Checking for Redundant Customer Intents ===")
        print(f"Successfully loaded {len(df)} classification results")
        print(f"Checking against {len(customer_intents)} customer intents")
        
        # Extract all customer intent names from the provided customer_intents
        intent_names = [intent['customer_intent'] for intent in customer_intents]
        
        print(f"Customer intents to check:")
        for intent in intent_names:
            print(f"  - {intent}")
        
        redundant_intents = []
        
        # Check if 'classification' column exists
        if 'classification' in df.columns:
            print(f"\n--- Classification Distribution Analysis ---")
            
            # Handle comma-separated classifications
            all_classifications = []
            for classification in df['classification']:
                if pd.notna(classification):  # Check for non-null values
                    # Split by comma and strip whitespace
                    individual_classifications = [cls.strip() for cls in str(classification).split(',')]
                    all_classifications.extend(individual_classifications)
            
            # Count individual classifications
            from collections import Counter
            classification_counts = Counter(all_classifications)
            
            # Check each customer intent for 0% distribution
            for intent_name in intent_names:
                count = classification_counts.get(intent_name, 0)
                if count == 0:
                    redundant_intents.append(intent_name)
                    print(f"  {intent_name}: 0 occurrences (0.0%) - REDUNDANT")
                else:
                    percentage = (count / len(all_classifications)) * 100
                    print(f"  {intent_name}: {count} occurrences ({percentage:.1f}%)")
        else:
            print(f"Error: 'classification' column not found in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return []
        
        print(f"\n--- Analysis Results ---")
        if redundant_intents:
            print(f"Found {len(redundant_intents)} redundant intents with 0% distribution:")
            for intent in redundant_intents:
                print(f"  - {intent}")
        else:
            print("No redundant customer intents found (all intents are being used).")
        
        return redundant_intents
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return []

def evaluate_ontology(customer_intent_categories_file_path: str):
    """Evaluate the mutual exclusivity of customer intent categories using LLM and cosine similarity."""
    
    print("Starting ontology evaluation...")
    
    # Load the clustered customer intents
    with open(customer_intent_categories_file_path, 'r') as f:
        common_intents = json.load(f)

    print("Checking for duplicate customer intents...")
    
    # Send the clustered_customer_intents.json file to an llm and ask it to score the list of categories on if they categories are mutually exclusive or not on a scale of 1 to 5 (5 being the most mutually exclusive), also return the reason for the score.
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
    {common_intents}
    </Customer Intent Categories>

    Return whether there are any exact duplicate customer intents and list the duplicate pairs if any exist.
    """

    class MutualExclusivityResponse(BaseModel):
        has_duplicates: str = Field(description="Whether there are exact duplicate customer intents - 'yes' or 'no'")
        duplicate_pairs: List[str] = Field(description="List of duplicate pairs found, empty if none")
        reason: str = Field(description="Brief explanation of the findings")

    mutual_exclusivity_response = client.responses.parse(
        model="o3",
        input=mutual_exclusivity_prompt,
        text_format=MutualExclusivityResponse,
    )

    print(f"Has duplicate intents: {mutual_exclusivity_response.output_parsed.has_duplicates}")
    if mutual_exclusivity_response.output_parsed.duplicate_pairs:
        print(f"Duplicate pairs found:")
        for pair in mutual_exclusivity_response.output_parsed.duplicate_pairs:
            print(f"  - {pair}")
    print(f"Reason: {mutual_exclusivity_response.output_parsed.reason}")

    # Evaluate mutual exclusivity using cosine similarity
    max_similarity, passes_exclusivity = evaluate_mutual_exclusivity_cosine_similarity(customer_intent_categories_file_path)
    print(f"Cosine similarity Mutual exclusivity score: {max_similarity:.3f}")
    print("Ontology is mutually-exclusive ✅" if passes_exclusivity else
          "Ontology has overlapping labels ❌")

    # Evaluate the collective exhaustivity of the customer intent categories

    """
    To evaluate for collective exhaustivity, we will:
    1. Load the customer_conversations.xlsx file into a df
    2. Run the LLM to classify each conversation into one of the customer intent categories or the 'other' category and then update the df synchronously.
    3. Store it as a new file called 'classified_customer_conversations.xlsx'
    4. Then use the df to calculate the unclassified conversations percentage.
    5. Finally return the unclassified conversations percentage and the mutual exclusivity score.
    """

    print("Evaluating collective exhaustivity...")

    # Load the clustered customer intents
    with open(customer_intent_categories_file_path, 'r') as f:
        customer_intent_categories = json.load(f)

    # Load the customer_conversations.xlsx file into a df
    df = pd.read_excel('customer_conversations.xlsx')
    
    print(f"Classifying {len(df)} conversations...")
    # Run the LLM to classify each conversation into one of the customer intent categories or the 'other' category in batches of 10.
    batch_size = 50
    
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        if batch_start % 100 == 0:
            print(f"  Progress: {batch_start}/{len(df)} conversations classified")
        
        # Create async function to handle individual classification
        async def classify_conversation(index, conversation):
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
                client.responses.parse,
                model="gpt-4.1-mini",
                input=classification_prompt,
                text_format=ClassificationResponse,
                temperature=0.0
            )
            return index, response.output_parsed.category
        
        # Process batch concurrently
        async def process_batch():
            tasks = []
            for index, row in batch_df.iterrows():
                conversation = row['conversation']
                task = classify_conversation(index, conversation)
                tasks.append(task)
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results
        
        # Run the async batch
        batch_results = asyncio.run(process_batch())
        
        # Update the df with the classification responses
        for index, category in batch_results:
            df.at[index, 'classification'] = category

    print("Classification complete. Saving results...")

    # Save the df to a new file called 'classified_customer_conversations.xlsx'
    df.to_excel('classified_customer_conversations.xlsx', index=False)

    # Calculate the unclassified conversations percentage
    unclassified_conversations_percentage = ((len(df[df['classification'] == 'Other']) + len(df[df['classification'] == 'other'])) / len(df)) * 100

    print(f"Evaluation complete. Unclassified conversations: {unclassified_conversations_percentage:.2f}%")

    # Load the clustered customer intents to get the number of clusters
    with open(customer_intent_categories_file_path, 'r') as f:
        clustered_intents = json.load(f)
    
    num_clusters = len(clustered_intents)

    # Check for redundant customer intents
    redundant_intents = check_redundant_intents('classified_customer_conversations.xlsx', clustered_intents)

    # Return the unclassified conversations percentage, mutual exclusivity score, and number of clusters
    return unclassified_conversations_percentage, mutual_exclusivity_response.output_parsed, num_clusters, max_similarity, passes_exclusivity, redundant_intents

# Non-examples - 83 customer intent categories

def compare_ontologies():
    """Compare ontologies generated with different distance thresholds."""
    distance_thresholds = [0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64]
    results = {}
    
    for threshold in distance_thresholds:
        print(f"\n{'='*50}")
        print(f"Testing distance threshold: {threshold}")
        print(f"{'='*50}")
        
        # Generate clusters with the current threshold
        generate_clusters(threshold)
        
        # Evaluate the ontology
        unclassified_percentage, mutual_exclusivity_result, num_clusters, max_similarity, passes_exclusivity, redundant_intents = evaluate_ontology(f'clustered_customer_intents_{threshold}.json')
        
        # Store results
        results[threshold] = {
            'coverage': 100 - unclassified_percentage,
            'duplicate_check': mutual_exclusivity_result.has_duplicates,
            'duplicate_check_reason': mutual_exclusivity_result.reason,
            'num_clusters': num_clusters,
            'max_similarity': max_similarity,
            'mutual_exclusivity_check': passes_exclusivity,
            'no_of_redundant_intents': len(redundant_intents),
            'redundant_intents': redundant_intents
        }
        
        print(f"Results for threshold {threshold}:")
        print(f"  Unclassified conversations: {unclassified_percentage:.2f}%")
        print(f"  Mutual exclusivity LLM check: {mutual_exclusivity_result.has_duplicates}")
        print(f"  Number of clusters: {num_clusters}")
        print(f"  Max similarity: {max_similarity:.3f}")
        print(f"  Passes exclusivity: {'✅' if passes_exclusivity else '❌'}")
        print(f"  Number of redundant intents: {len(redundant_intents)}")
        print(f"  Redundant intents: {redundant_intents}")

    # Print summary comparison
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print(f"{'='*50}")
    
    # Prepare data for Excel export
    comparison_data = []
    
    for threshold, result in results.items():
        print(f"Threshold {threshold}:")
        print(f"  Coverage: {result['coverage']:.2f}%")
        print(f"  Mutual exclusivity LLM check: {result['duplicate_check']}")
        print(f"  Number of clusters: {result['num_clusters']}")
        print(f"  Max similarity: {result['max_similarity']:.3f}")
        print(f"  Mutual exclusivity check: {'✅' if result['mutual_exclusivity_check'] else '❌'}")
        print(f"  Number of redundant intents: {result['no_of_redundant_intents']}")
        print(f"  Redundant intents: {result['redundant_intents']}")
        
        # Add to comparison data
        comparison_data.append({
            'Distance_Threshold': threshold,
            'Coverage': result['coverage'],
            'Duplicate_Check': result['duplicate_check'],
            'Duplicate_Check_Reason': result['duplicate_check_reason'],
            'Number_of_Clusters': result['num_clusters'],
            'Max_Similarity': result['max_similarity'],
            'Mutual_Exclusivity_Check': '✅' if result['mutual_exclusivity_check'] else '❌',
            'Number_of_Redundant_Intents': result['no_of_redundant_intents'],
            'Redundant_Intents': result['redundant_intents']
        })
    
    # Save comparison summary to Excel
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_excel('ontology_comparison_summary.xlsx', index=False)
    print(f"Comparison summary saved to 'ontology_comparison_summary.xlsx'")
    
    return results

# # compare_ontologies()

# To create the 3D PCA visualizations, uncomment the line below:
# create_clustering_visualizations()

def evaluate_ontology_hdbscan():
    """Evaluate the mutual exclusivity of customer intent categories."""
    
    print("Starting ontology evaluation using HDBSCAN clustering...")
    
    # Load the clustered customer intents
    with open('clustered_customer_intents_hdbscan_2.json', 'r') as f:
        common_intents = json.load(f)

    print("Evaluating mutual exclusivity using HDBSCAN clustering...")
    
    # Send the clustered_customer_intents.json file to an llm and ask it to score the list of categories on if they categories are mutually exclusive or not on a scale of 1 to 5 (5 being the most mutually exclusive), also return the reason for the score.
    mutual_exclusivity_prompt = f"""
    You are an ontology engineer. You are given a list of customer intent categories that are related to a specific customer support topic. Your job is to score the customer intent category mutual exclusivity on a scale of 1 to 10 on how mutually exclusive the list of items in that particular cluster are.

    If the goal the customer expects to achieve is exactly the same for two or more customer intent categories, then the customer intent categories are not mutually exclusive, and the score should be 1.

    If the goal the customer expects to achieve is completely different for two or more customer intent categories, then the customer intent categories are mutually exclusive, and the score should be 10.

    Score between the numbers 1 and 10 based on the degree to which the goal the customer expects to achieve is exactly the same for two or more customer intent categories.

    List of customer intent categories:
    <Customer Intent Categories>
    {common_intents}
    </Customer Intent Categories>

    Only return the score, reason for the score and nothing else, no other explanations. The score should be a number between 1 and 10.
    """

    class MutualExclusivityResponse(BaseModel):
        score: int = Field(description="The score of the mutual exclusivity of the customer intent categories on a scale of 1 to 10")
        reason: str = Field(description="The reason for the score")

    mutual_exclusivity_response = client.responses.parse(
        model="o3",
        input=mutual_exclusivity_prompt,
        text_format=MutualExclusivityResponse,
    )

    print(f"LLM Mutual exclusivity score: {mutual_exclusivity_response.output_parsed.score}/10")
    print(f"LLM Mutual exclusivity reason: \n{mutual_exclusivity_response.output_parsed.reason}")

    # Evaluate mutual exclusivity using cosine similarity
    max_similarity, passes_exclusivity = evaluate_mutual_exclusivity_cosine_similarity(f'clustered_customer_intents_hdbscan_2.json')
    print(f"Cosine similarity Mutual exclusivity score: {max_similarity:.3f}")
    print("Ontology is mutually-exclusive ✅" if passes_exclusivity else
          "Ontology has overlapping labels ❌")

    # Evaluate the collective exhaustivity of the customer intent categories

    """
    To evaluate for collective exhaustivity, we will:
    1. Load the customer_conversations.xlsx file into a df
    2. Run the LLM to classify each conversation into one of the customer intent categories or the 'other' category and then update the df synchronously.
    3. Store it as a new file called 'classified_customer_conversations.xlsx'
    4. Then use the df to calculate the unclassified conversations percentage.
    5. Finally return the unclassified conversations percentage and the mutual exclusivity score.
    """

    print("Evaluating collective exhaustivity using HDBSCAN clustering...")

    # Load the clustered customer intents
    with open('clustered_customer_intents_hdbscan_2.json', 'r') as f:
        customer_intent_categories = json.load(f)

    # Load the customer_conversations.xlsx file into a df
    df = pd.read_excel('customer_conversations.xlsx')
    
    print(f"Classifying {len(df)} conversations...")
    # Run the LLM to classify each conversation into one of the customer intent categories or the 'other' category in batches of 10.
    batch_size = 50
    
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        if batch_start % 100 == 0:
            print(f"  Progress: {batch_start}/{len(df)} conversations classified")
        
        # Create async function to handle individual classification
        async def classify_conversation(index, conversation):
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
            """

            # Run the synchronous API call in a thread pool to make it concurrent
            response = await asyncio.to_thread(
                client.responses.parse,
                model="gpt-4.1-mini",
                input=classification_prompt,
                text_format=ClassificationResponse,
                temperature=0.0
            )
            return index, response.output_parsed.category
        
        # Process batch concurrently
        async def process_batch():
            tasks = []
            for index, row in batch_df.iterrows():
                conversation = row['conversation']
                task = classify_conversation(index, conversation)
                tasks.append(task)
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results
        
        # Run the async batch
        batch_results = asyncio.run(process_batch())
        
        # Update the df with the classification responses
        for index, category in batch_results:
            df.at[index, 'classification'] = category

    print("Classification complete. Saving results...")

    # Save the df to a new file called 'classified_customer_conversations.xlsx'
    df.to_excel('classified_customer_conversations.xlsx', index=False)

    # Calculate the unclassified conversations percentage
    unclassified_conversations_percentage = ((len(df[df['classification'] == 'Other']) + len(df[df['classification'] == 'other'])) / len(df)) * 100

    print(f"Evaluation complete. Unclassified conversations: {unclassified_conversations_percentage:.2f}%")

    # Load the clustered customer intents to get the number of clusters
    with open('clustered_customer_intents_hdbscan_2.json', 'r') as f:
        clustered_intents = json.load(f)
    
    num_clusters = len(clustered_intents)

    print("\n" + "="*60)
    print("ONTOLOGY EVALUATION RESULTS - HDBSCAN CLUSTERING")
    print("="*60)
    print(f"Number of clusters: {num_clusters}")
    print(f"Mutual exclusivity score: {mutual_exclusivity_response.output_parsed.score}/10")
    print(f"Collective exhaustivity (classified): {100 - unclassified_conversations_percentage:.2f}%")
    print(f"Unclassified conversations: {unclassified_conversations_percentage:.2f}%")
    print("="*60)
    
    # Return the unclassified conversations percentage, mutual exclusivity score, and number of clusters
    return unclassified_conversations_percentage, mutual_exclusivity_response.output_parsed, num_clusters

def evaluate_mutual_exclusivity_cosine_similarity(customer_intent_categories_file_path: str):
    """Evaluate the mutual exclusivity of customer intent categories using cosine similarity."""
    
    print("Starting ontology evaluation using cosine similarity...")
    
    # Load the clustered customer intents
    with open(customer_intent_categories_file_path, 'r') as f:
        customer_intent_categories = json.load(f)
    
    # Create embeddings for all customer intents
    embeddings = []
    for intent in customer_intent_categories:
        text = f"{intent['customer_intent']}. {intent['customer_intent_description']}"
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-large"
        )
        embeddings.append(response.data[0].embedding)
    
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
    max_similarity = max(upper_triangular)
    threshold = 0.85
    passes_exclusivity = max_similarity < threshold
    
    print(f"Max pairwise similarity = {max_similarity:.3f}")
    print(f"Threshold = {threshold}")
    print("Ontology is mutually-exclusive ✅" if passes_exclusivity else
          "Ontology has overlapping labels ❌")
    
    return max_similarity, passes_exclusivity

def visualize_clustering_pca(embeddings_array, labels, ontology, output_file: str, clustering_method: str):
    """Visualize clustering results using PCA."""
    
    # Perform PCA to reduce dimensions to 3 for visualization
    pca = PCA(n_components=3)
    embeddings_pca = pca.fit_transform(embeddings_array)

    # Create a DataFrame for plotting
    df_pca = pd.DataFrame(embeddings_pca, columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca['Cluster'] = labels
    df_pca['Intent Name'] = [ontology[i].name for i in range(len(ontology))]
    df_pca['Description'] = [ontology[i].description for i in range(len(ontology))]
    
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
            aspectmode='cube'  # Ensure equal aspect ratio
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
    fig.write_html(output_file)
    print(f"{clustering_method} PCA visualization saved to {output_file}")
    
    return fig

def create_clustering_visualizations():
    """Create 3D PCA visualizations for both Agglomerative and HDBSCAN clustering results."""
    
    print("Creating 3D PCA visualizations for clustering results...")
    
    # Load the original ontology
    ontology_file_path = 'customer_intent_ontology.json'
    with open(ontology_file_path, 'r') as f:
        ontology_data = json.load(f)
    
    # Convert dictionaries back to CustomerIntent objects
    ontology = [CustomerIntent(**intent_data) for intent_data in ontology_data]
    
    # Create embeddings for the ontology
    sentences = [f"{intent.name}: {intent.description}. Examples: {', '.join(intent.examples)}" for intent in ontology]
    
    print("Generating embeddings...")
    embeddings = client.embeddings.create(
        input=sentences,
        model="text-embedding-3-large"
    )
    
    # Convert embeddings to numpy array
    embeddings_array = normalize(np.array([embedding.embedding for embedding in embeddings.data]))
    
    # 1. Visualize Agglomerative Clustering results
    print("\nProcessing Agglomerative Clustering results...")
    try:
        with open('reduced_cluster_agglomerative_0.61.json', 'r') as f:
            agg_cluster_data = json.load(f)
        
        # Extract cluster labels for agglomerative clustering
        agg_labels = np.full(len(ontology), -1)  # Initialize with -1
        
        for cluster_id, cluster_info in agg_cluster_data.items():
            cluster_id_int = int(cluster_id)
            for intent_info in cluster_info['customer_intents']:
                intent_index = intent_info['index']
                agg_labels[intent_index] = cluster_id_int
        
        # Create visualization for agglomerative clustering
        agg_fig = visualize_clustering_pca(
            embeddings_array, 
            agg_labels, 
            ontology, 
            'agglomerative_clustering_pca_visualization.html',
            'Agglomerative'
        )
        
        print(f"Agglomerative clustering: {len(np.unique(agg_labels))} clusters")
        
    except FileNotFoundError:
        print("Error: reduced_cluster_agglomerative_0.61.json not found. Please run agglomerative clustering first.")
    
    # 2. Visualize HDBSCAN Clustering results (with noise point assignment)
    print("\nProcessing HDBSCAN Clustering results...")
    try:
        # First, run the improved HDBSCAN clustering
        print("Running improved HDBSCAN clustering...")
        generate_clusters_with_hdbscan(min_cluster_size=2)
        
        with open('reduced_cluster_hdbscan_2.json', 'r') as f:
            hdb_cluster_data = json.load(f)
        
        # Extract cluster labels for HDBSCAN clustering
        hdb_labels = np.full(len(ontology), -1)  # Initialize with -1 (noise)
        
        for cluster_id, cluster_info in hdb_cluster_data.items():
            cluster_id_int = int(cluster_id)
            for intent_info in cluster_info['customer_intents']:
                intent_index = intent_info['index']
                hdb_labels[intent_index] = cluster_id_int
        
        # Create visualization for HDBSCAN clustering
        hdb_fig = visualize_clustering_pca(
            embeddings_array, 
            hdb_labels, 
            ontology, 
            'hdbscan_clustering_pca_visualization.html',
            'HDBSCAN'
        )
        
        # Count non-noise clusters
        non_noise_clusters = len(np.unique(hdb_labels[hdb_labels >= 0]))
        noise_points = len(hdb_labels[hdb_labels == -1])
        print(f"HDBSCAN clustering: {non_noise_clusters} clusters, {noise_points} noise points")
        
    except FileNotFoundError:
        print("Error: reduced_cluster_hdbscan.json not found. Please run HDBSCAN clustering first.")
    
    print("\nVisualization complete! Open the HTML files in your browser to view the interactive 3D plots.")
    print("- agglomerative_clustering_pca_visualization.html")
    print("- hdbscan_clustering_pca_visualization.html")

def merge_intents(cluster_file_path: str):
    """Merge intents that are similar to each other."""
    
    # Load the clustered customer intents
    with open(cluster_file_path, 'r') as f:
        customer_intent_categories = json.load(f)
    
    # Make an LLM call with structured output to ask to return a list of pairs of intents that are not mutually exclusive from each other. Then make a following LLM call and ask to merge the intents in each pair, then update the cluster file with the merged intents. Use the 'customer_intent' field to identify the intents.
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

    class IntentGroup(BaseModel):
        intents: List[str] = Field(description="List of intent names that should be merged together")
    
    class MergeIntentsResponse(BaseModel):
        groups: List[IntentGroup] = Field(description="A list of groups of intents that are not mutually exclusive from each other and should be merged")
    
    response = client.responses.parse(
        model="gpt-4.1",
        input=prompt,
        text_format=MergeIntentsResponse,
        temperature=0.0
    )
    
    # Make an LLM call to merge the intents in each group
    updated_intents = customer_intent_categories.copy()
    intents_to_remove = set()
    
    for group in response.output_parsed.groups:
        # Find all intents in the group
        group_intent_data = []
        
        for intent_name in group.intents:
            for intent in customer_intent_categories:
                if intent['customer_intent'] == intent_name:
                    group_intent_data.append(intent)
                    break
        
        if len(group_intent_data) >= 2:  # Only merge if we have at least 2 intents
            # Create a detailed description of all intents to merge
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
            
            class MergedIntent(BaseModel):
                customer_intent: str = Field(description="The merged customer intent name")
                customer_intent_description: str = Field(description="The merged customer intent description")
            
            merge_response = client.responses.parse(
                model="o3",
                input=merge_prompt,
                text_format=MergedIntent,

            )
            
            # Add the merged intent to the list
            merged_intent = {
                "customer_intent": merge_response.output_parsed.customer_intent,
                "customer_intent_description": merge_response.output_parsed.customer_intent_description
            }
            updated_intents.append(merged_intent)
            
            # Mark the original intents for removal
            intent_names = [intent['customer_intent'] for intent in group_intent_data]
            intents_to_remove.update(intent_names)
            
            print(f"Merged {len(group_intent_data)} intents: {', '.join(intent_names)} -> '{merged_intent['customer_intent']}'")
    
    # Remove the original intents that were merged
    final_intents = [intent for intent in updated_intents if intent['customer_intent'] not in intents_to_remove]
    
    # Save the updated intents to a new file
    merged_file_path = cluster_file_path.replace('.json', '_merged.json')
    with open(merged_file_path, 'w') as f:
        json.dump(final_intents, f, indent=4)
    
    print(f"Intents merged successfully. Saved to {merged_file_path}")
    print(f"Original intents: {len(customer_intent_categories)}")
    print(f"Final intents: {len(final_intents)}")
    print(f"Merged {len(response.output_parsed.groups)} groups, reducing by {len(customer_intent_categories) - len(final_intents)} intents")
    
    return final_intents


def generate_ontology():
    # Load the dataset
    df = load_dataset()
    
    # Split the dataset: first 250 for training, next 50 for testing
    train_df = df.iloc[:].copy()
    test_df = df.iloc[200:300].copy()
    
    print(f"Training set: {len(train_df)} conversations")
    print(f"Test set: {len(test_df)} conversations")

    # Build ontology from conversations
    generate_intent_list(train_df)

    # Reduce ontology to higher-level intents
    generate_clusters(0.6)

    evaluate_ontology(f'clustered_customer_intents_0.6.json')

    # merge_intents(f'clustered_customer_intents_0.6.json')

    # evaluate_ontology(f'clustered_customer_intents_0.6_merged.json')


compare_ontologies()

