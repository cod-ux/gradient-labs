from pydantic import BaseModel, Field
from typing import List, Optional


class CustomerIntent(BaseModel):
    """Model representing a customer intent category."""
    name: str = Field(description="The name of the customer intent category for classifying conversations")
    description: str = Field(description="A short description of the customer intent category")
    examples: List[str] = Field(description="A list of example conversations that fall under this intent category")


class IntentResponse(BaseModel):
    """Response model for intent generation API calls."""
    intents: List[CustomerIntent] = Field(description="List of new customer intent categories")


class ClusterNamingResponse(BaseModel):
    """Response model for cluster naming API calls."""
    cluster_name: str = Field(description="The common customer intent category name for the cluster")
    description: str = Field(description="A short description of the customer intent category")


class ClassificationResponse(BaseModel):
    """Response model for conversation classification API calls."""
    category: str = Field(description="The customer intent category that the conversation is most related to")


class MutualExclusivityResponse(BaseModel):
    """Response model for mutual exclusivity evaluation."""
    has_duplicates: str = Field(description="Whether there are exact duplicate customer intents - 'yes' or 'no'")
    duplicate_pairs: List[str] = Field(description="List of duplicate pairs found, empty if none")
    reason: str = Field(description="Brief explanation of the findings")


class MutualExclusivityScoreResponse(BaseModel):
    """Response model for mutual exclusivity scoring."""
    score: int = Field(description="The score of the mutual exclusivity on a scale of 1 to 10")
    reason: str = Field(description="The reason for the score")


class IntentGroup(BaseModel):
    """Model representing a group of intents to be merged."""
    intents: List[str] = Field(description="List of intent names that should be merged together")


class MergeIntentsResponse(BaseModel):
    """Response model for intent merging API calls."""
    groups: List[IntentGroup] = Field(description="A list of groups of intents that should be merged")


class MergedIntent(BaseModel):
    """Model representing a merged customer intent."""
    customer_intent: str = Field(description="The merged customer intent name")
    customer_intent_description: str = Field(description="The merged customer intent description")


class ClusterData(BaseModel):
    """Model representing cluster data structure."""
    cluster_size: int = Field(description="Number of intents in the cluster")
    customer_intents: List[dict] = Field(description="List of customer intents in the cluster")
    common_intent: Optional[str] = Field(default=None, description="Common intent name for the cluster")
    common_intent_description: Optional[str] = Field(default=None, description="Description of the common intent")


class EvaluationMetrics(BaseModel):
    """Model for storing evaluation metrics."""
    coverage: float = Field(description="Percentage of conversations classified (not 'Other')")
    mutual_exclusivity_score: float = Field(description="Cosine similarity based mutual exclusivity score")
    num_clusters: int = Field(description="Number of clusters/categories")
    max_similarity: float = Field(description="Maximum pairwise similarity between categories")
    passes_exclusivity: bool = Field(description="Whether the ontology passes mutual exclusivity threshold")
    redundant_intents: List[str] = Field(description="List of intents with 0% usage")