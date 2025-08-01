"""
Configuration settings for the ontology building system.
"""

# Default clustering parameters
DEFAULT_AGGLOMERATIVE_DISTANCE_THRESHOLD = 0.6
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 2

# Intent generation parameters
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_BATCHES = None  # None means process all batches

# Evaluation parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_CLASSIFICATION_BATCH_SIZE = 50

# LLM Configuration
DEFAULT_INTENT_GENERATION_MODEL = "gpt-4.1"
DEFAULT_CLUSTERING_MODEL = "gpt-4.1"
DEFAULT_CLASSIFICATION_MODEL = "gpt-4.1-mini"
DEFAULT_EVALUATION_MODEL = "o3"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

# File paths
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "outputs"