# Ontology Building System

A system for building and evaluating ontologies to classify conversations conversations based on intent from conversation data using machine learning clustering and LLM-based analysis.

## Overview

This system performs four main processes:
1. **Intent Generation** - Creates initial intents by analyzing conversation datasets
2. **Clustering** - Groups similar intents using clustering algorithms (Agglomerative, HDBSCAN)
3. **Ontology Building** - Processes clusters to create higher-level customer intent categories
4. **Evaluation** - Assesses ontology quality through mutual exclusivity and coverage metrics

## Project Structure

```
gradient-labs/
├── src/                           # Source code
│   ├── data/                      # Data handling and file management
│   ├── intent_generation/         # Intent generation from conversations
│   ├── clustering/                # Clustering algorithms (agglomerative, hdbscan)
│   ├── ontology/                  # Ontology building and merging
│   ├── evaluation/                # Quality assessment and metrics
│   ├── visualization/             # PCA plots and visualizations
│   └── utils/                     # Utilities and LLM client
├── scripts/                       # Executable scripts
│   ├── run_full_pipeline.py       # Main pipeline orchestrator
│   ├── compare_thresholds.py      # Parameter comparison
│   └── generate_visualizations.py # Visualization generation
├── data/                          # Organized data storage
│   ├── raw/                       # Original datasets (customer_conversations.xlsx)
│   ├── initial_intents/           # Generated initial intents
│   ├── ontologies/                # Generated ontologies by method and threshold
│   │   ├── agglomerative/         # Agglomerative clustering results
│   │   └── hdbscan/               # HDBSCAN clustering results
│   ├── clusters/                  # Clustering results by method and parameter
│   │   ├── agglomerative/         # Reduced clusters for different thresholds
│   │   └── hdbscan/               # HDBSCAN clustering outputs
│   └── evaluations/               # Evaluation results and reports
│       ├── classified_conversations/ # Classified conversation results
│       ├── comparison_reports/    # Ontology comparison summaries
│       └── metrics/               # Evaluation metrics
├── visualizations/                # Generated HTML visualizations
├── config/                        # Configuration settings
└── requirements.txt               # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install datasets pandas openai python-dotenv pydantic scikit-learn numpy hdbscan plotly
```

### 2. Set up Environment

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Full Pipeline

```bash
python scripts/run_full_pipeline.py
```

This will:
- Download the conversation dataset
- Generate initial intents
- Cluster similar intents
- Evaluate the resulting ontology
- Create visualizations

## Usage Examples

### Run Full Pipeline
```bash
python scripts/run_full_pipeline.py --mode pipeline
```

### Compare Different Clustering Parameters
```bash
python scripts/run_full_pipeline.py --mode compare
```

### Compare Specific Thresholds
```bash
python scripts/compare_thresholds.py --method agglomerative --thresholds 0.58 0.6 0.62
```

### Generate Visualizations
```bash
python scripts/generate_visualizations.py
```

### Generate Specific Visualization
```bash
python scripts/generate_visualizations.py --method agglomerative --parameter 0.6
```

## Key Components

### FileManager
Centralized file path management ensuring consistent data organization across all components.

### IntentGenerator
Processes conversation data in batches to generate initial customer intent categories using LLM analysis.

### Clustering Algorithms
- **AgglomerativeClusterer**: Hierarchical clustering with distance thresholds
- **HDBSCANClusterer**: Density-based clustering with noise detection

### OntologyEvaluator
Comprehensive evaluation including:
- Mutual exclusivity analysis
- Coverage assessment  
- Redundant intent detection

### PCAVisualizer
Creates interactive 3D visualizations of clustering results using PCA dimensionality reduction.

## Configuration

Modify `config/settings.py` to adjust:
- Default clustering parameters
- Batch sizes for processing
- LLM model selections
- Evaluation thresholds

## Advanced Usage

### Custom Clustering Parameters

```python
from src.data import FileManager
from src.ontology import OntologyBuilder

file_manager = FileManager()
builder = OntologyBuilder(file_manager)

# Test multiple thresholds
thresholds = [0.55, 0.6, 0.65]
results = builder.compare_clustering_thresholds(thresholds)
```

### Manual Intent Merging

```python
from src.ontology import IntentMerger
from src.data import FileManager, DataLoader, DataPreprocessor
from src.utils import LLMClient

file_manager = FileManager()
merger = IntentMerger(
    file_manager,
    DataLoader(file_manager),
    DataPreprocessor(file_manager),
    LLMClient()
)

# Merge similar intents
merged_intents = merger.merge_intents("agglomerative", 0.6)
```

### Custom Evaluation

```python
from src.evaluation import OntologyEvaluator

evaluator = OntologyEvaluator(file_manager, data_loader, preprocessor, llm_client)
metrics = evaluator.evaluate_ontology("agglomerative", 0.6)

print(f"Coverage: {metrics.coverage:.2f}%")
print(f"Mutual Exclusivity: {'✅' if metrics.passes_exclusivity else '❌'}")
```

## Output Files

### Data Structure
All outputs are organized in the `data/` directory:
- `raw/` - Original conversation datasets (customer_conversations.xlsx)
- `initial_intents/` - Generated initial intents from conversation analysis
- `ontologies/` - Final ontologies organized by clustering method and parameters
  - `agglomerative/` - Results for different distance thresholds (0.58-0.64)
  - `hdbscan/` - HDBSCAN clustering results
- `clusters/` - Intermediate clustering results with reduced cluster data
- `evaluations/` - Assessment results and reports
  - `classified_conversations/` - Conversation classification results by method/threshold
  - `comparison_reports/` - Ontology comparison summaries
  - `metrics/` - Detailed evaluation metrics

### Visualizations
Interactive HTML files in `visualizations/`:
- `agglomerative_clustering_pca_visualization.html` - 3D PCA plot for agglomerative clustering
- `hdbscan_clustering_pca_visualization.html` - 3D PCA plot for HDBSCAN clustering
- Hover details with intent names and descriptions

