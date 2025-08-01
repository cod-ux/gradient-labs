# Customer Intent Ontology Building System

A comprehensive system for building and evaluating customer intent ontologies from conversation data using machine learning clustering and LLM-based analysis.

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
│   ├── clustering/                # Clustering algorithms
│   ├── ontology/                  # Ontology building and merging
│   ├── evaluation/                # Quality assessment
│   ├── visualization/             # PCA plots and visualizations
│   └── utils/                     # Utilities and LLM client
├── scripts/                       # Executable scripts
│   ├── run_full_pipeline.py       # Main pipeline orchestrator
│   ├── compare_thresholds.py      # Parameter comparison
│   ├── generate_visualizations.py # Visualization generation
│   └── migrate_data.py            # Data migration utility
├── data/                          # Organized data storage
│   ├── raw/                       # Original datasets
│   ├── ontologies/                # Generated ontologies
│   ├── clusters/                  # Clustering results
│   ├── intent_categories/         # Final intent categories
│   └── evaluations/               # Evaluation results
├── outputs/                       # Generated outputs
│   ├── visualizations/            # HTML visualizations
│   └── reports/                   # Analysis reports
└── config/                        # Configuration settings
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
- `ontologies/` - JSON files with generated customer intents
- `clusters/` - Detailed clustering results by method
- `intent_categories/` - Final intent categories for each configuration
- `evaluations/` - Classification results and comparison reports

### Visualizations
Interactive HTML files in `outputs/visualizations/`:
- 3D PCA plots showing intent clustering
- Hover details with intent names and descriptions

## Migration

If you have existing files in the old structure, run:
```bash
python scripts/migrate_data.py
```

This will automatically move files to the new organized structure.

## Contributing

The system is designed to be modular and extensible:
- Add new clustering algorithms by extending `BaseClustering`
- Implement custom evaluation metrics in the `evaluation` module
- Create new visualization types in the `visualization` module

## License

[Add your license information here]