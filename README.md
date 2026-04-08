# A Dual-Transformer Network for Spatiotemporal Modeling of Carbon Dioxide Column Concentration (XCO2) Based on Dynamic Heterogeneous Graphs

## Overview

Geo-MAG is a framework for geological map understanding that integrates knowledge graphs with multimodal large language models. The system processes geological reports to build knowledge graphs, extracts metadata from geological maps using vision-language models, and generates semantic interpretations using multimodal large models.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### API Keys Configuration

Create a `.env` file in the project root directory with the following environment variables:

```env
# Vision-Language Model API (e.g., DashScope)
VL_API_KEY=your_vision_language_model_api_key
VL_API_BASE=https://api.openai.com/v1  # or OpenAI, or your provider's endpoint

# Multimodal Large Language Model API
MLLM_API_KEY=your_multimodal_llm_api_key
MLLM_API_BASE=https://api.openai.com/v1  # or your provider's endpoint

# Vector Database (if applicable)
VECTOR_DB_API_KEY=your_vector_database_api_key
VECTOR_DB_ENDPOINT=your_vector_database_endpoint

# Optional: Logging and Monitoring
LOG_LEVEL=INFO
```

## Data Preparation

### Required Data

1. **Geological Reports** (for knowledge graph construction)
   - Format: PDF or TXT files
   - Location: `data/reports/`
   - Each report should contain geological descriptions, stratigraphy, lithology, and structural information

2. **Geological Maps** (for interpretation)
   - Format: PNG or JPG images
   - Location: `data/maps/`
   - Images should be high-resolution and include legends

### Directory Structure

```
data/
├── reports/          # Geological report files
│   ├── report_001.pdf
│   ├── report_002.pdf
│   └── ...
└── maps/             # Geological map images
    ├── map_001.png
    ├── map_002.jpg
    └── ...
```

## Running the Project

### Step 1: Build Knowledge Graph

Process geological reports to construct the knowledge graph:

```bash
python scripts/build_kg.py --reports data/reports/ --output kg/geological_kg.json
```

**Output**: `kg/geological_kg.json` - Structured knowledge graph with entities and relationships

### Step 2: Extract Map Metadata

Use the vision-language model to extract metadata from geological maps:

```bash
python scripts/extract_metadata.py --maps data/maps/ --output metadata/map_metadata.json
```

**Output**: `metadata/map_metadata.json` - Extracted metadata including legends, strata, and structures

### Step 3: Retrieve Knowledge Subgraphs

Retrieve relevant subgraphs from the knowledge graph based on map metadata:

```bash
python scripts/retrieve_kg.py --metadata metadata/map_metadata.json --kg kg/geological_kg.json --output subgraphs/relevant_subgraph.json
```

**Output**: `subgraphs/relevant_subgraph.json` - Relevant knowledge subgraphs

### Step 4: Generate Interpretation

Generate geological interpretation using the multimodal large model:

```bash
python scripts/interpret.py --metadata metadata/map_metadata.json --subgraph subgraphs/relevant_subgraph.json --output results/interpretation.txt
```

**Output**: `results/interpretation.txt` - Generated geological interpretation report

## Quick Start

Run the complete pipeline:

```bash
bash run_pipeline.sh --reports data/reports/ --maps data/maps/ --output results/
```

## Notes

- Geological map data is not included in this repository due to copyright and sensitivity concerns
- Public geological datasets can be used for testing
- API keys must be obtained from respective service providers
- GPU acceleration is recommended for large-scale processing
