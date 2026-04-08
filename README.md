# A Dual-Transformer Network for Spatiotemporal Modeling of Carbon Dioxide Column Concentration (XCO2) Based on Dynamic Heterogeneous Graphs

## Overview

Dual-Transformer is a deep learning framework for complex spatiotemporal modeling tasks, aiming to integrate the advantages of HGT and Transformer architectures to enable the learning and predictive modeling of dynamic heterogeneous graphs.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

## Data Preparation

### Required Data

1. **Remote Sensing Data** (for build a graph structure)
   - Format: NetCDF files
   - Location: `data/`
   - The dataset should include CAMS-EGG4, CAMS-IO, OCO-2, ERA-5 and other remote sensing data.

### Directory Structure

```
data/
├── CAMS-EGG4/          
│   ├── cams_01.nc
│   ├── cams_02.nc
│   └── ...
└── OCO-2/             
    ├── oco2_01.nc
    ├── oco2_02.nc
    └── ...
```

## Running the Project

### Step 1: Build CAMS-IO-interpolation.nc

Interpolate CAMS-IO data onto the standard grid of CAMS-EGG4:

```bash
python graphs/interpolation.py 
```

**Output**: `processed_data/CAMS-IO-interpolation.nc` 

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
