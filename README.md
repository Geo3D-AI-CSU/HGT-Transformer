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

### Step 2: Spatiotemporal heterogeneous graph construction and HGT-Transformer training.

```bash
python models/HGT_Transformer.py 
```

**Output**: `trained_hgt_transformer.pt` - model weights `results/train_curve.png` - training curve 

## Notes

- Remote sensing dataset is not included in this repository
- Public remote sensing datasets can be used for testing
- GPU acceleration is recommended for large-scale processing
