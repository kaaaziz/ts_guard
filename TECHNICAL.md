# TSGuard Technical Documentation

**Version:** 0.1  
**Last Updated:** 2025

This document provides comprehensive technical details for TSGuard, including mathematical formulations, advanced usage patterns, detailed input requirements, and implementation specifics. For quick start and installation instructions, please refer to the main [README.md](README.md).

---

## Table of Contents

1. [Mathematical Formulations](#mathematical-formulations)
2. [Model Architecture Details](#model-architecture-details)
3. [Advanced Usage](#advanced-usage)
4. [Detailed Input Requirements](#detailed-input-requirements)
5. [API Reference](#api-reference)
6. [Implementation Details](#implementation-details)
7. [Performance Optimization](#performance-optimization)

---

## Mathematical Formulations

### Problem Formulation

Given a network of **N** sensors producing time series data **X_t ∈ ℝ^N** at discrete time steps **t**, where some entries may be missing (represented as NaN), TSGuard aims to reconstruct missing values.

Let:
- **X_t** = [x_{t,1}, x_{t,2}, ..., x_{t,N}]^T ∈ ℝ^N denote the observations at time t
- **M_t** = [m_{t,1}, m_{t,2}, ..., m_{t,N}]^T ∈ {0,1}^N denote the mask where m_{t,i} = 1 if x_{t,i} is missing, 0 otherwise
- **Ŷ_t** = [ŷ_{t,1}, ŷ_{t,2}, ..., ŷ_{t,N}]^T ∈ ℝ^N denote the imputed values

The objective is to learn a function **f** such that:

**Ŷ_t = f(X_{t-L:t-1}, M_t, A)**

where:
- **X_{t-L:t-1}** is the historical sequence of length L
- **A** is the spatial adjacency matrix
- **f** is the GCN-LSTM hybrid model

### Spatial Adjacency Matrix Construction

The adjacency matrix **A** captures spatial relationships between sensors:

1. **Distance Computation**: For sensors i and j with coordinates (lat_i, lon_i) and (lat_j, lon_j), compute haversine distance:

   **d_{ij} = 2R · arcsin(√(sin²(Δlat/2) + cos(lat_i) · cos(lat_j) · sin²(Δlon/2)))**

   where R = 6371 km (Earth's radius).

2. **Gaussian Kernel**: Apply Gaussian kernel to distances:

   **A_{ij} = exp(-d_{ij}² / σ²)**

   where σ² = sigma_sq_ratio × Var(d_{ij}) = sigma_sq_ratio × std(d_{ij})², with default sigma_sq_ratio = 0.1. The variance is computed from the standard deviation of all pairwise distances.

3. **Self-Loops**: Add self-connections:

   **A_{ii} = 1.0** for all i

4. **Symmetric Normalization**: Apply symmetric normalization for stable graph operations:

   **Â = D^{-1/2} A D^{-1/2}**

   where **D** is the degree matrix with **D_{ii} = Σ_j A_{ij}**.

### Graph Convolutional Network (GCN)

For each time step t, the spatial aggregation is performed as:

**h_i^(spatial) = ReLU(Σ_j x_{t,j} · Â_{ji} · W)**

where:
- **h_i^(spatial)** ∈ ℝ^d is the spatial embedding for sensor i
- **W** ∈ ℝ^{1×d} is the learnable weight matrix
- **d** is the GCN hidden dimension (default: 64)

In matrix form:

**H^(spatial) = ReLU(X_t · Â^T · W)**

Note: The implementation uses `X_t @ Â^T @ W` which is equivalent to aggregating over neighbors via the transposed adjacency matrix, followed by linear projection.

### Long Short-Term Memory (LSTM)

The temporal processing operates on the sequence of spatial embeddings:

**H^(spatial) = [h_1^(spatial), h_2^(spatial), ..., h_T^(spatial)]** ∈ ℝ^{T×d}

The LSTM processes this sequence:

**h_t^(temporal), c_t = LSTM(h_t^(spatial), h_{t-1}^(temporal), c_{t-1})**

where:
- **h_t^(temporal)** ∈ ℝ^h is the hidden state at time t
- **c_t** is the cell state
- **h** is the LSTM hidden dimension (default: 64)

### Output Projection

The final prediction for each sensor is obtained via linear projection:

**ŷ_{t,i} = W_out · h_t^(temporal)**

where **W_out** ∈ ℝ^{h×N} maps the temporal hidden state to per-sensor predictions.

### Training Objective

The model is trained using a masked Mean Squared Error (MSE) loss:

**L = (1/|M|) Σ_{(t,i) ∈ M} (ŷ_{t,i} - y_{t,i})²**

where:
- **M** = {(t,i) | m_{t,i} = 1} is the set of (time, sensor) pairs that were originally missing
- **|M|** is the cardinality of M
- **y_{t,i}** is the ground truth value
- **ŷ_{t,i}** is the model prediction

This ensures the model focuses learning on reconstructing missing values rather than memorizing observed data.

### Loss Function Implementation

```python
def masked_loss(outputs, targets, mask):
    """
    outputs, targets, mask: shape (batch, seq_len, num_nodes)
    """
    loss = criterion(outputs, targets)  # MSE per element
    masked_loss = loss * mask  # Apply mask
    denom = torch.sum(mask)
    if denom <= 0:
        return torch.tensor(0.0, device=outputs.device)
    return torch.sum(masked_loss) / denom
```

---

## Model Architecture Details

### GCNLSTMImputer Class

```python
class GCNLSTMImputer(nn.Module):
    def __init__(
        self,
        adj,                    # (N, N) normalized adjacency matrix
        num_nodes,              # Number of sensors
        in_features,            # Input feature dimension (typically num_nodes)
        gcn_hidden=64,         # GCN hidden dimension
        lstm_hidden=64,        # LSTM hidden dimension
        out_features=None,     # Output dimension (default: num_nodes)
        gcn_dropout=0.1,       # GCN dropout rate
        lstm_dropout=0.1,      # LSTM dropout rate
    ):
```

**Forward Pass**:
1. For each time step t in the sequence:
   - Apply GCN: `gcn_out = GCN(x[:, t, :], adj)`
   - Apply ReLU and dropout
2. Concatenate spatial embeddings: `gcn_sequence = [h_1, h_2, ..., h_T]`
3. Process with LSTM: `lstm_out, _ = LSTM(gcn_sequence)`
4. Project to output: `output = Linear(lstm_out)`

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_len` | 36 | Sequence length for temporal window |
| `gcn_hidden` | 64 | GCN hidden dimension |
| `lstm_hidden` | 64 | LSTM hidden dimension |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-3 | Adam optimizer learning rate |
| `epochs` | 20 | Number of training epochs |
| `gcn_dropout` | 0.1 | Dropout rate for GCN layer |
| `lstm_dropout` | 0.1 | Dropout rate for LSTM layer |
| `sigma_sq_ratio` | 0.1 | Variance ratio for Gaussian kernel |

### Training Process

1. **Data Splitting**: Temporal split by months
   - Training: January, February, April, May, July, August, October
   - Validation: March, June, September, December

2. **Normalization**: Min-max scaling per sensor
   - `x_scaled = (x - min) / (max - min)`
   - Min/max computed from training data only

3. **Window Generation**: Sliding windows with next-step prediction
   - Input: `X[t : t+L]` (times t to t+L-1)
   - Target: `Y[t+1 : t+L+1]` (times t+1 to t+L)
   - Mask: `M[t+1 : t+L+1]`

4. **Loss Computation**: Masked MSE on originally missing positions

5. **Checkpointing**: Saves model weights, scaler parameters, and adjacency matrix

---

## Advanced Usage

### Programmatic API

#### Training with Custom Parameters

```python
from models.simulation_original import train_model
import pandas as pd
import torch

# Load data
tr = pd.read_csv("training_data.csv")
df = pd.read_csv("sensor_data.csv")
pf = pd.read_csv("positions.csv")

# Train with custom parameters
model = train_model(
    tr=tr,                      # Ground truth DataFrame
    df=df,                      # Missing data DataFrame
    pf=pf,                      # Positions (DataFrame, dict, or path)
    epochs=30,                  # Number of training epochs
    model_path="custom_model.pth",
    seq_len=48,                 # Sequence length
    batch_size=64,              # Batch size
    lr=5e-4,                    # Learning rate
    sigma_sq_ratio=0.15,         # Adjacency kernel variance ratio
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

#### Inference and Simulation

```python
from models.simulation_original import run_simulation_with_live_imputation
import torch
import json

# Load trained model
model = torch.load("generated/model_TSGuard.pth", map_location="cpu")
model.eval()

# Load scaler
def load_scaler_from_json(scaler_json_path: str):
    with open(scaler_json_path, "r") as f:
        params = json.load(f)
    min_val = float(params["min_val"])
    max_val = float(params["max_val"])
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    def scaler(x): return (x - min_val) / denom
    def inv_scaler(x): return x * denom + min_val
    return scaler, inv_scaler

scaler, inv_scaler = load_scaler_from_json("generated/model_TSGuard_scaler.json")

# Run simulation
run_simulation_with_live_imputation(
    sim_df=tr,
    missing_df=df,
    positions=pf,
    model=model,
    scaler=scaler,
    inv_scaler=inv_scaler,
    device=torch.device("cpu"),
    window_hours=24,
    graph_placeholder=None,      # Streamlit placeholder (optional)
    sliding_chart_placeholder=None,
    gauge_placeholder=None,
)
```

#### Custom Model Architecture

```python
from models.simulation_original import GraphConvolution, GCNLSTMImputer
import torch.nn as nn

# Create custom adjacency matrix
adj = create_adjacency_matrix(
    latlng_df=positions_df,
    threshold_type="gaussian",
    sigma_sq_ratio=0.2  # Custom variance ratio
)

# Initialize custom model
model = GCNLSTMImputer(
    adj=adj,
    num_nodes=36,
    in_features=1,
    gcn_hidden=128,      # Larger GCN hidden dimension
    lstm_hidden=256,     # Larger LSTM hidden dimension
    gcn_dropout=0.2,     # Higher dropout
    lstm_dropout=0.2,
)

# Custom training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)

for epoch in range(epochs):
    model.train()
    for batch_x, batch_y, batch_mask in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = masked_loss(outputs, batch_y, batch_mask)
        loss.backward()
        optimizer.step()
    # Validation and scheduling...
```

### Custom Constraints

#### Spatial Constraints

```python
# Define spatial constraint programmatically
spatial_constraint = {
    "type": "Spatial",
    "distance in km": 5.0,      # Maximum distance for neighbor relationship
    "distance in miles": 3.107, # Alternative unit
    "diff": 10.0                 # Maximum allowed difference between neighbors
}

# Add to session state (for Streamlit) or constraint list
constraints = [spatial_constraint]
```

#### Temporal Constraints

```python
# Define temporal constraint
temporal_constraint = {
    "type": "Temporal",
    "month": "January",          # Month name or number (1-12)
    "option": "Less than",       # "Greater than" or "Less than"
    "temp_threshold": 50.0       # Threshold value
}

# Multiple temporal constraints
temporal_constraints = [
    {"type": "Temporal", "month": "January", "option": "Less than", "temp_threshold": 50.0},
    {"type": "Temporal", "month": "July", "option": "Greater than", "temp_threshold": 30.0},
]
```

### Integration with External Systems

#### Database Integration

```python
import pandas as pd
from sqlalchemy import create_engine
from helpers import load_training_data, normalize_df_columns

# Connect to database
engine = create_engine("postgresql://user:pass@localhost/dbname")

# Load data from database
query = """
    SELECT datetime, sensor_001, sensor_002, ...
    FROM sensor_readings
    WHERE datetime >= '2024-01-01'
    ORDER BY datetime
"""
training_data = pd.read_sql(query, engine)
training_data = normalize_df_columns(training_data)

# Continue with TSGuard pipeline
```

#### API Integration

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

# Fetch data from REST API
def fetch_sensor_data(start_date, end_date):
    url = "https://api.example.com/sensors/data"
    params = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "format": "csv"
    }
    response = requests.get(url, params=params)
    return pd.read_csv(io.StringIO(response.text))

# Use in TSGuard
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
data = fetch_sensor_data(start, end)
```

#### Model Serving

```python
import torch
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model = torch.load("generated/model_TSGuard.pth")
model.eval()

@app.route("/impute", methods=["POST"])
def impute():
    data = request.json
    # Preprocess input
    x = torch.FloatTensor(data["values"]).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
    return jsonify({"imputed": output.squeeze().tolist()})
```

### Batch Processing

```python
from models.simulation_original import train_model
import glob
import pandas as pd

# Process multiple datasets
datasets = glob.glob("data/*/")

for dataset_path in datasets:
    tr = pd.read_csv(f"{dataset_path}/ground_truth.csv")
    df = pd.read_csv(f"{dataset_path}/missing.csv")
    pf = pd.read_csv(f"{dataset_path}/positions.csv")
    
    model = train_model(
        tr=tr, df=df, pf=pf,
        model_path=f"models/{dataset_path.split('/')[-1]}.pth"
    )
    print(f"Trained model for {dataset_path}")
```

---

## Detailed Input Requirements

### Training Data (Ground Truth)

**File Format**: CSV or TXT (comma-separated or tab-separated)

**Required Structure**:
```csv
datetime,sensor_001,sensor_002,sensor_003,sensor_004,...
2024-01-01 00:00:00,25.3,24.8,26.1,25.9,...
2024-01-01 01:00:00,25.5,25.0,26.3,26.1,...
2024-01-01 02:00:00,25.7,25.2,26.5,26.3,...
...
```

**Column Requirements**:
- **Time column**: 
  - Must be named `datetime`, `timestamp`, `date`, or `time` (case-insensitive)
  - Alternative: Use pandas DatetimeIndex (no explicit datetime column)
  - Must be parseable by `pd.to_datetime()`
  - Supported formats: ISO 8601, common date formats (YYYY-MM-DD, MM/DD/YYYY, etc.)

- **Sensor columns**: 
  - All non-datetime columns are treated as sensor measurements
  - Column names are used as sensor IDs
  - Numeric sensor IDs are automatically zero-padded to 6 digits (e.g., `1` → `000001`, `123` → `000123`)
  - Non-numeric sensor IDs are preserved as-is

**Data Quality Requirements**:
- **Temporal Resolution**: Recommended hourly or sub-hourly (e.g., 15-min, 30-min, 1-hour intervals)
- **Missing Values**: Should be minimal in ground truth (used for training supervision)
  - Represented as: `NaN`, `NA`, empty cells, or `None`
- **Data Types**: All sensor columns must be numeric (int or float)
- **Temporal Continuity**: Gaps are acceptable, but very large gaps (>24 hours) may affect model performance
- **Value Range**: Should be within expected physical/statistical bounds for the measured variable

**Example Valid Formats**:

```csv
# Format 1: Explicit datetime column
datetime,000001,000002,000003
2024-01-01 00:00:00,25.3,24.8,26.1
2024-01-01 01:00:00,25.5,25.0,26.3

# Format 2: Timestamp column
timestamp,sensor_1,sensor_2,sensor_3
2024-01-01T00:00:00Z,25.3,24.8,26.1
2024-01-01T01:00:00Z,25.5,25.0,26.3

# Format 3: Date column with time
date,001,002,003
2024-01-01 00:00,25.3,24.8,26.1
2024-01-01 01:00,25.5,25.0,26.3
```

### Sensor Data (Incomplete/Missing)

**File Format**: Same as training data (CSV or TXT)

**Structure**: Identical column structure to training data, but with missing values

**Missing Value Representation**:
- `NaN` (preferred)
- `NA`
- Empty cells
- `None` (in Python)
- `-999` or other sentinel values (requires preprocessing)

**Purpose**: Represents real-world scenario where:
- Sensors fail temporarily
- Data transmission is interrupted
- Communication delays occur
- Network issues prevent data collection

**Requirements**:
- Must have same sensor columns as training data
- Must have overlapping timestamps with training data
- Missing pattern should be realistic (not completely random)
- At least some sensors should have some observed values at each time step

**Example**:
```csv
datetime,000001,000002,000003,000004
2024-01-01 00:00:00,25.3,NaN,26.1,25.9
2024-01-01 01:00:00,NaN,25.0,26.3,NaN
2024-01-01 02:00:00,25.7,25.2,NaN,26.3
```

### Positions File

**File Format**: CSV

**Required Columns**:
- `sensor_id`: Unique identifier matching sensor column names in data files
  - Must match exactly after normalization (zero-padding applied)
  - String type recommended
- `latitude`: Decimal degrees (WGS84 coordinate system)
  - Range: -90 to 90
  - Float type
- `longitude`: Decimal degrees (WGS84 coordinate system)
  - Range: -180 to 180
  - Float type

**Alternative Format**: Two-column CSV with `longitude`, `latitude` and sensor IDs as row index

**Example Format 1** (Preferred):
```csv
sensor_id,latitude,longitude
000001,48.8566,2.3522
000002,48.8606,2.3376
000003,48.8506,2.3622
000004,48.8666,2.3422
```

**Example Format 2** (Alternative):
```csv
,longitude,latitude
000001,2.3522,48.8566
000002,2.3376,48.8606
000003,2.3622,48.8506
```

**Example Format 3** (Dictionary/JSON):
```python
positions = {
    0: (2.3522, 48.8566),  # (longitude, latitude)
    1: (2.3376, 48.8606),
    2: (2.3622, 48.8506),
}
```

**Data Quality Requirements**:
- All sensors in data files must have corresponding entries in positions file
- Coordinates must be valid (within WGS84 bounds)
- No duplicate sensor IDs
- Coordinates should reflect actual sensor locations (affects spatial relationships)

### Data Alignment and Validation

TSGuard performs automatic alignment and validation:

1. **Column Alignment**: 
   - Matches sensor columns between training and sensor data files
   - Handles zero-padding automatically
   - Raises error if sensor mismatch detected

2. **Temporal Alignment**:
   - Finds intersection of timestamps
   - Sorts by datetime
   - Removes duplicates (keeps first occurrence)
   - Floors timestamps to hour precision

3. **Position Matching**:
   - Matches sensor IDs from data files with position file
   - Handles zero-padding and string normalization
   - Validates coordinate ranges

4. **Data Type Validation**:
   - Ensures numeric sensor values
   - Validates coordinate types
   - Checks datetime parseability

**Error Handling**:
- Missing sensors in positions file → `KeyError` with list of missing sensors
- No temporal overlap → `ValueError` with details
- Invalid coordinates → `ValueError` with problematic entries
- Type mismatches → `TypeError` with column details

---

## API Reference

### Core Functions

#### `train_model()`

```python
def train_model(
    tr: pd.DataFrame,          # Ground truth DataFrame
    df: pd.DataFrame,          # Missing data DataFrame
    pf,                        # Positions (DataFrame, dict, or path)
    epochs: int = 20,          # Number of training epochs
    model_path: str = "gcn_lstm_imputer.pth",
    seq_len: int = 36,         # Sequence length
    batch_size: int = 32,      # Batch size
    lr: float = 1e-3,          # Learning rate
    sigma_sq_ratio: float = 0.1,  # Adjacency kernel variance ratio
    device: torch.device = None,
) -> nn.Module:
    """
    Train GCN-LSTM imputation model.
    
    Returns:
        Trained PyTorch model
    """
```

#### `run_simulation_with_live_imputation()`

```python
def run_simulation_with_live_imputation(
    sim_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    positions,
    model: nn.Module,
    scaler: Callable,
    inv_scaler: Callable,
    device: torch.device,
    window_hours: int = 24,
    graph_placeholder=None,
    sliding_chart_placeholder=None,
    gauge_placeholder=None,
) -> None:
    """
    Run real-time simulation with live imputation and visualization.
    
    Updates Streamlit placeholders with real-time results.
    """
```

#### `create_adjacency_matrix()`

```python
def create_adjacency_matrix(
    latlng_df: pd.DataFrame,
    threshold_type: str = "gaussian",
    sigma_sq_ratio: float = 0.1,
) -> torch.Tensor:
    """
    Create normalized adjacency matrix from sensor positions.
    
    Args:
        latlng_df: DataFrame with columns ['sensor_id', 'latitude', 'longitude']
        threshold_type: 'gaussian' or 'threshold'
        sigma_sq_ratio: Variance ratio for Gaussian kernel
    
    Returns:
        Normalized adjacency matrix (N, N) as torch.Tensor
    """
```

### Helper Functions

#### `load_training_data()`

```python
@st.cache_data
def load_training_data(file) -> pd.DataFrame:
    """
    Load and validate training data from file.
    
    Handles multiple datetime column name formats.
    """
```

#### `normalize_positions_df()`

```python
def normalize_positions_df(pos) -> pd.DataFrame:
    """
    Normalize positions into DataFrame with standard columns.
    
    Accepts dict, DataFrame, or path-like.
    Returns DataFrame with columns: ['sensor_id', 'latitude', 'longitude']
    """
```

---

## Implementation Details

### Data Preprocessing Pipeline

1. **DateTime Normalization**:
   ```python
   df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
   df = df.dropna(subset=["datetime"])
   df.index = df["datetime"].floor("h")  # Floor to hour
   ```

2. **Sensor ID Canonicalization**:
   ```python
   def _id_norm(s: str) -> str:
       if s.lower() == "datetime":
           return "datetime"
       return s.zfill(6) if s.isdigit() else s
   ```

3. **Position Remapping**:
   - If positions have sequential IDs (0, 1, 2, ..., N-1), remap to actual sensor IDs from data
   - Ensures 1:1 correspondence between data columns and position entries

### Model Forward Pass

```python
def forward(self, x):
    """
    x shape: (batch_size, seq_len, num_nodes)
    """
    batch_size, seq_len, num_nodes = x.shape
    gcn_outputs = []
    
    # Spatial processing per time step
    for t in range(seq_len):
        gcn_out = self.gcn(x[:, t, :], self.adj)  # (B, gcn_hidden)
        gcn_out = self.relu(gcn_out)
        gcn_out = self.dropout_gcn(gcn_out)
        gcn_outputs.append(gcn_out.unsqueeze(1))
    
    # Concatenate spatial embeddings
    gcn_sequence = torch.cat(gcn_outputs, dim=1)  # (B, seq_len, gcn_hidden)
    
    # Temporal processing
    lstm_out, _ = self.lstm(gcn_sequence)  # (B, seq_len, lstm_hidden)
    lstm_out = self.dropout_lstm(lstm_out)
    
    # Output projection
    output = self.fc(lstm_out)  # (B, seq_len, num_nodes)
    return output
```

### Inference Scenarios

TSGuard handles three scenarios during inference:

1. **Scenario 1 - Delayed Data**:
   - Sensor value is missing but expected to arrive
   - Wait for delay threshold (σ minutes) before imputing
   - Alert: "Waiting for late data"

2. **Scenario 2 - Neighbors Available**:
   - Target sensor missing, but neighbors have values
   - Use GCN-LSTM model for imputation
   - Validate against constraints
   - Alert: "Imputation completed" or "Out of range"

3. **Scenario 3 - No Neighbors**:
   - Both target and neighbors missing
   - Fallback to historical patterns or rule-based imputation
   - Alert: "Using historical patterns" or "No reliable estimate"

---

## Performance Optimization

### Memory Optimization

1. **Sparse Adjacency Matrices**: For large networks (>1000 sensors), consider sparse matrix implementations:
   ```python
   from scipy.sparse import csr_matrix
   adj_sparse = csr_matrix(adj.numpy())
   ```

2. **Batch Processing**: Process data in smaller batches during inference:
   ```python
   batch_size = 16
   for i in range(0, len(data), batch_size):
       batch = data[i:i+batch_size]
       output = model(batch)
   ```

3. **Gradient Checkpointing**: For very deep models, use gradient checkpointing:
   ```python
   from torch.utils.checkpoint import checkpoint
   ```

### Computational Optimization

1. **GPU Acceleration**: Use CUDA for training and inference:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   ```

2. **Mixed Precision Training**: Use FP16 for faster training:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   ```

3. **DataLoader Optimization**: Use multiple workers and pin memory:
   ```python
   DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
   ```

### Scalability Considerations

- **Adjacency Matrix**: O(N²) memory complexity
  - For N > 1000: Consider k-nearest neighbors (k-NN) graph
  - For N > 5000: Use sparse representations

- **Sequence Length**: Longer sequences improve accuracy but increase memory
  - Balance: seq_len = 24-48 for hourly data

- **Batch Size**: Larger batches improve GPU utilization but require more memory
  - Recommended: 16-64 depending on GPU memory

---

## Additional Resources

- **Code Documentation**: Inline docstrings in source files
- **Example Notebooks**: See `examples/` directory (if available)
- **Issue Tracker**: GitHub issues for bug reports and feature requests
- **Community**: Contact contributors for questions and collaboration

---

**Last Updated**: January 2025  
**Maintained by**: TSGuard Development Team

