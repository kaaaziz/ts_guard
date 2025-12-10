from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import streamlit as st
from utils.config import DEFAULT_VALUES
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
from pathlib import Path
import torch.optim as optim
import json
from typing import List, Tuple, Optional
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path as _Path
import plotly.io as pio
import re
import pydeck as pdk
import shutil

import os, time, uuid
import sys, importlib, re
import pickle          
import yaml            

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


from uuid import uuid4
from contextlib import contextmanager

pio.templates.default = "plotly_white"
IMPUTATION_LOG = []

def flush_imputation_log_if_needed(
    path: str = "tsguard_imputations.csv",
    chunk_size: int = 100,
):
    """
    Écrit périodiquement le contenu de IMPUTATION_LOG dans un CSV, en mode append.
    """
    global IMPUTATION_LOG
    if len(IMPUTATION_LOG) < chunk_size:
        return

    df_log = pd.DataFrame(IMPUTATION_LOG)
    write_header = not os.path.exists(path)
    df_log.to_csv(path, index=False, mode="a", header=write_header)
    IMPUTATION_LOG.clear()

LIGHT_BG = "rgba(0,0,0,0)"
BASE_FONT = dict(
    family="Inter, Segoe UI, -apple-system, system-ui, sans-serif",
    size=14,
    color="#0F172A",  # dark text
)

# ---------------- Alert message templates ----------------
ALERT = {
    "SCENARIO_1_WAITING": (
        "Delayed data for sensor {sid} (t={ts}). "
        "Waiting for late data! ⏳"
    ),
    "SCENARIO_2_IMPUTED_OK": (
        "✅ Imputation completed for sensor {sid} at {ts}. "
        "The reconstructed value is within the expected range"
    ),
    "SCENARIO_2_IMPUTED_OOR": (
        "⚠️ Imputation out of range for sensor {sid} at {ts}. "
        "The reconstructed value violates domain constraints. Please review."
    ),
    "SCENARIO_2_NEIGHBOR_MISMATCH": (
        "🚨 Potential anomaly near sensor {sid} at {ts}. "
        "The imputed value is within range, but one or more neighboring stations are out of range. "
        #"Naïve imputation could hide an environmental event — immediate attention recommended."
    ),
    "SCENARIO_3_HIST_IMPUTE": (
        "ℹ️ Neighbor data unavailable. Imputed {sid} at {ts} using historical patterns. "
        #"Monitor closely until live data resumes."
    ),
    "SCENARIO_3_NO_ESTIMATE": (
        "🚨 No reliable estimate for sensor {sid} at {ts}. "
        "Target and neighboring stations are missing. Possible sensor or system fault."
    ),
}


def lightify(fig, *, title=None):
    """Make any Plotly fig blend with the light UI."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=LIGHT_BG,   # outside of axes
        plot_bgcolor=LIGHT_BG,    # inside axes
        font=BASE_FONT,
        title=title if title is not None else fig.layout.title.text if fig.layout.title else None,
        legend=dict(bgcolor="rgba(255,255,255,0.6)", borderwidth=0),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(showline=False, gridcolor="rgba(148,163,184,0.35)", zerolinecolor="rgba(148,163,184,0.4)")
    fig.update_yaxes(showline=False, gridcolor="rgba(148,163,184,0.35)", zerolinecolor="rgba(148,163,184,0.4)")
    return fig

def get_base64_icon(path):
    icon_bytes = Path(path).read_bytes()
    b64_str = base64.b64encode(icon_bytes).decode()
    return f"data:image/png;base64,{b64_str}"

# --- Icon spec (centered) ---
ICON_URL = get_base64_icon("images/captor_icon.png")  # you already have this helper

ICON_W, ICON_H = 128, 128  # keep high-res for crispness
ICON_SPEC = {
    "url": ICON_URL,
    "width": ICON_W,
    "height": ICON_H,
    "anchorX": ICON_W // 2,   # center the icon at the point
    "anchorY": ICON_H // 2
}

# ===================== Constraint & Scenario Helpers =====================

def _neighbors_within_km(latlng_df: pd.DataFrame, km: float) -> dict[str, list[str]]:
    """Build neighbor lists by distance threshold (km). Keys are sensor_id (string)."""
    # Drop duplicate columns (keep first occurrence)
    latlng_df = latlng_df.loc[:, ~latlng_df.columns.duplicated()].copy()

    # Force single Series even if duplicates exist upstream
    sid_series = latlng_df["sensor_id"]
    if isinstance(sid_series, pd.DataFrame):
        sid_series = sid_series.iloc[:, 0]

    ids = sid_series.astype(str).to_numpy().tolist()
    lat = pd.to_numeric(latlng_df["latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(latlng_df["longitude"], errors="coerce").to_numpy()

    # Simple O(N^2) neighbor search
    neigh = {sid: [] for sid in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = haversine_distance(lat[i], lon[i], lat[j], lon[j])
            if d <= km:
                a, b = ids[i], ids[j]
                neigh[a].append(b)
                neigh[b].append(a)
    return neigh


'''def positions_to_df(positions):
    if isinstance(positions, pd.DataFrame):
        df = positions.rename(columns={"lat": "latitude", "lng": "longitude", "lon": "longitude"}).copy()
        if "sensor_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "sensor_id"})
        return df[["sensor_id", "latitude", "longitude"]]
    rows = [{"sensor_id": str(k), "latitude": float(v[1]), "longitude": float(v[0])}
            for k, v in positions.items()]
    return pd.DataFrame(rows, columns=["sensor_id", "latitude", "longitude"])'''

def positions_to_df(positions):
    """
    Normalize static positions (from files) and merge any dynamic captors
    defined in st.session_state['dynamic_captors'].

    The returned DataFrame always has the columns:
        sensor_id, latitude, longitude
    """
    if isinstance(positions, pd.DataFrame):
        df = positions.rename(
            columns={"lat": "latitude", "lng": "longitude", "lon": "longitude"}
        ).copy()
        if "sensor_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "sensor_id"})
        base_df = df[["sensor_id", "latitude", "longitude"]]
    else:
        rows = [
            {"sensor_id": str(k), "latitude": float(v[1]), "longitude": float(v[0])}
            for k, v in positions.items()
        ]
        base_df = pd.DataFrame(rows, columns=["sensor_id", "latitude", "longitude"])

    # Merge dynamic captors from the UI (if any)
    try:
        dyn = st.session_state.get("dynamic_captors", {})
    except Exception:
        dyn = {}

    if dyn:
        dyn_rows = []
        for key, meta in dyn.items():
            # ✅ always prefer the explicit sensor_id from the UI if present
            sensor_id = str(meta.get("sensor_id", key))
            try:
                lat = float(meta.get("latitude"))
                lon = float(meta.get("longitude"))
            except (TypeError, ValueError):
                continue
            dyn_rows.append(
                {"sensor_id": sensor_id, "latitude": lat, "longitude": lon}
            )

        if dyn_rows:
            dyn_df = pd.DataFrame(dyn_rows)
            base_df = pd.concat([base_df, dyn_df], ignore_index=True)
            base_df = base_df.drop_duplicates(subset=["sensor_id"], keep="last")

    return base_df[["sensor_id", "latitude", "longitude"]]




# ---- GCN + LSTM model ----

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: (B, N), adj: (N, N) dense
        # We treat this as: spatial mixing via Â + linear projection (N → gcn_hidden)
        agg = x.matmul(adj.t())
        out = agg.matmul(self.weight)  # (B, N) @ (N, F_out) -> (B, F_out)
        return out


class GCNLSTMImputer(nn.Module):
    def __init__(
        self,
        adj,
        num_nodes,
        in_features,
        gcn_hidden,
        lstm_hidden,
        out_features,
        gcn_dropout: float = 0.1,
        lstm_dropout: float = 0.1,
    ):
        super(GCNLSTMImputer, self).__init__()
        self.adj = adj  # (N, N) normalized adjacency

        self.gcn = GraphConvolution(in_features, gcn_hidden)
        self.relu = nn.ReLU()
        self.dropout_gcn = nn.Dropout(p=gcn_dropout)

        # Single-layer LSTM over time; we add explicit dropout after it
        self.lstm = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.dropout_lstm = nn.Dropout(p=lstm_dropout)

        # Head back to N nodes
        self.fc = nn.Linear(lstm_hidden, out_features)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, num_nodes)

        Semantics:
        - For each time step t we do spatial mixing via GCN + ReLU + dropout.
        - Then feed the [batch, seq_len, gcn_hidden] sequence into LSTM.
        - FC head produces [batch, seq_len, num_nodes]; each step is a
          "next-step" prediction when trained with shifted targets.
        """
        batch_size, seq_len, num_nodes = x.shape
        gcn_outputs = []
        for t in range(seq_len):
            gcn_out = self.gcn(x[:, t, :], self.adj)     # (B, gcn_hidden)
            gcn_out = self.relu(gcn_out)
            gcn_out = self.dropout_gcn(gcn_out)
            gcn_outputs.append(gcn_out.unsqueeze(1))     # (B, 1, gcn_hidden)

        gcn_sequence = torch.cat(gcn_outputs, dim=1)      # (B, seq_len, gcn_hidden)
        lstm_out, _ = self.lstm(gcn_sequence)             # (B, seq_len, lstm_hidden)
        lstm_out = self.dropout_lstm(lstm_out)
        output = self.fc(lstm_out)                        # (B, seq_len, out_features=num_nodes)
        return output


class SpatioTemporalDataset(Dataset):
    """
    Dataset with explicit "next-step" semantics.

    Given arrays X, y, mask of shape (T, N), with seq_len = L:

        sample idx -> input window:  X[idx : idx+L]       (times t .. t+L-1)
                     -> target win:  y[idx+1 : idx+L+1]   (times t+1 .. t+L)
                     -> mask win:    mask[idx+1 : idx+L+1]

    So each output step corresponds to the next-time target relative to the
    input at that position, and the last step is "next-step after last history".
    """

    def __init__(self, X, y, mask, seq_len=36):
        assert X.shape == y.shape == mask.shape, "X, y, mask must have same shape"
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.mask = torch.FloatTensor(mask)
        self.seq_len = seq_len

    def __len__(self):
        # Need at least seq_len + 1 points to have one sample
        T = len(self.X)
        return max(0, T - self.seq_len)

    def __getitem__(self, idx):
        x_win = self.X[idx: idx + self.seq_len]                 # (L, N)
        y_win = self.y[idx + 1: idx + self.seq_len + 1]         # (L, N)
        m_win = self.mask[idx + 1: idx + self.seq_len + 1]      # (L, N)
        return x_win, y_win, m_win


criterion = nn.MSELoss(reduction="none")


def masked_loss(outputs, targets, mask):
    """
    MSE over only the masked positions, where mask==1.

    outputs, targets, mask: shape (batch, seq_len, num_nodes)
    """
    loss = criterion(outputs, targets)
    masked_loss = loss * mask
    denom = torch.sum(mask)
    if denom <= 0:
        # no valid positions; return 0 but caller should skip this batch
        return torch.tensor(0.0, device=outputs.device)
    return torch.sum(masked_loss) / denom


# ---- Geometry & adjacency ----

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(1 - a), sqrt(a))
    distance = R * c
    return distance


def create_adjacency_matrix(latlng_df, threshold_type="gaussian", sigma_sq_ratio=0.1):
    """
    Build spatial prior Â from lat/lon:

    - Haversine distances.
    - Gaussian kernel on distances: exp(-d^2 / σ^2), σ^2 = (std(d)^2)*sigma_sq_ratio.
    - Self-loops.
    - Symmetric normalization: D^{-1/2} A D^{-1/2}.
    """
    num_sensors = len(latlng_df)
    dist_matrix = np.zeros((num_sensors, num_sensors), dtype=np.float64)

    for i in range(num_sensors):
        lat1, lon1 = latlng_df.iloc[i]["latitude"], latlng_df.iloc[i]["longitude"]
        for j in range(i, num_sensors):
            lat2, lon2 = latlng_df.iloc[j]["latitude"], latlng_df.iloc[j]["longitude"]
            d = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist_matrix[j, i] = d

    if threshold_type == "gaussian":
        sigma = dist_matrix.std()
        sigma_sq = (sigma * sigma) * sigma_sq_ratio
        sigma_sq = max(sigma_sq, 1e-6)  # avoid div-by-zero
        adj_matrix = np.exp(-np.square(dist_matrix) / sigma_sq)
    else:
        threshold = np.mean(dist_matrix) * 0.5
        adj_matrix = (dist_matrix <= threshold).astype(float)

    np.fill_diagonal(adj_matrix, 1.0)

    D = np.diag(np.sum(adj_matrix, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    adj_norm = D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    return torch.tensor(adj_norm, dtype=torch.float32)


# ---- TRAINING TSGUARD -------------------------------------------------
def train_model(
    tr: pd.DataFrame,          # ground_df (ground truth)
    df: pd.DataFrame,          # missing_df (with gaps)
    pf,                        # positions (DataFrame or dict or path)
    epochs: int = 20,
    model_path: str = "gcn_lstm_imputer.pth",
    seq_len: int = 36,
    batch_size: int = 32,
    lr: float = 1e-3,
    sigma_sq_ratio: float = 0.1,
    device: torch.device = None,
):
    # ---------------- Helpers ----------------
    def _ensure_datetime_index(dfi: pd.DataFrame) -> pd.DataFrame:
        dfi = dfi.copy()
        if "datetime" in dfi.columns:
            dfi["datetime"] = pd.to_datetime(dfi["datetime"], errors="coerce")
            dfi = dfi.dropna(subset=["datetime"]).set_index("datetime")
        elif not isinstance(dfi.index, pd.DatetimeIndex):
            dfi.index = pd.to_datetime(dfi.index, errors="coerce")
            dfi = dfi[~dfi.index.isna()]
        dfi.index = dfi.index.floor("h")
        dfi = dfi[~dfi.index.duplicated(keep="first")]
        return dfi.sort_index()

    # ------------- Prepare DataFrames -------------
    ground_df  = tr.copy()
    missing_df = df.copy()

    ground_df  = _ensure_datetime_index(ground_df)
    missing_df = _ensure_datetime_index(missing_df)

    # Keep canonical string sensor IDs (zero-padded upstream)
    ground_df.columns  = [str(c).strip() for c in ground_df.columns]
    missing_df.columns = [str(c).strip() for c in missing_df.columns]

    # Positions: reuse normalisation logic (dict/df → sensor_id, latitude, longitude)
    def normalize_positions_df(pos) -> pd.DataFrame:
        """
        Normalize positions into a DataFrame with columns: sensor_id, latitude, longitude.
        Accepts dict or DataFrame.
        """
        # dict {idx: (lon, lat)}
        if isinstance(pos, dict):
            pos = (
                pd.DataFrame.from_dict(pos, orient="index", columns=["longitude", "latitude"])
                .rename_axis("sensor_id")
                .reset_index()
            )
        elif isinstance(pos, pd.DataFrame):
            pos = pos.copy()
        else:
            raise TypeError(f"Unsupported type for positions: {type(pos)}")

        pos = pos.rename(columns={"lat": "latitude", "lng": "longitude", "lon": "longitude"}).copy()
        if "sensor_id" not in pos.columns:
            pos = pos.reset_index().rename(columns={"index": "sensor_id"})

        pos["sensor_id"] = pos["sensor_id"].astype(str).str.strip()
        pos["sensor_id"] = pos["sensor_id"].apply(lambda x: x.zfill(6) if x.isdigit() else x)

        pos["latitude"]  = pd.to_numeric(pos["latitude"], errors="coerce")
        pos["longitude"] = pd.to_numeric(pos["longitude"], errors="coerce")

        return pos[["sensor_id", "latitude", "longitude"]]

    def remap_positions_if_indexed_0_to_n_minus_1(pos: pd.DataFrame, sensor_cols_in_data: list[str]) -> pd.DataFrame:
        """
        If positions have ids like '0','1',...,'N-1', replace them by real sensor ids
        from the *data* (exact order). Otherwise, just return normalized positions.
        """
        sid = pos["sensor_id"]
        if sid.str.fullmatch(r"\d+").all():
            nums = sid.astype(int)
            if nums.min() == 0 and nums.max() == len(pos) - 1:
                pos = pos.iloc[np.argsort(nums)].reset_index(drop=True).copy()
                wanted = [c for c in sensor_cols_in_data][:len(pos)]
                pos["sensor_id"] = wanted
        return pos

    latlng = normalize_positions_df(pf)
    sensor_cols_in_data = list(ground_df.columns)
    latlng = remap_positions_if_indexed_0_to_n_minus_1(latlng, sensor_cols_in_data)

    sensor_cols = latlng["sensor_id"].astype(str).tolist()

    missing_in_ground  = sorted(set(sensor_cols) - set(ground_df.columns))
    missing_in_missing = sorted(set(sensor_cols) - set(missing_df.columns))
    if missing_in_ground or missing_in_missing:
        raise KeyError(
            "Missing sensor columns after normalisation.\n"
            f"- Missing in ground: {missing_in_ground}\n"
            f"- Missing in missing: {missing_in_missing}\n"
            f"- Example ground cols: {list(ground_df.columns)[:8]}"
        )

    # Align columns and timestamps
    ground_df  = ground_df[sensor_cols]
    missing_df = missing_df[sensor_cols]

    common_idx = ground_df.index.intersection(missing_df.index)
    if common_idx.empty:
        raise ValueError("No temporal overlap between ground and missing.")
    ground_df  = ground_df.loc[common_idx].sort_index()
    missing_df = missing_df.loc[common_idx].sort_index()

    # ------------- Numpy & Imputed X -------------
    ground_data  = ground_df.to_numpy(dtype=np.float32)
    missing_data = missing_df.to_numpy(dtype=np.float32)

    imputed_df   = missing_df.ffill().bfill()
    imputed_data = imputed_df.to_numpy(dtype=np.float32)

    # Mask = 1 where originally missing in missing_df but finite in ground truth
    loss_mask = np.where(np.isnan(missing_data) & np.isfinite(ground_data), 1.0, 0.0).astype(np.float32)

    # ------------- Splits & Scaler -------------
    months = missing_df.index.month

    # Disjoint sets (if possible)
    train_month_list = [1, 2, 4, 5, 7, 8, 10]
    valid_month_list = [3, 6, 9, 12]

    train_slice = np.isin(months, train_month_list)
    valid_slice = np.isin(months, valid_month_list)

    if len(common_idx) < 2 * seq_len:
        raise ValueError(f"Not enough timestamps after alignment: {len(common_idx)} (< {2*seq_len}).")

    if not np.any(train_slice):
        raise ValueError("Empty train split (check months present in your data).")

    if not np.any(valid_slice):
        # Fallback: use non-train months as validation if available
        valid_slice = ~train_slice
        print("[train_model] Warning: no timestamps in validation months; "
              "using non-train months as validation.")
    if not np.any(valid_slice):
        # Last fallback: use train slice as validation (not ideal, but keeps training running)
        valid_slice = train_slice.copy()
        print("[train_model] Warning: using train slice as validation; metrics are not independent.")

    train_imputed = imputed_data[train_slice]
    if train_imputed.shape[0] <= seq_len:
        raise ValueError(
            f"Not enough train timestamps ({train_imputed.shape[0]}) "
            f"for seq_len={seq_len}."
        )

    min_val = float(np.nanmin(train_imputed))
    max_val = float(np.nanmax(train_imputed))
    denom   = (max_val - min_val) if (max_val - min_val) != 0 else 1.0

    scaler     = lambda x: (x - min_val) / denom
    inv_scaler = lambda x: x * denom + min_val

    scaler_params = {"min_val": min_val, "max_val": max_val}
    scaler_json   = str(_Path(model_path).with_suffix("")) + "_scaler.json"
    with open(scaler_json, "w") as f:
        json.dump(scaler_params, f, indent=2)

    X_norm = scaler(imputed_data)
    Y_norm = scaler(ground_data)
    Y_norm = np.nan_to_num(Y_norm, nan=0.0)

    X_train, y_train, m_train = X_norm[train_slice], Y_norm[train_slice], loss_mask[train_slice]
    X_val,   y_val,   m_val   = X_norm[valid_slice],   Y_norm[valid_slice],   loss_mask[valid_slice]

    # ------------- Datasets/Loaders -------------
    train_ds = SpatioTemporalDataset(X_train, y_train, m_train, seq_len=seq_len)
    val_ds   = SpatioTemporalDataset(X_val,   y_val,   m_val,   seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ------------- Model & Adj -------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adjacency from positions (latlng) in the same order as sensor_cols
    latlng = latlng.set_index("sensor_id").loc[sensor_cols].reset_index()
    adj = create_adjacency_matrix(
        latlng_df=latlng,
        threshold_type="gaussian",
        sigma_sq_ratio=sigma_sq_ratio,
    ).to(device)

    num_nodes = imputed_data.shape[1]
    model = GCNLSTMImputer(
        adj=adj,
        num_nodes=num_nodes,
        in_features=num_nodes,
        gcn_hidden=64,
        lstm_hidden=64,
        out_features=num_nodes,
        gcn_dropout=0.1,
        lstm_dropout=0.1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------- Training Loop -------------
    model.train()

    # Detect whether we’re actually running inside Streamlit.
    try:
        use_streamlit = st.runtime.exists()
    except Exception:
        use_streamlit = bool(getattr(st, "_is_running_with_streamlit", False))

    if use_streamlit:
        pbar = st.progress(0)
        status = st.container()
    else:
        pbar = None
        status = None

    for epoch in range(epochs):
        tot, n = 0.0, 0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = masked_loss(out, yb, mb)
            if torch.isfinite(loss) and torch.sum(mb) > 0:
                loss.backward()
                optimizer.step()
                tot += loss.item()
                n += 1

        # validation
        model.eval()
        with torch.no_grad():
            vtot, vn = 0.0, 0
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                out = model(xb)
                vloss = masked_loss(out, yb, mb)  # positional args only
                if torch.isfinite(vloss) and torch.sum(mb) > 0:
                    vtot += vloss.item()
                    vn += 1
        model.train()

        train_loss = tot / max(n, 1)
        val_loss   = vtot / max(vn, 1)

        if use_streamlit:
            # update progress bar
            pbar.progress(int((epoch + 1) * 100 / max(epochs, 1)))
            # write nicely under the "Training is running..." section
            with status:
                st.write(
                    f"🔹 **Epoch {epoch+1}** | "
                    f"📉 **Train:** `{train_loss:.4f}` | "
                    f"🧪 **Val:** `{val_loss:.4f}`"
                )
        else:
            print(
                f"Epoch {epoch+1}/{epochs} — "
                f"train {train_loss:.4f} | val {val_loss:.4f}",
                flush=True,
            )

    # ------------- Save & return -------------
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path} and scaler to {scaler_json}")

    # Save adjacency and imputer config for versioning
    adjacency_params = {
        "sigma_sq_ratio": sigma_sq_ratio,
    }
    adj_json = str(_Path(model_path).with_suffix("")) + "_adjacency.json"
    with open(adj_json, "w") as f:
        json.dump(adjacency_params, f, indent=2)

    imputer_config = {
        "seq_len": seq_len,
        "gcn_hidden": 64,
        "lstm_hidden": 64,
        "train_months": train_month_list,
        "valid_months": valid_month_list,
        # Online imputer defaults (can be overridden via run_simulation_with_live_imputation)
        "k_neighbors": 5,
        "trim_frac": 0.15,
        "distance_cap_km": None,
        "clim_beta": 0.3,
        "momentum_gamma": 0.99,
        "domain_clip_min": None,
        "domain_clip_max": None,
    }
    cfg_json = str(_Path(model_path).with_suffix("")) + "_imputer_config.json"
    with open(cfg_json, "w") as f:
        json.dump(imputer_config, f, indent=2)

    print(f"Saved adjacency params to {adj_json} and imputer config to {cfg_json}")
    return model


# ---- PriSTI integration helpers ----

@contextmanager
def pristi_workdir(root: str):
    """
    Historically this context manager changed the process working
    directory to the PriSTI repo:

        os.chdir(root)  ...  os.chdir(old_cwd)

    That works in a plain script, but in Streamlit it can race with the
    script reloader. If a rerun happens while the CWD is 'PriSTI/',
    Streamlit tries to open 'main_app.py' *inside* that folder and you
    get:

        [Errno 2] ... tsguard/PriSTI/main_app.py

    We now keep the same API (all existing 'with pristi_workdir(...)'
    calls still work) but make it a no-op so the global CWD never moves.
    PriSTI is already given absolute paths for its config, weights and
    data, so it does not need the CWD trick here.
    """
    yield # no-op for now


def _ensure_pristi_data_alias():
    try:
        # TSGuard project root: .../tsguard/
        root = Path(__file__).resolve().parent.parent

        src = root / "PRISTI" / "data"   # where PriSTI keeps its pm25 files
        dst = root / "data"              # what PriSTI's code expects

        # Nothing to do if there is no PriSTI data or alias already exists
        if not src.exists() or dst.exists():
            return

        try:
            # Prefer a symlink (cheap, no duplication)
            os.symlink(src, dst, target_is_directory=True)
            print(f"[PriSTI] created data alias {dst} -> {src}")
        except (OSError, AttributeError):
            # Symlinks not available (e.g. Windows without privileges) – fallback to copy
            shutil.copytree(src, dst)
            print(f"[PriSTI] copied PriSTI data into {dst}")
    except Exception as e:
        # Never break the app because of this – PriSTI stays best-effort
        print("[PriSTI] failed to set up data alias:", e)


# -------------------- PRISTI Imputation functions -------------
# --- PriSTI integration helpers ----

@st.cache_resource(show_spinner=False)
def load_pristi_artifacts(
    CONFIG_PATH: str,
    WEIGHTS_PATH: str,
    MEANSTD_PK: str,
    device: torch.device,
):
    """
    Load PriSTI model + scaling statistics.
    """

    _ensure_pristi_data_alias()

    CONFIG_PATH = os.path.abspath(CONFIG_PATH)
    WEIGHTS_PATH = os.path.abspath(WEIGHTS_PATH)
    MEANSTD_PK = os.path.abspath(MEANSTD_PK)

    # ----- locate PriSTI root folder from the config path -----
    # CONFIG_PATH looks like ".../PriSTI/config/base.yaml"
    config_dir = os.path.dirname(os.path.abspath(CONFIG_PATH))   # .../PriSTI/config
    pristi_root = os.path.dirname(config_dir)                    # .../PriSTI

    try:
        st.session_state["pristi_root"] = pristi_root
    except Exception:
        pass

    if pristi_root not in sys.path:
        sys.path.insert(0, pristi_root)

    # ----- load config -----
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}
    config.setdefault("diffusion", {})
    config["diffusion"].setdefault("adj_file", "AQI36")

    # ----- load mean/std -----
    with open(MEANSTD_PK, "rb") as f:
        meanstd = pickle.load(f)
    mean = np.asarray(meanstd[0], dtype=np.float32)
    std = np.asarray(meanstd[1], dtype=np.float32)
    std_safe = np.where(std == 0, 1.0, std)

    # ----- import PriSTI model -----
    with pristi_workdir(pristi_root):
        try:
            # Repo cloned as a folder; e.g. ./PriSTI/main_model.py
            mm = importlib.import_module("main_model")
        except ModuleNotFoundError:
            # Fallback if PriSTI is installed as a package
            mm = importlib.import_module("PriSTI.main_model")

        PriSTI_aqi36 = getattr(mm, "PriSTI_aqi36")

        model = PriSTI_aqi36(config, device).to(device)
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()

    return model, mean, std_safe


# these two mimic your Kaggle code (they read mean/std from session_state we’ll fill below)
def scale_window(x_2d: np.ndarray) -> np.ndarray:
    mean = st.session_state["pristi_mean"]
    std_safe = st.session_state["pristi_std"]
    return (x_2d - mean) / std_safe

def inv_scale_vec(x_1d: np.ndarray) -> np.ndarray:
    mean = st.session_state["pristi_mean"]
    std_safe = st.session_state["pristi_std"]
    return x_1d * std_safe + mean

def impute_window_with_pristi(
    missing_df: pd.DataFrame,
    sensor_cols: List[str],
    target_timestamp: pd.Timestamp,
    model: torch.nn.Module,
    device: torch.device,
    eval_len: int = 36,
    nsample: int = 100
) -> Tuple[pd.DataFrame, str]:
    """
    This is your tested Kaggle function, copied as-is (prints removed).
    It fills a (T=eval_len, N=len(sensor_cols)) window ending at target_timestamp.
    Returns (updated_df, "ok") or (original_df_copy, reason)
    """
    print(f"Imputing window with pristi ({nsample} samples)...")
    if "scale_window" not in globals() or "inv_scale_vec" not in globals():
        return missing_df.copy(), "Scaling functions not found."
    if list(missing_df.columns) != list(sensor_cols):
        return missing_df.copy(), "Columns mismatch."
    if target_timestamp not in missing_df.index:
        return missing_df.copy(), f"{target_timestamp} not in DataFrame index."

    end_loc = missing_df.index.get_loc(target_timestamp)
    if isinstance(end_loc, slice):
        return missing_df.copy(), "Ambiguous target timestamp."
    start_loc = end_loc - (eval_len - 1)
    if start_loc < 0:
        return missing_df.copy(), f"Not enough history (<{eval_len} rows)."

    time_index = missing_df.index[start_loc:end_loc + 1]
    filled_df = missing_df.ffill().bfill()

    window_filled = filled_df.iloc[start_loc:end_loc + 1].to_numpy(dtype=np.float32)
    window_orig   = missing_df.iloc[start_loc:end_loc + 1].to_numpy(dtype=np.float32)
    T, N = window_filled.shape

    # Mask (1=observed, 0=missing)
    mask_np = (~np.isnan(window_orig)).astype(np.float32)
    if not (mask_np == 0).any():
        return missing_df.copy(), "No missing values in window."

    # Scale and to tensors
    window_scaled = scale_window(window_filled)
    x_TN = torch.from_numpy(window_scaled).unsqueeze(0).to(device)  # (1,T,N)
    m_TN = torch.from_numpy(mask_np     ).unsqueeze(0).to(device)  # (1,T,N)
    x_NL = x_TN.permute(0, 2, 1).contiguous()  # (1,N,T)
    m_NL = m_TN.permute(0, 2, 1).contiguous()  # (1,N,T)

    # PriSTI API (inner model)
    inner = getattr(model, "model", getattr(model, "module", model))
    if not hasattr(inner, "get_side_info"):
        return missing_df.copy(), "PriSTI instance required."

    # PriSTI expects to run with its own repo root as the working directory
    # because it uses relative paths like './data/pm25/SampleData/pm25_latlng.txt'.
    pristi_root = st.session_state.get("pristi_root")
    if not pristi_root:
        # fall back to ./PriSTI relative to this file
        here = os.path.dirname(os.path.abspath(__file__))
        pristi_root = os.path.abspath(os.path.join(here, "..", "PriSTI"))
        st.session_state["pristi_root"] = pristi_root

    with pristi_workdir(pristi_root):
        observed_tp = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)  # (1,T)
        side_info = inner.get_side_info(observed_tp, m_NL)

        itp_info = None
        if getattr(inner, "use_guide", False):
            coeffs = torch.zeros((1, N, T), device=device, dtype=torch.float32)
            itp_info = coeffs.unsqueeze(1)

        inner.eval()
        with torch.no_grad():
            try:
                y_pred = inner.impute(x_NL, m_NL, side_info, int(nsample), itp_info)
            except TypeError:
                y_pred = inner.impute(x_NL, m_NL, side_info, int(nsample))

    if not isinstance(y_pred, torch.Tensor):
        return missing_df.copy(), "Non-tensor output."

    # Reduce samples & align shape
    if y_pred.dim() == 4:
        if y_pred.shape[0] == nsample:
            y_pred = y_pred.mean(dim=0)
        elif y_pred.shape[1] == nsample:
            y_pred = y_pred.mean(dim=1)
        else:
            y_pred = y_pred.mean(dim=0)

    if y_pred.dim() != 3:
        return missing_df.copy(), f"Unexpected output rank: {y_pred.dim()}."

    pred3 = y_pred[0]
    if pred3.shape == (N, T):
        pred_scaled_NT = pred3
    elif pred3.shape == (T, N):
        pred_scaled_NT = pred3.transpose(0, 1).contiguous()
    else:
        return missing_df.copy(), f"Unexpected output shape {tuple(pred3.shape)}."

    pred_scaled_TN = pred_scaled_NT.transpose(0, 1).detach().cpu().numpy()  # (T,N)
    pred_unscaled_TN = np.vstack([inv_scale_vec(pred_scaled_TN[t, :]) for t in range(T)])

    # Merge into a copy and return
    updated_df = missing_df.copy()
    miss_mask_bool = (mask_np == 0)
    for t_idx in range(T):
        ts = time_index[t_idx]
        for n_idx in range(N):
            if miss_mask_bool[t_idx, n_idx]:
                sensor_name = sensor_cols[n_idx]
                updated_df.loc[ts, sensor_name] = float(pred_unscaled_TN[t_idx, n_idx])

    return updated_df, "ok"


# ---------------- ORBITS offline results loader ----------------

def load_orbits_offline_results() -> pd.DataFrame | None:
    """
    Load ORBITS streaming-end recovered pm25 series.

    Assumes `extract_orbits_pm25.py` has already been run and produced
    `<TSGUARD_ROOT>/orbits_results/pm25_orbits_streaming.csv`.

    Returns a DataFrame with a DatetimeIndex and sensor columns,
    or None if the file does not exist.
    """
    root = Path(__file__).resolve().parent.parent  # .../tsguard-new
    path = root / "orbits_results" / "pm25_orbits_streaming.csv"
    #path = root / "orbits_results" / "pm25_missing_imputed_orbits.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError("ORBITS result file has no 'datetime' column.")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()
    return df



def _mpl_build_segments(times, values, imputed_flags, base_color, gap_hours):
    """
    Build line segments for a single sensor.

    times         : sequence of datetimes
    values        : sequence of floats
    imputed_flags : bool array, True where the point is imputed
    base_color    : color for non-imputed segments
    gap_hours     : break segments if time gap > gap_hours
    """
    times = pd.to_datetime(times)
    values = np.asarray(values, dtype=float)
    flags = np.asarray(imputed_flags, dtype=bool)

    if len(times) < 2:
        return [], []

    GAP = pd.Timedelta(hours=gap_hours) if gap_hours is not None else None

    t_num = mdates.date2num(times.to_numpy())

    segments = []
    colors = []

    for i in range(len(times) - 1):
        if not (np.isfinite(values[i]) and np.isfinite(values[i + 1])):
            continue

        if (GAP is not None) and ((times.iloc[i + 1] - times.iloc[i]) > GAP):
            continue

        segments.append([[t_num[i], values[i]], [t_num[i + 1], values[i + 1]]])
        colors.append("red" if (flags[i] or flags[i + 1]) else base_color)

    return segments, colors


def make_mpl_timeseries_figure(
    df: pd.DataFrame,
    imputed_mask: pd.DataFrame,
    sensor_cols: list[str],
    sensor_color_map: dict[str, str],
    title: str,
    gap_hours: float,
    show_legend: bool = True,
    style: str = "minimal",
) -> plt.Figure:
    """
    Build a single Matplotlib figure that mirrors the Plotly styling:

      - one line per sensor in `sensor_cols`
      - red segments where imputed_mask is True
      - optional legend

    df must contain a 'datetime' column and sensor columns.
    """
    # Normalize datetime
    df = df.copy()
    if "datetime" not in df.columns:
        idx_name = df.index.name or "datetime"
        df = df.reset_index().rename(columns={idx_name: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Make sure mask index is datetime
    imputed_mask = imputed_mask.copy()
    imputed_mask.index = pd.to_datetime(imputed_mask.index)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("none")

    any_points = False

    for col in sensor_cols:
        if col not in df.columns:
            continue

        sub = (
            df[["datetime", col]]
            .dropna(subset=[col])
            .sort_values("datetime")
        )
        if sub.empty:
            continue

        times = sub["datetime"]
        vals = sub[col].to_numpy(dtype=float)

        # Align mask on these timestamps (robust to small index issues)
        try:
            flags = imputed_mask.loc[times, col].to_numpy(dtype=bool)
        except KeyError:
            common = imputed_mask.index.intersection(times)
            tmp = imputed_mask.loc[common, col].reindex(times, fill_value=False)
            flags = tmp.to_numpy(dtype=bool)

        base_color = sensor_color_map.get(col, "#444")
        segments, seg_colors = _mpl_build_segments(
            times, vals, flags, base_color=base_color, gap_hours=gap_hours
        )
        if not segments:
            continue

        lc = LineCollection(
            segments,
            colors=seg_colors if seg_colors else base_color,
            linewidths=1.3,
        )
        ax.add_collection(lc)
        any_points = True

    # Axes limits (Matplotlib doesn’t auto-pick up LineCollection bounds)
    if any_points:
        x_all = mdates.date2num(df["datetime"].to_numpy())
        y_all = df[sensor_cols].to_numpy(dtype=float)
        y_all = y_all[np.isfinite(y_all)]
        if x_all.size > 0 and y_all.size > 0:
            ax.set_xlim(x_all.min(), x_all.max())
            ax.set_ylim(y_all.min(), y_all.max())

    # X-axis formatting
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    # --------- Style presets ----------
    if style == "minimal":
        ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="x", labelsize=9, pad=2)
        ax.tick_params(axis="y", labelsize=9, pad=2)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("none")

    elif style == "soft-dark":
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#020617")
        ax.grid(True, color="#1f2937", linewidth=0.7, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_color("#4b5563")
        ax.tick_params(colors="#e5e7eb", labelsize=9)
        ax.yaxis.label.set_color("#e5e7eb")
        ax.xaxis.label.set_color("#e5e7eb")
        ax.title.set_color("#e5e7eb")
    else:
        ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.9)

    ax.set_title(title, fontsize=12, fontweight="600")
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Sensor value", fontsize=10)

    
    #Won't be using this legend but keeping for reference
    if show_legend:
        legend_cols = sensor_cols[:12]   
        handles = [
            Line2D([0], [0],
                   color=sensor_color_map.get(c, "#444"),
                   lw=1.4,
                   label=f"Sensor {c}")
            for c in legend_cols
        ]
        handles.append(
            Line2D([0], [0], color="red", lw=1.4, label="Imputed segment")
        )
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            ncol=2,
            frameon=False,
        )

    fig.tight_layout()
    return fig


def predict_single_missing_value(
    historical_window: np.ndarray,
    target_sensor_index: int,
    model: torch.nn.Module,
    scaler: callable,
    inv_scaler: callable,
    device: torch.device,
) -> float:
    """
    Predict the next value for a single sensor using a historical window.

    historical_window: (L, num_sensors) with no NaNs.
        L does not have to equal the training seq_len; the GCNLSTM
        can handle variable sequence lengths, but performance is best
        when L ≈ seq_len used during training.
    """
    if historical_window.size == 0:
        return float("nan")

    if np.isnan(historical_window).any():
        raise ValueError("The input 'historical_window' cannot contain NaN values. Please pre-fill it.")

    normalized_window = scaler(historical_window)
    input_tensor = torch.FloatTensor(normalized_window).unsqueeze(0).to(device)  # (1, L, N)

    model.eval()
    with torch.no_grad():
        output_sequence = model(input_tensor)   # (1, L, N)

    last_step_prediction_normalized = output_sequence[0, -1, :]
    all_sensor_predictions = inv_scaler(last_step_prediction_normalized.cpu().numpy())
    imputed_value = all_sensor_predictions[target_sensor_index]

    return float(imputed_value)

# ---------- helpers for ground + comparison plot ----------

@st.cache_data(show_spinner=False)

def _stable_key(seed: str) -> str:
    # sanitize + add a stable hash suffix so weird chars (|, :, —) are safe
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(seed))
    return f"{base}_{abs(hash(seed)) & 0xFFFF_FFFF:X}"

def _init_group_store():
    SS = st.session_state
    if "_grp_alerts" not in SS:
        # dict: gid -> {gid, title, level, ts, items:[{iid,text,dismissed}], dismissed}
        SS["_grp_alerts"] = {}
    elif isinstance(SS["_grp_alerts"], list):
        # MIGRATION from the older list-based store to dict to avoid double-rendering
        d = {}
        for g in SS["_grp_alerts"]:
            gid = g.get("gid") or f"{g.get('ts')}|{g.get('title')}|{g.get('level','info')}"
            d[gid] = {
                "gid": gid,
                "title": g.get("title",""),
                "level": g.get("level","info"),
                "ts": pd.Timestamp(g.get("ts")),
                "items": g.get("items", []),
                "dismissed": g.get("dismissed", False),
            }
        SS["_grp_alerts"] = d
    if "_grp_ph" not in SS:
        SS["_grp_ph"] = st.empty()
    if "_grp_render_seq" not in SS:
        SS["_grp_render_seq"] = 0

def render_grouped_alerts():
    _init_group_store()
    SS = st.session_state

    # bump a per-render sequence to guarantee unique widget keys per run
    SS["_grp_render_seq"] += 1
    render_suffix = f"__r{SS['_grp_render_seq']}"

    ph = SS["_grp_ph"]
    ph.empty()

    st.markdown("""
    <style>
      div[data-testid="stVerticalBlock"]:has(> div#grp-alert-anchor){
        position: fixed; top: 86px; right: 22px; width: 420px; z-index: 99999;
      }
      .grp-card{background:#fff;border:1px solid #e5e7eb;border-left-width:6px;
                border-radius:12px;padding:10px 12px;margin:10px 0;box-shadow:0 8px 22px rgba(0,0,0,.06);}
      .grp-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;}
      .grp-title{font-weight:700}
      .grp-dot{margin-top:4px}
    </style>
    <div id="grp-alert-anchor"></div>
    """, unsafe_allow_html=True)

    palette = {"success":"#16a34a","info":"#2563eb","warning":"#f59e0b","error":"#dc2626"}
    icon    = {"success":"✅","info":"ℹ️","warning":"⚠️","error":"🚨"}

    groups = [g for g in SS["_grp_alerts"].values() if not g.get("dismissed")]
    groups.sort(key=lambda g: g["ts"], reverse=True)

    with ph.container():
        for g in groups:
            sev = g.get("level", "info")
            color = palette.get(sev, "#2563eb")
            gid = str(g["gid"])
            gid_key = _stable_key(gid)  # sanitized + hashed

            st.markdown(f"<div class='grp-card' style='border-left-color:{color}'>",
                        unsafe_allow_html=True)
            h1, h2 = st.columns([12,1], gap="small")
            with h1:
                st.markdown(
                    f"<div class='grp-head'><div class='grp-title'>{icon.get(sev,'ℹ️')} {g['title']}</div>"
                    f"<div style='opacity:.6;font-size:12px'>{g['ts']}</div></div>",
                    unsafe_allow_html=True
                )
            with h2:
                if st.button("✕", key=f"grpclose_{gid_key}{render_suffix}"):
                    g["dismissed"] = True

            live_items = [it for it in g["items"] if not it.get("dismissed")]
            for idx, it in enumerate(live_items):
                c1, c2, c3 = st.columns([1,22,1], gap="small")
                with c1:
                    st.markdown("<div class='grp-dot'>•</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(it["text"])
                with c3:
                    if st.button("✕", key=f"itemclose_{gid_key}_{idx}{render_suffix}"):
                        it["dismissed"] = True

            st.markdown("</div>", unsafe_allow_html=True)

class TickAlertBuffer:
    """Collecte les lignes puis pousse 1 groupe par (title, level) pour le timestamp ts."""
    def __init__(self):
        self.buckets = {}  # key=(title, level) -> list[str]

    def add(self, title: str, level: str, text: str):
        self.buckets.setdefault((title, level), []).append(text)

    def flush(self, ts):
        _init_group_store()
        store = st.session_state["_grp_alerts"]
        ts = pd.Timestamp(ts)
        for (title, level), lines in self.buckets.items():
            # gid unique = ts + title + level  (=> 1 boîte par type à ce tick)
            gid = f"{ts.isoformat()}|{title}|{level}"
            g = store.get(gid)
            if not g:
                g = {"gid": gid, "title": title, "level": level, "ts": ts,
                     "items": [], "dismissed": False}
                store[gid] = g
            # ajoute les lignes (chacune fermable)
            for line in lines:
                g["items"].append({"iid": uuid4().hex, "text": line, "dismissed": False})
        self.buckets.clear()



def verify_constraints_and_alerts_for_timestamp(
    ts: pd.Timestamp,
    latlng_df: pd.DataFrame,
    values_by_col: dict,
    imputed_mask_row: Optional[pd.Series],
    baseline_row: pd.Series,
    missing_streak_hours: Optional[dict] = None,
):
    """
    Emit alert/toast messages for the current timestamp:
      - Scenario 1: missing but delay below Δt_i  (waiting)
      - Scenario 2: delay above Δt_i and neighbors available (with 2.x sub-cases where applicable)
      - Scenario 3: delay above Δt_i and neighbors unavailable (historical imputation or no estimate)
      - Temporal and spatial constraint warnings
    All comments and alert texts are in English.

    Inputs:
      - latlng_df columns: ["sensor_id", "data_col", "latitude", "longitude"]
      - values_by_col keys: data_col
      - baseline_row: True where originally missing at ts (indexed by data_col)
      - imputed_mask_row: True where value was imputed at ts (indexed by data_col)
      - missing_streak_hours: per data_col ongoing missing streak in hours
    """
    # --- Track which sensors violate which constraints ---
    # sensor_id -> set of {"temporal", "spatial"}
    violating_imputed: dict[str, set] = {}
    violating_real: dict[str, set] = {}

    # Helper: imputed vs real
    def _value_source_label(sensor_col: str) -> str:
        if imputed_mask_row is None:
            return "real value"
        is_imp = bool(imputed_mask_row.get(sensor_col, False))
        return "imputed value" if is_imp else "real value"

    def _mark_violation(sensor_col: str, source: str, constraint_type: str):
        """
        constraint_type ∈ {"temporal", "spatial"}
        """
        if source == "imputed value":
            d = violating_imputed
        else:
            d = violating_real
        s = d.get(sensor_col)
        if s is None:
            s = set()
            d[sensor_col] = s
        s.add(constraint_type)

    # Helper pour savoir si une valeur est imputée ou réelle
    def _value_source_label(sensor_col: str) -> str:
        """
        Retourne un label texte pour indiquer si la valeur utilisée
        pour ce capteur à l'instant ts est imputée ou réelle.
        """
        if imputed_mask_row is None:
            return "real value"
        is_imp = bool(imputed_mask_row.get(sensor_col, False))
        return "imputed value" if is_imp else "real value"

    

    # Buffer for grouped cards (we keep your infra but primarily use toasts for real-time UX)
    buf = TickAlertBuffer()

    constraints = st.session_state.get("constraints", [])
    sigma_minutes = float(st.session_state.get("sigma_threshold", DEFAULT_VALUES.get("sigma_threshold", 30)))
    sigma_hours = max(0.0, sigma_minutes / 60.0)

    # Global constraint sensitivity (0.0 .. 1.0)
    sensitivity = float(
        st.session_state.get(
            "constraint_sensitivity",
            DEFAULT_VALUES.get("constraint_sensitivity", 1.0),
        )
    )
    sensitivity = max(0.0, min(1.0, sensitivity))

    # Minimum relative violation strength required to raise an alert:
    #   sensitivity = 1.0 → min_strength = 0.0  (any violation alerts)
    #   sensitivity = 0.0 → min_strength = 0.5  (only ≥ 50% beyond threshold)
    min_strength = (1.0 - sensitivity) * 0.5

    # Persistent fault/waiting state to avoid spam across ticks
    if "_fault_state" not in st.session_state:
        st.session_state["_fault_state"] = {}
    fault_state = st.session_state["_fault_state"]

    spatial_rules  = [c for c in constraints if c.get("type") == "Spatial"]
    temporal_rules = [c for c in constraints if c.get("type") == "Temporal"]

    # Track sensors whose REAL values violate any constraint at this timestamp.
    # "Real" = not originally missing at this ts (baseline_row[data_col] == False).
    real_constraint_violators: set[str] = set()


    # Prepare a quick lat/long lookup for spatial rules
    if spatial_rules:
        pos_map = {str(r["data_col"]): (float(r["latitude"]), float(r["longitude"]))
                   for _, r in latlng_df.iterrows() if str(r.get("data_col")) in values_by_col}
        def haversine_km(lat1, lon1, lat2, lon2):
            
            R=6371.0; dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
            a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
            return 2*R*atan2(sqrt(a), sqrt(1-a))

    # --- Neighbor availability map (use the largest spatial radius) ---
    try:
        max_km = max(float(c.get("distance in km", 0)) for c in spatial_rules) if spatial_rules else 0.0
    except Exception:
        max_km = 0.0

    if max_km > 0 and not latlng_df.empty:
        # Build a clean DF keyed by data_col (not the display sensor_id) to match values_by_col/baseline_row keys
        latlng_for_neigh = pd.DataFrame({
            "sensor_id": latlng_df["data_col"].astype(str),
            "latitude": pd.to_numeric(latlng_df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(latlng_df["longitude"], errors="coerce"),
        }).dropna(subset=["latitude", "longitude"])
        neigh_map = _neighbors_within_km(latlng_for_neigh, max_km)  # {data_col: [data_col, ...]}
    else:
        neigh_map = {}

    # A neighbor is considered "available" if it was originally present (not missing) at this timestamp
    present_cols = {k for k, miss in baseline_row.items() if not bool(miss)}

    # Helper for clean toast emission
    def emit_toast(level: str, text: str, dedup: str):
        toast_notify(level, text, dedup_key=dedup)

    # --- Helpers for relative magnitude of constraint violations ---
    def _rel_violation_greater(v: float, thr: float) -> float:
        """Constraint is 'v > thr'; return relative size of violation (0=no violation)."""
        if v > thr:
            return 0.0
        return (thr - v) / max(abs(thr), 1e-6)

    def _rel_violation_less(v: float, thr: float) -> float:
        """Constraint is 'v < thr'; return relative size of violation (0=no violation)."""
        if v < thr:
            return 0.0
        return (v - thr) / max(abs(thr), 1e-6)


    # ---- Constraint checks (Temporal + Spatial), independent of scenarios ----
    # Evaluate per-sensor temporal rules
    # ---- Constraint checks (Temporal + Spatial), independent of scenarios ----
    # Evaluate per-sensor temporal rules
    for data_col, val in values_by_col.items():
        v = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
        if v is None or not temporal_rules:
            continue
        month_name = ts.strftime("%B")
        for rule in temporal_rules:
            if rule.get("type") != "Temporal":
                continue
            if rule.get("month") != month_name:
                continue
            opt = rule.get("option", "").lower()
            try:
                thr = float(rule.get("temp_threshold", np.nan))
            except Exception:
                thr = np.nan
            if not np.isfinite(thr):
                continue

            source = _value_source_label(data_col)
            toast_type = "error" if source == "real value" else "warning"

            
            if opt.startswith("greater"):
                strength = _rel_violation_greater(v, thr)
                if strength > 0.0 and strength >= min_strength:
                    # 🔴 enregistrer la violation
                    _mark_violation(data_col, source, "temporal")

                    emit_toast(
                        toast_type,
                        f"Temporal constraint ({source}): expected > {thr:g}, got {v:.2f} for {data_col} at {ts}.",
                        f"tmp-gt-{data_col}-{ts}",
                    )
                    # mark this sensor as a real-valued violator at this timestamp
                    if not bool(baseline_row.get(data_col, False)):
                        real_constraint_violators.add(str(data_col))

            elif opt.startswith("less"):
                strength = _rel_violation_less(v, thr)
                if strength > 0.0 and strength >= min_strength:
                    _mark_violation(data_col, source, "temporal")

                    emit_toast(
                        toast_type,
                        f"Temporal constraint ({source}): expected < {thr:g}, got {v:.2f} for {data_col} at {ts}.",
                        f"tmp-lt-{data_col}-{ts}",
                    )
                    # mark this sensor as a real-valued violator at this timestamp
                    if not bool(baseline_row.get(data_col, False)):
                        real_constraint_violators.add(str(data_col))                    

    # Evaluate spatial rules pairwise (avoid duplicate A–B and B–A)
    if spatial_rules and pos_map:
        emitted_pairs = set()
        for rule in spatial_rules:
            try:
                dist_km = float(rule.get("distance in km", 0))
                max_diff = float(rule.get("diff", np.inf))
            except Exception:
                continue
            if not np.isfinite(dist_km) or dist_km <= 0 or not np.isfinite(max_diff):
                continue

            title = f"Spatial — ≤{dist_km:g} km | maxΔ {max_diff:g}"
            keys = list(values_by_col.keys())
            for i in range(len(keys)):
                a = keys[i]
                va = values_by_col.get(a)
                if va is None or (isinstance(va, float) and not np.isfinite(va)):
                    continue
                if a not in pos_map:
                    continue
                lat1, lon1 = pos_map[a]

                for j in range(i + 1, len(keys)):
                    b = keys[j]
                    vb = values_by_col.get(b)
                    if vb is None or (isinstance(vb, float) and not np.isfinite(vb)):
                        continue
                    if b not in pos_map:
                        continue

                    pair_key = (a, b, dist_km, max_diff, pd.Timestamp(ts))
                    if pair_key in emitted_pairs:
                        continue

                    lat2, lon2 = pos_map[b]
                    d = haversine_km(lat1, lon1, lat2, lon2)

                    if d <= dist_km:
                        delta = abs(float(va) - float(vb))
                        if delta > max_diff:
                            strength = (delta - max_diff) / max(max_diff, 1e-6)
                            if strength >= min_strength:
                                source_a = _value_source_label(a)
                                source_b = _value_source_label(b)

                                # rouge si au moins une vraie valeur
                                if (source_a == "real value") or (source_b == "real value"):
                                    toast_type = "error"
                                else:
                                    toast_type = "warning"

                                # 🔴 enregistrer la violation
                                _mark_violation(a, source_a, "spatial")
                                _mark_violation(b, source_b, "spatial")

                                emit_toast(
                                    toast_type,
                                    (
                                        f"{title}: |{a}-{b}| = {delta:.2f} > {max_diff:g} "
                                        f"(d≈{d:.1f} km) at {ts} "
                                        f"[{source_a} for {a}, {source_b} for {b}]"
                                    ),
                                    f"spat-{a}-{b}-{ts}",
                                )
                                emitted_pairs.add(pair_key)

                                # mark both ends as real-valued violators (if not originally missing)
                                if not bool(baseline_row.get(a, False)):
                                    real_constraint_violators.add(str(a))
                                if not bool(baseline_row.get(b, False)):
                                    real_constraint_violators.add(str(b))                                


    # ---- Scenario classification (per sensor) ----
    for data_col, val in values_by_col.items():
        v = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
        was_missing = bool(baseline_row.get(data_col, False))
        is_imputed  = bool(imputed_mask_row.get(data_col, False)) if imputed_mask_row is not None else False
        streak_h    = float((missing_streak_hours or {}).get(data_col, 0.0))

        prev = fault_state.get(data_col, "ok")
        curr = prev

        # Neighbor availability for this sensor
        has_present_neighbor = any(n in present_cols for n in neigh_map.get(data_col, []))

        # neighbors of this sensor that currently violate constraints with REAL values
        violating_neighbors = [
            n for n in neigh_map.get(data_col, [])
            if n in real_constraint_violators
        ]

        if was_missing:
            if (v is None) and not is_imputed:
                # No value to show yet
                if streak_h < sigma_hours:
                    # Scenario 1 — waiting under Δt
                    emit_toast("info",
                               ALERT["SCENARIO_1_WAITING"].format(sid=data_col, ts=ts),
                               f"s1-{data_col}-{ts}")
                    curr = "waiting"
                else:
                    # Δt exceeded
                    if has_present_neighbor and violating_neighbors:
                        # Scenario 2 — neighbors available and violating constraints
                        emit_toast(
                            "warning",
                            f"🚨 Potential anomaly near sensor {data_col} at {ts}. "
                            f"Neighbors {', '.join(violating_neighbors)} have REAL values "
                            f"violating constraints while this station is delayed or missing.",
                            f"s2-neigh-viol-miss-{data_col}-{ts}",
                        )
                        curr = "fault"
                    else:
                        # Scenario 3 — neighbors unavailable, no reliable estimate
                        emit_toast("warning",
                                   ALERT["SCENARIO_3_NO_ESTIMATE"].format(sid=data_col, ts=ts),
                                   f"s3-noest-{data_col}-{ts}")
                        curr = "fault"
            else:
                # We now have a value (real or imputed)
                if streak_h >= sigma_hours and not has_present_neighbor and is_imputed:
                    # Scenario 3 — historical imputation due to unavailable neighbors
                    emit_toast("warning",
                               ALERT["SCENARIO_3_HIST_IMPUTE"].format(sid=data_col, ts=ts),
                               f"s3-hist-{data_col}-{ts}")
                #elif is_imputed and has_present_neighbor:
                    # Scenario 2 — with neighbors; choose 2.1 / 2.2 / 2.3 depending on your checks
                    # By default, mark as OK; replace with OOR / NEIGHBOR_MISMATCH after your validations.
                #    emit_toast("info",
                #               ALERT["SCENARIO_2_IMPUTED_OK"].format(sid=data_col, ts=ts),
                #               f"s2-ok-{data_col}-{ts}")
                    
                elif is_imputed and has_present_neighbor:
                    if violating_neighbors:
                        # NEW: imputed OK here, but some neighbors have real constraint violations
                        emit_toast(
                            "warning",
                            ALERT["SCENARIO_2_NEIGHBOR_MISMATCH"].format(sid=data_col, ts=ts),
                            f"s2-neigh-viol-imp-{data_col}-{ts}",
                        )
                    else:
                        # No neighbor violations → normal “imputed OK”
                        emit_toast(
                            "info",
                            ALERT["SCENARIO_2_IMPUTED_OK"].format(sid=data_col, ts=ts),
                            f"s2-ok-{data_col}-{ts}",
                        )                

                # Show "Restored" once when transitioning out of waiting/fault
                if prev in ("waiting", "fault"):
                    emit_toast("success",
                               f"✅ Restored — {data_col} at {ts}",
                               f"restored-{data_col}-{ts}")
                curr = "ok"
        else:
            # Not originally missing at this timestamp; no scenario emission.
            curr = "ok"

        # Persist state for next ticks (spam control)
        fault_state[data_col] = curr

    # Flush grouped alert buffer (if you still use the grouped cards)
    buf.flush(ts)

    # ✅ retourner les capteurs et types de contraintes violées
    return {
        "violating_imputed": violating_imputed,
        "violating_real": violating_real,
    }



# ========= Toast notifications instead of grouped cards =========
_ICON = {"success":"✅","info":"ℹ️","warning":"⚠️","error":"🚨"}

def _init_toast_state():
    SS = st.session_state
    if "_toast_seen" not in SS:
        SS["_toast_seen"] = set()     # anti-dup court terme
    if "_toast_ttl" not in SS:
        SS["_toast_ttl"] = 8          # secondes d’affichage / anti-spam

def toast_notify(level: str, text: str, dedup_key: str|None = None):
    """
    Envoie une notification toast non bloquante.
    dedup_key permet d’éviter le spam si on rerend plusieurs fois dans la même seconde.
    """
    _init_toast_state()
    key = f"{level}|{dedup_key or text}"
    # anti-spam: si déjà toasté très récemment, on ignore
    if key in st.session_state["_toast_seen"]:
        return
    st.session_state["_toast_seen"].add(key)

    # st.toast accepte un "icon" (emoji) + du texte
    icon = _ICON.get(level, "ℹ️")
    st.toast(f"{icon} {text}", icon=None)  # icon dans le texte pour garder les couleurs Streamlit


def run_simulation_with_live_imputation(
    sim_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    positions,
    model: torch.nn.Module,
    scaler: callable,
    inv_scaler: callable,
    device: torch.device,
    graph_placeholder,              # unused, kept for signature
    sliding_chart_placeholder,      # unused, kept for signature
    gauge_placeholder,              # unused, kept for signature
    window_hours: int = 24,
    # Hybrid online imputer hyperparameters
    k_neighbors: int = 15, #Test K neighbors
    trim_frac: float = 0.15,
    distance_cap_km=None,
    clim_beta: float = 0.3,
    domain_clip_min=None,
    domain_clip_max=None,
):

    SS = st.session_state
    global IMPUTATION_LOG


    # ---- Models comparison (TSGuard vs others) defaults ----
    # Which models the user wants to see in the comparison tab
    SS.setdefault("comparison_models", ["TSGuard"])

    # Keep track of which models are actually available
    avail = SS.get("available_models")
    if isinstance(avail, set):
        avail.add("TSGuard")
        SS["available_models"] = avail
    elif isinstance(avail, (list, tuple)):
        SS["available_models"] = set(avail) | {"TSGuard"}
    else:
        SS["available_models"] = {"TSGuard"}

    # Load imputer config once (cache in session state)
    if "imputer_config" not in st.session_state:
        try:
            # assuming model_path was something like ".../gcn_lstm_imputer.pth"
            cfg_path = str(Path(model_path).with_suffix("")) + "_imputer_config.json"
            with open(cfg_path, "r") as f:
                st.session_state["imputer_config"] = json.load(f)
        except Exception:
            st.session_state["imputer_config"] = {}

    imputer_cfg = st.session_state["imputer_config"]
    seq_len = int(imputer_cfg.get("seq_len", 36))


    # Dynamic captor registry: sensors added from the Settings panel.
    # Stored as {sensor_id: {"sensor_id": str, "latitude": float, "longitude": float}}
    dynamic_captors = SS.get("dynamic_captors", {})
    # ✅ Use the same sensor_id convention as positions_to_df
    dynamic_ids = set()
    for key, meta in dynamic_captors.items():
        sid = meta.get("sensor_id", key)
        dynamic_ids.add(str(sid))


    # --- Sensors manually forced "offline" (hold-out mode) -------------------
    def canonical_sensor_id(x: str) -> str:
        x = str(x).strip()
        return x.zfill(6) if x.isdigit() else x

    forced_off_captors = {
        canonical_sensor_id(sid)
        for sid in SS.get("forced_off_captors", [])
    }


    # Simulation time scale (τ): real seconds per simulated hour.
    # Default falls back to config if user has not touched the slider.
    default_scale = float(DEFAULT_VALUES.get("sim_seconds_per_hour", 0.0))

    # placeholder persistant pour le centre d'alertes

    # ---------------- Alert message templates ----------------
    ALERT = {
        # Scenario 1 — missing value but still within the allowable delay Δt_i
        "SCENARIO_1_WAITING": (
            "Delayed data for sensor {sid} (t={ts}). "
            "⏳Waiting for data.. "
        ),

        # Scenario 2 — delay above Δt_i and neighbors available
        # (1) Imputed value within range & neighbors within range
        "SCENARIO_2_IMPUTED_OK": (
            "✅ Imputation completed for sensor {sid} at {ts}. "
             ),
        # (2) Imputed value outside acceptable range
        "SCENARIO_2_IMPUTED_OOR": (
            "⚠️ Imputation out of range for sensor {sid} at {ts}. "
            "The reconstructed value violates domain constraints. Please review."
        ),
        # (3) Imputed value within range, but at least one neighbor is out of range
        "SCENARIO_2_NEIGHBOR_MISMATCH": (
            "🚨 Potential anomaly near sensor {sid} at {ts}. "
            "The imputed value is within range, but one or more neighboring stations are out of range. "
        ),

        # Scenario 3 — delay above Δt_i and neighbors unavailable
        # Historical imputation is possible
        "SCENARIO_3_HIST_IMPUTE": (
            "ℹ️ Neighbor data unavailable. Imputed {sid} at {ts} using historical patterns. "
            "Monitor closely until live data resumes."
        ),
        # No reliable estimate possible
        "SCENARIO_3_NO_ESTIMATE": (
            "🚨 No reliable estimate for sensor {sid} at {ts}. "
            "Target and neighboring stations are missing. Possible sensor or system fault."
        ),
    }

    if "_alert_ph" not in SS:
        SS["_alert_ph"] = st.empty()

    # ---------- small helpers ----------
    def init_once(key, val):
        if key not in SS:
            SS[key] = val
        return SS[key]

    def zpad6(s: str) -> str:
        return s if not s.isdigit() else s.zfill(6)

    def strip0(s: str) -> str:
        t = s.lstrip("0")
        return t if t else "0"

    GREEN = [46, 204, 113, 200]
    RED   = [231, 76, 60, 200]

    def make_bg_layer(df):
        return pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["longitude", "latitude"],
            get_fill_color="bg_color",
            get_radius="bg_radius",
            radius_scale=1,
            radius_min_pixels=6,
            radius_max_pixels=22,
            stroked=True,
            get_line_color=[255, 255, 255, 180],
            line_width_min_pixels=1,
            pickable=False,
        )

    def make_icon_layer(df):
        return pdk.Layer(
            "IconLayer",
            data=df,
            get_icon="icon",
            get_position=["longitude", "latitude"],
            get_size="icon_size",
            size_scale=8,
            size_min_pixels=14,
            size_max_pixels=28,
            pickable=True,
        )

    def fit_view_simple(df: pd.DataFrame, padding_deg=0.02) -> pdk.ViewState:
        if df is None or df.empty:
            return pdk.ViewState(latitude=0, longitude=0, zoom=2, bearing=0, pitch=0)
        lat_min = float(df["latitude"].min());  lat_max = float(df["latitude"].max())
        lon_min = float(df["longitude"].min()); lon_max = float(df["longitude"].max())
        lat_c = (lat_min + lat_max) / 2.0
        lon_c = (lon_min + lon_max) / 2.0
        span = max(lat_max - lat_min, lon_max - lon_min) + padding_deg
        span = max(span, 1e-3)
        zoom = max(1.0, min(16.0, np.log2(360.0 / span)))
        return pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom, bearing=0, pitch=0)

    # ---------- select sensors for map/TS/snapshot10 ----------
    all_sensor_cols = [c for c in sim_df.columns if c != "datetime"]
    graph_size = int(SS.get("graph_size", DEFAULT_VALUES["graph_size"]))
    sensor_cols = [str(c) for c in all_sensor_cols[:graph_size]]
    col_to_idx = {c: i for i, c in enumerate(sensor_cols)}

    # ---------- positions ----------
    latlng_raw = positions_to_df(positions).copy()
    latlng_raw["sensor_id"] = latlng_raw["sensor_id"].astype(str).str.strip()
    latlng_raw["latitude"] = pd.to_numeric(latlng_raw["latitude"], errors="coerce")
    latlng_raw["longitude"] = pd.to_numeric(latlng_raw["longitude"], errors="coerce")
    latlng_raw = latlng_raw.dropna(subset=["latitude", "longitude"])

    #pos_ids = latlng_raw["sensor_id"].tolist()
    pos_ids = [str(pid) for pid in latlng_raw["sensor_id"]]    

    # Map static (modelled) sensors from positions → data columns in sim_df
    map_exact = {pid: pid for pid in pos_ids if pid in sensor_cols}
    map_pad6 = {pid: zpad6(pid) for pid in pos_ids if zpad6(pid) in sensor_cols}
    map_strip0 = {}
    for pid in pos_ids:
        s = strip0(pid)
        s6 = zpad6(s)
        if s in sensor_cols:
            map_strip0[pid] = s
        elif s6 in sensor_cols:
            map_strip0[pid] = s6
    map_index = {}
    #if all(p.isdigit() for p in pos_ids):
    #    nums = sorted(int(p) for p in pos_ids)
    #    if nums and nums[0] == 0 and nums[-1] == len(nums) - 1:
    #        for i, pid in enumerate(sorted(pos_ids, key=lambda x: int(x))):
    #            if i < len(sensor_cols):
    #                map_index[pid] = sensor_cols[i]
    for pid in pos_ids:
        if pid.isdigit():
            idx = int(pid)
            if 0 <= idx < len(sensor_cols):
                map_index[pid] = sensor_cols[idx]


    candidates = [map_exact, map_pad6, map_strip0, map_index]
    best_map = max(candidates, key=lambda m: len(m))

    if len(best_map) == 0:
        st.info("No matching positions for selected sensors.")
        return

    # Build lat/lon table for all sensors (static + dynamic).
    latlng = latlng_raw.copy()
    latlng["data_col"] = latlng["sensor_id"].map(best_map)


    # For captors that do not map to an existing data column (typically new/dynamic),
    # use their own ID (zero-padded if numeric) as the data column name.
    mask_missing = latlng["data_col"].isna()
    if mask_missing.any():
        latlng.loc[mask_missing, "data_col"] = (
            latlng.loc[mask_missing, "sensor_id"]
            .astype(str)
            .apply(lambda s: zpad6(s) if s.isdigit() else s)
        )

    # keep only sensors that belong to the selected graph
    # or were explicitly added as dynamic captors ---
    dyn_ids = set(dynamic_ids)  # dynamic_ids was built earlier from st.session_state["dynamic_captors"]

    latlng = latlng[
        latlng["data_col"].isin(sensor_cols) |    # modelled static sensors
        latlng["sensor_id"].isin(dyn_ids)        # user-added dynamic captors
    ].copy()


    latlng = latlng[latlng["data_col"].notna()].copy()

    # Expose coordinates for the Models Comparison tab
    try:
        SS["captor_coords_by_data_col"] = (
            latlng[["data_col", "latitude", "longitude"]]
            .dropna(subset=["latitude", "longitude"])
            .drop_duplicates(subset=["data_col"], keep="last")
            .set_index("data_col")
            .to_dict("index")
        )
        SS["captor_coords_by_sensor_id"] = (
            latlng[["sensor_id", "latitude", "longitude"]]
            .dropna(subset=["latitude", "longitude"])
            .drop_duplicates(subset=["sensor_id"], keep="last")
            .set_index("sensor_id")
            .to_dict("index")
        )
    except Exception:
        # Keep previous maps if something goes wrong
        pass



    # Static / modelled sensors = intersection of sensor_cols and latlng.data_col
    order_index = {c: i for i, c in enumerate(sensor_cols)}
    latlng["__ord"] = latlng["data_col"].map(order_index)
    latlng = latlng.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    sensor_cols = [c for c in sensor_cols if c in set(latlng["data_col"])]
    if not sensor_cols:
        st.info("After mapping, no sensors remain to plot.")
        return

    # Dynamic sensors = positions that do not correspond to an existing modelled column
    all_data_cols = list(dict.fromkeys(latlng["data_col"].tolist()))
    dynamic_sensor_cols = [c for c in all_data_cols if c not in sensor_cols]

    col_to_idx = {c: i for i, c in enumerate(sensor_cols)}

    # ---------- time alignment ----------
    def ensure_datetime_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "datetime" in df.columns:
            return df
        if isinstance(df.index, pd.DatetimeIndex):
            return df.reset_index().rename(columns={"index": "datetime"})
        for alt in ("timestamp", "date", "time"):
            if alt in df.columns:
                return df.rename(columns={alt: "datetime"})
        idx_as_dt = pd.to_datetime(df.index, errors="coerce")
        if idx_as_dt.notna().all():
            out = df.reset_index().rename(columns={"index": "datetime"})
            out["datetime"] = idx_as_dt
            return out
        raise KeyError(f"{name} has no 'datetime' column or datetime-like index.")

    sim_df     = ensure_datetime_column(sim_df, "sim_df")
    missing_df = ensure_datetime_column(missing_df, "missing_df")
    sim_df["datetime"]     = pd.to_datetime(sim_df["datetime"], errors="coerce")
    missing_df["datetime"] = pd.to_datetime(missing_df["datetime"], errors="coerce")
    sim_df     = sim_df.dropna(subset=["datetime"]).copy()
    missing_df = missing_df.dropna(subset=["datetime"]).copy()
    sim_df["datetime"]     = sim_df["datetime"].dt.floor("h")
    missing_df["datetime"] = missing_df["datetime"].dt.floor("h")
    sim_df.set_index("datetime", inplace=True)
    missing_df.set_index("datetime", inplace=True)
    sim_df     = sim_df[~sim_df.index.duplicated(keep="first")].copy()
    missing_df = missing_df[~missing_df.index.duplicated(keep="first")].copy()

    if "orig_missing_baseline" not in SS:
        SS.orig_missing_baseline = missing_df.isna().copy()
    else:
        if (not SS.orig_missing_baseline.index.equals(missing_df.index) or
            list(SS.orig_missing_baseline.columns) != list(missing_df.columns)):
            SS.orig_missing_baseline = missing_df.isna().copy()

    # >>> ADD (freeze the original values so PriSTI plots can show originals)
    if "orig_missing_values" not in SS:
        SS.orig_missing_values = missing_df.copy()
    # <<< END ADD

    base_index   = missing_df.index
    sim_df       = sim_df.reindex(base_index)
    common_index = base_index
    if common_index.empty or latlng.empty:
        st.info("No matching timeline or positions/sensors.")
        return

    # ----- ORBITS offline results (load once) -----
    if "orbits_df" not in SS:
        try:
            SS["orbits_df"] = load_orbits_offline_results()
        except Exception as e:
            SS["orbits_df"] = None
            SS["orbits_last_error"] = str(e)

    orbits_df = SS["orbits_df"]

    if orbits_df is not None:
        # Register ORBITS as an available model
        avail = SS.get("available_models")
        if isinstance(avail, set):
            avail.add("ORBITS")
            SS["available_models"] = avail
        elif isinstance(avail, (list, tuple)):
            SS["available_models"] = set(avail) | {"ORBITS"}
        else:
            SS["available_models"] = {"TSGuard", "ORBITS"}

        # If the user never touched the comparison tab, include ORBITS by default
        if "comparison_models" not in SS:
            SS["comparison_models"] = ["TSGuard", "PriSTI", "ORBITS"]
    else:
        # keep orbits_df defined for later code
        orbits_df = None


    # Baseline "present" mask (True where originally observed)
    SS.baseline_present = ~SS.orig_missing_baseline

    # ----- Precompute spatial + temporal stats for hybrid online imputer -----
    # sensor_cols has already been aligned to positions order above
    num_sensors = len(sensor_cols)

    # Distance matrix from lat/lon using the same order as sensor_cols
    dist_matrix = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    for i in range(num_sensors):
        lat1, lon1 = float(latlng.iloc[i]["latitude"]), float(latlng.iloc[i]["longitude"])
        for j in range(i, num_sensors):
            lat2, lon2 = float(latlng.iloc[j]["latitude"]), float(latlng.iloc[j]["longitude"])
            d = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist_matrix[j, i] = d

    # Neighbor indices per sensor (sorted by distance, optionally capped)
    neighbors_by_idx: list[list[int]] = []
    for i in range(num_sensors):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf  # exclude self
        order = np.argsort(dists)
        ordered_valid = [j for j in order if np.isfinite(dists[j])]
        if distance_cap_km is not None:
            ordered_valid = [j for j in ordered_valid if dists[j] <= distance_cap_km]
        neighbors_by_idx.append(ordered_valid[:max(0, min(k_neighbors, num_sensors - 1))])

    # Historical correlations from sim_df (ground truth / training-like)
    sim_df_stats = sim_df[sensor_cols].copy()
    corr_df = sim_df_stats.corr()
    corr_matrix = corr_df.to_numpy(dtype=np.float32)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    corr_boost_matrix = np.maximum(corr_matrix ** 2, 0.3 ** 2)  # corr^2 floored at 0.3^2

    # Hour-of-day climatology (median per hour)
    clim = sim_df_stats.groupby(sim_df_stats.index.hour).median()
    clim = clim.reindex(range(24))
    clim_values = clim.to_numpy(dtype=np.float32)

    def get_climatology(ts: pd.Timestamp, sensor_idx: int) -> float:
        try:
            hod = int(ts.hour)
            val = clim_values[hod, sensor_idx]
            return float(val) if np.isfinite(val) else float("nan")
        except Exception:
            return float("nan")

    def predict_all_sensors(historical_window: np.ndarray) -> np.ndarray | None:
        """
        One forward pass of the GCNLSTM for all sensors, given a history window.
        """
        if historical_window.size == 0 or model is None:
            return None
        #if np.isnan(historical_window).any():
        #    return None

        normalized_window = scaler(historical_window)
        input_tensor = torch.FloatTensor(normalized_window).unsqueeze(0).to(device)  # (1, L, N)
        with torch.no_grad():
            out_seq = model(input_tensor)    # (1, L, N)
        last_step = out_seq[0, -1, :].cpu().numpy()
        return inv_scaler(last_step)

    MIN_CONF = 0.1  # below this, a confidence is treated as "unreliable"



    # ---------- persistent state ----------
    uid = init_once("sim_uid", f"sim_{uuid.uuid4().hex[:8]}")
    init_once("sim_iter", 0)
    init_once("sim_ptr", 0)  # position dans la timeline

    # All sensors that appear on the map / time series (static + dynamic)
    imputed_cols = sensor_cols + dynamic_sensor_cols

    if "imputed_mask" not in SS:
        SS.imputed_mask = pd.DataFrame(
            False, index=common_index, columns=imputed_cols, dtype=bool
        )
    else:
        SS.imputed_mask = SS.imputed_mask.reindex(
            index=common_index, columns=imputed_cols, fill_value=False
        )


    init_once("sliding_window_df", pd.DataFrame(columns=["datetime"] + imputed_cols))
    init_once("global_df", pd.DataFrame(columns=["datetime"] + imputed_cols))
    init_once("impute_time_tsg", {})  # per-timestamp seconds
    init_once("impute_time_pri", {})




    # ---------- UI (created once) ----------
    if "_ui_inited" not in SS:
        SS["_ui_inited"] = True

        # Row 1: two columns: left (title, time, button, map) | right (gauge)
        col_left, col_right = st.columns([2, 1], gap="small")

        with col_left:
            st.markdown("### Sensor Visualization")
            hdr_l, hdr_r = st.columns([5, 2], gap="small")
            with hdr_l:
                SS["ph_time"] = st.markdown("**Current Time:** —")
            with hdr_r:
                SS["ph_fitbtn"] = st.empty()     # we render the button each run below

            # map placeholder lives in the left column
            SS["ph_map"] = st.empty()

        with col_right:
            st.markdown(
                "<div style='text-align:center;font-weight:700;margin:0.25rem 0'>Data Quality & Activity</div>",
                unsafe_allow_html=True
            )
            SS["ph_counts_active"]  = st.empty()
            SS["ph_counts_missing"] = st.empty()
            SS["ph_gauge"] = st.empty()

        st.markdown("---")

        # Row 2: Global TS (left) + legend (middle) + snapshot-10 (right)
        row2_l, row2_mid, row2_r = st.columns([3, 1, 3], gap="small")
        with row2_l:
            SS["ph_global"] = st.empty()
        with row2_mid:
            SS["ph_legend"] = st.empty()
        with row2_r:
            SS["ph_snap10"] = st.empty()

        st.markdown("---")


    # Render a single Fit button (in left header, aligned right)
    with SS["ph_fitbtn"]:
        # unique, stable key
        if st.button("Fit map to sensors", use_container_width=True, key=f"{uid}_fitbtn"):
            SS["_fit_event"] = True

    # ---------- deck.gl persistent ----------
    global ICON_SPEC
    if "ICON_SPEC" not in globals() or ICON_SPEC is None:
        ICON_SPEC = {"url": "", "width": 1, "height": 1, "anchorX": 0, "anchorY": 0}

    if "deck_obj" not in SS:
        base_df = latlng.copy()
        base_df["sensor_id"] = base_df["sensor_id"].astype(str)
        base_df["value"] = "NA"
        base_df["status"] = "Imputed"
        base_df["timestamp"] = "-"   # NEW: placeholder; real timestamps come from the sim loop
        base_df["bg_color"] = [[231, 76, 60, 200] for _ in range(len(base_df))]
        base_df["bg_radius"] = 10
        base_df["icon"] = [ICON_SPEC] * len(base_df)
        base_df["icon_size"] = 1.0

        init_view = fit_view_simple(base_df)
        #SS["deck_tooltip"] = {"text": "Sensor {sensor_id}\nValue: {value}\nStatus: {status}"}
        SS["deck_tooltip"] = {
            "html": (
                "<b>Sensor {sensor_id}</b><br/>"
                "Time: {timestamp}<br/>"
                "Value: {value}<br/>"
                "Status: {status}"
            ),
            "style": {
                "backgroundColor": "white",
                "color": "#111827",
                "fontSize": "12px",
                "padding": "8px 10px",
            },
        }
        SS.deck_obj = pdk.Deck(
            layers=[make_bg_layer(base_df), make_icon_layer(base_df)],
            initial_view_state=init_view,
            map_style="mapbox://styles/mapbox/light-v11",
            tooltip=SS["deck_tooltip"],
        )
        SS["_fit_base_df"] = base_df.copy()

    # Handle Fit event: REBUILD the deck with a NEW view state, then re-render
    if SS.get("_fit_event", False):
        df_to_fit = SS.get("_fit_base_df", latlng)
        new_view = fit_view_simple(df_to_fit)
        layers = SS.deck_obj.layers
        # rebuild deck to force refit
        SS.deck_obj = pdk.Deck(
            layers=layers,
            initial_view_state=new_view,
            map_style=getattr(SS.deck_obj, "map_style", "mapbox://styles/mapbox/light-v11"),
            tooltip=SS.get("deck_tooltip", {"text": "Sensor {sensor_id}\nValue: {value}\nStatus: {status}"}),
        )
        SS["ph_map"].pydeck_chart(SS.deck_obj, use_container_width=True)
        SS["_fit_event"] = False

    # Initial map draw (if not drawn by fit)
    SS["ph_map"].pydeck_chart(SS.deck_obj, use_container_width=True)

    # ---------- palettes ----------
    base_palette = ["#000000", "#003366", "#009999", "#006600", "#66CC66",
                    "#FF9933", "#FFD700", "#708090", "#4682B4", "#99FF33"]
    plot_sensor_cols = sensor_cols + dynamic_sensor_cols
    sensor_color_map = {
        c: base_palette[i % len(base_palette)]
        for i, c in enumerate(plot_sensor_cols)
    }

    # --- Scrollable sensor legend between the plots ---
    if "_legend_css_injected" not in SS:
        st.markdown(
            """
            <style>
              .tsguard-legend-panel {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 8px 10px;
                font-size: 13px;
              }
              .tsguard-legend-title {
                font-weight: 600;
                margin-bottom: 6px;
              }
              .tsguard-legend-scroll {
                max-height: 260px;
                overflow-y: auto;
                padding-right: 4px;
              }
              .tsguard-legend-item {
                display: flex;
                align-items: center;
                margin-bottom: 4px;
                white-space: nowrap;
              }
              .tsguard-legend-color {
                width: 10px;
                height: 10px;
                border-radius: 999px;
                margin-right: 6px;
                border: 1px solid #cbd5e1;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
        SS["_legend_css_injected"] = True

    with SS["ph_legend"]:
        legend_html_parts = [
            "<div class='tsguard-legend-panel'>",
            "<div class='tsguard-legend-title'>Sensors</div>",
            "<div class='tsguard-legend-scroll'>",
        ]
        for c in plot_sensor_cols:
            color = sensor_color_map.get(c, "#444")
            label = f"Sensor {c}"
            legend_html_parts.append(
                f"<div class='tsguard-legend-item'>"
                f"<span class='tsguard-legend-color' style='background:{color}'></span>"
                f"{label}</div>"
            )
        legend_html_parts.append("</div></div>")
        st.markdown("".join(legend_html_parts), unsafe_allow_html=True)



    # add a place to show alerts (non-intrusive)
    if "ph_alerts" not in SS:
        SS["ph_alerts"] = st.empty()

    # missing-streak tracker (by data_col) for scenario 1 thresholding
    if "_missing_streak_hours" not in SS:
        SS["_missing_streak_hours"] = {c: 0 for c in sensor_cols}
    else:
        # keep in sync with current selection
        for c in sensor_cols:
            SS["_missing_streak_hours"].setdefault(c, 0)


    # ---------- main loop: ONE TICK PER RERUN ----------
    use_model = model is not None
    if use_model:
        model.eval()

    SNAP10 = 10
    ptr = int(SS["sim_ptr"])
    total_steps = len(common_index)
    SS["sim_total_steps"] = total_steps

    # Nothing left to simulate → stop cleanly
    if ptr >= total_steps:
        st.session_state["running"] = False
        SS["sim_ptr"] = 0  # rewind so a new run starts from the beginning
        # Dernier flush pour vider le buffer si nécessaire
        flush_imputation_log_if_needed(
            path="tsguard_imputations.csv",
            chunk_size=1,  # flush même s'il reste peu de lignes
        )
        st.success("✅ Simulation finished. You can adjust settings and start it again.")
        return     

    # ---- ONE logical simulation tick for timestamp ts ----
    ts = pd.Timestamp(common_index[ptr])
    SS["sim_iter"] += 1
    iter_key = SS["sim_iter"]
    #baseline_row_ts = SS.orig_missing_baseline.reindex(
    #    index=[ts], columns=sensor_cols
    #).iloc[0]

    # time label
    SS["ph_time"].markdown(
        f"<div style='font-weight:600'>Current Time: {ts}</div>",
        unsafe_allow_html=True,
    )

    # history window strictly before ts
    hist_end = ts - pd.Timedelta(hours=1)
    if hist_end in missing_df.index:
        #hist_idx = missing_df.loc[:hist_end].index[-window_hours:]
        hist_idx = missing_df.loc[:hist_end].index[-seq_len:]
    else:
        #hist_idx = missing_df.index[missing_df.index < ts][-window_hours:]
        hist_idx = missing_df.index[missing_df.index < ts][-seq_len:]
    hist_win = (
        missing_df.loc[hist_idx, sensor_cols]
        if len(hist_idx) > 0
        else pd.DataFrame()
    )

    hist_win_dense = hist_win.ffill().bfill()

    # --- TSGuard imputation (alignée baseline) ---
    # --- TSGuard hybrid imputation (temporal + spatial + climatology) ---
    #tsg_start = time.perf_counter()
    #baseline_row_ts = SS.orig_missing_baseline.reindex(index=[ts], columns=sensor_cols).iloc[0]
    #present_row_ts  = SS.baseline_present.reindex(index=[ts], columns=sensor_cols).iloc[0]

    # --- TSGuard imputation (temporal + spatial + climatology) ---
    tsg_start = time.perf_counter()

    # 1) Original missing pattern from the dataset
    baseline_row_ts = SS.orig_missing_baseline.reindex(
        index=[ts], columns=sensor_cols
    ).iloc[0].copy()

    # 2) Present mask = complement of (possibly modified) baseline
    present_row_ts = (~baseline_row_ts).copy()

    # ORBITS offline row for this timestamp
    orbits_row = None
    orbits_flags = None
    if orbits_df is not None and ts in orbits_df.index:
        row = orbits_df.loc[ts]
        # keep only the sensors we are plotting
        orbits_row = row.reindex(sensor_cols)
        # For the "imputed?" flag, we reuse the same original-missing mask
        orbits_flags = baseline_row_ts.reindex(sensor_cols)


    # 3) Apply manual deactivation: forced-off captors are treated as missing
    #    and are not considered "present" neighbors.
    for sid in forced_off_captors:
        if sid in baseline_row_ts.index:
            baseline_row_ts.loc[sid] = True
            present_row_ts.loc[sid] = False


    # Temporal predictions for all sensors at ts (single forward pass)
    temporal_vector = None
    if not hist_win.empty and use_model:
        try:
            temporal_vector = predict_all_sensors(
                historical_window=np.asarray(hist_win_dense.values, dtype=np.float32)
            )
        except Exception:
            temporal_vector = None

    original_imputed_values = {}
    svals, sstatus = [], []

    for col in sensor_cols:
        col_idx = col_to_idx[col]
        is_missing_now = bool(baseline_row_ts.get(col, False))

        if is_missing_now:
            # ------- TEMPORAL ESTIMATE T_s(t) -------
            T_val = float("nan")
            conf_T = 0.0

            if temporal_vector is not None:
                T_val = float(temporal_vector[col_idx])

                # Confidence for temporal: fraction of valid history + stability
                vals_hist_col = hist_win[col].to_numpy(dtype=float) if not hist_win.empty else np.array([])
                vals_hist_col = vals_hist_col[np.isfinite(vals_hist_col)]
                if vals_hist_col.size >= 2:
                    mu = float(np.mean(vals_hist_col))
                    sigma = float(np.std(vals_hist_col))
                    cv = sigma / (abs(mu) + 1e-3)
                    stability = 1.0 / (1.0 + cv)
                    frac_valid = min(1.0, vals_hist_col.size / max(1, window_hours))
                    conf_T = frac_valid * stability
                elif vals_hist_col.size == 1:
                    conf_T = 0.3  # a single point is not great, but not zero either
                else:
                    conf_T = 0.0

            # ------- SPATIAL ESTIMATE S_s(t) -------
            neighbors = neighbors_by_idx[col_idx]
            S_val = float("nan")
            conf_S = 0.0

            if len(neighbors) > 0:
                vals = []
                weights = []
                for j in neighbors:
                    nbr_col = sensor_cols[j]
                    # only use neighbors that were originally present at time t
                    if not bool(present_row_ts.get(nbr_col, False)):
                        continue
                    try:
                        v = float(missing_df.at[ts, nbr_col])
                    except Exception:
                        v = float("nan")
                    if not np.isfinite(v):
                        continue

                    d = dist_matrix[col_idx, j]
                    inv_d = 1.0 if d <= 0 else 1.0 / d
                    corr_boost = float(corr_boost_matrix[col_idx, j])
                    w = inv_d * corr_boost
                    vals.append(v)
                    weights.append(w)

                vals = np.asarray(vals, dtype=float)
                weights = np.asarray(weights, dtype=float)

                if vals.size > 0:
                    # MAD-based trimming around median
                    if trim_frac > 0.0 and vals.size > 2:
                        median = float(np.median(vals))
                        abs_dev = np.abs(vals - median)
                        order = np.argsort(abs_dev)
                        keep_n = max(1, int(round((1.0 - 2.0 * trim_frac) * vals.size)))
                        keep_idx = order[:keep_n]
                        vals_trimmed = vals[keep_idx]
                        weights_trimmed = weights[keep_idx]
                    else:
                        vals_trimmed = vals
                        weights_trimmed = weights

                    if np.all(weights_trimmed == 0):
                        weights_trimmed = np.ones_like(weights_trimmed)
                    w_norm = weights_trimmed / np.sum(weights_trimmed)
                    S_val = float(np.sum(w_norm * vals_trimmed))

                    # Confidence for spatial: neighbor coverage + low spread
                    if vals_trimmed.size > 1:
                        median = float(np.median(vals_trimmed))
                        mad = float(np.median(np.abs(vals_trimmed - median)))
                        spread = mad / (abs(median) + 1e-3)
                        coverage = min(1.0, vals_trimmed.size / max(1.0, k_neighbors))
                        conf_S = coverage * (1.0 / (1.0 + spread))
                    else:
                        conf_S = 0.3  # single neighbor, low but non-zero confidence

            # ------- Climatology -------
            clim_val = get_climatology(ts, col_idx)
            conf_clim = 0.0
            if np.isfinite(clim_val):
                conf_clim = 0.2  # simple proxy

            # ------- Blend T and S (and climatology as backup) -------
            use_T = np.isfinite(T_val) and conf_T >= MIN_CONF
            use_S = np.isfinite(S_val) and conf_S >= MIN_CONF

            # Convenience: last *actually observed* value in history (if any)
            last_observed_val = float("nan")
            if not hist_win.empty:
                last = pd.to_numeric(hist_win[col].dropna(), errors="coerce")
                if len(last):
                    last_observed_val = float(last.iloc[-1])

            # does THIS sensor have ANY neighbour with an observed value at ts?
            has_present_neighbour = False
            if len(neighbors) > 0:
                for j in neighbors:
                    nbr_col = sensor_cols[j]
                    if bool(present_row_ts.get(nbr_col, False)):
                        has_present_neighbour = True
                        break

            # used to down-weight T even more at the very beginning
            short_history = hist_win.shape[0] < (seq_len // 2)

            final_val = float("nan")

            if use_T and use_S:
                # Confidence-weighted blend between temporal & spatial
                denom = conf_T + conf_S
                alpha = conf_T / (denom + 1e-6)
                final_val = alpha * T_val + (1.0 - alpha) * S_val

            elif use_S:
                # Spatial-only but confident enough
                final_val = S_val

            elif use_T:
                # Temporal-only. If this sensor has *no present neighbours* at ts,
                # soften GCNLSTM by blending with a smoother baseline.
                if not has_present_neighbour:
                    # Build a baseline from climatology and/or last observed
                    base_candidates = []
                    if np.isfinite(clim_val):
                        base_candidates.append(clim_val)
                    if np.isfinite(last_observed_val):
                        base_candidates.append(last_observed_val)

                    if base_candidates:
                        base = float(np.mean(base_candidates))
                        # Stronger down-weighting if the history is also short
                        alpha = 0.3 if short_history else 0.4
                        final_val = alpha * T_val + (1.0 - alpha) * base
                    else:
                        # No reasonable baseline → fall back to pure T
                        final_val = T_val
                else:
                    # Neighbours exist but spatial estimate didn't pass MIN_CONF → keep T
                    final_val = T_val

            else:
                # No reliable T or S → fall back to climatology / last observed
                if np.isfinite(clim_val) and conf_clim > 0.0:
                    if conf_T > 0.0 and np.isfinite(T_val):
                        # Blend temporal with climatology when temporal confidence is non-zero
                        alpha = conf_T / (conf_T + clim_beta + 1e-6)
                        final_val = alpha * T_val + (1.0 - alpha) * clim_val
                    else:
                        final_val = clim_val
                else:
                    # Last-resort: last observed value in history (if any)
                    if np.isfinite(last_observed_val):
                        final_val = last_observed_val
                    else:
                        final_val = float("nan")

            # Domain clipping if requested
            if np.isfinite(final_val):
                if domain_clip_min is not None:
                    final_val = max(domain_clip_min, final_val)
                if domain_clip_max is not None:
                    final_val = min(domain_clip_max, final_val)


            # write + flags
            try:
                missing_df.at[ts, col] = final_val if pd.notna(final_val) else np.nan
            except Exception:
                pass
            svals.append(final_val)
            sstatus.append(False)  # imputed
            SS.imputed_mask.at[ts, col] = pd.notna(final_val)
            original_imputed_values[col] = final_val


        else:
            # originally present, pass through
            v = missing_df.at[ts, col] if (ts in missing_df.index and col in missing_df.columns) else np.nan
            svals.append(v)
            sstatus.append(True)   # original
            SS.imputed_mask.at[ts, col] = False

#    SS["impute_time_tsg"][ts] = time.perf_counter() - tsg_start

#    # Safety: keep vectors aligned with static sensor list
#    if len(svals) < len(sensor_cols):
#        svals += [np.nan] * (len(sensor_cols) - len(svals))
#    if len(sstatus) < len(sensor_cols):
#        sstatus += [False] * (len(sstatus) - len(sstatus))


    # ----- TSGuard timing (per-timestamp and per-value) -----
    tsg_elapsed = time.perf_counter() - tsg_start
    SS["impute_time_tsg"][ts] = tsg_elapsed

    # NEW: seconds per imputed value (TSGuard, static sensors only)
    # We look only at the static sensors in `sensor_cols`, not dynamic captors.
    imputed_row_static = SS.imputed_mask.reindex(
        index=[ts], columns=sensor_cols, fill_value=False
    ).iloc[0]
    n_imputed_static = int(imputed_row_static.sum())

    if "impute_time_tsg_per_value" not in SS:
        SS["impute_time_tsg_per_value"] = {}

    if n_imputed_static > 0:
        per_val_time = tsg_elapsed / n_imputed_static
    else:
        per_val_time = float("nan")  # or 0.0 if you prefer

    SS["impute_time_tsg_per_value"][ts] = per_val_time

    # Print seconds per imputed value to the console (server logs)
    try:
        print(
            f"[TSGuard] {ts} — per-imputed-value time = {per_val_time:.6e} s "
            f"(total={tsg_elapsed:.6f} s, n_imputed={n_imputed_static})",
            flush=True,
        )
    except Exception:
        # Never break the app because printing failed
        pass

    # Safety: keep vectors aligned with static sensor list
    if len(svals) < len(sensor_cols):
        svals += [np.nan] * (len(sensor_cols) - len(svals))
    if len(sstatus) < len(sensor_cols):
        sstatus += [False] * (len(sensor_cols) - len(sstatus))


    # ---- Dynamic captors: purely rule-based spatial imputer ----
    dyn_vals = []
    dyn_status = []
    latlng_by_col = latlng.set_index("data_col")

    for dyn_col in dynamic_sensor_cols:
        # Compute a distance-weighted average of the nearest static neighbours
        try:
            lat_s = float(latlng_by_col.at[dyn_col, "latitude"])
            lon_s = float(latlng_by_col.at[dyn_col, "longitude"])
        except Exception:
            # If we cannot find coordinates, skip this captor
            SS.imputed_mask.at[ts, dyn_col] = False
            dyn_vals.append(np.nan)
            dyn_status.append(False)
            continue

        neighbour_candidates = []
        for core_col in sensor_cols:
            try:
                lat_n = float(latlng_by_col.at[core_col, "latitude"])
                lon_n = float(latlng_by_col.at[core_col, "longitude"])
            except Exception:
                continue
            d = haversine_distance(lat_s, lon_s, lat_n, lon_n)
            if not np.isfinite(d):
                continue
            neighbour_candidates.append((core_col, d))

        neighbour_candidates.sort(key=lambda t: t[1])
        if k_neighbors > 0:
            neighbour_candidates = neighbour_candidates[:k_neighbors]

        neighbour_values = []
        weights = []
        for core_col, d in neighbour_candidates:
            v = None
            if core_col in sensor_cols:
                idx = col_to_idx[core_col]
                if idx < len(svals):
                    v = svals[idx]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            # simple inverse-distance weighting
            w = 1.0 if d <= 0 else 1.0 / d
            neighbour_values.append(float(v))
            weights.append(w)

        if neighbour_values:
            w_arr = np.asarray(weights, dtype=float)
            v_arr = np.asarray(neighbour_values, dtype=float)
            w_norm = w_arr / np.sum(w_arr)
            dyn_val = float(np.sum(w_norm * v_arr))
            SS.imputed_mask.at[ts, dyn_col] = True
            dyn_status.append(False)  # always imputed / predicted
            dyn_vals.append(dyn_val)
        else:
            SS.imputed_mask.at[ts, dyn_col] = False
            dyn_status.append(False)
            dyn_vals.append(np.nan)

    # ---- Dicts captor -> value / state (static + dynamic) ----
    vals_by_col = dict(zip(sensor_cols, svals))
    real_by_col = dict(zip(sensor_cols, sstatus))

    for col, val, st_flag in zip(dynamic_sensor_cols, dyn_vals, dyn_status):
        vals_by_col[col] = val
        real_by_col[col] = st_flag

    imputed_row = SS.imputed_mask.reindex(
        index=[ts], columns=sensor_cols + dynamic_sensor_cols, fill_value=False
    ).iloc[0]

    # ---- Optional PriSTI per-tick imputation (for comparison panel only) ----
    pristi_row = None
    pristi_flags = None

    # 1) Detect whether PriSTI artifacts are present and mark the model as available
    pristi_available = False
    try:
        SS["pristi_last_error"] = None
        #PRISTI_ROOT = "./PriSTI"
        PRISTI_ROOT = "./PRISTI"
        CONFIG_PATH = f"{PRISTI_ROOT}/config/base.yaml"
        WEIGHTS_PATH = f"{PRISTI_ROOT}/save/aqi36/model.pth"
        MEANSTD_PK = f"{PRISTI_ROOT}/data/pm25/pm25_meanstd.pk"


        #CONFIG_PATH = f"PRISTI/config/base.yaml"
        #WEIGHTS_PATH = f"PRISTI/save/aqi36/model.pth"
        #MEANSTD_PK = f"PRISTI/data/pm25/pm25_meanstd.pk"


        have_files = (
            os.path.exists(CONFIG_PATH)
            and os.path.exists(WEIGHTS_PATH)
            and os.path.exists(MEANSTD_PK)
        )

        if have_files and ("orig_missing_values" in SS):
            pristi_available = True

            # Register PriSTI as an available model
            if isinstance(SS.get("available_models"), set):
                SS["available_models"].add("PriSTI")
            else:
                SS["available_models"] = set(SS.get("available_models", [])) | {"PriSTI"}

            # If user never touched the comparison tab, show PriSTI by default
            if "comparison_models" not in SS:
                SS["comparison_models"] = ["TSGuard", "PriSTI"]
    except Exception:
        pristi_available = False  # never break the sim because of PriSTI problems


    # 2) Only run PriSTI imputation if user actually selected it
    if pristi_available and ("PriSTI" in SS.get("comparison_models", [])):
        try:
            PRISTI_ROOT = "./PRISTI"
            CONFIG_PATH = f"{PRISTI_ROOT}/config/base.yaml"
            WEIGHTS_PATH = f"{PRISTI_ROOT}/save/aqi36/model.pth"
            MEANSTD_PK = f"{PRISTI_ROOT}/data/pm25/pm25_meanstd.pk"

            # Lazy-load PriSTI artifacts (cached by Streamlit)
            if ("pristi_model" not in SS) or (SS.get("pristi_device") != str(device)):
                pm, mean, std = load_pristi_artifacts(
                    CONFIG_PATH, WEIGHTS_PATH, MEANSTD_PK, device=device
                )
                SS["pristi_model"] = pm
                SS["pristi_mean"] = mean
                SS["pristi_std"] = std
                SS["pristi_device"] = str(device)

            # Work on the original missing-data frame kept in session_state
            base_df = SS.orig_missing_values

            if ts in base_df.index:
                pristi_cols = list(base_df.columns)
                if pristi_cols:
                    base_df_36 = base_df[pristi_cols].copy()

                    # ---------- NEW: lightweight interactive settings ----------
                    # eval_len: at most 20, but never more than available rows
                    eval_len = min(20, len(base_df_36))

                    # nsample: much smaller for interactive use (default 2, capped in [1, 5])
                    raw_nsample = st.session_state.get("pristi_nsample", 2)
                    try:
                        nsample = int(raw_nsample)
                    except Exception:
                        nsample = 2
                    nsample = max(1, min(nsample, 5))

                    # ---------- NEW: skip PriSTI when there is no missing data
                    # in the window we would feed it ----------
                    run_pristi = False
                    baseline = SS.orig_missing_baseline
                    if ts in baseline.index:
                        loc = baseline.index.get_loc(ts)
                        if isinstance(loc, int):
                            start = max(0, loc - (eval_len - 1))
                            window_mask = baseline.iloc[start : loc + 1, :]
                            if window_mask.values.any():
                                run_pristi = True

                    if run_pristi:
                        updated_df, info = impute_window_with_pristi(
                            missing_df=base_df_36,
                            sensor_cols=pristi_cols,
                            target_timestamp=ts,
                            model=SS["pristi_model"],
                            device=device,
                            eval_len=eval_len,
                            nsample=nsample,
                        )

                        if ts in updated_df.index:
                            # Keep PriSTI’s imputations so the next window can reuse them
                            base_df.loc[updated_df.index, pristi_cols] = updated_df[pristi_cols]
                            SS.orig_missing_values.loc[updated_df.index, pristi_cols] = updated_df[pristi_cols]

                            pristi_row = updated_df.loc[ts, pristi_cols]

                            # For the “imputed?” flag we just reuse the original missing mask
                            pristi_flags = SS.orig_missing_baseline.reindex(
                                index=[ts], columns=pristi_cols
                            ).iloc[0]
        except Exception as e:
            # PriSTI is strictly best-effort; never break the simulation because of it
            pristi_row = None
            pristi_flags = None
            SS["pristi_last_error"] = str(e)
            print("[PriSTI] ERROR:", e, flush=True)          

    # ---- Publish snapshot for the Models Comparison tab ----
    try:
        # TSGuard part (unchanged)
        snapshot = {
            "timestamp": ts,
            "TSGuard_values": dict(vals_by_col),
            "TSGuard_imputed": {
                str(k): bool(imputed_row.get(k, False))
                for k in vals_by_col.keys()
            },
        }

        # ---- ORBITS offline values (same captor ids as TSGuard) ----
        all_ids = list(vals_by_col.keys())

        if orbits_row is not None and orbits_flags is not None:
            orbits_values = {}
            orbits_imputed = {}
            for k in all_ids:
                # value from ORBITS df, or NaN if ORBITS doesn't have this captor
                v = orbits_row.get(k, np.nan)
                orbits_values[str(k)] = float(v) if pd.notna(v) else np.nan
                # we mark the same originally-missing cells as "imputed" for ORBITS
                orbits_imputed[str(k)] = bool(orbits_flags.get(k, False))

            snapshot["ORBITS_values"] = orbits_values
            snapshot["ORBITS_imputed"] = orbits_imputed        
            

        # ✅ NEW: always build PriSTI dicts over the SAME captor ids as TSGuard
        all_ids = list(vals_by_col.keys())
        pri_values = {str(k): np.nan for k in all_ids}
        pri_imputed = {str(k): False for k in all_ids}

        if pristi_row is not None and pristi_flags is not None:
            for k in pristi_row.index:
                if k not in pri_values:
                    # PriSTI may cover only a subset of stations; ignore others
                    continue
                v = pristi_row[k]
                pri_values[str(k)] = float(v) if pd.notna(v) else np.nan
                pri_imputed[str(k)] = bool(pristi_flags.get(k, False))

            snapshot["PriSTI_values"] = pri_values
            snapshot["PriSTI_imputed"] = pri_imputed

        SS["model_comparison_snapshot"] = snapshot
        SS["current_sim_timestamp"] = ts
        SS["tsguard_sensor_ids"] = list(all_ids)
    except Exception:
        # Snapshot is strictly best-effort; never break the simulation
        pass


    # --- Update missing streak (Scénario 1) ---
    if "_missing_streak_hours" not in SS:
        SS["_missing_streak_hours"] = {c: 0.0 for c in sensor_cols}

    for c in sensor_cols:
        originally_missing = bool(baseline_row_ts.get(c, False))
        val_now = vals_by_col.get(c, np.nan)
        no_value_now = (val_now is None) or (isinstance(val_now, float) and np.isnan(val_now))
        # On incrémente uniquement quand il manquait à l’origine ET qu’on n’a toujours pas de valeur à afficher
        if originally_missing and no_value_now:
            SS["_missing_streak_hours"][c] = SS["_missing_streak_hours"].get(c, 0.0) + 1.0  # +1h par tick
        else:
            SS["_missing_streak_hours"][c] = 0.0

    # ---- Vérification des contraintes & alertes (appelle TA fonction) ----
    viol_info = verify_constraints_and_alerts_for_timestamp(
        ts=ts,
        latlng_df=latlng[["sensor_id", "data_col", "latitude", "longitude"]],
        values_by_col=vals_by_col,
        imputed_mask_row=imputed_row,
        baseline_row=baseline_row_ts,
        missing_streak_hours=SS["_missing_streak_hours"],
    )
    render_grouped_alerts()

    # dict: sensor -> set({"temporal","spatial"})
    violating_imputed = viol_info.get("violating_imputed", {}) or {}
    violating_real = viol_info.get("violating_real", {}) or {}

    # ---------- Fallback pour les valeurs imputées qui violent une contrainte ----------
    if violating_imputed:
        for col, ct_types in violating_imputed.items():
            if col not in sensor_cols:
                continue
            col_idx = col_to_idx.get(col)
            if col_idx is None:
                continue

            neighbours = neighbors_by_idx[col_idx]
            vals = []
            weights = []

            for j in neighbours:
                nbr_col = sensor_cols[j]

                # ignorer les voisins eux-mêmes en violation
                if (nbr_col in violating_imputed) or (nbr_col in violating_real):
                    continue

                try:
                    v = float(missing_df.at[ts, nbr_col])
                except Exception:
                    v = float("nan")
                if not np.isfinite(v):
                    continue

                d = dist_matrix[col_idx, j]
                inv_d = 1.0 if d <= 0 else 1.0 / d
                corr_boost = float(corr_boost_matrix[col_idx, j])
                w = inv_d * corr_boost

                vals.append(v)
                weights.append(w)

            if vals:
                vals = np.asarray(vals, dtype=float)
                weights = np.asarray(weights, dtype=float)
                if np.all(weights == 0):
                    weights = np.ones_like(weights)
                w_norm = weights / np.sum(weights)
                fb_val = float(np.sum(w_norm * vals))

                # bornes domaine éventuelles
                if domain_clip_min is not None:
                    fb_val = max(domain_clip_min, fb_val)
                if domain_clip_max is not None:
                    fb_val = min(domain_clip_max, fb_val)

                # ✅ mettre à jour les valeurs utilisées ensuite
                missing_df.at[ts, col] = fb_val
                vals_by_col[col] = fb_val
                SS.imputed_mask.at[ts, col] = True

                ct_str = "+".join(sorted(ct_types)) if ct_types else "unknown"

                # 🟠 NEW : logger la valeur fallback
                IMPUTATION_LOG.append({
                    "timestamp": ts,
                    "sensor": col,
                    "value_type": "fallback",
                    "value": float(fb_val),
                    "violated_constraint": ct_str,
                })

                toast_notify(
                    "warning",
                    f"Fallback applied for imputed value at sensor {col} (t={ts}) after {ct_str} constraint violation.",
                    f"fb-{col}-{ts}",
                )
            else:
                toast_notify(
                    "warning",
                    f"No valid neighbors for fallback at sensor {col} (t={ts}). Keeping previous imputed value.",
                    f"fb-none-{col}-{ts}",
                )

    # ---------- Log des valeurs imputées initiales (modèle TSGuard) ----------
    for col in sensor_cols:
        was_missing = bool(baseline_row_ts.get(col, False))
        if not was_missing:
            continue  # on ne logge que les valeurs imputées

        # types de contraintes violées par cette imputation (si c'est le cas)
        ct_types = violating_imputed.get(col, set())
        ct_str = "+".join(sorted(ct_types)) if ct_types else "none"

        # valeur imputée initiale par TSGuard
        orig_val = original_imputed_values.get(col, vals_by_col.get(col, np.nan))

        IMPUTATION_LOG.append({
            "timestamp": ts,
            "sensor": col,
            "value_type": "imputed",  # valeur du modèle, avant fallback
            "value": float(orig_val) if np.isfinite(orig_val) else float("nan"),
            "violated_constraint": ct_str,
        })

    # 🔁 Écriture progressive dans le CSV
    flush_imputation_log_if_needed(
        path="tsguard_imputations.csv",
        chunk_size=100,  # adapte selon ce que tu veux
    )

    # ---- Buffers de rendu (fixed sliding window) ----

    row = {"datetime": ts}
    # static sensors
    for c in sensor_cols:
        row[c] = vals_by_col.get(c, np.nan)
    # dynamic sensors (if any)
    for c in dynamic_sensor_cols:
        row[c] = vals_by_col.get(c, np.nan)

    new_row = pd.DataFrame([row])      

    # Append as a new row (ignore old indices so length really increases)
    SS.sliding_window_df = (
        pd.concat([SS.sliding_window_df, new_row], ignore_index=True)
        if not SS.sliding_window_df.empty
        else new_row.copy()
    )

    SS.global_df = (
        pd.concat([SS.global_df, new_row], ignore_index=True)
        if not SS.global_df.empty
        else new_row.copy()
    )

    # Keep only the last 36 rows for the sliding window
    if len(SS.sliding_window_df) > 36:
        SS.sliding_window_df = SS.sliding_window_df.tail(36).reset_index(drop=True)


    # --- Map update & store base for next Fit ---
    if len(svals) != len(sensor_cols) or len(sstatus) != len(sensor_cols):
        if not st.session_state.get("_warned_len_mismatch", False):
            st.warning(
                f"[Guard] Mismatch tailles — sensors={len(sensor_cols)}, "
                f"svals={len(svals)}, sstatus={len(sstatus)}. "
                "On complète seulement ce tick."
            )
            st.session_state["_warned_len_mismatch"] = True

    # Use the combined dict (static + dynamic) built above
    def _fmt_val(v):
        try:
            if v is None or (_isinstance := isinstance(v, float)) and not np.isfinite(v):
                return "NA"
        except TypeError:
            return "NA"
        # adjust formatting as you like
        return f"{v:.1f}"

    tick_df = latlng.copy()
    tick_df["value"] = tick_df["data_col"].map(vals_by_col).apply(_fmt_val)

    tick_df["status"] = tick_df["data_col"].map(
        lambda c: "Real" if real_by_col.get(c, False) else "Imputed"
    )
    tick_df["timestamp"] = ts.strftime("%Y-%m-%d %H:%M")
    tick_df["bg_color"] = [GREEN if s == "Real" else RED for s in tick_df["status"]]
    tick_df["bg_radius"] = 10
    tick_df["icon"] = [ICON_SPEC] * len(tick_df)
    tick_df["icon_size"] = 1.0

    SS.deck_obj.layers = [make_bg_layer(tick_df), make_icon_layer(tick_df)]
    SS["_fit_base_df"] = tick_df.copy()
    SS["ph_map"].pydeck_chart(SS.deck_obj, use_container_width=True)


    # --- Gauge & counts (CUMULATIVE MISSED from the beginning to now) ---
    baseline_mask_to_now = SS.orig_missing_baseline.loc[:ts, sensor_cols]
    cumulative_missed = int(baseline_mask_to_now.values.sum())  # total # of originally-missing cells up to ts
    total_cells_to_now = baseline_mask_to_now.size
    pct_missed_to_now = (cumulative_missed / total_cells_to_now * 100.0) if total_cells_to_now else 0.0

    # Active sensors NOW (same as before, just for info)
    row_imp = SS.imputed_mask.reindex(index=[ts], columns=sensor_cols, fill_value=False).iloc[0]
    imputed_now = int(row_imp.sum())
    sensors_total = max(1, len(sensor_cols))
    real_now = sensors_total - imputed_now

    # Per-timestamp (NOW) counts that match the map colors:
    missed_now = int(baseline_row_ts.sum())
    active_now = len(sensor_cols) - missed_now

    SS["ph_counts_active"].markdown(f"Active sensors now: **{active_now}**")
    SS["ph_counts_missing"].markdown(f"Delayed sensors now: **{missed_now}**")

    # Use dynamic gauge thresholds if the user defined them; otherwise fall back to defaults
    mv_thresh = SS.get("missing_value_thresholds")
    if isinstance(mv_thresh, dict):
        g_min, g_max = mv_thresh["Green"]
        y_min, y_max = mv_thresh["Yellow"]
        r_min, r_max = mv_thresh["Red"]
    else:
        g_min = DEFAULT_VALUES["gauge_green_min"]
        g_max = DEFAULT_VALUES["gauge_green_max"]
        y_min = DEFAULT_VALUES["gauge_yellow_min"]
        y_max = DEFAULT_VALUES["gauge_yellow_max"]
        r_min = DEFAULT_VALUES["gauge_red_min"]
        r_max = DEFAULT_VALUES["gauge_red_max"]

    # Choose bar color based on which zone pct_missed_to_now falls into
    if pct_missed_to_now >= r_min:
        bar_color = "red"
    elif pct_missed_to_now >= y_min:
        bar_color = "yellow"
    else:
        bar_color = "green"

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct_missed_to_now,
        title={"text": "Missed Data (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [g_min, g_max], "color": "lightgreen"},
                {"range": [y_min, y_max], "color": "yellow"},
                {"range": [r_min, r_max], "color": "red"},
            ],
        },
    ))


    gauge_fig.update_layout(title="",margin=dict(l=10, r=10, t=30, b=10))
    lightify(gauge_fig)
    SS["ph_gauge"].plotly_chart(gauge_fig, use_container_width=True, key=f"{uid}_gauge_{iter_key}")

    # --- Global TS (Matplotlib) ---
    global_df_plot = SS.global_df.copy()
    fig_global = make_mpl_timeseries_figure(
        df=global_df_plot,
        imputed_mask=SS.imputed_mask,
        #sensor_cols=sensor_cols,
        sensor_cols=plot_sensor_cols,
        sensor_color_map=sensor_color_map,
        title="Global Sensors Time Series",
        gap_hours=12,
        #show_legend=True,
        show_legend=False, #legend moved to middle panel
        style="minimal",
    )
    SS["ph_global"].pyplot(fig_global, clear_figure=True)
    plt.close(fig_global)

    # --- Snapshot 10 (Matplotlib, last 10 timestamps, no legend) ---
    snap10_df = SS.sliding_window_df.tail(SNAP10).copy()
    fig_snap = make_mpl_timeseries_figure(
        df=snap10_df,
        imputed_mask=SS.imputed_mask,
        #sensor_cols=sensor_cols,
        sensor_cols=plot_sensor_cols,
        sensor_color_map=sensor_color_map,
        title="Snapshot (last 10)",
        gap_hours=6,
        show_legend=False,
        style="minimal",
    )
    SS["ph_snap10"].pyplot(fig_snap, clear_figure=True)
    plt.close(fig_snap)    

    # advance pointer for next tick and persist it
    ptr += 1
    SS["sim_ptr"] = ptr

    # Configurable Simulation Time:
    #   τ = sim_seconds_per_hour  (real seconds per simulated hour)
    sim_seconds_per_hour = float(
        SS.get("sim_seconds_per_hour", default_scale)
    )
    sim_seconds_per_hour = max(0.0, sim_seconds_per_hour)

    if sim_seconds_per_hour > 0.0 and ptr < total_steps:
        next_ts = pd.Timestamp(common_index[ptr])
        dt_hours = max(0.0, (next_ts - ts).total_seconds() / 3600.0)

        # If timestamps are identical or out-of-order, fall back to 1 hour
        if dt_hours <= 0.0:
            dt_hours = 1.0

        sleep_seconds = sim_seconds_per_hour * dt_hours
        time.sleep(sleep_seconds)

    # If we’re still running and not at the end, immediately schedule
    # the next tick. This re-runs `main()` from the top.
    if st.session_state.get("running", False) and ptr < total_steps:
        st.rerun()
    else:
        # End reached or user stopped
        st.session_state["running"] = False