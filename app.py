# app.py
"""
Business Forecast & Anomaly Detection Automation Platform (Tiny DL models)
- Upload CSV (datetime + numeric target)
- Trains 6 tiny DL models (PyTorch) in parallel (sequentially run)
- Compares MAE / RMSE / MAPE and ranks models
- Produces iterative multi-step forecast and anomaly detection
- Intended for demo / small datasets and Streamlit free tier (use Quick Mode to reduce time)
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go

# ---------------------------
# Config / Helpers
# ---------------------------
st.set_page_config(layout="wide", page_title="AutoForecast DL Platform")
st.title("AutoForecast — Business Forecast & Anomaly Detection (Tiny DL)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_download_link(df, filename="results.csv"):
    towrite = io.StringIO()
    df.to_csv(towrite, index=False)
    b64 = base64.b64encode(towrite.getvalue().encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def detect_datetime_numeric_columns(df):
    # detect datetime-like and numeric columns
    dt_candidates = []
    num_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c])
            if parsed.notna().sum() / len(parsed) > 0.6:
                dt_candidates.append(c)
        except Exception:
            continue
    return dt_candidates, num_candidates

def make_supervised(series_values, lookback):
    X, y = [], []
    for i in range(lookback, len(series_values)):
        X.append(series_values[i-lookback:i])
        y.append(series_values[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def iterative_forecast(model, last_window, n_steps, scaler=None, model_type="seq", device=DEVICE):
    # last_window: numpy array shape (lookback,)
    preds = []
    cur = last_window.copy()
    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.tensor(cur.astype(np.float32)).unsqueeze(0).to(device)  # (1, lookback) or (1, lookback, 1)
            if model_type in ("mlp", "cnn", "perceptron"):
                out = model(x).cpu().numpy().ravel()
            else:  # seq models expect (batch, seq_len, feat)
                if x.ndim == 2:
                    x_seq = x.unsqueeze(-1)  # (1, lookback, 1)
                else:
                    x_seq = x
                out = model(x_seq).cpu().numpy().ravel()
            p = out[0]
            preds.append(p)
            # slide
            cur = np.concatenate([cur[1:], [p]])
    # inverse scale if scaler provided (assume StandardScaler)
    if scaler is not None:
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).ravel()
    return np.array(preds)

# ---------------------------
# Tiny PyTorch model definitions (very small)
# ---------------------------
class TinyMLP(nn.Module):
    def __init__(self, input_size, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        # x shape: (batch, input_size)
        h = self.act(self.fc1(x))
        return self.fc2(h)

class MicroMLP(nn.Module):
    def __init__(self, input_size, h1=32, h2=16):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return self.fc3(h)

class MiniCNN1D(nn.Module):
    def __init__(self, input_size, kernel=3, n_filters=8):
        super().__init__()
        self.conv = nn.Conv1d(1, n_filters, kernel_size=kernel)  # input channel 1
        conv_out = input_size - kernel + 1
        self.fc = nn.Linear(n_filters * conv_out, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        # x shape (batch, input_size) -> conv expects (batch, in_channels, L)
        x = x.unsqueeze(1)  # (batch, 1, L)
        h = self.act(self.conv(x))
        h = h.view(h.size(0), -1)
        return self.fc(h)

class MiniRNN(nn.Module):
    def __init__(self, input_size, hidden=16):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        # x shape (batch, seq_len, 1)
        out, h = self.rnn(x)
        return self.fc(out[:, -1, :])

class MiniLSTM(nn.Module):
    def __init__(self, input_size, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class MiniGRU(nn.Module):
    def __init__(self, input_size, hidden=16):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ---------------------------
# Training helper
# ---------------------------
def train_torch_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device=DEVICE):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = None
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # adapt shapes: if sequential models expect (batch, seq_len, 1)
            if xb.dim() == 2 and isinstance(model, (MiniRNN, MiniLSTM, MiniGRU)):
                xb_in = xb.unsqueeze(-1)
            else:
                xb_in = xb
            pred = model(xb_in)
            loss = loss_fn(pred.view(-1,1), yb.view(-1,1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        # validation
        if val_loader is not None:
            model.eval()
            vals = []
            with torch.no_grad():
                for xb_val, yb_val in val_loader:
                    xb_val = xb_val.to(device)
                    yb_val = yb_val.to(device)
                    if xb_val.dim() == 2 and isinstance(model, (MiniRNN, MiniLSTM, MiniGRU)):
                        xb_in = xb_val.unsqueeze(-1)
                    else:
                        xb_in = xb_val
                    p = model(xb_in)
                    vals.append(((p.view(-1).cpu().numpy() - yb_val.view(-1).cpu().numpy())**2).mean())
            val_loss = float(np.mean(vals)) if vals else None
            # simple early stop logic
            if best_val is None or (val_loss is not None and val_loss < best_val):
                best_val = val_loss
    return model

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(truth, pred):
    mae = mean_absolute_error(truth, pred)
    rmse = mean_squared_error(truth, pred, squared=False)
    # MAPE handle zeros
    denom = np.where(np.abs(truth) < 1e-8, 1e-8, truth)
    mape = np.mean(np.abs((truth - pred) / denom)) * 100
    return mae, rmse, mape

# ---------------------------
# Streamlit UI: Upload + options
# ---------------------------
st.sidebar.header("Options")
quick_mode = st.sidebar.checkbox("Quick Mode (fewer epochs, faster)", value=True)
forecast_steps = st.sidebar.number_input("Forecast steps (periods)", min_value=1, max_value=365, value=30)
lookback = st.sidebar.slider("Lookback (timesteps)", min_value=3, max_value=64, value=12)
val_fraction = st.sidebar.slider("Validation fraction", min_value=0.05, max_value=0.5, value=0.15)
epochs = 8 if quick_mode else 30
batch_size = 32

uploaded = st.file_uploader("Upload CSV (must contain datetime and at least one numeric column)", type=["csv"])
use_demo = st.sidebar.selectbox("Or use demo dataset", options=["-- none --","sales_monthly.csv","energy_hourly.csv","stock_daily.csv"])

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif use_demo != "-- none --":
    try:
        df = pd.read_csv(use_demo)
        st.info(f"Using demo file: {use_demo}")
    except Exception as e:
        st.error(f"Failed to load demo CSV: {e}")

if df is None:
    st.stop()

st.write("Preview:")
st.dataframe(df.head())

# detect columns
dt_cols, num_cols = detect_datetime_numeric_columns(df)
st.sidebar.write("Detected datetime columns:", dt_cols)
st.sidebar.write("Detected numeric columns:", num_cols)

date_col = st.sidebar.selectbox("Datetime column (auto-detected)", options=dt_cols if dt_cols else df.columns.tolist())
value_col = st.sidebar.selectbox("Target numeric column", options=num_cols if num_cols else df.columns.tolist())

# prepare series: sort by date and take target
try:
    df[date_col] = pd.to_datetime(df[date_col])
except Exception:
    st.error("Could not parse datetime column. Please choose another or format dates as ISO.")
    st.stop()

df = df[[date_col, value_col]].dropna().sort_values(date_col)
if len(df) < 20:
    st.warning("Small dataset — models will be tiny and results are illustrative.")

series = df[value_col].values.astype(np.float32)
dates = df[date_col].values
# scaler
scaler = StandardScaler()
series_scaled = scaler.fit_transform(series.reshape(-1,1)).ravel()

# supervised windows
X_all, y_all = make_supervised(series_scaled, lookback)
if len(X_all) < 5:
    st.error("Not enough data after applying lookback. Reduce lookback or upload more rows.")
    st.stop()

# split
n_total = len(X_all)
n_val = max(1, int(n_total * val_fraction))
n_train = n_total - n_val
X_train, y_train = X_all[:n_train], y_all[:n_train]
X_val, y_val = X_all[n_train:], y_all[n_train:]

# create DataLoaders
def make_loader(X, y, batch_size=batch_size, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
train_loader = make_loader(X_train, y_train)
val_loader = make_loader(X_val, y_val, shuffle=False)

st.write(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# initialize models dictionary
models = {
    "TinyMLP": TinyMLP(input_size=lookback, hidden=24),
    "MicroMLP": MicroMLP(input_size=lookback, h1=48, h2=24),
    "MiniCNN": MiniCNN1D(input_size=lookback, kernel=3, n_filters=8),
    "MiniRNN": MiniRNN(input_size=lookback, hidden=24),
    "MiniLSTM": MiniLSTM(input_size=lookback, hidden=24),
    "MiniGRU": MiniGRU(input_size=lookback, hidden=24)
}

st.info(f"Training {len(models)} tiny DL models (epochs={epochs}). This runs sequentially — allow some time. Use Quick Mode for demo speed.")
if st.button("Train & Compare Models"):
    results_preds = {}
    metrics = {}
    progress = st.progress(0)
    total = len(models)
    idx = 0
    for name, model in models.items():
        st.write(f"Training {name} ...")
        # adapt input shapes for loaders: for MLP/CNN models we expect (batch, lookback)
        # For seq models (RNN/LSTM/GRU), training function will reshape to (batch, seq_len, 1)
        mdl = train_torch_model(model, train_loader, val_loader, epochs=epochs, lr=1e-3, device=DEVICE)
        # Forecast on validation set (one-step predictions) for metrics
        mdl.eval()
        preds_val = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb_cpu = xb.to(DEVICE)
                if isinstance(mdl, (MiniRNN, MiniLSTM, MiniGRU)):
                    out = mdl(xb_cpu.unsqueeze(-1)).cpu().numpy().ravel()
                else:
                    out = mdl(xb_cpu).cpu().numpy().ravel()
                preds_val.extend(out)
        preds_val = np.array(preds_val)
        # inverse scale
        preds_val_inv = scaler.inverse_transform(preds_val.reshape(-1,1)).ravel()
        y_val_inv = scaler.inverse_transform(y_val.reshape(-1,1)).ravel()
        mae, rmse, mape = compute_metrics(y_val_inv, preds_val_inv)
        metrics[name] = {"MAE": float(mae), "RMSE": float(rmse), "MAPE%": float(mape)}
        # full forecast: iterative forecast n_steps using last window from full series
        last_window = series_scaled[-lookback:]
        preds_future_scaled = iterative_forecast(mdl, last_window, forecast_steps, scaler=None,
                                                 model_type="seq" if isinstance(mdl,(MiniRNN,MiniLSTM,MiniGRU)) else "mlp",
                                                 device=DEVICE)
        preds_future = scaler.inverse_transform(preds_future_scaled.reshape(-1,1)).ravel()
        # assemble dates for forecast (use daily increments average if times not evenly spaced)
        try:
            # compute average delta
            deltas = np.diff(pd.to_datetime(df[date_col]).values.astype("datetime64[s]").astype(int))
            avg_delta = int(np.median(deltas)) if len(deltas)>0 else 86400
        except Exception:
            avg_delta = 86400
        last_date = pd.to_datetime(df[date_col].iloc[-1])
        future_dates = [last_date + pd.to_timedelta((i+1)*avg_delta, unit='s') for i in range(forecast_steps)]
        df_fore = pd.DataFrame({"datetime": future_dates, "forecast": preds_future})
        results_preds[name] = df_fore
        idx += 1
        progress.progress(idx/total)
    # show metrics table
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").sort_values("RMSE")
    st.subheader("Model comparison (validation)")
    st.dataframe(metrics_df)
    best = metrics_df.index[0]
    st.success(f"Best model on validation: {best}")
    # Plot historical + forecasts side-by-side for top 3 models
    topk = list(metrics_df.index[:3])
    st.subheader("Forecast comparison (top 3 models)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=series, name="history"))
    for name in topk:
        fr = results_preds[name]
        fig.add_trace(go.Scatter(x=fr["datetime"], y=fr["forecast"], name=name))
    st.plotly_chart(fig, use_container_width=True)
    # Anomalies using best model residuals on training history (z-score of residuals)
    st.subheader("Anomaly detection (residual z-score) - using best model")
    best_preds_train_scaled = []
    # produce one-step-in-sample predictions on entire history using sliding window and best model
    mdl_best = models[best].to(DEVICE)
    all_X = X_all
    with torch.no_grad():
        xb_all = torch.tensor(all_X, dtype=torch.float32).to(DEVICE)
        if isinstance(mdl_best, (MiniRNN, MiniLSTM, MiniGRU)):
            out_all = mdl_best(xb_all.unsqueeze(-1)).cpu().numpy().ravel()
        else:
            out_all = mdl_best(xb_all).cpu().numpy().ravel()
    out_all_inv = scaler.inverse_transform(out_all.reshape(-1,1)).ravel()
    # truth aligned for windows:
    truth_windows = series[lookback:]
    residuals = truth_windows - out_all_inv
    resid_mean = residuals.mean()
    resid_std = residuals.std() if residuals.std()>0 else 1.0
    anomaly_mask = np.abs((residuals - resid_mean)/resid_std) > 2.5
    anomaly_idx = np.where(anomaly_mask)[0]
    anomalies = [{"datetime": df[date_col].iloc[lookback + int(i)], "value": float(truth_windows[int(i)])} for i in anomaly_idx]
    if anomalies:
        st.warning(f"Detected {len(anomalies)} anomalies (z-score > 2.5). Showing up to 10:")
        st.write(anomalies[:10])
    else:
        st.success("No strong anomalies detected (z-score test).")
    # let user download best model forecast or full table
    st.subheader("Download forecasts")
    combined = []
    for name, dfr in results_preds.items():
        tmp = dfr.copy()
        tmp["model"] = name
        combined.append(tmp)
    all_forecasts = pd.concat(combined, ignore_index=True)
    st.markdown(make_download_link(all_forecasts, "all_forecasts.csv"), unsafe_allow_html=True)
    st.markdown(make_download_link(pd.DataFrame(metrics_df), "model_metrics.csv"), unsafe_allow_html=True)
else:
    st.info("Configure options on left and press 'Train & Compare Models' to run tiny DL automation.")
