#!/usr/bin/env python3
"""
MVP Recreation ── Tsoku & Makatjane (2026)
"Deep learning-based pairs trading: real-time forecasting of co-integrated
cryptocurrency pairs", Front. Appl. Math. Stat. 12:1749337.

This project recreates the core pipeline of the above paper and extends the
data window to February 2026 to include the most recent market conditions.

Pipeline
--------
1.  Download five crypto close prices  (ETH, BNB, LTC, XRP, USDT)
2.  Rolling Johansen co-integration → dynamic spread        (Table 2)
3.  Rolling z-score of spread → dynamic score               (Section 3.2)
4.  Train DNN & LSTM on dynamic-score sequences             (Sec 2.2–2.3)
5.  Dynamic Weighted Ensemble of DNN + LSTM                 (DWE)
6.  Error metrics: MSE / RMSE / MAE / MAPE / MFE / Theil U (Table 3)
7.  Percentile-threshold trading signals                    (Figure 6)
8.  Risk-performance metrics                                (Table 4)
9.  99 % Prediction Intervals                               (Figure 8)
"""

import copy
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ══════════════════════════════════════════════════════════════
# Configuration  (paper: Section 2 & Table 2)
# ══════════════════════════════════════════════════════════════
TICKERS: list[str] = ["ETH-USD", "BNB-USD", "LTC-USD", "XRP-USD", "USDT-USD"]
TICKER_LABELS: list[str] = ["Ethereum", "BNB", "Litecoin", "Ripple", "USDT"]
START_DATE = "2018-01-02"           # paper's original start
END_DATE = "2026-02-01"            # extended beyond paper (2025-10-31)

# COINT_WINDOW: size of each rolling window for the Johansen test.
# 252 ≈ 1 year of daily trading data.  A larger window produces more
# stable estimates but is slower to adapt to regime changes.
COINT_WINDOW = 252

# Z_WINDOW: how many past days to use when computing the rolling z-score
# (the "dynamic score").  60 ≈ 3 calendar months.
Z_WINDOW = 60

# SEQ_WINDOW: the number of past daily z-scores that each deep-learning
# model receives as input (i.e. the "look-back" length of every sample).
SEQ_WINDOW = 30

# Temporal train / validation / test split.  We keep strict chronological
# order — no shuffling — to avoid look-ahead bias.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (implicit)

# ── DNN hyper-parameters  (Section 2.2, Figure 1) ──
# Two hidden layers: 128 → 64 neurons, each followed by BatchNorm,
# ReLU activation, and 20 % Dropout.
DNN_HIDDEN: list[int] = [128, 64]
DNN_DROP = 0.2
DNN_EPOCHS = 100
DNN_LR = 5e-4          # Adam initial learning rate
DNN_BATCH = 32
DNN_PAT = 15            # early-stopping patience (epochs)

# ── LSTM hyper-parameters  (Section 2.3, Figure 2) ──
# 2-layer stacked LSTM with hidden dimension 64, 20 % inter-layer dropout.
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROP = 0.2
LSTM_EPOCHS = 100
LSTM_LR = 5e-4
LSTM_BATCH = 32
LSTM_PAT = 15

# DWE_WINDOW: the Dynamic Weighted Ensemble looks back this many test-set
# steps to evaluate each model's recent accuracy before assigning weights.
DWE_WINDOW = 50

# Trading signal thresholds (percentiles of the predicted dynamic score).
# Scores above the 90th percentile → SELL; below the 10th → BUY.
UPPER_PCT = 90
LOWER_PCT = 10

# ── Improvements over base paper (addressing professional review) ──
# Transaction cost per unit of position change (in z-score space).
# Even a small cost erodes profits when turnover is high.
TC_COST = 0.01

# Minimum observations before computing expanding-window percentile thresholds
MIN_OBS_PCT = 50

# EWM span for smoothing daily Johansen vectors → reduces hedge-ratio turnover
SMOOTH_SPAN = 20

SEED = 42
CSV_FILE = "crypto_data.csv"


# ══════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════
def set_seed(seed: int = SEED) -> None:
    """Fix every random seed so results are fully reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
# 1  Data download
# ══════════════════════════════════════════════════════════════
def download_prices(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame:
    """Download daily Close prices via *yfinance* for multiple tickers.

    Returns a DataFrame indexed by date with one column per ticker.
    Missing values are forward-filled first, then back-filled, so the
    very first rows (before a coin was listed) are also covered.
    """
    raw = yf.download(
        tickers, start=start, end=end, progress=True, auto_adjust=False
    )
    # yfinance returns a MultiIndex (field, ticker) when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    avail = [t for t in tickers if t in close.columns]
    if not avail:
        raise ValueError("No price data returned from yfinance.")
    return close[avail].ffill().bfill().dropna()


# ══════════════════════════════════════════════════════════════
# 2  Dynamic Johansen co-integration  (Section 3.1, Table 2)
# ══════════════════════════════════════════════════════════════
def rolling_johansen(
    log_prices: pd.DataFrame,
    window: int,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Rolling-window multivariate Johansen co-integration test.

    **Why co-integration?**
    Individual crypto prices are non-stationary (they trend upward or
    downward with no fixed mean).  However, certain *linear combinations*
    of their log-prices may be stationary — i.e. they fluctuate around a
    fixed mean and tend to revert back.  Finding such a combination is
    called co-integration, and the stationary linear combination is the
    "spread" we trade.

    **Why Johansen (not Engle–Granger)?**
    Engle–Granger only tests *one pair* at a time and requires choosing
    which asset is "dependent".  The Johansen procedure handles *all five
    assets simultaneously* via an eigenvalue decomposition of the Vector
    Error-Correction Model (VECM).  The eigenvector associated with the
    largest eigenvalue gives the strongest co-integrating relationship.

    **Why rolling windows?**
    A static Johansen test assumes the co-integrating relationship is
    constant over the entire sample.  In crypto markets that assumption
    is unrealistic — relationships shift due to regulation, sentiment, and
    liquidity changes.  By re-estimating the test over a sliding 252-day
    window, we obtain *time-varying* co-integrating vectors that adapt to
    the current market regime (consistent with Lo's Adaptive Market
    Hypothesis, referenced in the paper).

    **Look-ahead bias prevention**
    The resulting vector DataFrame is shifted forward by 1 day, so the
    spread at date *t* only uses coefficients estimated from data up to
    *t − 1*.

    Parameters
    ----------
    log_prices : DataFrame of log(close prices), shape (T, K)
    window     : number of days in each rolling estimation window
    det_order  : 0 = constant-only VECM (no linear trend)
    k_ar_diff  : number of lagged differences in the VECM

    Returns
    -------
    vectors : DataFrame – time-varying 1st co-integrating vector
    spread  : Series   – Σ coeff_i · log(price_i)
    """
    n = len(log_prices)
    vectors = pd.DataFrame(
        np.nan, index=log_prices.index, columns=log_prices.columns
    )
    total_iters = n - window + 1
    for i, end in enumerate(range(window, n + 1)):
        chunk = log_prices.iloc[end - window : end]
        try:
            res = coint_johansen(chunk.values, det_order, k_ar_diff)
            # evec[:, 0] is the eigenvector for the *largest* eigenvalue,
            # representing the most statistically significant co-integrating
            # relationship among the five assets.
            vec = res.evec[:, 0]
            vectors.iloc[end - 1] = vec
        except Exception:
            pass  # some windows may be numerically degenerate
        if (i + 1) % 500 == 0 or (i + 1) == total_iters:
            print(f"    Johansen window {i + 1:,}/{total_iters:,}")

    # Shift by 1 to prevent look-ahead bias
    vectors = vectors.shift(1)
    spread = (log_prices * vectors).sum(axis=1)
    spread[vectors.isna().any(axis=1)] = np.nan
    return vectors, spread


# ══════════════════════════════════════════════════════════════
# 3  Sequence builder
# ══════════════════════════════════════════════════════════════
def build_sequences(
    values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping (X, y) pairs for supervised next-step prediction.

    Given a 1-D array of length T and a look-back *window* of length W,
    this produces:
        X[i] = values[i : i+W]       (the past W values)
        y[i] = values[i+W]           (the value we want to predict)

    We use numpy's ``sliding_window_view`` for zero-copy efficiency — no
    Python loop, no data duplication.

    Returns
    -------
    X : ndarray of shape (T − W, W, 1)   – the trailing "1" is the
        feature dimension expected by both the DNN and LSTM.
    y : ndarray of shape (T − W, 1)
    """
    if len(values) <= window:
        raise ValueError("Series too short for the requested window.")
    wins = np.lib.stride_tricks.sliding_window_view(values, window)[:-1]
    X = wins[..., np.newaxis]   # (N, W, 1)
    y = values[window:, np.newaxis]  # (N, 1)
    return X, y


# ══════════════════════════════════════════════════════════════
# 4  DNN model  (Section 2.2, Figure 1)
# ══════════════════════════════════════════════════════════════
class SpreadDNN(nn.Module):
    """Deep (feed-forward) Neural Network for spread prediction.

    Architecture per hidden layer:
        Linear  → BatchNorm1d → ReLU → Dropout

    **BatchNorm1d** normalises each mini-batch so every neuron receives
    inputs with zero mean and unit variance.  This stabilises gradients
    and lets us use higher learning rates.  It also acts as a mild
    regulariser, reducing overfitting.

    **ReLU** (Rectified Linear Unit):  f(x) = max(0, x).  Unlike sigmoid
    or tanh, the gradient is either 0 or 1 — this avoids the "vanishing
    gradient" problem and is the standard activation for modern DNNs.

    **Dropout** randomly zeroes 20 % of neurons during each training step.
    This forces the network to spread information across many neurons
    instead of relying on a few, improving generalisation.

    The final layer is a single Linear neuron that outputs the predicted
    dynamic score (a scalar).
    """

    def __init__(
        self, input_dim: int, hidden: list[int], drop: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(drop),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len, 1).
        # The DNN treats the whole window as a flat feature vector, so we
        # squeeze the trailing dimension → (batch, seq_len).
        return self.net(x.squeeze(-1))


# ══════════════════════════════════════════════════════════════
# 5  LSTM model  (Section 2.3, Figure 2)
# ══════════════════════════════════════════════════════════════
class SpreadLSTM(nn.Module):
    """Long Short-Term Memory network for spread prediction.

    Unlike the DNN (which sees the whole window at once), the LSTM reads
    the input **one time-step at a time** and maintains an internal "cell
    state" that acts as a memory lane.  At each step three gates decide
    what to remember and what to forget:

    1. **Forget gate (f_t)**  — sigmoid that decides how much of the old
       cell state to keep.  Values near 0 ⇒ forget; near 1 ⇒ remember.
    2. **Input gate (i_t)**   — sigmoid that decides which new information
       to write into the cell state.
    3. **Output gate (o_t)**  — sigmoid that determines how much of the
       cell state to expose as the hidden state (the layer's "output").

    These gates let the LSTM capture long-range temporal dependencies
    while avoiding the vanishing-gradient problem that plagues vanilla
    RNNs — which is why it is widely used in financial time-series
    forecasting.

    We stack two LSTM layers (``num_layers=2``) so the second layer can
    learn higher-order temporal patterns from the first layer's hidden
    states.  Inter-layer dropout (20 %) regularises this stacking.

    The final Linear layer maps the last hidden state → a single scalar
    prediction.
    """

    def __init__(
        self, input_dim: int, hidden: int, n_layers: int, drop: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=drop if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # out[:, -1, :] takes the hidden state at the *last* time-step
        return self.fc(out[:, -1, :])


# ══════════════════════════════════════════════════════════════
# 6  Generic trainer  (early stopping + LR scheduler)
# ══════════════════════════════════════════════════════════════
def train_model(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int,
    lr: float,
    bs: int,
    patience: int,
    name: str = "",
) -> tuple[nn.Module, list[float], list[float]]:
    """Train a PyTorch model with several best-practice techniques.

    **Adam optimiser** — an adaptive-learning-rate variant of SGD.  It
    maintains per-parameter moving averages of past gradients (momentum)
    and squared gradients, so each weight effectively gets its own tuned
    learning rate.

    **ReduceLROnPlateau scheduler** — if validation loss stops improving
    for a few epochs, the scheduler *halves* the learning rate.  This
    lets the model take large steps early on, then fine-tune later.

    **Early stopping** — if validation loss has not improved for
    ``patience`` consecutive epochs, training halts and the best model
    weights are restored.  This is the primary defence against overfitting.

    **shuffle=False** — critical for time-series!  Shuffling the training
    data would let the model "see" future patterns during training,
    creating look-ahead bias.
    """
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5,
        patience=max(1, patience // 2), min_lr=1e-6,
    )

    ds = torch.utils.data.TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)

    xv = torch.tensor(x_va, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.float32)

    best_w: Optional[dict] = None
    best_v = float("inf")
    wait = 0
    t_hist: list[float] = []
    v_hist: list[float] = []

    for ep in range(1, epochs + 1):
        model.train()       # enable dropout + batch-norm training mode
        e_loss = 0.0
        for bx, by in dl:
            opt.zero_grad()             # reset gradients
            loss = crit(model(bx), by)  # forward pass → MSE loss
            loss.backward()             # back-propagation
            opt.step()                  # update weights
            e_loss += loss.item() * bx.size(0)

        model.eval()        # disable dropout, use running BN stats
        with torch.no_grad():
            vl = crit(model(xv), yv).item()
        sched.step(vl)      # tell scheduler about the new val loss

        tl = e_loss / len(ds)
        t_hist.append(tl)
        v_hist.append(vl)

        if vl < best_v:
            best_v = vl
            best_w = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if ep % 10 == 0:
            print(
                f"    {name} ep {ep:03d}/{epochs}  "
                f"train={tl:.6f}  val={vl:.6f}"
            )

        if wait >= patience:
            print(
                f"    {name} early-stop ep {ep}  best_val={best_v:.6f}"
            )
            break

    if best_w is not None:
        model.load_state_dict(best_w)
    model.eval()
    return model, t_hist, v_hist


# ══════════════════════════════════════════════════════════════
# 7  Dynamic Weighted Ensemble  (DWE, Eq 12)
# ══════════════════════════════════════════════════════════════
def dynamic_weighted_ensemble(
    p_dnn: np.ndarray,
    p_lstm: np.ndarray,
    actual: np.ndarray,
    adapt_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptively combine DNN and LSTM predictions.

    **Key idea (paper Eq 12):** instead of a fixed 50/50 average, give
    *more* weight to whichever model has been more accurate *recently*.

    At each test step *t* we look back over the previous ``adapt_w`` steps
    and compute each model's recent Mean Squared Error.  The weight
    assigned to each model is inversely proportional to its MSE:

        w_DNN  ∝  1 / MSE_DNN(recent)
        w_LSTM ∝  1 / MSE_LSTM(recent)

    This means: a model with *half* the recent error receives *twice* the
    weight.  The weights are re-normalised to sum to 1.

    **Causal structure:** at step *t* we only use actual values from
    [t − w, t − 1].  The actual value at *t* (which we are predicting)
    is never used for weighting at *t*, so there is no look-ahead bias.

    For the first ``adapt_w`` steps (where we lack enough history) we
    default to equal weights.
    """
    n = len(p_dnn)
    ens = np.empty(n)
    wts = np.empty((n, 2))
    eps = 1e-10     # prevent division by zero

    for t in range(n):
        if t < adapt_w:
            wd, wl = 0.5, 0.5
        else:
            s = slice(t - adapt_w, t)
            mse_d = np.mean((p_dnn[s] - actual[s]) ** 2) + eps
            mse_l = np.mean((p_lstm[s] - actual[s]) ** 2) + eps
            inv_d, inv_l = 1.0 / mse_d, 1.0 / mse_l
            tot = inv_d + inv_l
            wd, wl = inv_d / tot, inv_l / tot
        wts[t] = [wd, wl]
        ens[t] = wd * p_dnn[t] + wl * p_lstm[t]

    return ens, wts


# ══════════════════════════════════════════════════════════════
# 8  Error metrics  (Table 3)
# ══════════════════════════════════════════════════════════════
def error_metrics(
    actual: np.ndarray, pred: np.ndarray
) -> dict[str, float]:
    """Compute six standard forecasting-error metrics.

    - **MSE**  (Mean Squared Error)   – average of squared errors; heavily
      penalises large deviations.
    - **RMSE** (Root MSE)             – same unit as the target; easier to
      interpret than MSE.
    - **MAE**  (Mean Absolute Error)  – average of |error|; more robust to
      outliers than MSE.
    - **MAPE** (Mean Absolute Percentage Error) – error as a % of the
      actual value; scale-free, but undefined when actual ≈ 0.
    - **MFE**  (Mean Forecast Error / Bias) – positive ⇒ over-predicting,
      negative ⇒ under-predicting.
    - **Theil U** – RMSE of the model divided by RMSE of a naïve
      random-walk forecast (predict tomorrow = today).  Theil U < 1 means
      the model beats the naïve baseline.
    """
    e = pred - actual
    mse = float(np.mean(e ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(e)))

    mask = np.abs(actual) > 1e-10
    mape = (
        float(np.mean(np.abs(e[mask] / actual[mask])) * 100)
        if mask.any()
        else np.nan
    )
    mfe = float(np.mean(e))

    naive_e = actual[1:] - actual[:-1]
    n_rmse = float(np.sqrt(np.mean(naive_e ** 2)))
    theil = rmse / n_rmse if n_rmse > 0 else np.nan

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "MFE (Bias)": mfe,
        "Theil U": theil,
    }


# ══════════════════════════════════════════════════════════════
# 9  Trading signals  (Section 3.2, Figure 6)
# ══════════════════════════════════════════════════════════════
def trading_signals(
    score: np.ndarray,
    upper_pct: float,
    lower_pct: float,
    min_obs: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position-aware trading signals with expanding-window thresholds.

    **Fixes over naïve approach (professional review):**

    1. **No look-ahead bias.**  The original implementation computed
       percentile thresholds over the *entire* test set at once — meaning
       the threshold at day 1 already used information from day 393.
       Now, thresholds at time *t* are computed from ``score[0 : t]``
       only (expanding window), so no future data is ever used.

    2. **Proper exit strategy.**  The original code flipped between
       BUY / SELL / HOLD every day, causing excessive turnover.  Now a
       finite-state machine manages positions:
       - **Enter LONG** when the score drops below the lower threshold
         (undervalued → expect mean-reversion upward).
       - **Exit LONG** when the score reverts above 0 (mean).
       - **Enter SHORT** when the score rises above the upper threshold
         (overvalued → expect mean-reversion downward).
       - **Exit SHORT** when the score reverts below 0.

       This creates complete trade cycles (entry → hold → exit) and
       prevents the constant position-flipping that erodes profits.

    Returns
    -------
    sig    : int array of positions at each step (+1 long, −1 short, 0 flat)
    up_arr : float array of expanding upper thresholds (NaN where < min_obs)
    lo_arr : float array of expanding lower thresholds (NaN where < min_obs)
    """
    n = len(score)
    sig = np.zeros(n, dtype=int)
    up_arr = np.full(n, np.nan)
    lo_arr = np.full(n, np.nan)
    position = 0  # current state: +1 long, −1 short, 0 flat

    for t in range(n):
        if t < min_obs:
            # Not enough history for reliable percentile estimates
            sig[t] = position
            continue

        # Expanding-window thresholds — strictly causal (uses [0..t-1])
        up = np.percentile(score[:t], upper_pct)
        lo = np.percentile(score[:t], lower_pct)
        up_arr[t] = up
        lo_arr[t] = lo

        # ── State machine ──
        if position == 0:
            # Flat → look for entry
            if score[t] < lo:
                position = 1    # enter long
            elif score[t] > up:
                position = -1   # enter short
        elif position == 1:
            # Long → exit when score reverts above mean (0)
            if score[t] >= 0:
                position = 0
        elif position == -1:
            # Short → exit when score reverts below mean (0)
            if score[t] <= 0:
                position = 0

        sig[t] = position

    return sig, up_arr, lo_arr


# ══════════════════════════════════════════════════════════════
# 10  Risk-performance metrics  (Table 4)
# ══════════════════════════════════════════════════════════════
def risk_metrics(
    signals: np.ndarray,
    spread: np.ndarray,
    tc_cost: float = 0.0,
    market_ret: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute portfolio-level risk metrics for the trading strategy.

    **Transaction cost model (improvement over base paper):**
    Every time the position changes (e.g. flat→long, long→flat), a cost
    proportional to ``tc_cost × |Δposition|`` is deducted.  This
    penalises high turnover and produces realistic net-of-cost metrics.

    - **Hit Rate** – fraction of active position-days that were profitable.
    - **Avg P/L per Trade** – mean daily P/L on active position-days.
    - **Max Drawdown (MDD)** – largest peak-to-trough decline in equity.
    - **Sharpe Ratio** – annualised (mean / σ) of daily net returns.
    - **Sortino Ratio** – like Sharpe but only penalises downside.
    - **Avg Position Size** – fraction of days with an open position.
    - **Num Trades** – total number of position changes (entries + exits).
    - **Total TC** – cumulative transaction cost deducted.
    - **Signal Lag** – cross-correlation lag; 0 is ideal.
    - **Market Correlation** – ρ vs ETH; negative = good diversifier.
    """
    d_sp = np.diff(spread)
    sig = signals[:-1]   # signal at t → P/L realised at t+1
    ret_gross = sig * d_sp

    # Transaction costs: proportional to |position change|
    pos_changes = np.abs(np.diff(np.concatenate([[0], signals])))[:-1]
    tc_total = tc_cost * pos_changes
    ret = ret_gross - tc_total      # net returns

    trades = ret[sig != 0]
    hit = float(np.mean(trades > 0)) if len(trades) else 0.0
    avg = float(np.mean(trades)) if len(trades) else 0.0

    # Additive equity curve (z-score changes, not % returns)
    eq = 1.0 + np.cumsum(ret)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / np.maximum(running_max, 1e-10)
    mdd = float(dd.min())

    mu, sd = float(np.mean(ret)), float(np.std(ret))
    sharpe = float(np.sqrt(252) * mu / sd) if sd > 0 else 0.0

    ds = ret[ret < 0]
    dsd = float(np.std(ds)) if len(ds) > 0 else 1e-10
    sortino = float(np.sqrt(252) * mu / dsd)

    pos = float(np.mean(np.abs(sig)))
    n_trades = int(pos_changes.sum())

    # Signal lag via cross-correlation
    if len(d_sp) > 1:
        c = np.correlate(
            sig - sig.mean(), d_sp - d_sp.mean(), mode="full"
        )
        lags = np.arange(-len(sig) + 1, len(sig))
        lag = float(lags[np.argmax(c)])
    else:
        lag = 0.0

    # Correlation with market returns
    if market_ret is not None and len(market_ret) == len(ret):
        mcorr = float(np.corrcoef(ret, market_ret)[0, 1])
    else:
        mcorr = np.nan

    return {
        "Hit Rate": hit,
        "Avg P/L per Trade": avg,
        "Max Drawdown": mdd,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Avg Position Size": pos,
        "Num Trades": n_trades,
        "Total TC": float(tc_total.sum()),
        "Signal Lag (days)": lag,
        "Market Correlation": mcorr,
    }


# ══════════════════════════════════════════════════════════════
# 11  Prediction intervals  (Section 2.5, Eqs 10-14)
# ══════════════════════════════════════════════════════════════
def prediction_intervals(
    p_dnn: np.ndarray,
    p_lstm: np.ndarray,
    p_ens: np.ndarray,
    actual: np.ndarray,
    confidence: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Construct dynamic prediction intervals around the ensemble forecast.

    **Why prediction intervals?**
    A point forecast ("tomorrow's score will be 0.5") tells you nothing
    about *uncertainty*.  A prediction interval ("… between −0.2 and 1.2
    with 99 % confidence") is far more actionable for risk management.

    **How they are built (paper Eqs 10–14):**
    1. σ̂_t = standard deviation across the two base-learner forecasts
       at time *t* (a measure of model *disagreement* — Eq 13).
    2. q   = the empirical 99th percentile of |standardised errors|,
       capturing how far off the ensemble typically is in σ̂ units.
    3. PI  = ŷ ± q · σ̂   (Eq 10).

    Wider σ̂ (models disagree a lot) ⇒ wider interval ⇒ more uncertainty.

    Returns
    -------
    lo, hi   : lower and upper bounds of the interval at each step
    metrics  : dict with Mean / Min / Max PIW and Coverage Probability
    """
    sigma = np.std(np.stack([p_dnn, p_lstm]), axis=0)
    sigma = np.maximum(sigma, 1e-4)        # numerical floor
    std_err = (p_ens - actual) / sigma
    q = float(np.percentile(np.abs(std_err), confidence * 100))

    lo = p_ens - q * sigma
    hi = p_ens + q * sigma
    piw = hi - lo

    cp = float(np.mean((actual >= lo) & (actual <= hi)))
    return lo, hi, {
        "Mean PIW": float(piw.mean()),
        "Min PIW": float(piw.min()),
        "Max PIW": float(piw.max()),
        "Coverage Prob": cp,
    }


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def main() -> None:  # noqa: C901
    set_seed()

    # ── Step 1: Download prices ──────────────────────────────
    print("=" * 65)
    print(f"Step 1 · Downloading price data  ({START_DATE} → {END_DATE})")
    print("=" * 65)
    try:
        prices = download_prices(TICKERS, START_DATE, END_DATE)
        prices.to_csv(CSV_FILE)
        print(f"  Downloaded {len(prices):,} obs  ×  {prices.shape[1]} assets")
    except Exception as exc:
        print(f"  Download failed ({exc}), loading from {CSV_FILE} …")
        prices = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)

    tickers_avail = [t for t in TICKERS if t in prices.columns]
    labels_avail = [
        TICKER_LABELS[TICKERS.index(t)] for t in tickers_avail
    ]
    print(f"  Available tickers: {tickers_avail}")

    # ── Figure 3: closing-price chart ────────────────────────
    colors = ["black", "blue", "green", "red", "purple"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (tk, lb) in enumerate(zip(tickers_avail, labels_avail)):
        ax.plot(
            prices.index, prices[tk],
            label=lb, color=colors[i % len(colors)], lw=0.8,
        )
    ax.set_title("Closing Prices of Five Cryptocurrencies (2018–2026)  [Figure 3]")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure3_prices.png", dpi=150)
    plt.close(fig)

    # ── Step 2: Dynamic Johansen co-integration ──────────────
    print("\n" + "=" * 65)
    print("Step 2 · Rolling Johansen co-integration  (Table 2)")
    print("=" * 65)
    # We take logarithms of prices before co-integration analysis.
    # Reason: log-prices make multiplicative relationships additive and
    # percentage changes approximately equal to differences, which is the
    # standard pre-processing in financial econometrics.
    log_p = np.log(prices[tickers_avail].clip(lower=1e-8))
    vectors, spread_raw = rolling_johansen(log_p, window=COINT_WINDOW)

    # ── Smooth Johansen vectors to reduce hedge-ratio turnover ──
    # Daily re-estimation of the co-integrating vector means the
    # portfolio weights change every day, generating large rebalancing
    # costs.  Exponential smoothing (EWM) stabilises the vectors while
    # preserving the adaptive nature of rolling estimation.
    turnover_raw = float(vectors.diff().abs().mean().sum())
    vectors = vectors.ewm(span=SMOOTH_SPAN, min_periods=1).mean()
    turnover_smooth = float(vectors.diff().abs().mean().sum())

    # Recompute spread with smoothed vectors
    spread_raw = (log_p * vectors).sum(axis=1)
    spread_raw[vectors.isna().any(axis=1)] = np.nan
    spread_raw = spread_raw.dropna()
    print(f"  Raw spread length after dropna: {len(spread_raw):,}")
    print(
        f"  Hedge-ratio turnover:  raw = {turnover_raw:.4f}   "
        f"smoothed = {turnover_smooth:.4f}   "
        f"(−{(1 - turnover_smooth / max(turnover_raw, 1e-10)) * 100:.0f} %)"
    )

    # Full-sample Johansen for trace-test table  (Table 2b)
    full_data = log_p.dropna()
    full_res = coint_johansen(full_data.values, 0, 1)
    print("\n  Johansen Trace Test  (full sample, Table 2b):")
    header = f"  {'r≤':>4}  {'Trace':>10}  {'90% CV':>9}  {'95% CV':>9}  {'99% CV':>9}"
    print(header)
    for i in range(min(len(full_res.lr1), len(tickers_avail))):
        print(
            f"  {i:>4}  {full_res.lr1[i]:>10.3f}  "
            f"{full_res.cvt[i, 0]:>9.3f}  "
            f"{full_res.cvt[i, 1]:>9.3f}  "
            f"{full_res.cvt[i, 2]:>9.3f}"
        )

    # Latest co-integrating vector (cf. Table 2a)
    last_vec = vectors.dropna().iloc[-1]
    print("\n  Co-integrating vector (latest window, Table 2a):")
    for tk, c in zip(tickers_avail, last_vec):
        print(f"    {tk:>10}  {c:+.4f}")

    adf = adfuller(spread_raw.values, autolag="AIC")
    print(f"\n  ADF on spread:  stat = {adf[0]:.4f}   p-value = {adf[1]:.6f}")

    # ── Figure 4: raw spread time-series ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(spread_raw.index, spread_raw.values, lw=0.8)
    mu_s, sd_s = spread_raw.mean(), spread_raw.std()
    ax.axhline(mu_s, color="k", ls="--", lw=1, label="Mean")
    ax.axhline(mu_s + sd_s, color="r", ls="--", lw=0.8, label="±1 σ")
    ax.axhline(mu_s - sd_s, color="r", ls="--", lw=0.8)
    ax.set_title("Dynamic Co-integration Spread (2018–2026)  [Figure 4]")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure4_spread.png", dpi=150)
    plt.close(fig)

    # ── Step 3: rolling z-score → "dynamic score" ────────────
    # The paper models the *standardised* spread (the "dynamic score"),
    # not the raw spread.  A rolling z-score subtracts the local mean and
    # divides by the local standard deviation over the past Z_WINDOW days:
    #
    #     z_t = (spread_t − μ_rolling) / σ_rolling
    #
    # This removes slow-moving trends and volatility changes, ensuring
    # that the train / val / test distributions stay comparable —
    # critical for deep-learning models that assume stationary inputs.
    print("\n" + "=" * 65)
    print("Step 3 · Building dynamic-score sequences  (rolling z-score)")
    print("=" * 65)
    roll_mu = spread_raw.rolling(window=Z_WINDOW).mean()
    roll_sd = spread_raw.rolling(window=Z_WINDOW).std().clip(lower=1e-8)
    z_spread = ((spread_raw - roll_mu) / roll_sd).dropna()
    print(f"  Dynamic score length: {len(z_spread):,}")

    z_vals = z_spread.values.astype(np.float64)
    X, y = build_sequences(z_vals, SEQ_WINDOW)
    N = X.shape[0]
    n_tr = int(N * TRAIN_RATIO)
    n_va = int(N * VAL_RATIO)

    Xtr, Xva, Xte = X[:n_tr], X[n_tr : n_tr + n_va], X[n_tr + n_va :]
    ytr, yva, yte = y[:n_tr], y[n_tr : n_tr + n_va], y[n_tr + n_va :]
    test_dates = z_spread.index[SEQ_WINDOW + n_tr + n_va :]
    print(f"  train = {n_tr:,}   val = {n_va:,}   test = {len(Xte):,}")

    # ── Step 4: train DNN ────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 4 · Training DNN  (Section 2.2)")
    print("=" * 65)
    dnn_model = SpreadDNN(SEQ_WINDOW, DNN_HIDDEN, DNN_DROP)
    dnn_model, dnn_tl, dnn_vl = train_model(
        dnn_model, Xtr, ytr, Xva, yva,
        DNN_EPOCHS, DNN_LR, DNN_BATCH, DNN_PAT, "DNN",
    )

    # ── Step 5: train LSTM ───────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 5 · Training LSTM  (Section 2.3)")
    print("=" * 65)
    lstm_model = SpreadLSTM(1, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROP)
    lstm_model, lstm_tl, lstm_vl = train_model(
        lstm_model, Xtr, ytr, Xva, yva,
        LSTM_EPOCHS, LSTM_LR, LSTM_BATCH, LSTM_PAT, "LSTM",
    )

    # ── Step 6: test-set predictions + DWE ───────────────────
    print("\n" + "=" * 65)
    print("Step 6 · Predictions & Dynamic Weighted Ensemble")
    print("=" * 65)
    Xt = torch.tensor(Xte, dtype=torch.float32)
    with torch.no_grad():
        p_dnn = dnn_model(Xt).numpy().flatten()
        p_lstm = lstm_model(Xt).numpy().flatten()
    y_actual = yte.flatten()

    p_ens, dwe_w = dynamic_weighted_ensemble(
        p_dnn, p_lstm, y_actual, DWE_WINDOW
    )
    print(f"  DNN  mean weight: {dwe_w[:, 0].mean():.3f}")
    print(f"  LSTM mean weight: {dwe_w[:, 1].mean():.3f}")

    # ── Step 7: error metrics  (Table 3) ─────────────────────
    # Metrics computed in dynamic-score (z-score) space,
    # matching the paper's scale.
    print("\n" + "=" * 65)
    print("Step 7 · Performance Comparison  [Table 3]")
    print("=" * 65)
    m_d = error_metrics(y_actual, p_dnn)
    m_l = error_metrics(y_actual, p_lstm)
    m_e = error_metrics(y_actual, p_ens)

    tbl3 = pd.DataFrame(
        {"DNN": m_d, "LSTM": m_l, "Dynamic Ensemble": m_e}
    )
    best_col = []
    for metric in tbl3.index:
        row = tbl3.loc[metric].dropna()
        best_col.append(row.abs().idxmin() if len(row) else "")
    tbl3["Best"] = best_col
    print(tbl3.to_string())

    # ── Figure 5: model performance ──────────────────────────
    td = test_dates[: len(y_actual)]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, preds, nm in zip(
        axes,
        [p_dnn, p_lstm, p_ens],
        ["DNN", "LSTM", "Dynamic Ensemble"],
    ):
        ax.plot(td, y_actual, label="Actual", color="black", lw=0.8)
        ax.plot(td, preds, label=f"{nm} Predicted", alpha=0.85)
        ax.set_ylabel("Dynamic Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{nm} Forecast vs Actual")
    axes[-1].set_xlabel("Date")
    fig.suptitle(
        "Model Performance Visualisation  [Figure 5]",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig("figure5_performance.png", dpi=150)
    plt.close(fig)

    # ── Step 8: trading signals  (Figure 6, Table 5) ─────────
    print("\n" + "=" * 65)
    print("Step 8 · Trading Signals  [Figure 6 / Table 5]")
    print("=" * 65)
    dynamic_score = p_ens
    sigs, up_arr, lo_arr = trading_signals(
        dynamic_score, UPPER_PCT, LOWER_PCT, MIN_OBS_PCT
    )

    # Position-day counts
    n_long_days = int((sigs == 1).sum())
    n_short_days = int((sigs == -1).sum())
    n_flat_days = int((sigs == 0).sum())

    # Trade counts (entries + exits)
    pos_changes = np.abs(np.diff(np.concatenate([[0], sigs])))
    n_trades = int(pos_changes.sum())
    n_round_trips = n_trades // 2

    # Win rate: fraction of active position-days where P/L > 0
    d_sp = np.diff(y_actual)
    active_days = 0
    wins = 0
    for t in range(len(sigs) - 1):
        if sigs[t] != 0:
            active_days += 1
            if sigs[t] * d_sp[t] > 0:  # position direction × move > 0
                wins += 1
    win_pct = wins / max(active_days, 1) * 100

    print(f"  Long days = {n_long_days}   Short days = {n_short_days}   "
          f"Flat days = {n_flat_days}")
    print(f"  Round-trip trades: {n_round_trips}   "
          f"Total position changes: {n_trades}")
    print(f"  Win rate (active position-days): {win_pct:.1f}%")

    # Figure 6 — with time-varying thresholds (no look-ahead)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(td, dynamic_score, lw=0.8, color="steelblue",
            label="Ensemble Dynamic Score")

    # Expanding-window thresholds (time-varying)
    valid_th = ~np.isnan(up_arr)
    ax.plot(td[valid_th], up_arr[valid_th], color="red", ls="--", lw=0.7,
            label=f"Sell threshold ({UPPER_PCT}th pct, expanding)")
    ax.plot(td[valid_th], lo_arr[valid_th], color="green", ls="--", lw=0.7,
            label=f"Buy threshold ({LOWER_PCT}th pct, expanding)")
    ax.axhline(0, color="k", ls=":", lw=0.5, label="Mean (exit level)")

    # Mark trade entries only (not all position-days)
    prev_sig = np.concatenate([[0], sigs[:-1]])
    long_entry = (sigs == 1) & (prev_sig != 1)
    short_entry = (sigs == -1) & (prev_sig != -1)
    if long_entry.any():
        ax.scatter(
            td[long_entry], dynamic_score[long_entry],
            c="green", marker="^", s=40, label="Enter LONG", zorder=3,
        )
    if short_entry.any():
        ax.scatter(
            td[short_entry], dynamic_score[short_entry],
            c="red", marker="v", s=40, label="Enter SHORT", zorder=3,
        )
    ax.set_title(
        "Dynamic Ensemble Forecast Score & Trading Signals  [Figure 6]"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Dynamic Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure6_signals.png", dpi=150)
    plt.close(fig)

    # Table 5 (sample of dynamic scores and actions)
    labels = np.where(
        sigs == 1, "LONG", np.where(sigs == -1, "SHORT", "FLAT")
    )
    idx5 = np.linspace(0, len(dynamic_score) - 1, 10, dtype=int)
    t5 = pd.DataFrame({
        "Date": td[idx5].values,
        "Ensemble Score": np.round(dynamic_score[idx5], 4),
        "Action": labels[idx5],
    })
    print("\n  Sample Trading Actions  [Table 5]:")
    print(t5.to_string(index=False))

    # ── Step 9: risk metrics  (Table 4) ──────────────────────
    print("\n" + "=" * 65)
    print("Step 9 · Risk Performance Metrics  [Table 4]")
    print("=" * 65)
    eth_test = (
        prices[tickers_avail[0]]
        .reindex(z_spread.index)
        .iloc[SEQ_WINDOW + n_tr + n_va :]
    )
    eth_vals = np.clip(eth_test.values.astype(float), 1e-8, None)
    eth_ret = np.diff(np.log(eth_vals))

    rm = risk_metrics(
        sigs, y_actual,
        tc_cost=TC_COST,
        market_ret=eth_ret if len(eth_ret) == len(sigs) - 1 else None,
    )
    for k, v in rm.items():
        if isinstance(v, float) and not np.isnan(v):
            print(f"  {k:>25s} : {v:+.4f}")
        else:
            print(f"  {k:>25s} : {v}")

    # ── Step 10: 99% prediction intervals  (Figure 8) ────────
    print("\n" + "=" * 65)
    print("Step 10 · 99 % Prediction Intervals  [Figure 8]")
    print("=" * 65)
    pi_lo, pi_hi, pi_m = prediction_intervals(
        p_dnn, p_lstm, p_ens, y_actual, 0.99,
    )
    for k, v in pi_m.items():
        print(f"  {k:>20s} : {v:.4f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(
        td, pi_lo, pi_hi,
        color="lightblue", alpha=0.5, label="99 % Prediction Interval",
    )
    ax.plot(td, y_actual, color="black", lw=0.8, label="Actual Score")
    ax.plot(
        td, p_ens, color="tab:orange", lw=0.8,
        label="Ensemble Predicted",
    )
    ax.set_title(
        "Dynamic Weighted Ensemble — 99 % Prediction Intervals  [Figure 8]"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Dynamic Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure8_intervals.png", dpi=150)
    plt.close(fig)

    # ── Bonus: training loss curves ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tl, vl, nm in zip(
        axes,
        [dnn_tl, lstm_tl],
        [dnn_vl, lstm_vl],
        ["DNN", "LSTM"],
    ):
        ax.plot(tl, label="Train Loss")
        ax.plot(vl, label="Val Loss")
        ax.set_title(f"{nm} Training vs Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close(fig)

    # ── Bonus: DWE weight evolution ──────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(td[: len(dwe_w)], dwe_w[:, 0], label="DNN weight")
    ax.plot(td[: len(dwe_w)], dwe_w[:, 1], label="LSTM weight")
    ax.set_title("Dynamic Weighted Ensemble — Weight Evolution")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dwe_weights.png", dpi=150)
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Pipeline complete.  Saved figures:")
    print("  figure3_prices.png        – Crypto price chart       [Fig 3]")
    print("  figure4_spread.png        – Dynamic spread           [Fig 4]")
    print("  figure5_performance.png   – Model forecasts          [Fig 5]")
    print("  figure6_signals.png       – Trading signals          [Fig 6]")
    print("  figure8_intervals.png     – 99% prediction intervals [Fig 8]")
    print("  training_curves.png       – DNN & LSTM loss curves")
    print("  dwe_weights.png           – DWE weight evolution")
    print("=" * 65)


if __name__ == "__main__":
    main()
