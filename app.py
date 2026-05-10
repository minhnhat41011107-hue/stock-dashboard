# app.py
# VN Stock Dashboard + Roboadvisor + LONG Screener
# Giao diện 100% tiếng Việt
# Phiên bản ổn định cho Streamlit Cloud
# - Chỉ tải 4 file lõi khi khởi động
# - Dữ liệu phụ tải theo nút bấm (không sync folder lúc startup)
# - LONG realtime dùng form để tránh rerun liên tục
# - Hỗ trợ đăng ký API key vnstock để giảm giới hạn request

import os
import json
import tempfile
from pathlib import Path
from datetime import date, timedelta

import gdown
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from vnstock import Quote, Trading, Listing, register_user

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cố vấn danh mục & LONG Screener",
    page_icon="📈",
    layout="wide",
)

st.title("🇻🇳 Cố vấn danh mục cổ phiếu Việt Nam")
st.caption(
    "Tạo danh mục mẫu, backtest so với VNINDEX, cập nhật giá mỗi ngày, "
    "và chấm điểm LONG theo mô hình đã huấn luyện."
)

# =========================================================
# CONFIG
# =========================================================
def get_cfg(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

# 4 file lõi (bắt buộc)
LONG_MODEL_ID = get_cfg("LONG_MODEL_ID", "1nQbV59VCT5HLEGAcZCcTLk5ackhPyPS6")
FEATURE_COLS_ID = get_cfg("FEATURE_COLS_ID", "1RSvGie6w_Xl9OHo-X9EqgfqC9Z5Oc-4_")
PRICE_ALIGNED_COPY_ID = get_cfg("PRICE_ALIGNED_COPY_ID", "1CCaW7V10VPRF6r33fvdJOn-yj6nnffxD")
METADATA_ID = get_cfg("METADATA_ID", "1LmGfNmAyvQ94hGqZJi0hR1oGp--hzFAp")

# Dữ liệu phụ: chỉ tải khi người dùng bấm nút
OPTIONAL_FOLDER_URL = get_cfg(
    "KHOALUAN_FOLDER_URL",
    "https://drive.google.com/drive/folders/1VcKf2mWjmeiN16kpj25I7zrbPhxlXmqg?usp=sharing",
)

# API key vnstock
VNSTOCK_API_KEY = get_cfg("vnstock_aaab902776edf70323db0f169b4ee80c", "").strip()

VNSTOCK_PRIMARY_SOURCE = get_cfg("VNSTOCK_SOURCE", "KBS").strip().upper() or "KBS"
VNSTOCK_FALLBACK_SOURCE = "VCI" if VNSTOCK_PRIMARY_SOURCE == "KBS" else "KBS"

CORE_CACHE_DIR = Path(tempfile.gettempdir()) / "khoaluan_core_files"
CORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_CACHE_DIR = Path(tempfile.gettempdir()) / "khoaluan_extra_files"
EXTRA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = [
    "models/long_model.pkl",
    "models/feature_cols.pkl",
    "data/price_aligned_copy.csv",
    "metadata/metadata.json",
]

OPTIONAL_FILES = [
    "data/ml_df.csv",
    "data/walk_forward_df.csv",
    "predictions/wf_oos_predictions.csv",
    "predictions/latest_top30_predictions.csv",
    "results/oos_decile_performance.csv",
    "results/walk_forward_summary.csv",
    "results/long_model_feature_importance.csv",
    "results/feature_future_return_correlation.csv",
    "backtest/oos_topk_backtest_vs_vnindex.csv",
    "backtest/regime_filtered_oos_topk_backtest_vs_vnindex.csv",
]

# =========================================================
# VNSTOCK AUTH
# =========================================================
@st.cache_resource(show_spinner=False)
def init_vnstock_auth() -> bool:
    if not VNSTOCK_API_KEY:
        return False
    try:
        register_user(VNSTOCK_API_KEY)
        return True
    except Exception:
        return False

VNSTOCK_AUTH_OK = init_vnstock_auth()

# =========================================================
# HELPERS: DOWNLOAD FILES BY ID
# =========================================================
def download_one(file_id: str, out_name: str, cache_dir: Path) -> Path | None:
    if not file_id:
        return None

    out_path = cache_dir / out_name
    if out_path.exists():
        return out_path

    url = f"https://drive.google.com/uc?id={file_id}"
    ok = gdown.download(url, str(out_path), quiet=True)
    if not ok:
        return None
    return out_path if out_path.exists() else None


@st.cache_resource(show_spinner=False)
def sync_optional_folder() -> Path | None:
    """Tải dữ liệu phụ theo nút bấm. Không gọi ở startup."""
    marker = EXTRA_CACHE_DIR / ".synced"
    if marker.exists():
        return EXTRA_CACHE_DIR

    if not OPTIONAL_FOLDER_URL.strip():
        return None

    try:
        gdown.download_folder(
            url=OPTIONAL_FOLDER_URL.strip(),
            output=str(EXTRA_CACHE_DIR),
            quiet=True,
        )
        marker.write_text("ok", encoding="utf-8")
        return EXTRA_CACHE_DIR
    except Exception:
        return None


def resolve_required_artifact(rel_path: str) -> Path | None:
    """Chỉ dùng cho 4 file lõi."""
    local_candidates = [
        Path(rel_path),
        Path("data") / rel_path,
        Path("models") / rel_path,
        Path("metadata") / rel_path,
    ]
    for p in local_candidates:
        if p.exists():
            return p

    mapping = {
        "models/long_model.pkl": (LONG_MODEL_ID, "long_model.pkl"),
        "models/feature_cols.pkl": (FEATURE_COLS_ID, "feature_cols.pkl"),
        "data/price_aligned_copy.csv": (PRICE_ALIGNED_COPY_ID, "price_aligned_copy.csv"),
        "metadata/metadata.json": (METADATA_ID, "metadata.json"),
    }

    if rel_path in mapping:
        file_id, out_name = mapping[rel_path]
        return download_one(file_id, out_name, CORE_CACHE_DIR)

    return None


def resolve_optional_artifact(rel_path: str, allow_remote: bool = False) -> Path | None:
    """Chỉ dùng cho file phụ."""
    local_candidates = [
        Path(rel_path),
        Path("data") / rel_path,
        Path("results") / rel_path,
        Path("predictions") / rel_path,
        Path("backtest") / rel_path,
    ]
    for p in local_candidates:
        if p.exists():
            return p

    if not allow_remote or not st.session_state.get("extras_loaded", False):
        return None

    root = sync_optional_folder()
    if root is None:
        return None

    direct = root / rel_path
    if direct.exists():
        return direct

    matches = list(root.rglob(Path(rel_path).name))
    if matches:
        return matches[0]

    return None


def check_required_files():
    return [p for p in REQUIRED_FILES if resolve_required_artifact(p) is None]


@st.cache_data(show_spinner=False)
def load_csv_required(rel_path: str) -> pd.DataFrame | None:
    p = resolve_required_artifact(rel_path)
    if p is None:
        return None
    return pd.read_csv(p, low_memory=False)


@st.cache_data(show_spinner=False)
def load_json_required(rel_path: str) -> dict | None:
    p = resolve_required_artifact(rel_path)
    if p is None:
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_pickle_required(rel_path: str):
    p = resolve_required_artifact(rel_path)
    if p is None:
        return None
    return joblib.load(p)


@st.cache_data(show_spinner=False)
def load_csv_optional(rel_path: str, allow_remote: bool) -> pd.DataFrame | None:
    p = resolve_optional_artifact(rel_path, allow_remote=allow_remote)
    if p is None:
        return None
    return pd.read_csv(p, low_memory=False)


# =========================================================
# HELPERS: PRICE MATRIX
# =========================================================
@st.cache_data(show_spinner=False)
def load_price_wide() -> pd.DataFrame:
    """
    price_aligned_copy.csv là wide:
    time, ticker1, ticker2, ..., VNINDEX
    """
    p = resolve_required_artifact("data/price_aligned_copy.csv")
    if p is None:
        raise FileNotFoundError("Không tìm thấy data/price_aligned_copy.csv trong repo hoặc qua file ID.")

    df = pd.read_csv(p, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "time" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "time"})
        else:
            df = df.rename(columns={df.columns[0]: "time"})

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index("time").sort_index()
    return df


def build_returns(price_wide: pd.DataFrame) -> pd.DataFrame:
    ret = price_wide.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return ret


# =========================================================
# HELPERS: VNSTOCK LIVE UPDATE
# =========================================================
@st.cache_data(show_spinner=False, ttl=30)
def fetch_ohlc_history(ticker: str, start_date: str, end_date: str, source: str = "KBS") -> pd.DataFrame | None:
    for src in [source, VNSTOCK_FALLBACK_SOURCE]:
        try:
            q = Quote(symbol=ticker, source=src)
            hist = q.history(start=start_date, end=end_date, interval="d")

            if hist is None or hist.empty:
                continue

            hist = hist.copy()
            hist.columns = [str(c).strip().lower() for c in hist.columns]

            if "time" in hist.columns:
                hist = hist.rename(columns={"time": "date"})
            elif "date" not in hist.columns:
                hist = hist.rename(columns={hist.columns[0]: "date"})

            if "ticker" not in hist.columns:
                hist["ticker"] = ticker

            for c in ["open", "high", "low", "close", "volume"]:
                if c in hist.columns:
                    hist[c] = pd.to_numeric(hist[c], errors="coerce")

            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            hist = hist.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            hist["ticker"] = ticker
            return hist
        except Exception:
            continue

    return None


def fetch_latest_close_vnstock(ticker: str, source: str = "KBS") -> dict | None:
    # lấy cửa sổ ngắn để giảm tải và cho dữ liệu mới nhất
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=10)
    hist = fetch_ohlc_history(
        ticker=ticker,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        source=source,
    )
    if hist is None or hist.empty:
        return None

    last = hist.iloc[-1]
    return {
        "ticker": ticker,
        "date": pd.to_datetime(last["date"], errors="coerce"),
        "close": float(last["close"]),
        "source": source,
    }


def merge_live_updates_into_price_wide(price_wide: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    if updates is None or updates.empty:
        return price_wide

    out = price_wide.copy()
    updates = updates.dropna(subset=["date", "ticker", "close"]).copy()
    updates["date"] = pd.to_datetime(updates["date"], errors="coerce")
    updates = updates.dropna(subset=["date"])

    latest_date = updates["date"].max()

    for tk in updates["ticker"].unique():
        if tk not in out.columns:
            out[tk] = np.nan

    if latest_date in out.index:
        for _, r in updates.iterrows():
            out.loc[latest_date, r["ticker"]] = r["close"]
    else:
        new_row = out.iloc[-1].copy()
        new_row.name = latest_date
        for _, r in updates.iterrows():
            new_row[r["ticker"]] = r["close"]
        out = pd.concat([out, pd.DataFrame([new_row], index=[latest_date])], axis=0)

    out = out.sort_index()
    out.index.name = "time"
    return out


# =========================================================
# HELPERS: NOTEBOOK "DANH MỤC TEST"
# =========================================================
def build_universe(price_wide: pd.DataFrame, returns_wide: pd.DataFrame) -> pd.DataFrame:
    market_col = "VNINDEX" if "VNINDEX" in price_wide.columns else None
    stock_cols = [c for c in price_wide.columns if c != market_col]

    TRADING_DAYS = 252
    mu = returns_wide[stock_cols].mean() * TRADING_DAYS
    vol = returns_wide[stock_cols].std() * np.sqrt(TRADING_DAYS)
    latest_price = price_wide[stock_cols].iloc[-1] * 1000

    universe = pd.DataFrame(
        {
            "mean_return": mu,
            "volatility": vol,
            "latest_price": latest_price,
        }
    ).sort_index()

    return universe


def map_profile(risk_score: int):
    if risk_score <= 3:
        return "Thận trọng", 0.25, 0.75, 8, 0.18
    elif risk_score <= 7:
        return "Cân bằng", 0.50, 0.50, 10, 0.15
    else:
        return "Tăng trưởng", 0.75, 0.25, 12, 0.12


def construct_portfolio(
    universe: pd.DataFrame,
    risk_score: int,
    target_return: float,
    horizon_month: int,
    capital: float,
    vol_quantile: float,
    min_price: float,
):
    profile, growth_weight, defensive_weight, n_stocks, max_weight = map_profile(risk_score)

    positive_mask = universe["mean_return"] > 0
    vol_cutoff = universe["volatility"].quantile(vol_quantile)
    vol_mask = universe["volatility"] <= vol_cutoff
    price_mask = universe["latest_price"] >= min_price

    filtered = universe[positive_mask & vol_mask & price_mask].copy()

    growth_pool = filtered.sort_values("mean_return", ascending=False).head(30).copy()
    defensive_pool = filtered.sort_values("volatility", ascending=True).head(30).copy()

    if filtered.empty:
        stats = {
            "profile": profile,
            "growth_weight": growth_weight,
            "defensive_weight": defensive_weight,
            "n_stocks": n_stocks,
            "max_weight": max_weight,
            "vol_cutoff": vol_cutoff,
            "expected_return": np.nan,
            "expected_risk": np.nan,
            "return_risk": np.nan,
            "target_return": target_return,
            "horizon_month": horizon_month,
            "capital": capital,
        }
        return profile, filtered, growth_pool, defensive_pool, pd.DataFrame(), stats

    n_growth = n_stocks // 2
    n_def = n_stocks - n_growth

    growth_pool["score"] = growth_pool["mean_return"] / growth_pool["volatility"]
    defensive_pool["score"] = 1 / defensive_pool["volatility"]

    growth_pick = growth_pool.sort_values("score", ascending=False).head(n_growth)
    defensive_pick = defensive_pool.sort_values("score", ascending=False).head(n_def)

    portfolio = pd.concat([growth_pick, defensive_pick])
    portfolio = portfolio[~portfolio.index.duplicated()].copy()

    if portfolio.empty:
        stats = {
            "profile": profile,
            "growth_weight": growth_weight,
            "defensive_weight": defensive_weight,
            "n_stocks": n_stocks,
            "max_weight": max_weight,
            "vol_cutoff": vol_cutoff,
            "expected_return": np.nan,
            "expected_risk": np.nan,
            "return_risk": np.nan,
            "target_return": target_return,
            "horizon_month": horizon_month,
            "capital": capital,
        }
        return profile, filtered, growth_pool, defensive_pool, portfolio, stats

    portfolio["raw_weight"] = portfolio["score"] / portfolio["score"].sum()
    portfolio["weight"] = portfolio["raw_weight"].clip(upper=max_weight)
    portfolio["weight"] = portfolio["weight"] / portfolio["weight"].sum()
    portfolio["allocation_vnd"] = capital * portfolio["weight"]
    portfolio["shares"] = (portfolio["allocation_vnd"] / portfolio["latest_price"]).astype(int)

    port_return = (portfolio["mean_return"] * portfolio["weight"]).sum()
    port_risk = np.sqrt(((portfolio["volatility"] ** 2) * (portfolio["weight"] ** 2)).sum())

    stats = {
        "profile": profile,
        "growth_weight": growth_weight,
        "defensive_weight": defensive_weight,
        "n_stocks": n_stocks,
        "max_weight": max_weight,
        "vol_cutoff": vol_cutoff,
        "expected_return": port_return,
        "expected_risk": port_risk,
        "return_risk": (port_return / port_risk) if port_risk > 0 else np.nan,
        "target_return": target_return,
        "horizon_month": horizon_month,
        "capital": capital,
    }

    return profile, filtered, growth_pool, defensive_pool, portfolio, stats


def backtest_portfolio(price_wide: pd.DataFrame, returns_wide: pd.DataFrame, portfolio: pd.DataFrame):
    selected = portfolio.index.tolist()
    weights = portfolio["weight"].values

    port_ret_series = (returns_wide[selected] * weights).sum(axis=1)
    benchmark = returns_wide["VNINDEX"] if "VNINDEX" in returns_wide.columns else None

    portfolio_nav = (1 + port_ret_series).cumprod()
    result = pd.DataFrame({"Portfolio": portfolio_nav})

    if benchmark is not None:
        vnindex_nav = (1 + benchmark).cumprod()
        result["VNINDEX"] = vnindex_nav

    years = len(port_ret_series) / 252 if len(port_ret_series) > 0 else np.nan
    cagr = result["Portfolio"].iloc[-1] ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = port_ret_series.std() * np.sqrt(252)
    sharpe = cagr / vol if vol and vol > 0 else np.nan

    out = {
        "result": result,
        "portfolio_ret_series": port_ret_series,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
    }

    if benchmark is not None:
        vn_cagr = result["VNINDEX"].iloc[-1] ** (1 / years) - 1 if years and years > 0 else np.nan
        vn_vol = benchmark.std() * np.sqrt(252)
        vn_sharpe = vn_cagr / vn_vol if vn_vol and vn_vol > 0 else np.nan
        out.update(
            {
                "vn_cagr": vn_cagr,
                "vn_vol": vn_vol,
                "vn_sharpe": vn_sharpe,
            }
        )

    return out


def make_price_overlay(d: pd.DataFrame):
    d = d.copy()
    d["MA20"] = d["close"].rolling(20).mean()
    d["MA50"] = d["close"].rolling(50).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    d["RSI14"] = 100 - (100 / (1 + rs))

    d["BUY_overlay"] = (d["RSI14"] < 35) & (d["close"] > d["MA20"])
    d["SELL_overlay"] = (d["RSI14"] > 65) & (d["close"] < d["MA20"])
    return d


# =========================================================
# HELPERS: LONG FEATURE ENGINE (LIVE SCORES)
# =========================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def compute_bbands(series, window=20):
    ma = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return ma + 2 * std, ma - 2 * std


def classify_regime_value(r, ma20, ma50):
    if pd.isna(r):
        return "UNKNOWN"
    if r < -0.10:
        return "BEAR"
    if (r > 0.12) and (ma20 > ma50):
        return "BULL_STRONG"
    if r > 0.03:
        return "BULL"
    return "SIDEWAY"


def smooth_regime(regimes, win=10):
    smooth = []
    for i in range(len(regimes)):
        window = regimes[max(0, i - win + 1) : i + 1]
        mode_val = pd.Series(window).mode()
        if len(mode_val) > 0:
            smooth.append(mode_val.iloc[0])
        else:
            smooth.append("SIDEWAY")
    return smooth


def detect_fractals(temp):
    lows = temp["low"].values
    highs = temp["high"].values

    if "volume" in temp.columns and temp["volume"].notna().any():
        vol = temp["volume"].fillna(0).values
    else:
        vol = temp["RET"].abs().fillna(0).values

    support_points = []
    resistance_points = []
    vol_mean_20 = pd.Series(vol).rolling(20, min_periods=10).mean().values

    for i in range(2, len(temp) - 2):
        if (
            lows[i] < lows[i - 1]
            and lows[i] < lows[i - 2]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i + 2]
        ):
            if i < len(vol_mean_20) and vol[i] >= vol_mean_20[i]:
                support_points.append(lows[i])

        if (
            highs[i] > highs[i - 1]
            and highs[i] > highs[i - 2]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i + 2]
        ):
            if i < len(vol_mean_20) and vol[i] >= vol_mean_20[i]:
                resistance_points.append(highs[i])

    return np.array(support_points), np.array(resistance_points)


def cluster_levels(levels, max_k=4):
    levels = np.array(levels)
    if len(levels) == 0:
        return []

    levels = levels.reshape(-1, 1)
    k = min(max_k, len(levels))

    if k <= 1:
        return sorted(levels.flatten().tolist())

    try:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(levels)
        return sorted(model.cluster_centers_.flatten().tolist())
    except Exception:
        return sorted(levels.flatten().tolist())


def build_live_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Tái tạo gần nhất pipeline feature từ notebook LONG cho một mã."""
    temp = ohlc.copy().sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in temp.columns:
            temp[c] = temp["close"]

    temp["RET"] = temp["close"].pct_change()
    temp["RSI14"] = compute_rsi(temp["close"])
    temp["MACD"], temp["MACD_signal"], temp["MACD_hist"] = compute_macd(temp["close"])
    temp["BB_upper"], temp["BB_lower"] = compute_bbands(temp["close"])
    temp["MA20"] = temp["close"].rolling(20, min_periods=20).mean()
    temp["MA50"] = temp["close"].rolling(50, min_periods=50).mean()
    temp["MA100"] = temp["close"].rolling(100, min_periods=100).mean()
    temp["Volatility20"] = temp["RET"].rolling(20, min_periods=20).std()

    hl = temp["high"] - temp["low"]
    hc = (temp["high"] - temp["close"].shift()).abs()
    lc = (temp["low"] - temp["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    temp["ATR14"] = tr.rolling(14, min_periods=14).mean()

    temp["RollingRet60"] = temp["close"].pct_change(60)
    temp["Trend_MA20_vs_50"] = temp["MA20"] - temp["MA50"]
    temp["Trend_MA50_vs_100"] = temp["MA50"] - temp["MA100"]

    temp["Regime_raw"] = temp.apply(
        lambda row: classify_regime_value(row["RollingRet60"], row["MA20"], row["MA50"]),
        axis=1,
    )
    temp["Regime"] = smooth_regime(temp["Regime_raw"].values)

    supports, resistances = detect_fractals(temp)

    if len(supports) < 2:
        supports = temp["close"].rolling(60, min_periods=30).min().dropna().values
    if len(resistances) < 2:
        resistances = temp["close"].rolling(60, min_periods=30).max().dropna().values
    if len(supports) == 0:
        supports = np.array([temp["close"].min()])
    if len(resistances) == 0:
        resistances = np.array([temp["close"].max()])

    support_levels = cluster_levels(supports)
    resistance_levels = cluster_levels(resistances)

    dist_sup = []
    dist_res = []

    for price in temp["close"]:
        below = [s for s in support_levels if s <= price]
        above = [r for r in resistance_levels if r >= price]

        dist_sup.append((price - max(below)) / price if below else np.nan)
        dist_res.append((min(above) - price) / price if above else np.nan)

    temp["dist_support"] = pd.Series(dist_sup, index=temp.index).bfill().ffill().fillna(0)
    temp["dist_resistance"] = pd.Series(dist_res, index=temp.index).ffill().bfill().fillna(0)

    temp["MA200"] = temp["close"].rolling(200, min_periods=150).mean()
    temp["High250"] = temp["close"].rolling(250, min_periods=200).max()
    temp["Low250"] = temp["close"].rolling(250, min_periods=200).min()

    temp["Above_MA100"] = (temp["close"] > temp["MA100"]).astype(int)
    temp["Below_MA100"] = (temp["close"] < temp["MA100"]).astype(int)
    temp["MajorBreakout"] = (temp["close"] >= temp["High250"]).astype(int)
    temp["MajorBreakdown"] = (temp["close"] <= temp["Low250"]).astype(int)

    temp["MotherBull"] = (
        ((temp["Above_MA100"] == 1) & (temp["Regime"].isin(["BULL", "BULL_STRONG"])))
        | (temp["MajorBreakout"] == 1)
    ).astype(int)

    temp["MotherBear"] = (
        ((temp["Below_MA100"] == 1) & (temp["Regime"] == "BEAR"))
        | (temp["MajorBreakdown"] == 1)
    ).astype(int)

    temp = temp.dropna().copy()
    return temp


def score_long_ticker(ticker: str, history_days: int = 420, source: str = "KBS"):
    if long_model is None or feature_cols is None:
        return None

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=max(600, history_days))

    ohlc = fetch_ohlc_history(
        ticker=ticker,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        source=source,
    )
    if ohlc is None or ohlc.empty:
        return None

    feat = build_live_features(ohlc)
    if feat.empty:
        return None

    latest = feat.iloc[-1:].copy()
    for c in feature_cols:
        if c not in latest.columns:
            latest[c] = np.nan

    latest = latest.dropna(subset=feature_cols)
    if latest.empty:
        return None

    prob = float(long_model.predict_proba(latest[feature_cols])[:, 1][0])

    if prob >= 0.80:
        signal = "MUA MẠNH"
    elif prob >= 0.60:
        signal = "CHỜ MUA"
    elif prob >= 0.40:
        signal = "GIỮ"
    else:
        signal = "THẬN TRỌNG"

    row = latest.iloc[0].to_dict()
    row["ticker"] = ticker
    row["long_probability"] = prob
    row["signal"] = signal
    row["history_rows"] = len(feat)

    return row, feat


# =========================================================
# LOAD CORE DATA
# =========================================================
with st.spinner("Đang tải dữ liệu lõi..."):
    missing_required = check_required_files()
    if missing_required:
        st.error("Thiếu file bắt buộc:")
        st.code("\n".join(missing_required))
        st.stop()

    price_wide_base = load_price_wide()

if "price_wide" not in st.session_state:
    st.session_state.price_wide = price_wide_base.copy()
    st.session_state.returns_wide = build_returns(st.session_state.price_wide)

st.session_state.setdefault("extras_loaded", False)
st.session_state.setdefault("live_tickers_selected", [])

price_wide = st.session_state.price_wide
returns_wide = st.session_state.returns_wide

ticker_list = [c for c in price_wide.columns if c != "VNINDEX"]
date_min = price_wide.index.min()
date_max = price_wide.index.max()

long_model = load_pickle_required("models/long_model.pkl")
feature_cols = load_pickle_required("models/feature_cols.pkl")
metadata = load_json_required("metadata/metadata.json")

df_ml = load_csv_optional("data/ml_df.csv", allow_remote=st.session_state.extras_loaded)
df_wf = load_csv_optional("data/walk_forward_df.csv", allow_remote=st.session_state.extras_loaded)
df_oos = load_csv_optional("predictions/wf_oos_predictions.csv", allow_remote=st.session_state.extras_loaded)
df_latest_top = load_csv_optional("predictions/latest_top30_predictions.csv", allow_remote=st.session_state.extras_loaded)
df_decile = load_csv_optional("results/oos_decile_performance.csv", allow_remote=st.session_state.extras_loaded)
df_wf_summary = load_csv_optional("results/walk_forward_summary.csv", allow_remote=st.session_state.extras_loaded)
df_importance = load_csv_optional("results/long_model_feature_importance.csv", allow_remote=st.session_state.extras_loaded)
df_corr = load_csv_optional("results/feature_future_return_correlation.csv", allow_remote=st.session_state.extras_loaded)
df_backtest = load_csv_optional("backtest/oos_topk_backtest_vs_vnindex.csv", allow_remote=st.session_state.extras_loaded)
df_regime_backtest = load_csv_optional("backtest/regime_filtered_oos_topk_backtest_vs_vnindex.csv", allow_remote=st.session_state.extras_loaded)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Điều khiển")

if VNSTOCK_API_KEY:
    st.sidebar.success("vnstock API key đã được cấu hình")
else:
    st.sidebar.warning("Chưa cấu hình VNSTOCK_API_KEY")

if VNSTOCK_AUTH_OK:
    st.sidebar.success("vnstock đã đăng ký tài khoản thành công")
else:
    st.sidebar.info("Đang dùng chế độ truy cập thông thường hoặc đăng ký chưa thành công")

if st.sidebar.button("Tải dữ liệu phụ từ Drive"):
    with st.spinner("Đang tải dữ liệu phụ..."):
        root = sync_optional_folder()
    if root is None:
        st.sidebar.error("Không tải được dữ liệu phụ.")
    else:
        st.session_state.extras_loaded = True
        st.sidebar.success("Đã tải dữ liệu phụ. Hãy reload trang nếu cần.")
        st.rerun()

if st.session_state.get("extras_loaded", False):
    st.sidebar.success("Dữ liệu phụ đã được nạp trong phiên hiện tại")
else:
    st.sidebar.info("Dữ liệu phụ chưa nạp")

selected_ticker = st.sidebar.selectbox(
    "Chọn mã cổ phiếu",
    ticker_list,
    index=0 if ticker_list else None,
)

risk_score = st.sidebar.slider("Mức chịu rủi ro", 1, 10, 7, 1)
target_return = st.sidebar.slider("Lợi nhuận kỳ vọng / năm", 0.05, 0.30, 0.18, 0.01)
horizon_month = st.sidebar.slider("Thời gian đầu tư (tháng)", 3, 60, 12, 1)
capital = st.sidebar.number_input("Vốn đầu tư (VND)", min_value=1_000_000, value=100_000_000, step=1_000_000)

vol_quantile = st.sidebar.slider("Ngưỡng volatility (quantile)", 0.50, 0.99, 0.90, 0.01)
min_price = st.sidebar.number_input("Giá tối thiểu (VND)", min_value=0, value=5000, step=500)

show_overlay = st.sidebar.checkbox("Hiển thị BUY/SELL overlay", value=True)

update_mode = st.sidebar.radio(
    "Cập nhật giá bằng vnstock",
    ["Mã đang chọn"],
)

update_benchmark = st.sidebar.checkbox("Cập nhật thêm VNINDEX", value=True)
active_source = st.sidebar.selectbox("Nguồn vnstock", ["KBS", "VCI"], index=0 if VNSTOCK_PRIMARY_SOURCE == "KBS" else 1)

history_days_input = st.sidebar.number_input(
    "Số phiên lịch sử để chấm LONG",
    min_value=260,
    max_value=1200,
    value=420,
    step=20,
)

refresh_btn = st.sidebar.button("Cập nhật giá hôm nay", type="primary")

st.sidebar.caption(
    "Cập nhật qua vnstock chỉ lưu trong phiên hiện tại. Muốn lưu vĩnh viễn thì cần job đồng bộ riêng."
)

# =========================================================
# LIVE UPDATE ACTION
# =========================================================
if refresh_btn:
    tickers_to_update = [selected_ticker]
    if update_benchmark:
        tickers_to_update = tickers_to_update + ["VNINDEX"]

    progress = st.sidebar.progress(0)
    updates = []
    # clear cached history before refreshing
    fetch_ohlc_history.clear()
    fetch_latest_close_vnstock.clear()

    for i, tk in enumerate(tickers_to_update, start=1):
        rec = fetch_latest_close_vnstock(tk, source=active_source)
        if rec is not None:
            updates.append(rec)
        progress.progress(i / len(tickers_to_update))

    if updates:
        updates_df = pd.DataFrame(updates)
        st.session_state.price_wide = merge_live_updates_into_price_wide(
            st.session_state.price_wide, updates_df
        )
        st.session_state.returns_wide = build_returns(st.session_state.price_wide)
        price_wide = st.session_state.price_wide
        returns_wide = st.session_state.returns_wide
        st.sidebar.success(f"Đã cập nhật {len(updates_df)} mã trong phiên hiện tại.")
    else:
        st.sidebar.error("Không lấy được giá mới từ vnstock.")

price_wide = st.session_state.price_wide
returns_wide = st.session_state.returns_wide

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Tổng quan dữ liệu",
        "Tạo danh mục mẫu",
        "Backtest danh mục",
        "Cập nhật giá",
        "Mô hình LONG",
        "Cổ phiếu chi tiết",
    ]
)

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    st.subheader("Tổng quan dữ liệu")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số ngày", f"{len(price_wide):,}")
    c2.metric("Số mã", f"{len(ticker_list):,}")
    c3.metric("Từ ngày", f"{date_min.date()}")
    c4.metric("Đến ngày", f"{date_max.date()}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Price matrix", f"{price_wide.shape[0]:,} x {price_wide.shape[1]:,}")
    c6.metric("Returns matrix", f"{returns_wide.shape[0]:,} x {returns_wide.shape[1]:,}")
    c7.metric("Benchmark VNINDEX", "Có" if "VNINDEX" in price_wide.columns else "Không")
    c8.metric("Model LONG", "Có" if long_model is not None else "Không")

    st.write("### Bảng giá mẫu")
    st.dataframe(price_wide.reset_index().tail(10), width="stretch")

    st.write("### Trạng thái file đã nạp")
    rows = []
    for fn in REQUIRED_FILES:
        rows.append({"file": fn, "status": "Đã nạp" if resolve_required_artifact(fn) is not None else "Thiếu"})
    for fn in OPTIONAL_FILES:
        rows.append(
            {
                "file": fn,
                "status": "Đã nạp" if resolve_optional_artifact(fn, allow_remote=st.session_state.extras_loaded) is not None else "Chưa nạp / tùy chọn",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")

    if metadata:
        st.write("### Metadata mô hình")
        st.json(metadata)

# =========================================================
# TAB 2 — PORTFOLIO CONSTRUCTION
# =========================================================
with tab2:
    st.subheader("Tạo danh mục mẫu cho nhà đầu tư")

    universe = build_universe(price_wide, returns_wide)
    profile, filtered, growth_pool, defensive_pool, portfolio, stats = construct_portfolio(
        universe=universe,
        risk_score=risk_score,
        target_return=target_return,
        horizon_month=horizon_month,
        capital=capital,
        vol_quantile=vol_quantile,
        min_price=min_price,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nhóm nhà đầu tư", stats.get("profile", "N/A"))
    c2.metric("Tỷ trọng tăng trưởng", f"{stats.get('growth_weight', 0):.0%}")
    c3.metric("Tỷ trọng phòng thủ", f"{stats.get('defensive_weight', 0):.0%}")
    c4.metric("Số mã mục tiêu", f"{stats.get('n_stocks', 0)}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Vol cutoff", f"{stats.get('vol_cutoff', np.nan):.4f}")
    c6.metric("Lợi nhuận kỳ vọng", f"{stats.get('expected_return', np.nan):.2%}" if pd.notna(stats.get("expected_return", np.nan)) else "N/A")
    c7.metric("Rủi ro kỳ vọng", f"{stats.get('expected_risk', np.nan):.2%}" if pd.notna(stats.get("expected_risk", np.nan)) else "N/A")
    c8.metric("Return/Risk", f"{stats.get('return_risk', np.nan):.2f}" if pd.notna(stats.get("return_risk", np.nan)) else "N/A")

    st.write(f"**Số mã đủ điều kiện:** {len(filtered):,}")
    if not filtered.empty:
        st.write("### Universe sau khi lọc")
        st.dataframe(filtered.sort_values("mean_return", ascending=False).head(20), width="stretch")

    if not growth_pool.empty and not defensive_pool.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("#### Nhóm tăng trưởng")
            st.dataframe(growth_pool.head(15), width="stretch")
        with col_b:
            st.write("#### Nhóm phòng thủ")
            st.dataframe(defensive_pool.head(15), width="stretch")

    st.write("### Danh mục đề xuất")
    if portfolio is None or portfolio.empty:
        st.warning("Danh mục rỗng sau khi lọc. Hãy nới ngưỡng volatility hoặc giảm giá tối thiểu.")
    else:
        show_port = portfolio.rename_axis("ticker").reset_index()
        show_port = show_port[["ticker", "mean_return", "volatility", "weight", "allocation_vnd", "shares"]].sort_values("weight", ascending=False)
        st.dataframe(show_port, width="stretch")

        fig = px.bar(show_port, x="ticker", y="weight", title="Trọng số danh mục", labels={"ticker": "Mã", "weight": "Trọng số"})
        st.plotly_chart(fig, width="stretch")

# =========================================================
# TAB 3 — BACKTEST
# =========================================================
with tab3:
    st.subheader("Backtest danh mục so với VNINDEX")

    universe = build_universe(price_wide, returns_wide)
    _, filtered, growth_pool, defensive_pool, portfolio, stats = construct_portfolio(
        universe=universe,
        risk_score=risk_score,
        target_return=target_return,
        horizon_month=horizon_month,
        capital=capital,
        vol_quantile=vol_quantile,
        min_price=min_price,
    )

    if portfolio is None or portfolio.empty:
        st.warning("Không thể backtest vì danh mục đang rỗng.")
    else:
        bt = backtest_portfolio(price_wide, returns_wide, portfolio)
        result = bt["result"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.index, y=result["Portfolio"], name="Danh mục", line=dict(width=2)))
        if "VNINDEX" in result.columns:
            fig.add_trace(go.Scatter(x=result.index, y=result["VNINDEX"], name="VNINDEX", line=dict(width=2)))

        fig.update_layout(height=600, xaxis_title="Ngày", yaxis_title="NAV", legend_title="Đường cong")
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3 = st.columns(3)
        c1.metric("CAGR danh mục", f"{bt['cagr']:.2%}" if pd.notna(bt["cagr"]) else "N/A")
        c2.metric("Volatility", f"{bt['vol']:.2%}" if pd.notna(bt["vol"]) else "N/A")
        c3.metric("Sharpe-like", f"{bt['sharpe']:.2f}" if pd.notna(bt["sharpe"]) else "N/A")

        if "vn_cagr" in bt:
            c4, c5, c6 = st.columns(3)
            c4.metric("CAGR VNINDEX", f"{bt['vn_cagr']:.2%}" if pd.notna(bt["vn_cagr"]) else "N/A")
            c5.metric("Vol VNINDEX", f"{bt['vn_vol']:.2%}" if pd.notna(bt["vn_vol"]) else "N/A")
            c6.metric("Sharpe VNINDEX", f"{bt['vn_sharpe']:.2f}" if pd.notna(bt["vn_sharpe"]) else "N/A")

        st.write("### Dữ liệu backtest")
        st.dataframe(result.tail(20), width="stretch")

# =========================================================
# TAB 4 — LIVE UPDATE
# =========================================================
with tab4:
    st.subheader("Cập nhật giá mỗi ngày bằng vnstock")

    st.write(f"Ưu tiên nguồn: **{active_source}**, fallback: **{VNSTOCK_FALLBACK_SOURCE}**.")
    if VNSTOCK_API_KEY:
        st.caption("vnstock API key đã được nạp để giảm giới hạn request.")

    st.write("### Giá mới nhất của mã đang chọn")
    latest_one = fetch_latest_close_vnstock(selected_ticker, source=active_source)
    if latest_one is None:
        st.warning("Không lấy được giá mới nhất cho mã đang chọn.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mã", latest_one["ticker"])
        c2.metric("Giá đóng cửa mới nhất", f"{latest_one['close']:.2f}")
        c3.metric("Nguồn", latest_one["source"])
        st.json({"ticker": latest_one["ticker"], "date": str(latest_one["date"]), "close": latest_one["close"], "source": latest_one["source"]})

    st.write("### Cập nhật hàng loạt trong phiên")
    st.info("Nút cập nhật ở sidebar sẽ kéo giá hôm nay cho mã đang chọn và VNINDEX. Toàn bộ giá mới chỉ lưu trong phiên hiện tại.")

    st.write("### Bảng cập nhật gần nhất trong phiên")
    st.dataframe(st.session_state.price_wide.reset_index().tail(5), width="stretch")

    st.caption("Muốn lưu vĩnh viễn dữ liệu cập nhật thì cần một job đồng bộ riêng. Streamlit Cloud chỉ giữ thay đổi trong phiên hiện tại.")

# =========================================================
# TAB 5 — LONG MODEL
# =========================================================
with tab5:
    st.subheader("Mô hình LONG đã lưu")

    if long_model is None:
        st.error("Chưa nạp được long_model.pkl.")
    else:
        st.success(f"Đã nạp mô hình: {type(long_model)}")

    if feature_cols is not None:
        st.write("### Danh sách feature")
        st.write(feature_cols)

    if metadata:
        st.write("### Metadata")
        st.json(metadata)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Walk-forward", "Có" if df_wf_summary is not None else "Thiếu")
    c2.metric("OOS decile", "Có" if df_decile is not None else "Thiếu")
    c3.metric("Backtest OOS", "Có" if df_backtest is not None else "Thiếu")
    c4.metric("LONG picks", "Có" if df_latest_top is not None else "Thiếu")

    if st.session_state.get("extras_loaded", False):
        if df_wf_summary is not None:
            st.write("### Walk-forward summary")
            st.dataframe(df_wf_summary, width="stretch")

        if df_decile is not None:
            st.write("### OOS decile performance")
            st.dataframe(df_decile, width="stretch")

        if df_backtest is not None:
            st.write("### Backtest OOS top-K vs VNINDEX")
            st.dataframe(df_backtest.tail(20), width="stretch")

        if df_regime_backtest is not None:
            st.write("### Backtest có lọc regime")
            st.dataframe(df_regime_backtest.tail(20), width="stretch")

        if df_latest_top is not None:
            st.write("### Top 30 tín hiệu LONG gần nhất")
            st.dataframe(df_latest_top, width="stretch")

        if df_importance is not None:
            st.write("### Feature importance")
            st.dataframe(df_importance.head(20), width="stretch")
            fig = px.bar(df_importance.sort_values("importance", ascending=True), x="importance", y="feature", orientation="h", title="Mức độ quan trọng của feature")
            st.plotly_chart(fig, width="stretch")

        if df_corr is not None:
            st.write("### Tương quan feature với forward return")
            st.dataframe(df_corr.head(20), width="stretch")
    else:
        st.info("Bấm nút 'Tải dữ liệu phụ từ Drive' ở sidebar nếu muốn xem các bảng kết quả/backtest đã lưu.")

    st.divider()
    st.write("### Chấm điểm LONG realtime")

    if "live_tickers_selected" not in st.session_state:
        st.session_state.live_tickers_selected = [selected_ticker]

    with st.form("long_scoring_form", clear_on_submit=False):
        mode = st.radio("Chế độ", ["Mã đang chọn", "Nhiều mã"], horizontal=True, key="long_mode_radio")

        if mode == "Mã đang chọn":
            live_tickers = [selected_ticker]
            st.info(f"Đang chấm theo mã đang chọn: {selected_ticker}")
        else:
            max_symbols = st.slider("Số lượng mã tối đa", min_value=5, max_value=20, value=10, step=5, key="long_max_symbols_slider")
            default_symbols = [t for t in st.session_state.live_tickers_selected if t in ticker_list][:max_symbols]
            if not default_symbols:
                default_symbols = ticker_list[: min(max_symbols, len(ticker_list))]

            selected_symbols = st.multiselect("Chọn mã để chấm", ticker_list, default=default_symbols, key="long_multiselect")
            live_tickers = [t for t in selected_symbols if t in ticker_list][:max_symbols]

        submitted = st.form_submit_button("Chấm điểm LONG ngay")

    if submitted:
        if long_model is None or feature_cols is None:
            st.error("Thiếu model hoặc feature_cols.")
        elif not live_tickers:
            st.warning("Chưa chọn mã nào.")
        else:
            st.session_state.live_tickers_selected = live_tickers[:]

            results = []
            prog = st.progress(0)
            status = st.empty()
            total = len(live_tickers)

            for i, tk in enumerate(live_tickers, start=1):
                try:
                    status.info(f"Đang xử lý: {tk} ({i}/{total})")
                    scored = score_long_ticker(tk, history_days=int(history_days_input), source=active_source)
                    if scored is not None:
                        row, _feat = scored
                        results.append(
                            {
                                "ticker": row["ticker"],
                                "date": row["date"],
                                "close": round(float(row["close"]), 2),
                                "long_probability": round(float(row["long_probability"]), 4),
                                "signal": row["signal"],
                                "history_rows": int(row["history_rows"]),
                            }
                        )
                except Exception as e:
                    st.warning(f"Lỗi mã {tk}: {str(e)}")
                prog.progress(i / total)

            prog.empty()
            status.empty()

            if not results:
                st.warning("Không chấm được mã nào. Có thể dữ liệu vnstock thiếu hoặc mạng quá chậm.")
            else:
                score_df = pd.DataFrame(results).sort_values("long_probability", ascending=False).reset_index(drop=True)
                st.session_state["live_long_scores"] = score_df

                st.success(f"Đã chấm {len(score_df)} mã.")
                st.write("### Bảng xếp hạng LONG realtime")
                st.dataframe(score_df, width="stretch", height=500)

                st.write("### Top 10 cơ hội LONG")
                top10 = score_df.head(10)
                st.dataframe(top10, width="stretch")

                fig = px.bar(top10, x="ticker", y="long_probability", color="signal", title="Top xác suất LONG", labels={"ticker": "Mã", "long_probability": "Xác suất LONG"})
                st.plotly_chart(fig, width="stretch")

                if selected_ticker in score_df["ticker"].values:
                    one = score_df[score_df["ticker"] == selected_ticker].iloc[0]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mã", one["ticker"])
                    c2.metric("Giá gần nhất", f"{one['close']:.2f}")
                    c3.metric("Xác suất LONG", f"{one['long_probability']:.2%}")
                    c4.metric("Tín hiệu", one["signal"])

# =========================================================
# TAB 6 — STOCK EXPLORER
# =========================================================
with tab6:
    st.subheader("Cổ phiếu chi tiết")

    start_date_detail = st.date_input("Từ ngày", value=date_min.date(), key="start_detail")
    end_date_detail = st.date_input("Đến ngày", value=date_max.date(), key="end_detail")

    d = price_wide[[selected_ticker]].copy()
    d = d.loc[pd.to_datetime(start_date_detail) : pd.to_datetime(end_date_detail)].dropna().reset_index()

    # chuẩn hoá tên cột thời gian sau reset_index()
    if "time" not in d.columns:
        if "date" in d.columns:
            d = d.rename(columns={"date": "time"})
        elif "index" in d.columns:
            d = d.rename(columns={"index": "time"})
        else:
            d = d.rename(columns={d.columns[0]: "time"})

    d = d.rename(columns={selected_ticker: "close"}).sort_values("time")

    if d.empty:
        st.warning("Không có dữ liệu trong khoảng ngày này.")
    else:
        d_overlay = make_price_overlay(d.rename(columns={"time": "date"}).copy())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giá đầu kỳ", f"{d['close'].iloc[0]:.2f}")
        c2.metric("Giá cuối kỳ", f"{d['close'].iloc[-1]:.2f}")
        c3.metric("Cao nhất", f"{d['close'].max():.2f}")
        c4.metric("Thấp nhất", f"{d['close'].min():.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d_overlay["date"], y=d_overlay["close"], name="Giá", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=d_overlay["date"], y=d_overlay["MA20"], name="MA20", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=d_overlay["date"], y=d_overlay["MA50"], name="MA50", line=dict(width=2)))

        if show_overlay:
            buy = d_overlay[d_overlay["BUY_overlay"]]
            sell = d_overlay[d_overlay["SELL_overlay"]]

            fig.add_trace(go.Scatter(
                x=buy["date"],
                y=buy["close"],
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", size=11),
            ))
            fig.add_trace(go.Scatter(
                x=sell["date"],
                y=sell["close"],
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", size=11),
            ))

        fig.update_layout(
            height=700,
            xaxis_title="Ngày",
            yaxis_title="Giá",
            xaxis_rangeslider_visible=True,
        )
        st.plotly_chart(fig, width="stretch")

        st.write("### Bảng dữ liệu")
        st.dataframe(d_overlay.sort_values("date", ascending=False), width="stretch")

        st.write("### Thống kê nhanh")
        daily_ret = d["close"].pct_change().fillna(0)
        stats = pd.DataFrame(
            {
                "Chỉ tiêu": ["Lợi nhuận TB/ngày", "Biến động/ngày", "Tổng lợi nhuận", "Số phiên"],
                "Giá trị": [
                    f"{daily_ret.mean():.4%}",
                    f"{daily_ret.std():.4%}",
                    f"{(d['close'].iloc[-1] / d['close'].iloc[0] - 1):.2%}",
                    f"{len(d):,}",
                ],
            }
        )
        st.dataframe(stats, width="stretch")
