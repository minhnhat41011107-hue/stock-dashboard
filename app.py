# app.py
# Giao diện tiếng Việt: tạo danh mục mẫu + backtest + cập nhật giá qua vnstock
# Dữ liệu model/artifact sẽ được tải từ Google Drive (public file ID) hoặc từ file local trong repo.

import os
import json
import tempfile
from pathlib import Path

import joblib
import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vnstock import Quote

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cố vấn danh mục & Long Screener",
    page_icon="📈",
    layout="wide",
)

st.title("🇻🇳 Cố vấn danh mục cổ phiếu Việt Nam")
st.caption(
    "Kết hợp danh mục mẫu cho nhà đầu tư, backtest so với VNINDEX và bộ tín hiệu LONG đã huấn luyện."
)

# =========================================================
# CONFIG
# =========================================================
def cfg(name: str, default: str = "") -> str:
    """Đọc từ Streamlit secrets trước, nếu không có thì đọc từ env var."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


ASSET_IDS = {
    # dữ liệu gốc cho danh mục mẫu
    "price_aligned_copy.csv": cfg("PRICE_ALIGNED_COPY_ID"),
    # artifact của mô hình LONG
    "long_model.pkl": cfg("LONG_MODEL_ID"),
    "feature_cols.pkl": cfg("FEATURE_COLS_ID"),
    "metadata.json": cfg("METADATA_ID"),

    # output đã lưu từ Colab
    "ml_df.csv": cfg("ML_DF_ID"),
    "walk_forward_df.csv": cfg("WALK_FORWARD_DF_ID"),
    "wf_oos_predictions.csv": cfg("WF_OOS_PRED_ID"),
    "latest_top30_predictions.csv": cfg("LATEST_TOP30_ID"),
    "oos_decile_performance.csv": cfg("DECILE_PERF_ID"),
    "walk_forward_summary.csv": cfg("WF_SUMMARY_ID"),
    "long_model_feature_importance.csv": cfg("FEATURE_IMPORTANCE_ID"),
    "feature_future_return_correlation.csv": cfg("FEATURE_CORR_ID"),
    "oos_topk_backtest_vs_vnindex.csv": cfg("BACKTEST_ID"),
    "regime_filtered_oos_topk_backtest_vs_vnindex.csv": cfg("REGIME_BACKTEST_ID"),
}

CACHE_DIR = Path(tempfile.gettempdir()) / "khoaluan_assets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VNSTOCK_PRIMARY_SOURCE = cfg("VNSTOCK_SOURCE", "KBS")
VNSTOCK_FALLBACK_SOURCE = "VCI" if VNSTOCK_PRIMARY_SOURCE == "KBS" else "KBS"

# =========================================================
# HELPERS: DOWNLOAD / LOAD ARTIFACTS
# =========================================================
def download_from_drive(file_id: str, filename: str) -> Path | None:
    if not file_id:
        return None
    out_path = CACHE_DIR / filename
    if out_path.exists():
        return out_path
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(out_path), quiet=True)
    return out_path if out_path.exists() else None


def resolve_artifact(filename: str) -> Path | None:
    """Ưu tiên file local/repo, sau đó mới tải từ Drive."""
    candidates = [
        Path(filename),
        Path("data") / filename,
        Path("models") / filename,
        Path("results") / filename,
        Path("predictions") / filename,
        Path("backtest") / filename,
        Path("metadata") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    file_id = ASSET_IDS.get(filename, "")
    return download_from_drive(file_id, filename)


@st.cache_data(show_spinner=False)
def load_csv_asset(filename: str) -> pd.DataFrame | None:
    path = resolve_artifact(filename)
    if path is None:
        return None
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def load_json_asset(filename: str) -> dict | None:
    path = resolve_artifact(filename)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_pickle_asset(filename: str):
    path = resolve_artifact(filename)
    if path is None:
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_price_wide() -> pd.DataFrame:
    """
    File price_aligned_copy.csv là wide:
    time, ticker1, ticker2, ..., VNINDEX
    """
    path = resolve_artifact("price_aligned_copy.csv")
    if path is None:
        raise FileNotFoundError(
            "Không tìm thấy price_aligned_copy.csv. Hãy tạo public file ID hoặc để file trong repo."
        )

    df = pd.read_csv(path, low_memory=False)
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
def fetch_latest_close_vnstock(ticker: str) -> dict | None:
    """
    Kéo giá daily mới nhất bằng vnstock.
    Ưu tiên source KBS, nếu lỗi thì fallback VCI.
    """
    for source in [VNSTOCK_PRIMARY_SOURCE, VNSTOCK_FALLBACK_SOURCE]:
        try:
            q = Quote(symbol=ticker, source=source)
            hist = q.history(length="5", interval="d")

            if hist is None or hist.empty:
                continue

            hist = hist.copy()
            hist.columns = [str(c).strip().lower() for c in hist.columns]

            date_col = "time" if "time" in hist.columns else "date"
            close_col = "close" if "close" in hist.columns else None
            if close_col is None:
                continue

            last = hist.iloc[-1]
            return {
                "ticker": ticker,
                "date": pd.to_datetime(last[date_col], errors="coerce"),
                "close": float(last[close_col]),
                "source": source,
            }
        except Exception:
            continue
    return None


def merge_live_updates_into_price_wide(
    price_wide: pd.DataFrame, updates: pd.DataFrame
) -> pd.DataFrame:
    """
    Cập nhật giá cuối ngày vào matrix wide.
    Nếu ngày mới chưa có thì append row mới.
    """
    if updates is None or updates.empty:
        return price_wide

    out = price_wide.copy()
    updates = updates.dropna(subset=["date", "ticker", "close"]).copy()
    updates["date"] = pd.to_datetime(updates["date"], errors="coerce")
    updates = updates.dropna(subset=["date"])

    latest_date = updates["date"].max()

    # đảm bảo cột ticker tồn tại
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
    return out


# =========================================================
# HELPERS: PORTFOLIO ENGINE (TỪ NOTEBOOK "DANH MỤC TEST")
# =========================================================
def build_universe(price_wide: pd.DataFrame, returns_wide: pd.DataFrame) -> pd.DataFrame:
    market_col = "VNINDEX" if "VNINDEX" in price_wide.columns else None
    stock_cols = [c for c in price_wide.columns if c != market_col]

    TRADING_DAYS = 252
    mu = returns_wide[stock_cols].mean() * TRADING_DAYS
    vol = returns_wide[stock_cols].std() * np.sqrt(TRADING_DAYS)

    # giữ đúng logic notebook: price * 1000
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

    if filtered.empty:
        return profile, filtered, pd.DataFrame(), pd.DataFrame(), {
            "vol_cutoff": vol_cutoff,
            "profile": profile,
            "growth_weight": growth_weight,
            "defensive_weight": defensive_weight,
            "n_stocks": n_stocks,
            "max_weight": max_weight,
        }

    growth_pool = filtered.sort_values("mean_return", ascending=False).head(30).copy()
    defensive_pool = filtered.sort_values("volatility", ascending=True).head(30).copy()

    n_growth = n_stocks // 2
    n_def = n_stocks - n_growth

    growth_pool["score"] = growth_pool["mean_return"] / growth_pool["volatility"]
    defensive_pool["score"] = 1 / defensive_pool["volatility"]

    growth_pick = growth_pool.sort_values("score", ascending=False).head(n_growth)
    defensive_pick = defensive_pool.sort_values("score", ascending=False).head(n_def)

    portfolio = pd.concat([growth_pick, defensive_pick])
    portfolio = portfolio[~portfolio.index.duplicated()].copy()

    if portfolio.empty:
        return profile, filtered, growth_pool, defensive_pool, {
            "vol_cutoff": vol_cutoff,
            "profile": profile,
            "growth_weight": growth_weight,
            "defensive_weight": defensive_weight,
            "n_stocks": n_stocks,
            "max_weight": max_weight,
        }

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


def backtest_portfolio(
    price_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    portfolio: pd.DataFrame,
):
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


def make_price_overlay(df: pd.DataFrame):
    d = df.copy()
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
# LOAD SESSION DATA
# =========================================================
with st.spinner("Đang tải dữ liệu gốc từ Drive / repo..."):
    price_wide_base = load_price_wide()

if "price_wide" not in st.session_state:
    st.session_state.price_wide = price_wide_base.copy()
    st.session_state.returns_wide = build_returns(st.session_state.price_wide)

price_wide = st.session_state.price_wide
returns_wide = st.session_state.returns_wide

ticker_list = [c for c in price_wide.columns if c != "VNINDEX"]
date_min = price_wide.index.min()
date_max = price_wide.index.max()

# =========================================================
# LOAD SAVED MODEL ARTIFACTS
# =========================================================
long_model = load_pickle_asset("long_model.pkl")
feature_cols = load_pickle_asset("feature_cols.pkl")
metadata = load_json_asset("metadata.json")

df_ml = load_csv_asset("ml_df.csv")
df_wf = load_csv_asset("walk_forward_df.csv")
df_oos = load_csv_asset("wf_oos_predictions.csv")
df_latest_top = load_csv_asset("latest_top30_predictions.csv")
df_decile = load_csv_asset("oos_decile_performance.csv")
df_wf_summary = load_csv_asset("walk_forward_summary.csv")
df_importance = load_csv_asset("long_model_feature_importance.csv")
df_corr = load_csv_asset("feature_future_return_correlation.csv")
df_backtest = load_csv_asset("oos_topk_backtest_vs_vnindex.csv")
df_regime_backtest = load_csv_asset("regime_filtered_oos_topk_backtest_vs_vnindex.csv")

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("Điều khiển")

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

update_mode = st.sidebar.radio(
    "Cập nhật giá bằng vnstock",
    ["Mã đang chọn", "Toàn bộ danh mục"],
)

source_choice = st.sidebar.selectbox(
    "Nguồn vnstock",
    [VNSTOCK_PRIMARY_SOURCE, VNSTOCK_FALLBACK_SOURCE],
    index=0,
)

refresh_btn = st.sidebar.button("Cập nhật giá hôm nay", type="primary")

st.sidebar.caption(
    "Lưu ý: cập nhật bằng vnstock chỉ lưu trong phiên hiện tại của Streamlit Cloud. "
    "Muốn lưu vĩnh viễn thì cần job đồng bộ riêng."
)

# =========================================================
# LIVE UPDATE ACTION
# =========================================================
if refresh_btn:
    tickers_to_update = [selected_ticker] if update_mode == "Mã đang chọn" else ticker_list

    if len(tickers_to_update) == 0:
        st.sidebar.warning("Không có mã để cập nhật.")
    else:
        progress = st.sidebar.progress(0)
        updates = []
        for i, tk in enumerate(tickers_to_update, start=1):
            rec = fetch_latest_close_vnstock(tk)
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

# refresh local refs after possible update
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
    st.dataframe(price_wide.reset_index().tail(10), use_container_width=True)

    if metadata:
        st.write("### Metadata mô hình")
        st.json(metadata)

    st.write("### Trạng thái file đã nạp từ Drive")
    loaded_files = []
    for fn in ASSET_IDS.keys():
        if resolve_artifact(fn) is not None:
            loaded_files.append({"file": fn, "status": "Đã nạp"})
        else:
            loaded_files.append({"file": fn, "status": "Thiếu / chưa cấu hình ID"})
    st.dataframe(pd.DataFrame(loaded_files), use_container_width=True)

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
    c6.metric("Lợi nhuận kỳ vọng", f"{stats.get('expected_return', np.nan):.2%}")
    c7.metric("Rủi ro kỳ vọng", f"{stats.get('expected_risk', np.nan):.2%}")
    c8.metric("Return/Risk", f"{stats.get('return_risk', np.nan):.2f}")

    st.write("### Universe sau khi lọc")
    st.write(f"Số mã đủ điều kiện: **{len(filtered):,}**")
    if not filtered.empty:
        st.dataframe(
            filtered.sort_values("mean_return", ascending=False).head(20),
            use_container_width=True,
        )

    if not growth_pool.empty and not defensive_pool.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("#### Nhóm tăng trưởng")
            st.dataframe(growth_pool.head(15), use_container_width=True)
        with col_b:
            st.write("#### Nhóm phòng thủ")
            st.dataframe(defensive_pool.head(15), use_container_width=True)

    st.write("### Danh mục đề xuất")
    if portfolio is None or portfolio.empty:
        st.warning("Danh mục rỗng sau khi lọc. Hãy nới ngưỡng volatility hoặc giảm giá tối thiểu.")
    else:
        show_port = portfolio.copy()
        show_port = show_port[[
            "mean_return", "volatility", "weight", "allocation_vnd", "shares"
        ]].sort_values("weight", ascending=False)

        st.dataframe(show_port, use_container_width=True)

        fig = px.bar(
            portfolio.reset_index(),
            x="index",
            y="weight",
            title="Trọng số danh mục",
            labels={"index": "Mã", "weight": "Trọng số"},
        )
        st.plotly_chart(fig, use_container_width=True)

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
        fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result["Portfolio"],
                name="Danh mục",
                line=dict(width=2),
            )
        )

        if "VNINDEX" in result.columns:
            fig.add_trace(
                go.Scatter(
                    x=result.index,
                    y=result["VNINDEX"],
                    name="VNINDEX",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            height=600,
            xaxis_title="Ngày",
            yaxis_title="NAV",
            legend_title="Đường cong",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("CAGR danh mục", f"{bt['cagr']:.2%}")
        c2.metric("Volatility", f"{bt['vol']:.2%}")
        c3.metric("Sharpe-like", f"{bt['sharpe']:.2f}")

        if "vn_cagr" in bt:
            c4, c5, c6 = st.columns(3)
            c4.metric("CAGR VNINDEX", f"{bt['vn_cagr']:.2%}")
            c5.metric("Vol VNINDEX", f"{bt['vn_vol']:.2%}")
            c6.metric("Sharpe VNINDEX", f"{bt['vn_sharpe']:.2f}")

        st.write("### Dữ liệu backtest")
        st.dataframe(result.tail(20), use_container_width=True)

# =========================================================
# TAB 4 — LIVE UPDATE VIA VNSTOCK
# =========================================================
with tab4:
    st.subheader("Cập nhật giá mỗi ngày bằng vnstock")

    st.write(
        f"Ưu tiên nguồn: **{VNSTOCK_PRIMARY_SOURCE}**. "
        f"Nếu lỗi, tự fallback sang **{VNSTOCK_FALLBACK_SOURCE}**."
    )

    st.write("### Giá mới nhất của mã đang chọn")
    latest_one = fetch_latest_close_vnstock(selected_ticker)
    if latest_one is None:
        st.warning("Không lấy được giá mới nhất cho mã đang chọn.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mã", latest_one["ticker"])
        c2.metric("Giá đóng cửa mới nhất", f'{latest_one["close"]:.2f}')
        c3.metric("Nguồn", latest_one["source"])
        st.write(latest_one)

    st.write("### Cập nhật hàng loạt trong phiên")
    st.info(
        "Nút cập nhật ở sidebar sẽ kéo giá hôm nay cho mã đang chọn hoặc toàn bộ danh mục "
        "và cập nhật ngay trong phiên hiện tại."
    )

    st.write("### Bảng cập nhật gần nhất trong phiên")
    if "price_wide" in st.session_state:
        st.dataframe(st.session_state.price_wide.reset_index().tail(5), use_container_width=True)

    st.caption(
        "Nếu muốn lưu vĩnh viễn dữ liệu cập nhật, cần một job riêng để đồng bộ ngược vào Drive hoặc GitHub. "
        "Streamlit Cloud chỉ giữ thay đổi trong phiên hiện tại."
    )

# =========================================================
# TAB 5 — LONG MODEL ARTIFACTS
# =========================================================
with tab5:
    st.subheader("Mô hình LONG đã lưu")

    if long_model is None:
        st.warning("Chưa nạp được long_model.pkl. Hãy cấu hình LONG_MODEL_ID hoặc đặt file trong repo.")
    else:
        st.success(f"Đã nạp mô hình: {type(long_model)}")

    if feature_cols is not None:
        st.write("### Danh sách feature")
        st.write(feature_cols)

    if metadata:
        st.write("### Metadata")
        st.json(metadata)

    st.write("### Walk-forward summary")
    if df_wf_summary is not None:
        st.dataframe(df_wf_summary, use_container_width=True)
    else:
        st.info("Chưa có walk_forward_summary.csv.")

    st.write("### OOS decile performance")
    if df_decile is not None:
        st.dataframe(df_decile, use_container_width=True)
    else:
        st.info("Chưa có oos_decile_performance.csv.")

    st.write("### Backtest OOS top-K vs VNINDEX")
    if df_backtest is not None:
        st.dataframe(df_backtest.tail(20), use_container_width=True)
    else:
        st.info("Chưa có oos_topk_backtest_vs_vnindex.csv.")

    st.write("### Top 30 tín hiệu LONG gần nhất")
    if df_latest_top is not None:
        st.dataframe(df_latest_top, use_container_width=True)
    else:
        st.info("Chưa có latest_top30_predictions.csv.")

    st.write("### Feature importance")
    if df_importance is not None:
        st.dataframe(df_importance.head(20), use_container_width=True)
        fig = px.bar(
            df_importance.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Mức độ quan trọng của feature",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chưa có long_model_feature_importance.csv.")

# =========================================================
# TAB 6 — STOCK EXPLORER
# =========================================================
with tab6:
    st.subheader("Cổ phiếu chi tiết")

    start_date = st.date_input("Từ ngày", value=date_min.date(), key="start_detail")
    end_date = st.date_input("Đến ngày", value=date_max.date(), key="end_detail")

    d = price_wide[[selected_ticker]].copy()
    d = d.loc[pd.to_datetime(start_date): pd.to_datetime(end_date)].dropna().reset_index()
    d = d.rename(columns={selected_ticker: "close"})
    d = d.sort_values("time")

    if d.empty:
        st.warning("Không có dữ liệu trong khoảng ngày này.")
    else:
        d_overlay = make_price_overlay(d.rename(columns={"time": "date"}).assign(date=d["time"]))
        # d_overlay có cột date, close, MA20, MA50, RSI14, BUY_overlay, SELL_overlay

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giá đầu kỳ", f"{d['close'].iloc[0]:.2f}")
        c2.metric("Giá cuối kỳ", f"{d['close'].iloc[-1]:.2f}")
        c3.metric("Cao nhất", f"{d['close'].max():.2f}")
        c4.metric("Thấp nhất", f"{d['close'].min():.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d_overlay["date"], y=d_overlay["close"], name="Giá", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=d_overlay["date"], y=d_overlay["MA20"], name="MA20", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=d_overlay["date"], y=d_overlay["MA50"], name="MA50", line=dict(width=2)))

        buy = d_overlay[d_overlay["BUY_overlay"]]
        sell = d_overlay[d_overlay["SELL_overlay"]]

        fig.add_trace(go.Scatter(
            x=buy["date"], y=buy["close"],
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=11)
        ))
        fig.add_trace(go.Scatter(
            x=sell["date"], y=sell["close"],
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=11)
        ))

        fig.update_layout(
            height=700,
            xaxis_title="Ngày",
            yaxis_title="Giá",
            xaxis_rangeslider_visible=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Bảng dữ liệu")
        st.dataframe(d_overlay.sort_values("date", ascending=False), use_container_width=True)

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
        st.dataframe(stats, use_container_width=True)