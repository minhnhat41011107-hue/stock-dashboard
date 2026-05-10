import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import gdown

st.set_page_config(page_title="VN Stock Dashboard", layout="wide")

# =========================
# CONFIG
# =========================
# Dán FILE_ID của data.parquet ở đây
DATA_FILE_ID = "1bInFLKm6A-Zu9lCqH2c3_svUNlbsSTw7"

# Optional: nếu muốn so sánh đúng kiểu Stage E với VNINDEX,
# hãy dán FILE_ID của file benchmark vào đây (nếu có).
BENCHMARK_FILE_ID = ""

# Nếu đang chạy local, có thể trỏ về file sẵn có
LOCAL_DATA_PATH = "data.parquet"


# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False)
def download_from_drive(file_id: str, output_name: str) -> str:
    url = f"https://drive.google.com/uc?id={file_id}"
    out_path = Path(tempfile.gettempdir()) / output_name
    gdown.download(url, str(out_path), quiet=True)
    return str(out_path)


@st.cache_data(show_spinner=False)
def load_long_data() -> pd.DataFrame:
    if os.path.exists(LOCAL_DATA_PATH):
        df = pd.read_parquet(LOCAL_DATA_PATH)
    else:
        path = download_from_drive(DATA_FILE_ID, "data.parquet")
        df = pd.read_parquet(path)

    # Accept either long format or wide format
    cols = set(df.columns)

    if {"date", "ticker", "close"}.issubset(cols):
        long_df = df[["date", "ticker", "close"]].copy()
    elif "time" in cols:
        tmp = df.rename(columns={"time": "date"}).copy()
        ticker_cols = [c for c in tmp.columns if c != "date"]
        long_df = tmp.melt(
            id_vars=["date"],
            value_vars=ticker_cols,
            var_name="ticker",
            value_name="close",
        )
    else:
        # fallback: try first column as date/time
        first_col = df.columns[0]
        tmp = df.rename(columns={first_col: "date"}).copy()
        ticker_cols = [c for c in tmp.columns if c != "date"]
        long_df = tmp.melt(
            id_vars=["date"],
            value_vars=ticker_cols,
            var_name="ticker",
            value_name="close",
        )

    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.dropna(subset=["close"])
    long_df["ticker"] = long_df["ticker"].astype(str)
    long_df = long_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return long_df


@st.cache_data(show_spinner=False)
def load_benchmark_optional(file_id: str) -> pd.Series | None:
    if not file_id.strip():
        return None

    path = download_from_drive(file_id.strip(), "benchmark.parquet")
    df = pd.read_parquet(path)

    if {"date", "ticker", "close"}.issubset(df.columns):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        # If multiple tickers exist, prefer VNINDEX; else first ticker
        if (df["ticker"] == "VNINDEX").any():
            s = df[df["ticker"] == "VNINDEX"].set_index("date")["close"].sort_index()
        else:
            t = df["ticker"].iloc[0]
            s = df[df["ticker"] == t].set_index("date")["close"].sort_index()
        return s

    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = [c for c in df.columns if c != "date"]
        if "VNINDEX" in numeric_cols:
            s = df.set_index("date")["VNINDEX"].sort_index()
            return s
        else:
            s = df.set_index("date")[numeric_cols[0]].sort_index()
            return s

    # fallback: first column as date/time
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = [c for c in df.columns if c != "date"]
    s = df.set_index("date")[numeric_cols[0]].sort_index()
    return s


def build_wide_matrices(long_df: pd.DataFrame):
    price = long_df.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return price, returns


def compute_metrics(price_wide: pd.DataFrame, returns_wide: pd.DataFrame):
    TRADING_DAYS = 252
    mu = returns_wide.mean() * TRADING_DAYS
    vol = returns_wide.std() * np.sqrt(TRADING_DAYS)
    latest_price = price_wide.iloc[-1] * 1000
    universe = pd.DataFrame(
        {"mean_return": mu, "volatility": vol, "latest_price": latest_price}
    ).sort_index()
    return universe


def make_buy_sell_overlay(d: pd.DataFrame):
    # Simple visual overlay for the stock explorer tab.
    # You can keep it ON/OFF without affecting the portfolio notebook logic.
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


# =========================
# LOAD DATA
# =========================
with st.spinner("Đang tải dữ liệu..."):
    long_df = load_long_data()

price_wide, returns_wide = build_wide_matrices(long_df)
universe = compute_metrics(price_wide, returns_wide)

date_min = long_df["date"].min()
date_max = long_df["date"].max()
ticker_list = sorted(long_df["ticker"].unique())

benchmark_series = None
if BENCHMARK_FILE_ID.strip():
    try:
        benchmark_series = load_benchmark_optional(BENCHMARK_FILE_ID)
    except Exception:
        benchmark_series = None


# =========================
# HEADER
# =========================
st.title("🇻🇳 VN Stock Dashboard")
st.caption("Bản đầy đủ theo pipeline trong notebook: Data → Universe → Profile → Portfolio → Backtest → Stock Explorer")


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Điều khiển")

selected_tab = st.sidebar.radio(
    "Chọn mục",
    [
        "Stage A — Data",
        "Stage B — Universe",
        "Stage C — Profile",
        "Stage D — Portfolio",
        "Stage E — Backtest",
        "Stage F — Stock Explorer",
    ],
)

ticker_default = ticker_list[0] if ticker_list else None
selected_ticker = st.sidebar.selectbox("Chọn cổ phiếu", ticker_list, index=0 if ticker_default else None)

start_date = st.sidebar.date_input("Từ ngày", value=date_min.date())
end_date = st.sidebar.date_input("Đến ngày", value=date_max.date())

st.sidebar.divider()
risk_score = st.sidebar.slider("Risk score", 1, 10, 7, 1)
target_return = st.sidebar.slider("Target return / year", 0.05, 0.30, 0.18, 0.01)
horizon_month = st.sidebar.slider("Horizon (months)", 3, 60, 12, 1)
capital = st.sidebar.number_input("Capital (VND)", min_value=1_000_000, value=100_000_000, step=1_000_000)

st.sidebar.divider()
vol_quantile = st.sidebar.slider("Universe volatility cutoff quantile", 0.50, 0.99, 0.90, 0.01)
min_price = st.sidebar.number_input("Min latest price (VND)", min_value=0, value=5000, step=500)

show_overlay = st.sidebar.checkbox("Show BUY/SELL overlay in stock tab", value=True)


# =========================
# STAGE A — DATA FOUNDATION
# =========================
if selected_tab == "Stage A — Data":
    st.subheader("Stage A — Data Foundation")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (long)", f"{len(long_df):,}")
    c2.metric("Tickers", f"{long_df['ticker'].nunique():,}")
    c3.metric("Date range", f"{date_min.date()} → {date_max.date()}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Price wide shape", f"{price_wide.shape[0]:,} x {price_wide.shape[1]:,}")
    c5.metric("Returns wide shape", f"{returns_wide.shape[0]:,} x {returns_wide.shape[1]:,}")
    c6.metric("Benchmark loaded", "Yes" if benchmark_series is not None else "No")

    st.write("### Sample data")
    st.dataframe(long_df.head(20), use_container_width=True)

    st.write("### Missing values")
    miss = long_df["close"].isna().sum()
    st.write(f"Missing close values: {miss:,}")

    st.write("### Price matrix preview")
    st.dataframe(price_wide.head(10), use_container_width=True)


# =========================
# STAGE B — INVESTABLE UNIVERSE
# =========================
elif selected_tab == "Stage B — Universe":
    st.subheader("Stage B — Investable Universe Filtering")

    positive_mask = universe["mean_return"] > 0
    vol_cutoff = universe["volatility"].quantile(vol_quantile)
    vol_mask = universe["volatility"] <= vol_cutoff
    price_mask = universe["latest_price"] >= min_price

    filtered = universe[positive_mask & vol_mask & price_mask].copy()

    st.write(f"**Final investable stocks:** {len(filtered)}")
    st.write(f"**Vol cutoff:** {vol_cutoff:.4f}")

    c1, c2 = st.columns(2)
    with c1:
        st.write("#### Top returns")
        st.dataframe(filtered.sort_values("mean_return", ascending=False).head(15), use_container_width=True)
    with c2:
        st.write("#### Lowest volatility")
        st.dataframe(filtered.sort_values("volatility", ascending=True).head(15), use_container_width=True)

    st.write("#### Universe scatter")
    fig = px.scatter(
        universe.reset_index(),
        x="volatility",
        y="mean_return",
        hover_data=["index", "latest_price"],
        title="Mean Return vs Volatility",
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# STAGE C — INVESTOR PROFILING
# =========================
elif selected_tab == "Stage C — Profile":
    st.subheader("Stage C — Investor Profiling Model")

    if risk_score <= 3:
        profile = "Conservative"
        growth_weight = 0.25
        defensive_weight = 0.75
        n_stocks = 8
        max_weight = 0.18
    elif risk_score <= 7:
        profile = "Balanced"
        growth_weight = 0.50
        defensive_weight = 0.50
        n_stocks = 10
        max_weight = 0.15
    else:
        profile = "Aggressive"
        growth_weight = 0.75
        defensive_weight = 0.25
        n_stocks = 12
        max_weight = 0.12

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profile", profile)
    c2.metric("Growth tilt", f"{growth_weight:.0%}")
    c3.metric("Defensive tilt", f"{defensive_weight:.0%}")
    c4.metric("Stocks", n_stocks)

    st.write(f"**Target return:** {target_return:.0%}")
    st.write(f"**Horizon:** {horizon_month} months")
    st.write(f"**Capital:** {capital:,.0f} VND")
    st.write(f"**Max weight:** {max_weight:.0%}")

    growth_pool = universe.sort_values("mean_return", ascending=False).head(30)
    defensive_pool = universe.sort_values("volatility", ascending=True).head(30)

    c5, c6 = st.columns(2)
    with c5:
        st.write("#### Growth pool")
        st.dataframe(growth_pool.head(15), use_container_width=True)
    with c6:
        st.write("#### Defensive pool")
        st.dataframe(defensive_pool.head(15), use_container_width=True)


# =========================
# STAGE D — PORTFOLIO CONSTRUCTION
# =========================
elif selected_tab == "Stage D — Portfolio":
    st.subheader("Stage D — Portfolio Construction Engine")

    # Rebuild the profile logic
    if risk_score <= 3:
        profile = "Conservative"
        growth_weight = 0.25
        defensive_weight = 0.75
        n_stocks = 8
        max_weight = 0.18
    elif risk_score <= 7:
        profile = "Balanced"
        growth_weight = 0.50
        defensive_weight = 0.50
        n_stocks = 10
        max_weight = 0.15
    else:
        profile = "Aggressive"
        growth_weight = 0.75
        defensive_weight = 0.25
        n_stocks = 12
        max_weight = 0.12

    growth_pool = universe.sort_values("mean_return", ascending=False).head(30).copy()
    defensive_pool = universe.sort_values("volatility", ascending=True).head(30).copy()

    n_growth = n_stocks // 2
    n_def = n_stocks - n_growth

    growth_pool["score"] = growth_pool["mean_return"] / growth_pool["volatility"]
    defensive_pool["score"] = 1 / defensive_pool["volatility"]

    growth_pick = growth_pool.sort_values("score", ascending=False).head(n_growth)
    defensive_pick = defensive_pool.sort_values("score", ascending=False).head(n_def)

    portfolio = pd.concat([growth_pick, defensive_pick])
    portfolio = portfolio[~portfolio.index.duplicated()].copy()

    if len(portfolio) == 0:
        st.warning("Portfolio rỗng sau khi chọn danh mục.")
    else:
        portfolio["raw_weight"] = portfolio["score"] / portfolio["score"].sum()
        portfolio["weight"] = portfolio["raw_weight"].clip(upper=max_weight)
        portfolio["weight"] = portfolio["weight"] / portfolio["weight"].sum()
        portfolio["allocation_vnd"] = capital * portfolio["weight"]
        portfolio["shares"] = (portfolio["allocation_vnd"] / portfolio["latest_price"]).astype(int)

        port_return = (portfolio["mean_return"] * portfolio["weight"]).sum()
        port_risk = np.sqrt(((portfolio["volatility"] ** 2) * (portfolio["weight"] ** 2)).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Expected return", f"{port_return:.2%}")
        c2.metric("Risk", f"{port_risk:.2%}")
        c3.metric("Return/Risk", f"{(port_return / port_risk):.2f}" if port_risk > 0 else "N/A")

        st.write("#### Selected portfolio")
        st.dataframe(
            portfolio[["mean_return", "volatility", "weight", "allocation_vnd", "shares"]]
            .sort_values("weight", ascending=False),
            use_container_width=True,
        )

        fig = px.bar(
            portfolio.reset_index(),
            x="index",
            y="weight",
            title="Portfolio weights",
            labels={"index": "ticker", "weight": "weight"},
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================
# STAGE E — BACKTEST
# =========================
elif selected_tab == "Stage E — Backtest":
    st.subheader("Stage E — Backtest Engine vs Benchmark")

    # Rebuild portfolio exactly like notebook
    if risk_score <= 3:
        profile = "Conservative"
        n_stocks = 8
        max_weight = 0.18
    elif risk_score <= 7:
        profile = "Balanced"
        n_stocks = 10
        max_weight = 0.15
    else:
        profile = "Aggressive"
        n_stocks = 12
        max_weight = 0.12

    growth_pool = universe.sort_values("mean_return", ascending=False).head(30).copy()
    defensive_pool = universe.sort_values("volatility", ascending=True).head(30).copy()

    n_growth = n_stocks // 2
    n_def = n_stocks - n_growth

    growth_pool["score"] = growth_pool["mean_return"] / growth_pool["volatility"]
    defensive_pool["score"] = 1 / defensive_pool["volatility"]

    growth_pick = growth_pool.sort_values("score", ascending=False).head(n_growth)
    defensive_pick = defensive_pool.sort_values("score", ascending=False).head(n_def)

    portfolio = pd.concat([growth_pick, defensive_pick])
    portfolio = portfolio[~portfolio.index.duplicated()].copy()

    portfolio["raw_weight"] = portfolio["score"] / portfolio["score"].sum()
    portfolio["weight"] = portfolio["raw_weight"].clip(upper=max_weight)
    portfolio["weight"] = portfolio["weight"] / portfolio["weight"].sum()

    selected = portfolio.index.tolist()
    weights = portfolio["weight"].values

    # backtest on daily returns
    port_ret_series = (returns_wide[selected] * weights).sum(axis=1)
    portfolio_nav = (1 + port_ret_series).cumprod()

    result_df = pd.DataFrame({"Portfolio": portfolio_nav})

    has_benchmark = benchmark_series is not None
    if has_benchmark:
        bench = benchmark_series.copy()
        bench.index = pd.to_datetime(bench.index)
        bench = bench.sort_index()

        # align index/date with our portfolio series
        bench_ret = bench.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        common_index = result_df.index.intersection(bench_ret.index)
        result_df = result_df.loc[common_index]
        bench_ret = bench_ret.loc[common_index]
        vn_nav = (1 + bench_ret).cumprod()
        result_df["Benchmark"] = vn_nav

    st.write("#### NAV curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df["Portfolio"], name="Portfolio", line=dict(width=2)))
    if has_benchmark:
        fig.add_trace(go.Scatter(x=result_df.index, y=result_df["Benchmark"], name="Benchmark", line=dict(width=2)))
    fig.update_layout(height=600, xaxis_title="Date", yaxis_title="NAV")
    st.plotly_chart(fig, use_container_width=True)

    years = len(port_ret_series) / 252 if len(port_ret_series) > 0 else np.nan
    cagr = result_df["Portfolio"].iloc[-1] ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = port_ret_series.std() * np.sqrt(252)
    sharpe = cagr / vol if vol and vol > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("CAGR", f"{cagr:.2%}" if pd.notna(cagr) else "N/A")
    c2.metric("Volatility", f"{vol:.2%}" if pd.notna(vol) else "N/A")
    c3.metric("Sharpe-like", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")

    if has_benchmark:
        bn_ret = bench_ret
        bn_years = len(bn_ret) / 252 if len(bn_ret) > 0 else np.nan
        bn_cagr = result_df["Benchmark"].iloc[-1] ** (1 / bn_years) - 1 if bn_years and bn_years > 0 else np.nan
        bn_vol = bn_ret.std() * np.sqrt(252)
        bn_sharpe = bn_cagr / bn_vol if bn_vol and bn_vol > 0 else np.nan

        c4, c5, c6 = st.columns(3)
        c4.metric("Benchmark CAGR", f"{bn_cagr:.2%}" if pd.notna(bn_cagr) else "N/A")
        c5.metric("Benchmark Vol", f"{bn_vol:.2%}" if pd.notna(bn_vol) else "N/A")
        c6.metric("Benchmark Sharpe", f"{bn_sharpe:.2f}" if pd.notna(bn_sharpe) else "N/A")
    else:
        st.info("Chưa nạp benchmark VNINDEX. Nếu muốn đúng Stage E như notebook, hãy bổ sung FILE_ID benchmark vào BENCHMARK_FILE_ID.")


# =========================
# STAGE F — STOCK EXPLORER
# =========================
else:
    st.subheader("Stage F — Stock Explorer (xem giá từng ngày)")

    d = long_df[long_df["ticker"] == selected_ticker].copy()
    d = d[(d["date"] >= pd.to_datetime(start_date)) & (d["date"] <= pd.to_datetime(end_date))]
    d = d.sort_values("date")

    if d.empty:
        st.warning("Không có dữ liệu trong khoảng ngày này.")
    else:
        d_overlay = make_buy_sell_overlay(d) if show_overlay else d.copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("First close", f"{d['close'].iloc[0]:.2f}")
        c2.metric("Last close", f"{d['close'].iloc[-1]:.2f}")
        c3.metric("Max close", f"{d['close'].max():.2f}")
        c4.metric("Min close", f"{d['close'].min():.2f}")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=d_overlay["date"],
            y=d_overlay["close"],
            name="Price",
            line=dict(width=2)
        ))

        fig.add_trace(go.Scatter(
            x=d_overlay["date"],
            y=d_overlay["close"].rolling(20).mean(),
            name="MA20",
            line=dict(width=2)
        ))

        fig.add_trace(go.Scatter(
            x=d_overlay["date"],
            y=d_overlay["close"].rolling(50).mean(),
            name="MA50",
            line=dict(width=2)
        ))

        if show_overlay:
            buy = d_overlay[d_overlay["BUY_overlay"]]
            sell = d_overlay[d_overlay["SELL_overlay"]]

            fig.add_trace(go.Scatter(
                x=buy["date"], y=buy["close"],
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", size=10)
            ))

            fig.add_trace(go.Scatter(
                x=sell["date"], y=sell["close"],
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", size=10)
            ))

        fig.update_layout(
            height=700,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("### Bảng dữ liệu từng ngày")
        st.dataframe(d_overlay.sort_values("date", ascending=False), use_container_width=True)

        st.write("### Thống kê nhanh")
        daily_ret = d["close"].pct_change().fillna(0)
        stats = pd.DataFrame(
            {
                "metric": ["Avg daily return", "Daily volatility", "Total return", "Trading days"],
                "value": [
                    f"{daily_ret.mean():.4%}",
                    f"{daily_ret.std():.4%}",
                    f"{(d['close'].iloc[-1] / d['close'].iloc[0] - 1):.2%}",
                    f"{len(d):,}",
                ],
            }
        )
        st.dataframe(stats, use_container_width=True)