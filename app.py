import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gdown

st.set_page_config(layout="wide")

st.title("📊 VN STOCK DASHBOARD")

# =========================
# LOAD DATA (FROM DRIVE)
# =========================
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1bInFLKm6A-Zu9lCqH2c3_svUNlbsSTw7"
    gdown.download(url, "data.parquet", quiet=False)
    df = pd.read_parquet("data.parquet")
    return df

df = load_data()

# =========================
# SIDEBAR
# =========================
tickers = sorted(df["ticker"].unique())

ticker = st.sidebar.selectbox("Chọn cổ phiếu", tickers)

# =========================
# FILTER DATA
# =========================
d = df[df["ticker"] == ticker].copy()

# =========================
# INDICATORS
# =========================
d["MA20"] = d["close"].rolling(20).mean()
d["MA50"] = d["close"].rolling(50).mean()

# RSI
delta = d["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
d["RSI"] = 100 - (100/(1+rs))

# =========================
# SIGNAL (CẢI TIẾN)
# =========================
d["BUY"] = (
    (d["RSI"] < 35) &
    (d["close"] > d["MA20"])
)

d["SELL"] = (
    (d["RSI"] > 65) &
    (d["close"] < d["MA20"])
)

# =========================
# PLOT (QUAN TRỌNG)
# =========================
fig = go.Figure()

# price
fig.add_trace(go.Scatter(
    x=d["date"],
    y=d["close"],
    name="Price",
    line=dict(width=2)
))

# MA20
fig.add_trace(go.Scatter(
    x=d["date"],
    y=d["MA20"],
    name="MA20"
))

# MA50
fig.add_trace(go.Scatter(
    x=d["date"],
    y=d["MA50"],
    name="MA50"
))

# BUY
buy = d[d["BUY"]]
fig.add_trace(go.Scatter(
    x=buy["date"],
    y=buy["close"],
    mode="markers",
    marker=dict(symbol="triangle-up", size=10),
    name="BUY"
))

# SELL
sell = d[d["SELL"]]
fig.add_trace(go.Scatter(
    x=sell["date"],
    y=sell["close"],
    mode="markers",
    marker=dict(symbol="triangle-down", size=10),
    name="SELL"
))

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=True
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# TABLE
# =========================
st.subheader("📅 Dữ liệu gần nhất")

st.dataframe(
    d.sort_values("date", ascending=False).head(50),
    use_container_width=True
)