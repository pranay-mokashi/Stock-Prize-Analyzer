import streamlit as st
from backend import get_stock_data, train_lstm, predict_price  
from database import save_prediction, get_history 
import plotly.graph_objects as go 
import yfinance as yf
import pandas as pd
import math


st.set_page_config(page_title="Pro Trading System", layout="wide")

st.markdown("""
<style>

/* 🌟 MAIN BACKGROUND (light premium gradient) */
.stApp {
    background: linear-gradient(135deg, #eef2f7, #dfe9f3);
    color: #1a1a1a;
}

/* 🌟 CARD DESIGN (glass effect) */
.card {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}

/* 🌟 BUTTONS (premium gradient) */
.stButton>button {
    background: linear-gradient(90deg, #2962ff, #00c853);
    color: white;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: bold;
    border: none;
}

/* 🌟 METRICS (thick cards) */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

/* 🌟 TABS */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    padding: 10px;
    color: #333;
}

/* 🌟 INPUT BOX */
input {
    background-color: white !important;
    color: black !important;
    border-radius: 8px;
}

/* 🌟 SELECT BOX */
div[data-baseweb="select"] {
    background-color: white;
    border-radius: 8px;
}

/* 🌟 SUCCESS / BUY */
.success-box {
    background: #e8f5e9;
    color: #1b5e20;
    padding: 15px;
    border-radius: 10px;
}

/* 🌟 SELL */
.danger-box {
    background: #ffebee;
    color: #b71c1c;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.title("📈  STOCK TRADING SYSTEM") #maain heading

# =========================
# 📊 STOCK LIST
# =========================


stock_options = {
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "Suzlon": "SUZLON.NS",
    "Ola Electric": "OLAELEC.NS",
    "Vidya Wires" : "VIDYAWIRES.NS",
    "Nifty 50" : "NIFTY^NSEI",
    "hfcl" : "HFCL.NS",
    "hdfc bank" : "HDFCBANK.NS",
    "wipro" : "WIPRO.NS",
    "Axis Bank": "AXISBANK.NS",
    "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Titan": "TITAN.NS",
    "UltraTech": "ULTRACEMCO.NS",
    "Nestle": "NESTLEIND.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bharat electronic limited" : "BEL.NS"
}

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Prediction", "📜 History"])

# =========================
# 📊 DASHBOARD
# =========================
with tab1:
    st.subheader("📊 Live Market Dashboard") #ssubtitle

    stock = st.selectbox("Select Stock", list(stock_options.keys()), key="dash")
    symbol = stock_options[stock]

    data = get_stock_data(symbol)

    if data is not None and not data.empty:

        # ✅ FIX MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna() 

        # Candlestick
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        st.plotly_chart(fig, use_container_width=True)

        # ================= Indicators =================
        data['EMA20'] = data['Close'].ewm(span=20).mean()
        data['EMA50'] = data['Close'].ewm(span=50).mean()

         # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        st.subheader("📊 Indicators")

        chart_data = data[['Close', 'EMA20', 'EMA50']].dropna()
        if not chart_data.empty:
            st.line_chart(chart_data)

        rsi_data = data['RSI'].dropna()
        if not rsi_data.empty:
            st.line_chart(rsi_data)

    else:
        st.error("❌ No data available")


# =========================
# 🤖 AI PREDICTION
# =========================
with tab2:
    st.subheader("🤖 AI + Strategy Prediction")

    stock = st.selectbox("Select Stock", list(stock_options.keys()), key="pred")
    symbol = stock_options[stock]

    timeframe = st.selectbox(
        "Select Timeframe",
        ["1m", "5m", "15m", "1d", "1wk", "1mo"]
    )

    if st.button("🚀 Predict Now"):

        # Fetch data
        if timeframe in ["1m", "5m", "15m"]:
            data = yf.download(symbol, period="5d", interval=timeframe)
        else:
            data = yf.download(symbol, period="2y")

        if data is None or data.empty:
            st.error("❌ No data")
            st.stop()

        # ✅ FIX MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna()

        if data.empty:
            st.error("❌ Data empty after cleaning")
            st.stop()

        current_price = float(data['Close'].iloc[-1])

        # =========================
        # ⚡ INTRADAY
        # =========================
        if timeframe in ["1m", "5m", "15m"]:

            if len(data) < 10:
                st.error("❌ Not enough intraday data")
                st.stop()

            short_ma = data['Close'].rolling(5).mean().iloc[-1]
            long_ma = data['Close'].rolling(10).mean().iloc[-1]

            st.info(f"Price: ₹{current_price:.2f}")

            if float(short_ma) > float(long_ma):
                st.success("🟢 STRONG BUY")
            else:
                st.error("🔴 STRONG SELL")

            st.line_chart(data['Close'])  #display price trend
            st.stop()

        # =========================
        # 🤖 LSTM
        # =========================
        if len(data) < 60:
            st.error("❌ Not enough data")
            st.stop()

        model, scaler, X = train_lstm(data)

        if model is None or len(X) == 0:
            st.error("❌ Model failed")
            st.stop()

        predicted_price = predict_price(model, scaler, X)

        if predicted_price is None or math.isnan(predicted_price): # predict invalid output
            st.error("❌ Prediction error")
            st.stop()

        # Indicators
        data['EMA20'] = data['Close'].ewm(span=20).mean()
        data['EMA50'] = data['Close'].ewm(span=50).mean()

        rsi = data['RSI'].dropna().iloc[-1] if 'RSI' in data else 50

        # =========================
        # 📊 FINAL DECISION
        # =========================
        st.subheader("📊 Final Decision")

        st.write(f"Predicted: ₹{predicted_price:.2f}")
        st.write(f"Current: ₹{current_price:.2f}")

        score = 0

        if predicted_price > current_price:
            score += 1
        else:
            score -= 1

        if data['EMA20'].iloc[-1] > data['EMA50'].iloc[-1]:
            score += 1
        else:
            score -= 1

        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1

        if score >= 2:
            st.success("🟢 STRONG BUY 🚀")
        elif score == 1:
            st.success("🟢 BUY")
        
        elif score == -1:
            st.error("🔴 SELL")
        else:
            st.error("🔴 STRONG SELL")

        # Save
        try:

            st.write("⚡ Saving to database...")

            save_prediction(
            stock,
            float(predicted_price),
            float(current_price)
        )

            st.success("✅ Saved Successfully")

        except Exception as e:

            st.error(f"❌ DB ERROR: {e}")


# =========================
# 📜 HISTORY
# =========================
with tab3:

    st.subheader("📜 Prediction History")

    history = get_history()

    if history:

        df = pd.DataFrame(
            history,
            columns=[
                "ID",
                "Stock",
                "Predicted Price",
                "Actual Price",
                "Date & Time"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info("No history yet")