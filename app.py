# ======================
# 1️⃣ PAGE CONFIG
# ======================
import streamlit as st
st.set_page_config(
    page_title="AI-Powered Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# 2️⃣ IMPORTS
# ======================
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import pytz
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ======================
# 3️⃣ STYLING
# ======================
st.markdown("""
<style>
.stApp {background-color: #0b0e11; color: #ffffff; font-family: 'Inter', sans-serif;}
[data-testid="stSidebar"] {background-color: #111418; color: #ffffff; padding: 2rem 1rem;}
h1,h2,h3,h4 {color: #00bcd4; font-weight:600;}
label,p,span,div {color:#e0e0e0 !important;}
.stTextInput input, .stSelectbox, .stSlider {background-color:#1a1f25 !important; color:white !important; border:1px solid #2c3138; border-radius:6px;}
button[kind="primary"] {background-color:#00bcd4 !important; color:white !important; border-radius:8px; border:none; font-weight:600;}
.stMetric {background-color:#1a1f25 !important; border-radius:12px; padding:1rem; margin:0.5rem; box-shadow:0px 2px 8px rgba(0,0,0,0.3);}
</style>
""", unsafe_allow_html=True)

# ======================
# 4️⃣ HEADER
# ======================
st.markdown("""
<h1 style='text-align:center; color:#00bcd4; font-size:2.5rem;'>🤖 AI-Powered Stock Dashboard</h1>
<p style='text-align:center; color:#b0bec5;'>Advanced Machine Learning & Technical Analysis</p>
""", unsafe_allow_html=True)

# ======================
# 5️⃣ SIDEBAR INPUTS
# ======================
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("🔄 Refresh Data", help="Clear cache and fetch latest data"):
    st.cache_data.clear()
    st.rerun()

ticker = st.sidebar.text_input("Stock Ticker (e.g. RELIANCE.NS, TCS.NS)", "TCS.NS")

today_ist = datetime.now(pytz.timezone('Asia/Kolkata')).date()
default_start = today_ist - timedelta(days=180)
start_date = st.sidebar.date_input("Start Date", default_start, max_value=today_ist)
end_date = st.sidebar.date_input("End Date", today_ist, min_value=start_date, max_value=today_ist)

model_type = st.sidebar.selectbox("Model Type", ["Random Forest", "XGBoost", "Gradient Boosting", "Ensemble (Best)"], index=3)
train_size = st.sidebar.slider("Train/Test Split %", 60, 95, 80)

st.sidebar.markdown("### 🤖 AI Features")
enable_sentiment = st.sidebar.checkbox("Enable Sentiment Analysis", value=True, help="Analyze market sentiment from indicators")
enable_anomaly = st.sidebar.checkbox("Detect Anomalies", value=True, help="Flag unusual price movements")
enable_pattern = st.sidebar.checkbox("Pattern Recognition", value=True, help="Identify chart patterns")
prediction_days = st.sidebar.slider("Forecast Days Ahead", 1, 7, 3)

st.sidebar.markdown("---")

# ======================
# 6️⃣ FETCH DATA
# ======================
@st.cache_data(show_spinner=True, ttl=1800)
def fetch_data(ticker, start, end):
    try:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        buffered_end = end_dt + timedelta(days=7)
        
        df = yf.download(ticker, start=start_dt, end=buffered_end, interval="1d", 
                         auto_adjust=True, progress=False)
        
        if df is None or df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        df.dropna(subset=["Close"], inplace=True)
        if df.empty:
            return None
        
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        mask = (df.index >= start_dt) & (df.index <= end_dt + timedelta(days=1))
        df = df[mask]
        
        if df.empty:
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

with st.spinner(f"Fetching data for {ticker}..."):
    df = fetch_data(ticker, start_date, end_date)

if df is None or df.empty:
    st.error(f"❌ No data found for {ticker} between {start_date} and {end_date}. Please check:")
    st.info("• Ticker symbol is correct (e.g., TCS.NS, RELIANCE.NS)")
    st.info("• Date range includes trading days")
    st.info("• Stock was actively traded during this period")
    st.stop()

latest_date = df.index[-1].date()
st.success(f"✅ Fetched {len(df)} trading days of data (Latest: {latest_date})")

today_ist = datetime.now(pytz.timezone('Asia/Kolkata')).date()
if latest_date < today_ist - timedelta(days=1):
    st.warning(f"⚠️ Data may be delayed. Yahoo Finance often has 1-2 day lag for NSE stocks. Latest available: {latest_date}")

# ======================
# 7️⃣ ADD TECHNICAL INDICATORS
# ======================
def add_indicators(df):
    df = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain/(avg_loss + 1e-9)
    df["RSI14"] = 100 - (100/(1+rs))
    
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(14).mean()
    
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()*np.sqrt(252)
    df["Log_Return"] = np.log(df["Close"]/df["Close"].shift(1))
    
    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]
    
    df["ROC"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100
    df["Momentum"] = df["Close"] - df["Close"].shift(4)
    
    df["Pivot"] = (df["High"] + df["Low"] + df["Close"])/3
    df["R1"] = 2*df["Pivot"] - df["Low"]
    df["S1"] = 2*df["Pivot"] - df["High"]
    
    df["Trend_Strength"] = (df["EMA20"] - df["EMA50"]) / df["EMA50"] * 100
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    
    df.dropna(inplace=True)
    return df

df = add_indicators(df)

if df.empty:
    st.error("❌ Not enough data to calculate technical indicators. Try a longer date range.")
    st.stop()

# ======================
# 8️⃣ PRICE FIGURE
# ======================
def make_price_figure(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(width=2, color="cyan")))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], mode="lines", name="EMA20", line=dict(width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA50", line=dict(width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA100"], mode="lines", name="EMA100", line=dict(width=1.5, dash="dot")))
    last = df.iloc[-1]
    for key, color in zip(["Pivot","R1","S1"], ["gray","green","red"]):
        fig.add_hline(y=last[key], line_dash="dot", annotation_text=key, line_color=color)
    fig.update_layout(title=f"{ticker} — Price & EMA", template="plotly_white", height=400)
    return fig

st.plotly_chart(make_price_figure(df, ticker), use_container_width=True)

# ======================
# 🤖 AI FEATURE 1: ANOMALY DETECTION
# ======================
if enable_anomaly:
    st.markdown("## 🚨 AI Anomaly Detection")
    
    def detect_anomalies(df):
        df_copy = df.copy()
        returns = df_copy["Return"]
        mean_return = returns.mean()
        std_return = returns.std()
        df_copy["Z_Score"] = (returns - mean_return) / std_return
        df_copy["Is_Anomaly"] = np.abs(df_copy["Z_Score"]) > 2.5
        df_copy["Volume_Z"] = (df_copy["Volume"] - df_copy["Volume"].mean()) / df_copy["Volume"].std()
        df_copy["Volume_Anomaly"] = np.abs(df_copy["Volume_Z"]) > 2.5
        return df_copy
    
    df_anomaly = detect_anomalies(df)
    anomalies = df_anomaly[df_anomaly["Is_Anomaly"]]
    
    if len(anomalies) > 0:
        col1, col2 = st.columns(2)
        col1.metric("Price Anomalies Detected", len(anomalies))
        col2.metric("Volume Anomalies", df_anomaly["Volume_Anomaly"].sum())
        
        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(go.Scatter(x=df_anomaly.index, y=df_anomaly["Close"], 
                                         mode="lines", name="Price", line=dict(color="cyan")))
        fig_anomaly.add_trace(go.Scatter(x=anomalies.index, y=anomalies["Close"],
                                         mode="markers", name="Anomalies",
                                         marker=dict(size=10, color="red", symbol="x")))
        fig_anomaly.update_layout(title="Anomalous Price Movements", template="plotly_white", height=300)
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        with st.expander("📊 View Anomaly Details"):
            st.dataframe(anomalies[["Close", "Return", "Volume", "Z_Score"]].tail(10))
    else:
        st.info("✅ No significant anomalies detected in recent data")
    
    st.markdown("---")

# ======================
# 🤖 AI FEATURE 2: PATTERN RECOGNITION
# ======================
if enable_pattern:
    st.markdown("## 🔍 AI Pattern Recognition")
    
    def detect_patterns(df):
        patterns = []
        last_row = df.iloc[-1]
        
        if last_row["Close"] > last_row["EMA20"] > last_row["EMA50"]:
            patterns.append(("🟢 Golden Cross", "Bullish momentum - price above both EMAs"))
        if last_row["RSI14"] < 30:
            patterns.append(("🟢 Oversold (RSI)", "Potential buying opportunity"))
        if last_row["MACD"] > last_row["MACD_Signal"] and df.iloc[-2]["MACD"] <= df.iloc[-2]["MACD_Signal"]:
            patterns.append(("🟢 MACD Bullish Crossover", "Momentum turning positive"))
        if last_row["Close"] < last_row["BB_Lower"]:
            patterns.append(("🟢 Below Lower BB", "Potential reversal upward"))
        if last_row["Close"] < last_row["EMA20"] < last_row["EMA50"]:
            patterns.append(("🔴 Death Cross", "Bearish momentum - price below both EMAs"))
        if last_row["RSI14"] > 70:
            patterns.append(("🔴 Overbought (RSI)", "Potential selling pressure"))
        if last_row["MACD"] < last_row["MACD_Signal"] and df.iloc[-2]["MACD"] >= df.iloc[-2]["MACD_Signal"]:
            patterns.append(("🔴 MACD Bearish Crossover", "Momentum turning negative"))
        if last_row["Close"] > last_row["BB_Upper"]:
            patterns.append(("🔴 Above Upper BB", "Potential reversal downward"))
        if 45 <= last_row["RSI14"] <= 55:
            patterns.append(("⚪ RSI Neutral", "Market in equilibrium"))
        if last_row["BB_Width"] > df["BB_Width"].quantile(0.8):
            patterns.append(("⚡ High Volatility", "Increased price movement expected"))
        elif last_row["BB_Width"] < df["BB_Width"].quantile(0.2):
            patterns.append(("🔇 Low Volatility", "Consolidation phase - breakout likely"))
        return patterns
    
    patterns = detect_patterns(df)
    
    if patterns:
        st.markdown("### 📍 Detected Patterns:")
        for pattern_name, description in patterns:
            st.markdown(f"**{pattern_name}**: {description}")
    else:
        st.info("No significant patterns detected")
    
    st.markdown("---")

# ======================
# 🤖 AI FEATURE 3: SENTIMENT ANALYSIS
# ======================
sentiment_score = 0
if enable_sentiment:
    st.markdown("## 💭 AI Market Sentiment Score")
    
    def calculate_sentiment(df):
        last = df.iloc[-1]
        scores = []
        
        if last["RSI14"] < 30:
            scores.append(1)
        elif last["RSI14"] > 70:
            scores.append(-1)
        else:
            scores.append((50 - last["RSI14"]) / 20)
        
        if last["MACD"] > last["MACD_Signal"]:
            scores.append(1)
        else:
            scores.append(-1)
        
        if last["Close"] > last["EMA20"] > last["EMA50"]:
            scores.append(1)
        elif last["Close"] < last["EMA20"] < last["EMA50"]:
            scores.append(-1)
        else:
            scores.append(0)
        
        if last["Volume_Ratio"] > 1.5:
            scores.append(0.5 if last["Return"] > 0 else -0.5)
        else:
            scores.append(0)
        
        if last["BB_Position"] > 0.8:
            scores.append(-0.5)
        elif last["BB_Position"] < 0.2:
            scores.append(0.5)
        else:
            scores.append(0)
        
        overall_sentiment = np.mean(scores)
        return overall_sentiment, scores
    
    sentiment_score, individual_scores = calculate_sentiment(df)
    
    col1, col2, col3 = st.columns(3)
    
    if sentiment_score > 0.3:
        sentiment_label = "🟢 BULLISH"
        sentiment_color = "green"
    elif sentiment_score < -0.3:
        sentiment_label = "🔴 BEARISH"
        sentiment_color = "red"
    else:
        sentiment_label = "⚪ NEUTRAL"
        sentiment_color = "gray"
    
    col2.markdown(f"<h2 style='text-align:center;color:{sentiment_color};'>{sentiment_label}</h2>", unsafe_allow_html=True)
    col2.metric("Sentiment Score", f"{sentiment_score:.2f}", f"{abs(sentiment_score)*100:.0f}% confidence")
    
    with st.expander("📊 Sentiment Breakdown"):
        sentiment_df = pd.DataFrame({
            "Indicator": ["RSI", "MACD", "EMA Trend", "Volume", "Bollinger Position"],
            "Score": individual_scores
        })
        fig_sent = px.bar(sentiment_df, x="Indicator", y="Score", 
                          title="Individual Indicator Sentiment",
                          color="Score", color_continuous_scale=["red", "gray", "green"])
        st.plotly_chart(fig_sent, use_container_width=True)
    
    st.markdown("---")

# ======================
# 9️⃣ ENHANCED AI MODEL TRAINING
# ======================
st.markdown("## 🧠 Advanced AI Model Training & Prediction")

feature_cols = ["EMA20","EMA50","EMA100","RSI14","MACD","MACD_Signal","MACD_Hist",
                "BB_Upper","BB_Lower","BB_Width","BB_Position","ATR","Volatility",
                "Volume_Ratio","ROC","Momentum","Trend_Strength","Pivot","R1","S1"]

df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

if len(df) < 50:
    st.error("❌ Not enough data for model training. Need at least 50 data points.")
    st.stop()

X = df[feature_cols]
y = df["Target"]
split_idx = int(len(df)*(train_size/100))

if split_idx < 30 or (len(df) - split_idx) < 10:
    st.error("❌ Insufficient data for train/test split. Adjust date range or split percentage.")
    st.stop()

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with st.spinner("Training AI models..."):
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_type == "XGBoost":
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model1 = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        model2 = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
        model3 = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        model3.fit(X_train_scaled, y_train)
        
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        pred3 = model3.predict(X_test_scaled)
        
        y_pred = (pred1 * 0.35 + pred2 * 0.40 + pred3 * 0.25)
        model = model2

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", f"₹{rmse:.2f}")
col2.metric("MAE", f"₹{mae:.2f}")
col3.metric("R² Score", f"{r2:.3f}")
col4.metric("MAPE", f"{mape:.2f}%")

st.markdown("---")
st.markdown(f"### 🎯 {prediction_days}-Day AI Forecast")

last_features = df[feature_cols].iloc[-1].values
predictions = []
feature_history = last_features.copy()

for day in range(prediction_days):
    if model_type == "Gradient Boosting":
        pred = model.predict(scaler.transform(feature_history.reshape(1, -1)))[0]
    else:
        pred = model.predict(feature_history.reshape(1, -1))[0]
    predictions.append(pred)
    feature_history = feature_history * 0.98 + pred * 0.02

current_price = df["Close"].iloc[-1]
forecast_df = pd.DataFrame({
    "Day": [f"Day {i+1}" for i in range(prediction_days)],
    "Predicted Price": predictions,
    "Change from Current": [p - current_price for p in predictions],
    "% Change": [(p - current_price)/current_price * 100 for p in predictions]
})

st.dataframe(forecast_df.style.format({
    "Predicted Price": "₹{:.2f}",
    "Change from Current": "₹{:.2f}",
    "% Change": "{:.2f}%"
}))

fig_forecast = go.Figure()
historical_dates = df.index[-30:]
fig_forecast.add_trace(go.Scatter(x=historical_dates, y=df["Close"].iloc[-30:],
                                  mode="lines", name="Historical", line=dict(color="cyan")))

future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=prediction_days, freq='D')
fig_forecast.add_trace(go.Scatter(x=future_dates, y=predictions,
                                  mode="lines+markers", name="Forecast",
                                  line=dict(color="orange", dash="dash")))
fig_forecast.update_layout(title=f"{prediction_days}-Day Price Forecast", template="plotly_white", height=400)
st.plotly_chart(fig_forecast, use_container_width=True)

if len(predictions) > 0:
    avg_change = np.mean(forecast_df["% Change"])
    if avg_change > 1:
        st.success(f"📈 AI predicts upward trend: +{avg_change:.2f}% over {prediction_days} days")
    elif avg_change < -1:
        st.error(f"📉 AI predicts downward trend: {avg_change:.2f}% over {prediction_days} days")
    else:
        st.info(f"➡️ AI predicts sideways movement: {avg_change:.2f}% over {prediction_days} days")

# ======================
# 1️⃣0️⃣ ACTUAL VS PREDICTED
# ======================
st.markdown("### 📊 Actual vs Predicted Prices")
results_df = pd.DataFrame({
    "Date": df.index[split_idx:],
    "Actual": y_test.values,
    "Predicted": y_pred
})
fig = go.Figure()
fig.add_trace(go.Scatter(x=results_df["Date"], y=results_df["Actual"], mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=results_df["Date"], y=results_df["Predicted"], mode='lines+markers', name='Predicted'))
fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ======================
# 1️⃣1️⃣ AI FEATURE IMPORTANCE & EDA
# ======================
st.markdown("## 🧩 AI Insights & Analysis")

if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title='Top 10 Most Important Features (AI-Driven)',
                     color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("### 📊 Correlation Analysis")

def plot_correlation_matrix(df):
    corr_cols = ["Close","EMA20","EMA50","EMA100","RSI14","MACD","MACD_Signal","Volatility"]
    corr_df = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="vlag", ax=ax)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    return fig

colA, colB = st.columns((2,1))
with colA:
    st.pyplot(plot_correlation_matrix(df))
with colB:
    fig3 = px.scatter(df, x="RSI14", y="Return", title="RSI vs Return", color="MACD")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ======================
# 1️⃣2️⃣ AI TRADING SUMMARY REPORT
# ======================
st.markdown("## 📋 AI Trading Summary Report")
summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown("### 📊 Technical Summary")
    last = df.iloc[-1]
    st.write(f"**Current Price:** ₹{last['Close']:.2f}")
    st.write(f"**RSI (14):** {last['RSI14']:.2f}")
    st.write(f"**MACD:** {last['MACD']:.2f}")
    st.write(f"**Volatility:** {last['Volatility']:.2%}")
    st.write(f"**Volume Ratio:** {last['Volume_Ratio']:.2f}x")

with summary_col2:
    st.markdown("### 🎯 AI Recommendations")
    
    score = 0
    if enable_sentiment:
        score += sentiment_score * 30
    
    if last["RSI14"] < 40:
        score += 20
    elif last["RSI14"] > 60:
        score -= 20
    
    if last["Close"] > last["EMA20"]:
        score += 15
    else:
        score -= 15
    
    if last["MACD"] > last["MACD_Signal"]:
        score += 15
    else:
        score -= 15
    
    if len(predictions) > 0:
        avg_forecast_change = np.mean([(p - current_price)/current_price for p in predictions])
        score += avg_forecast_change * 100
    
    if score > 30:
        
        st.success("🟢 **STRONG BUY**")
        st.write("AI signals indicate strong bullish momentum. Consider accumulating position.")
    elif score > 10:
        st.success("🟢 **BUY**")
        st.write("Positive indicators suggest upward movement likely.")
    elif score > -10:
        st.info("⚪ **HOLD**")
        st.write("Mixed signals. Wait for clearer trend confirmation.")
    elif score > -30:
        st.warning("🔴 **SELL**")
        st.write("Bearish signals detected. Consider reducing exposure.")
    else:
        st.error("🔴 **STRONG SELL**")
        st.write("Strong bearish momentum. Exit or hedge position.")
    
    st.write(f"**Confidence Score:** {abs(score):.1f}/100")

st.markdown("---")
st.dataframe(df.tail(10))

# Risk Metrics
st.markdown("### ⚠️ AI Risk Analysis")
risk_col1, risk_col2, risk_col3 = st.columns(3)

sharpe_ratio = (df["Return"].mean() / df["Return"].std()) * np.sqrt(252) if df["Return"].std() > 0 else 0
max_drawdown = ((df["Close"].cummax() - df["Close"]) / df["Close"].cummax()).max()
var_95 = df["Return"].quantile(0.05)

risk_col1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", 
                 "Good" if sharpe_ratio > 1 else "Moderate" if sharpe_ratio > 0 else "Poor")
risk_col2.metric("Max Drawdown", f"{max_drawdown*100:.2f}%",
                 delta=None, delta_color="inverse")
risk_col3.metric("VaR (95%)", f"{var_95*100:.2f}%",
                 help="Maximum expected loss in 95% of scenarios")

st.markdown("<div class='small-muted'>⚠️ Note: This is an educational AI demo. Not financial advice. Always do your own research and consult financial advisors.</div>", unsafe_allow_html=True)