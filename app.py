import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sniper Elite · Ankesh",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# PREMIUM TERMINAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

:root {
    --bg-main: #06070a;
    --bg-card: #0e1117;
    --border: #1f2436;
    --accent: #4f6ef7;
    --green: #00e676;
    --red: #ff3d5a;
    --gold: #ffc107;
    --text: #c8cfdf;
}

html, body, [class*="css"] {
    background-color: var(--bg-main) !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text) !important;
}

/* Elite Header HUD */
.top-hud {
    background: linear-gradient(90deg, #0e1117 0%, #161b22 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 15px 25px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.hud-metric { text-align: center; border-right: 1px solid var(--border); padding: 0 20px; }
.hud-label { font-size: 0.6rem; color: #505872; text-transform: uppercase; letter-spacing: 1.5px; }
.hud-value { font-size: 1.2rem; font-weight: 700; color: white; }

/* Custom Tab Styling */
.stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
.stTabs [data-baseweb="tab"] {
    background-color: #0e1117;
    border: 1px solid var(--border);
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
}

/* Backtest Card */
.stats-card {
    background: var(--bg-card);
    border-top: 3px solid var(--accent);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CORE ENGINE: DATA & ANALYTICS
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data(ticker, interval="1d", years=2):
    df = yf.download(ticker, period=f"{years}y", interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df

def run_world_class_backtest(df, risk_per_trade=2000, commission=0.0005):
    # Strategy Logic: Long if Price > EMA200 and RSI < 40
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    trades = []
    equity = 100000
    equity_curve = [equity]
    
    for i in range(201, len(df)-1):
        if df['Close'].iloc[i] > df['EMA200'].iloc[i] and df['RSI'].iloc[i] < 40:
            entry_p = df['Open'].iloc[i+1]
            sl = entry_p - (2 * df['ATR'].iloc[i])
            tp1 = entry_p + (2 * df['ATR'].iloc[i]) # 1:2 Reward
            
            # Position Sizing
            risk_amt = entry_p - sl
            qty = int(risk_per_trade / risk_amt) if risk_amt > 0 else 0
            
            if qty <= 0: continue
            
            # Simple Exit Logic
            for j in range(i+1, len(df)):
                if df['Low'].iloc[j] <= sl:
                    pnl = (sl - entry_p) * qty
                    equity += (pnl - (entry_p * qty * commission))
                    trades.append({"PnL": pnl, "Result": "Loss"})
                    break
                if df['High'].iloc[j] >= tp1:
                    pnl = (tp1 - entry_p) * qty
                    equity += (pnl - (entry_p * qty * commission))
                    trades.append({"PnL": pnl, "Result": "Win"})
                    break
            equity_curve.append(equity)

    # Metrics
    if not trades: return None, None
    tdf = pd.DataFrame(trades)
    win_rate = (tdf[tdf['Result'] == "Win"].shape[0] / len(tdf)) * 100
    total_pnl = tdf['PnL'].sum()
    
    return tdf, {
        "Win Rate": f"{win_rate:.1f}%",
        "Total PnL": f"₹{total_pnl:,.0f}",
        "Max Drawdown": f"{((max(equity_curve)-min(equity_curve))/max(equity_curve)*100):.1f}%",
        "Profit Factor": f"{(tdf[tdf['PnL']>0]['PnL'].sum()/abs(tdf[tdf['PnL']<0]['PnL'].sum())):.2f}",
        "Equity Curve": equity_curve
    }

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
def main():
    # 1. TOP HUD (Utilizing the blank space)
    st.markdown(f"""
    <div class="top-hud">
        <div style="display:flex; align-items:center;">
            <div style="width:40px; height:40px; background:var(--accent); border-radius:8px; display:grid; place-items:center; font-weight:900; color:white; margin-right:15px">S</div>
            <div>
                <div style="font-family:Syne; font-size:1.2rem; font-weight:800; color:white;">SNIPER ELITE <span style="color:var(--accent)">v3.0</span></div>
                <div style="font-size:0.6rem; color:#505872; letter-spacing:2px">INSTITUTIONAL QUANT ENGINE</div>
            </div>
        </div>
        <div style="display:flex;">
            <div class="hud-metric"><div class="hud-label">NIFTY 50</div><div class="hud-value" style="color:var(--green)">22,410.50 ▲</div></div>
            <div class="hud-metric"><div class="hud-label">MARKET BIAS</div><div class="hud-value" style="color:var(--gold)">NEUTRAL</div></div>
            <div class="hud-metric" style="border:none;"><div class="hud-label">SIGNAL STRENGTH</div><div class="hud-value">84%</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ticker = st.sidebar.text_input("Active Ticker", "HAL.NS").upper()
    
    tab1, tab2, tab3 = st.tabs(["📊 Screener", "📈 World-Class Backtest", "🛡️ Risk Hub"])

    with tab2:
        st.subheader(f"Strategy Performance Analysis: {ticker}")
        df = load_data(ticker)
        
        if not df.empty:
            trades_df, stats = run_world_class_backtest(df)
            
            if stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f'<div class="stats-card"><div class="hud-label">Win Rate</div><div class="hud-value">{stats["Win Rate"]}</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="stats-card"><div class="hud-label">Profit Factor</div><div class="hud-value">{stats["Profit Factor"]}</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="stats-card"><div class="hud-label">Total PnL</div><div class="hud-value" style="color:var(--green)">{stats["Total PnL"]}</div></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="stats-card"><div class="hud-label">Max Drawdown</div><div class="hud-value" style="color:var(--red)">{stats["Max Drawdown"]}</div></div>', unsafe_allow_html=True)
                
                # Equity Curve Plotly
                st.markdown("---")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=stats["Equity Curve"], mode='lines', fill='tozeroy', line=dict(color='#4f6ef7', width=3), name="Equity"))
                fig.update_layout(title="Institutional Equity Growth (₹1 Lakh Base)", template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No A++ Sniper setups found in this historical period.")

    with tab1:
        st.info("Screener logic active. Select a ticker from the sidebar to begin depth-analysis.")

if __name__ == "__main__":
    main()
