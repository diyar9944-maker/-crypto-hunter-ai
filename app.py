import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://fapi.binance.com"
INTERVALS = {"15m": "Scalping", "1h": "Gün içi", "4h": "Swing", "1d": "Ana trend"}
TIMEOUT = 12

st.set_page_config(page_title="Crypto Hunter AI", page_icon="📈", layout="wide")
st.title("📈 Crypto Hunter AI")
st.caption("Binance USDT-M vadeli piyasa verileri • API anahtarı gerekmez • Yalnızca analiz, otomatik emir yok")

st.info(
    "Bu ilk sürüm bir analiz prototipidir. Sinyaller garanti değildir; gerçek para ile işlem açmadan önce "
    "test edin. Bu uygulama Binance hesabınıza bağlanmaz ve emir göndermez."
)

with st.sidebar:
    st.header("Tarama ayarları")
    coin_count = st.slider("Taranacak altcoin sayısı", min_value=5, max_value=30, value=15)
    min_score = st.slider("En düşük sinyal puanı", min_value=40, max_value=90, value=65, step=5)
    st.caption("Likiditesi yüksek USDT sürekli vadeli altcoinler taranır. BTC ve ETH liste dışıdır.")

@st.cache_data(ttl=60, show_spinner=False)
def get_json(path, params=None):
    response = requests.get(BASE + path, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()

def get_universe(limit):
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = get_json("/fapi/v1/ticker/24hr")
    tradable = {
        s["symbol"] for s in info["symbols"]
        if s.get("status") == "TRADING"
        and s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("symbol") not in {"BTCUSDT", "ETHUSDT"}
    }
    rows = []
    for t in tickers:
        sym = t.get("symbol")
        if sym in tradable:
            try:
                qv = float(t.get("quoteVolume", 0))
                price = float(t.get("lastPrice", 0))
                change = float(t.get("priceChangePercent", 0))
                if qv > 0 and price > 0:
                    rows.append({"symbol": sym, "quoteVolume": qv, "price": price, "change24h": change})
            except (TypeError, ValueError):
                pass
    rows.sort(key=lambda x: x["quoteVolume"], reverse=True)
    return rows[:limit]

def candles(symbol, interval, limit=160):
    raw = get_json("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Exclude the currently forming candle to reduce false signals.
    if len(df) > 2:
        df = df.iloc[:-1].copy()
    return df.reset_index(drop=True)

def add_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df = df.copy()
    for n in [20, 50, 100, 200]:
        if len(df) >= n:
            df[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
        else:
            df[f"ema{n}"] = np.nan
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["bb_mid"] = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    low14 = df["rsi"].rolling(14).min()
    high14 = df["rsi"].rolling(14).max()
    stoch = 100 * (df["rsi"] - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_rsi"] = stoch.rolling(3).mean()
    df["vol_avg"] = v.rolling(20).mean()
    # Approximate rolling VWAP from recent candles.
    typical = (h + l + c) / 3
    df["vwap"] = (typical * v).rolling(20).sum() / v.rolling(20).sum()
    return df

def timeframe_score(df):
    df = add_indicators(df)
    r = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(r["close"])
    score = 0
    reasons = []
    # Trend alignment
    if pd.notna(r["ema20"]) and pd.notna(r["ema50"]):
        if price > r["ema20"] > r["ema50"]:
            score += 2; reasons.append("Fiyat EMA20/50 üzerinde")
        elif price < r["ema20"] < r["ema50"]:
            score -= 2; reasons.append("Fiyat EMA20/50 altında")
    if pd.notna(r["ema100"]):
        if price > r["ema100"]: score += 1
        elif price < r["ema100"]: score -= 1
    if pd.notna(r["ema200"]):
        if price > r["ema200"]: score += 1
        elif price < r["ema200"]: score -= 1
    # RSI
    if pd.notna(r["rsi"]):
        if 52 <= r["rsi"] <= 68:
            score += 1; reasons.append("RSI pozitif bölgede")
        elif 32 <= r["rsi"] <= 48:
            score -= 1; reasons.append("RSI negatif bölgede")
        elif r["rsi"] > 75:
            score -= 1; reasons.append("RSI aşırı alıma yakın")
        elif r["rsi"] < 25:
            score += 1; reasons.append("RSI aşırı satıma yakın")
    # MACD momentum
    if r["macd"] > r["macd_signal"] and r["macd"] > prev["macd"]:
        score += 1; reasons.append("MACD momentumu yukarı")
    elif r["macd"] < r["macd_signal"] and r["macd"] < prev["macd"]:
        score -= 1; reasons.append("MACD momentumu aşağı")
    # Volume confirmation
    if pd.notna(r["vol_avg"]) and r["vol_avg"] > 0 and r["volume"] > 1.3 * r["vol_avg"]:
        if r["close"] > r["open"]:
            score += 1; reasons.append("Hacim artışı yükselişi destekliyor")
        elif r["close"] < r["open"]:
            score -= 1; reasons.append("Hacim artışı düşüşü destekliyor")
    # VWAP and Bollinger context
    if pd.notna(r["vwap"]):
        if price > r["vwap"]: score += 1
        elif price < r["vwap"]: score -= 1
    if pd.notna(r["bb_upper"]) and price > r["bb_upper"]:
        reasons.append("Üst Bollinger bandı aşılmış: kovalamaya dikkat")
    return score, r, reasons, df

def analyze_symbol(item):
    symbol = item["symbol"]
    frames = {}
    reasons = []
    total = 0
    latest = None
    for interval, weight in [("1d", 3), ("4h", 2), ("1h", 2), ("15m", 1)]:
        df = candles(symbol, interval, 220)
        s, r, why, full = timeframe_score(df)
        total += s * weight
        frames[interval] = {"score": s, "row": r}
        if interval == "15m":
            latest = r
        if interval in ("1d", "4h"):
            reasons.extend([f"{interval}: {x}" for x in why[:2]])
    premium = get_json("/fapi/v1/premiumIndex", {"symbol": symbol})
    oi = get_json("/fapi/v1/openInterest", {"symbol": symbol})
    funding = float(premium.get("lastFundingRate", 0))
    price = float(latest["close"])
    atr = float(latest["atr"]) if pd.notna(latest["atr"]) else price * 0.02
    # Funding is a context flag, not a directional signal by itself.
    funding_note = "Funding pozitif" if funding > 0 else ("Funding negatif" if funding < 0 else "Funding sıfıra yakın")
    # Score normalized to an approximate 0-100 display, not a probability.
    max_abs = 3*6 + 2*6 + 2*6 + 1*6
    confidence = int(max(0, min(100, 50 + 50 * total / max_abs)))
    if total >= 5:
        direction = "LONG adayı"
        stop = price - 1.5 * atr
        tp1, tp2, tp3 = price + 1.5*atr, price + 2.5*atr, price + 3.5*atr
    elif total <= -5:
        direction = "SHORT adayı"
        stop = price + 1.5 * atr
        tp1, tp2, tp3 = price - 1.5*atr, price - 2.5*atr, price - 3.5*atr
    else:
        direction = "NÖTR / BEKLE"
        stop = tp1 = tp2 = tp3 = np.nan
    return {
        "Coin": symbol, "Yön": direction, "Analiz puanı": confidence,
        "Fiyat": price, "24s %": item["change24h"], "24s hacim (USDT)": item["quoteVolume"],
        "RSI 15m": float(latest["rsi"]) if pd.notna(latest["rsi"]) else np.nan,
        "Funding": funding, "Open Interest": float(oi.get("openInterest", 0)),
        "Giriş referansı": price, "Stop (ATR)": stop, "TP1": tp1, "TP2": tp2, "TP3": tp3,
        "Detay": " • ".join(reasons[:5] + [funding_note]),
        "_rawscore": total
    }

def fmt_price(x):
    if pd.isna(x): return "—"
    if abs(x) >= 1000: return f"{x:,.2f}"
    if abs(x) >= 1: return f"{x:.4f}"
    return f"{x:.7f}"

if st.button("🔎 Altcoinleri tara", type="primary", use_container_width=True):
    results = []
    progress = st.progress(0, text="Likiditesi yüksek vadeli pariteler alınıyor…")
    try:
        universe = get_universe(coin_count)
        progress.progress(5, text=f"{len(universe)} coin taranıyor…")
        errors = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(analyze_symbol, item): item["symbol"] for item in universe}
            for i, future in enumerate(as_completed(futures)):
                symbol = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    errors.append(f"{symbol}: {str(e)[:100]}")
                progress.progress(min(95, int(5 + 90*(i+1)/len(futures))), text=f"Analiz: {i+1}/{len(futures)}")
        progress.progress(100, text="Tarama tamamlandı.")
        if results:
            df = pd.DataFrame(results).sort_values("Analiz puanı", ascending=False)
            df = df[df["Analiz puanı"] >= min_score]
            st.session_state["results"] = df
            st.session_state["scan_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["errors"] = errors
        else:
            st.error("Sonuç alınamadı. Bağlantıyı tekrar deneyin.")
            st.session_state["errors"] = errors
    except Exception as e:
        st.error(f"Binance verilerine ulaşılamadı: {e}")
        st.caption("İnternet bağlantısını veya Binance erişimini kontrol edip tekrar deneyin.")

if "results" in st.session_state:
    df = st.session_state["results"]
    st.caption(f"Son tarama: {st.session_state.get('scan_time', '—')} • Eşik üstünde {len(df)} aday")
    if df.empty:
        st.warning("Seçilen puan eşiğini geçen aday yok. Bu, işlem açmamak için de geçerli bir sonuçtur.")
    else:
        display = df.copy()
        for col in ["Fiyat", "Giriş referansı", "Stop (ATR)", "TP1", "TP2", "TP3"]:
            display[col] = display[col].map(fmt_price)
        display["24s %"] = display["24s %"].map(lambda x: f"{x:.2f}%")
        display["RSI 15m"] = display["RSI 15m"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        display["Funding"] = display["Funding"].map(lambda x: f"{x*100:.4f}%")
        display["24s hacim (USDT)"] = display["24s hacim (USDT)"].map(lambda x: f"{x:,.0f}")
        display["Open Interest"] = display["Open Interest"].map(lambda x: f"{x:,.2f}")
        st.dataframe(display.drop(columns=["_rawscore"]), use_container_width=True, hide_index=True)
        csv = df.drop(columns=["_rawscore"]).to_csv(index=False).encode("utf-8-sig")
        st.download_button("Sonuçları CSV indir", csv, "crypto_hunter_signals.csv", "text/csv")
        chosen = st.selectbox("Detayını görmek istediğin coin", df["Coin"].tolist())
        row = df[df["Coin"] == chosen].iloc[0]
        st.subheader(f"{chosen} — {row['Yön']}")
        st.write(f"**Analiz puanı:** {row['Analiz puanı']}/100 (kazanma olasılığı değildir)")
        st.write(f"**Analiz notları:** {row['Detay']}")
        st.write(f"**Giriş referansı:** {fmt_price(row['Giriş referansı'])} · **Stop:** {fmt_price(row['Stop (ATR)'])}")
        st.write(f"**TP1:** {fmt_price(row['TP1'])} · **TP2:** {fmt_price(row['TP2'])} · **TP3:** {fmt_price(row['TP3'])}")
    if st.session_state.get("errors"):
        with st.expander(f"Veri alınamayan coinler ({len(st.session_state['errors'])})"):
            st.write("\n".join(st.session_state["errors"]))

st.divider()
st.caption("Veri kaynağı: Binance USDⓈ-M Futures public market-data endpoints. Gecikme, rate limit ve bağlantı hataları olabilir. Bu bir yatırım tavsiyesi değildir.")
