# ============================================================
# BOT TRADING V300 – INSTITUTIONAL (LLAMA 70B + NISON LOGIC)
# ============================================================
# ✔ Llama 3.3 70B (Groq)
# ✔ Contexto completo estilo Nison (sin imágenes para IA)
# ✔ Gráfico enviado a Telegram
# ✔ TP1 (50%) + BE + Trailing dinámico
# ✔ Logs PRO (console + Telegram)
# ✔ Estado de trade detallado
# ✔ Historial persistente (JSON)
# ✔ Autoanálisis cada 10 trades
# ✔ Validaciones y reintentos IA
# ============================================================

import os
import time
import json
import math
import requests
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from groq import Groq

# ================= CONFIG =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "BTCUSDT"
INTERVAL = "5"  # minutos
SLEEP_SECONDS = 60
MODEL = "llama-3.3-70b-versatile"

DATA_LIMIT = 200
STATE_FILE = "state.json"
TRADES_FILE = "trades.json"

ATR_MULT_SL = 1.2
TRAILING_FACTOR = 0.5  # % de ATR para trailing dinámico
TP1_R_MULT = 1.0  # 1R para TP1

# ================= CLIENT =================
client = Groq(api_key=GROQ_API_KEY)

# ================= UTILS =================
def now_utc():
    return datetime.now(timezone.utc)


def log_console(msg):
    ts = now_utc().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")


def tg_send(text):
    log_console(f"TG >> {text}")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text}
        )
    except Exception as e:
        log_console(f"TG send error: {e}")


def tg_send_img(path, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        with open(path, 'rb') as f:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": f}
            )
    except Exception as e:
        log_console(f"TG img error: {e}")


# ================= DATA =================
BASE_URL = "https://api.bybit.com"


def fetch_klines():
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": DATA_LIMIT
    }
    r = requests.get(url, params=params, timeout=15).json()
    if r.get("retCode") not in (0, None):
        raise ValueError(f"Bybit error: {r}")

    df = pd.DataFrame(r["result"]["list"][::-1],
                      columns=["time", "open", "high", "low", "close", "volume", "turnover"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df["time"] = pd.to_datetime(df["time"].astype(np.int64), unit='ms', utc=True)
    df.set_index("time", inplace=True)

    # indicadores
    df["ema20"] = df["close"].ewm(span=20).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(abs(df["high"] - df["close"].shift()),
                               abs(df["low"] - df["close"].shift())))
    df["atr"] = pd.Series(tr, index=df.index).rolling(14).mean()

    return df.dropna()


# ================= CONTEXTO NISON =================
def candle_features(row):
    body = abs(row.close - row.open)
    upper = row.high - max(row.open, row.close)
    lower = min(row.open, row.close) - row.low
    direction = "bullish" if row.close > row.open else "bearish"
    return {
        "type": direction,
        "body": round(body, 4),
        "upper_wick": round(upper, 4),
        "lower_wick": round(lower, 4),
        "range": round(row.high - row.low, 4)
    }


def build_context(df):
    last = df.tail(30)
    candles = [candle_features(r) for r in last.itertuples()]

    support = float(last["low"].rolling(10).min().iloc[-1])
    resistance = float(last["high"].rolling(10).max().iloc[-1])

    slope = np.polyfit(np.arange(len(last)), last["close"].values, 1)[0]
    trend = "up" if slope > 0 else "down"

    ctx = {
        "symbol": SYMBOL,
        "tf": f"{INTERVAL}m",
        "price": float(df["close"].iloc[-1]),
        "ema20": float(df["ema20"].iloc[-1]),
        "rsi": float(df["rsi"].iloc[-1]),
        "atr": float(df["atr"].iloc[-1]),
        "support": support,
        "resistance": resistance,
        "trend": trend,
        "slope": float(slope),
        "candles": candles
    }

    return json.dumps(ctx, indent=2)


# ================= IA =================
def extract_json(text):
    if not text:
        return "{}"
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return text[start:end+1]


def call_llama(context, retries=2):
    prompt = f"""
You are a professional trader inspired by Steve Nison.

Rules:
- Use candlestick logic deeply (wicks, bodies, rejection)
- Consider EMA20 as dynamic S/R
- Use RSI context
- Prefer confluence
- If unclear -> HOLD

Return ONLY JSON:
{{
  "decision": "Buy/Sell/Hold",
  "tp1": 0,
  "sl": 0,
  "pattern": "",
  "reasons": ["", "", ""]
}}

Context:
{context}
"""

    last_err = None
    for i in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )
            raw = resp.choices[0].message.content
            log_console(f"RAW IA: {raw}")
            clean = extract_json(raw)
            data = json.loads(clean)

            if "decision" not in data:
                raise ValueError("No decision field")

            return data
        except Exception as e:
            last_err = e
            log_console(f"IA error (try {i+1}): {e}")
            time.sleep(1.5)

    return {"decision": "Hold", "tp1": 0, "sl": 0, "pattern": "", "reasons": ["fallback"]}


# ================= STATE =================
def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return default


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


state = load_json(STATE_FILE, {"current_trade": None})
trades = load_json(TRADES_FILE, [])


# ================= CHART =================
def plot_chart(df):
    d = df.tail(120)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, r in enumerate(d.itertuples()):
        color = 'green' if r.close >= r.open else 'red'
        ax.vlines(i, r.low, r.high, color=color)
        ax.add_patch(plt.Rectangle((i - 0.3, min(r.open, r.close)),
                                   0.6, abs(r.close - r.open), color=color))

    ax.plot(x, d.ema20, label='EMA20')
    ax.set_title(f"{SYMBOL} {INTERVAL}m")
    ax.legend()

    path = 'chart.png'
    plt.savefig(path)
    plt.close()
    return path


# ================= TRADE MGMT =================
def open_trade(side, price, sl, tp1, ctx, ai):
    trade = {
        "id": int(time.time()),
        "side": side,
        "entry": price,
        "sl": sl,
        "tp1": tp1,
        "tp1_hit": False,
        "open_time": now_utc().isoformat(),
        "context": ctx,
        "ai": ai,
        "events": []
    }
    state["current_trade"] = trade
    save_json(STATE_FILE, state)

    caption = (
        f"📊 {SYMBOL} {INTERVAL}m\n"
        f"🧠 {side} @ {price:.2f}\n"
        f"🎯 TP1: {tp1:.2f}\n"
        f"🛑 SL: {sl:.2f}\n"
        f"📌 Pattern: {ai.get('pattern','')}\n"
        f"📋 Reasons: {'; '.join(ai.get('reasons', []))}"
    )

    tg_send(caption)


def close_trade(price, reason):
    trade = state.get("current_trade")
    if not trade:
        return

    trade["close_price"] = price
    trade["close_time"] = now_utc().isoformat()
    trade["result"] = "win" if (
        (trade["side"] == "Buy" and price > trade["entry"]) or
        (trade["side"] == "Sell" and price < trade["entry"]) else "loss"
    )
    trade["close_reason"] = reason

    trades.append(trade)
    save_json(TRADES_FILE, trades)

    state["current_trade"] = None
    save_json(STATE_FILE, state)

    tg_send(f"❌ Trade cerrado ({reason}) @ {price:.2f} -> {trade['result']}")


def manage_trade(df):
    trade = state.get("current_trade")
    if not trade:
        return

    price = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1])

    # TP1
    if not trade["tp1_hit"]:
        if (trade["side"] == "Buy" and price >= trade["tp1"]) or \
           (trade["side"] == "Sell" and price <= trade["tp1"]):
            trade["tp1_hit"] = True
            trade["sl"] = trade["entry"]  # BE
            trade["events"].append({"t": now_utc().isoformat(), "e": "TP1"})
            tg_send(f"✅ TP1 alcanzado. SL a BE @ {trade['sl']:.2f}")
            save_json(STATE_FILE, state)

    # trailing dinámico (runner)
    if trade["tp1_hit"]:
        if trade["side"] == "Buy":
            new_sl = price - atr * TRAILING_FACTOR
            if new_sl > trade["sl"]:
                trade["sl"] = new_sl
                tg_send(f"🔄 Trailing SL actualizado: {trade['sl']:.2f}")
                save_json(STATE_FILE, state)
        else:
            new_sl = price + atr * TRAILING_FACTOR
            if new_sl < trade["sl"]:
                trade["sl"] = new_sl
                tg_send(f"🔄 Trailing SL actualizado: {trade['sl']:.2f}")
                save_json(STATE_FILE, state)

    # SL hit
    if (trade["side"] == "Buy" and price <= trade["sl"]) or \
       (trade["side"] == "Sell" and price >= trade["sl"]):
        close_trade(price, "SL")


# ================= VALIDATION =================
def validate_levels(price, sl, tp1, side):
    if side == "Buy":
        if not (sl < price < tp1):
            return False
    else:
        if not (tp1 < price < sl):
            return False
    return True


# ================= LEARNING =================
def auto_learning():
    if len(trades) and len(trades) % 10 == 0:
        last = trades[-10:]
        wins = sum(1 for t in last if t.get("result") == "win")
        tg_send(f"📊 Últimos 10 trades: {wins}/10 winrate")


# ================= MAIN =================
def run():
    tg_send("🚀 BOT V300 INICIADO")

    while True:
        try:
            df = fetch_klines()
            price = float(df["close"].iloc[-1])

            log_console(f"Precio: {price}")

            # gestionar trade actual
            manage_trade(df)

            if not state.get("current_trade"):
                ctx = build_context(df)
                log_console(f"Contexto enviado a IA: {ctx[:500]}...")

                ai = call_llama(ctx)
                log_console(f"IA decision: {ai}")

                decision = ai.get("decision", "Hold")

                if decision in ("Buy", "Sell"):
                    sl = float(ai.get("sl", 0))
                    tp1 = float(ai.get("tp1", 0))

                    # fallback si IA no define bien niveles
                    atr = float(df["atr"].iloc[-1])
                    if sl == 0 or tp1 == 0:
                        if decision == "Buy":
                            sl = price - atr * ATR_MULT_SL
                            tp1 = price + (price - sl) * TP1_R_MULT
                        else:
                            sl = price + atr * ATR_MULT_SL
                            tp1 = price - (sl - price) * TP1_R_MULT

                    if validate_levels(price, sl, tp1, decision):
                        open_trade(decision, price, sl, tp1, ctx, ai)
                        img = plot_chart(df)
                        tg_send_img(img, f"{decision} setup")
                    else:
                        log_console("Niveles inválidos, se ignora trade")

            auto_learning()

        except Exception as e:
            log_console(f"ERROR LOOP: {e}")
            log_console(traceback.format_exc())
            tg_send(f"⚠️ ERROR: {e}")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    run()
