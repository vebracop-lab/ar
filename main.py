
# BOT TRADING V104.0 - FULL IA (SIN PATRONES)
# ===========================================
# - Eliminados TODOS los patrones de velas
# - IA (Groq) toma TODAS las decisiones
# - Usa EMA + RSI + ATR + contexto + imagen
# - TP1 50% + TP2 dinámico con trailing inteligente

import os
import time
import io
import base64
import json
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq

# ================= CONFIG =================
SYMBOL = "BTCUSDT"
INTERVAL = "5"
SLEEP_SECONDS = 60

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ================= DATA =================
def obtener_velas():
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category":"linear","symbol":SYMBOL,"interval":INTERVAL,"limit":200}
    data = requests.get(url, params=params).json()["result"]["list"][::-1]
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','turnover'])

    for col in ['open','high','low','close']:
        df[col] = df[col].astype(float)

    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms')
    df.set_index('time', inplace=True)

    return df

def indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()

    tr = pd.concat([
        df['high']-df['low'],
        (df['high']-df['close'].shift()).abs(),
        (df['low']-df['close'].shift()).abs()
    ], axis=1).max(axis=1)

    df['atr'] = tr.rolling(14).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + rs))

    return df.dropna()

# ================= IA =================
def analizar_con_groq(df, img_b64):
    if not client_groq:
        return "WAIT", {}

    precio = df['close'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    ema20 = df['ema20'].iloc[-1]
    ema50 = df['ema50'].iloc[-1]
    atr = df['atr'].iloc[-1]

    prompt = f"""
    Eres trader institucional.

    Datos:
    Precio: {precio}
    RSI: {rsi}
    EMA20: {ema20}
    EMA50: {ema50}
    ATR: {atr}

    Analiza:
    - estructura
    - tendencia
    - momentum
    - velas
    - EMA como soporte/resistencia

    Decide:
    BUY / SELL / WAIT

    JSON:
    {{
      "decision":"BUY|SELL|WAIT",
      "confidence":0-100,
      "trailing":"AGGRESSIVE|NORMAL|CONSERVATIVE",
      "razon":"explicacion"
    }}
    """

    try:
        res = client_groq.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role":"user",
                "content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{img_b64}"}}
                ]
            }]
        )

        txt = res.choices[0].message.content
        data = json.loads(re.search(r'\{.*\}', txt, re.S).group())

        return data["decision"], data

    except Exception as e:
        return "WAIT", {"error":str(e)}

# ================= BOT =================
class Bot:
    def __init__(self):
        self.pos = None
        self.entry = None
        self.sl = None
        self.tp1 = None
        self.partial = False
        self.trailing_mode = "NORMAL"

    def abrir(self, side, price, atr, trailing):
        self.pos = side
        self.entry = price
        self.trailing_mode = trailing

        if side == "BUY":
            self.sl = price - atr * 1.5
            self.tp1 = price + atr * 2.5
        else:
            self.sl = price + atr * 1.5
            self.tp1 = price - atr * 2.5

    def gestionar(self, price, atr):
        if not self.pos:
            return

        # TP1
        if not self.partial:
            if (self.pos=="BUY" and price>=self.tp1) or (self.pos=="SELL" and price<=self.tp1):
                self.partial = True
                self.sl = self.entry
                print("TP1 -> SL BE")

        # trailing dinámico
        if self.partial:
            dist = abs(price - self.entry) / atr

            if dist >= 6:
                mult = 0.6
            elif dist >= 4:
                mult = 0.9
            elif dist >= 2:
                mult = 1.2
            else:
                mult = 1.8

            if self.trailing_mode == "AGGRESSIVE":
                mult *= 0.7
            elif self.trailing_mode == "CONSERVATIVE":
                mult *= 1.3

            if self.pos == "BUY":
                new_sl = price - atr * mult
                if new_sl > self.sl:
                    self.sl = new_sl
            else:
                new_sl = price + atr * mult
                if new_sl < self.sl:
                    self.sl = new_sl

        # cierre
        if (self.pos=="BUY" and price<=self.sl) or (self.pos=="SELL" and price>=self.sl):
            print("Cierre trade")
            self.reset()

    def reset(self):
        self.pos = None
        self.entry = None
        self.sl = None
        self.tp1 = None
        self.partial = False

# ================= LOOP =================
bot = Bot()

while True:
    try:
        df = indicadores(obtener_velas())
        price = df['close'].iloc[-1]

        # gráfico simple
        fig = plt.figure()
        plt.plot(df['close'])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        decision, data = analizar_con_groq(df, img_b64)

        if bot.pos is None and decision in ["BUY","SELL"]:
            bot.abrir(decision, price, df['atr'].iloc[-1], data.get("trailing","NORMAL"))
            print("ENTRADA:", decision, data)

        bot.gestionar(price, df['atr'].iloc[-1])

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("ERROR:", e)
        time.sleep(10)
