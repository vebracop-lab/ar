
# BOT TRADING V105.0 - IA FULL + LOG PRO + SIN ATR + SIN PATRONES
# ================================================================
# ✔ IA decide TODO (Groq)
# ✔ EMA20 / EMA50
# ✔ Soportes / Resistencias
# ✔ Tendencia con slope
# ✔ Gráfico profesional
# ✔ Logs completos + estadísticas
# ✔ TP1 fijo + TP2 trailing SIN ATR

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
from datetime import datetime
from groq import Groq
from scipy.stats import linregress

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

    x = np.arange(len(df))
    slope, intercept, _, _, _ = linregress(x, df['close'])

    df['trend'] = slope
    return df.dropna(), slope, intercept

# ================= SOPORTES =================
def soportes_resistencias(df):
    soporte = df['low'].rolling(20).min().iloc[-1]
    resistencia = df['high'].rolling(20).max().iloc[-1]
    return soporte, resistencia

# ================= IA =================
def analizar_con_groq(df, img_b64, soporte, resistencia, slope):
    if not client_groq:
        return "WAIT", {}

    precio = df['close'].iloc[-1]
    ema20 = df['ema20'].iloc[-1]
    ema50 = df['ema50'].iloc[-1]

    prompt = f"""
    Eres un trader institucional.

    Datos:
    Precio: {precio}
    EMA20: {ema20}
    EMA50: {ema50}
    Soporte: {soporte}
    Resistencia: {resistencia}
    Pendiente tendencia: {slope}

    Analiza TODO el gráfico.

    Devuelve JSON:
    {{
      "decision":"BUY|SELL|WAIT",
      "confidence":0-100,
      "sl":precio,
      "tp1":precio,
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
        self.balance = 100
        self.trades = 0
        self.wins = 0

    def abrir(self, side, price, sl, tp1):
        self.pos = side
        self.entry = price
        self.sl = sl
        self.tp1 = tp1
        self.partial = False

    def gestionar(self, price):
        if not self.pos:
            return

        if not self.partial:
            if (self.pos=="BUY" and price>=self.tp1) or (self.pos=="SELL" and price<=self.tp1):
                self.partial = True
                self.sl = self.entry
                print("TP1 alcanzado")

        if self.partial:
            profit = abs(price - self.entry)
            if self.pos == "BUY":
                new_sl = price - profit*0.3
                if new_sl > self.sl:
                    self.sl = new_sl
            else:
                new_sl = price + profit*0.3
                if new_sl < self.sl:
                    self.sl = new_sl

        if (self.pos=="BUY" and price<=self.sl) or (self.pos=="SELL" and price>=self.sl):
            self.trades += 1
            if (self.pos=="BUY" and price>self.entry) or (self.pos=="SELL" and price<self.entry):
                self.wins += 1
                self.balance += 1
            else:
                self.balance -= 1

            print("Cierre trade")
            self.reset()

    def reset(self):
        self.pos = None
        self.entry = None
        self.sl = None
        self.tp1 = None
        self.partial = False

    def stats(self):
        winrate = (self.wins/self.trades*100) if self.trades>0 else 0
        return f"Balance: {self.balance} | Trades: {self.trades} | Winrate: {winrate:.2f}%"

# ================= LOOP =================
bot = Bot()

while True:
    try:
        df_raw = obtener_velas()
        df, slope, intercept = indicadores(df_raw)

        soporte, resistencia = soportes_resistencias(df)

        price = df['close'].iloc[-1]

        # gráfico
        fig = plt.figure()
        plt.plot(df['close'])
        plt.axhline(soporte)
        plt.axhline(resistencia)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        decision, data = analizar_con_groq(df, img_b64, soporte, resistencia, slope)

        print("="*80)
        print(datetime.now())
        print("Precio:", price)
        print("Tendencia slope:", slope)
        print("Soporte:", soporte, "Resistencia:", resistencia)
        print("IA:", data)

        if bot.pos is None and decision in ["BUY","SELL"]:
            bot.abrir(decision, price, data.get("sl",price), data.get("tp1",price))
            print("ENTRADA:", decision)

        bot.gestionar(price)

        print(bot.stats())
        print("="*80)

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("ERROR:", e)
        time.sleep(10)
