import os
import time
import json
import requests
import numpy as np
import pandas as pd
from groq import Groq
from scipy.stats import linregress

# ==========================
# CONFIG
# ==========================
SYMBOL = "BTCUSDT"
INTERVAL = "5"
SLEEP_SECONDS = 60

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ==========================
# DATA
# ==========================
def obtener_velas():
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": 200
    }

    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(
        data["result"]["list"],
        columns=['time','open','high','low','close','volume','turnover']
    )

    df = df.iloc[::-1]

    df[['open','high','low','close','volume']] = df[
        ['open','high','low','close','volume']
    ].astype(float)

    return df

# ==========================
# INDICADORES
# ==========================
def indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['atr'] = (df['high'] - df['low']).rolling(14).mean()

    return df.dropna()

# ==========================
# SOPORTE / RESISTENCIA
# ==========================
def soporte_resistencia(df):
    soporte = df['low'].rolling(40).min().iloc[-2]
    resistencia = df['high'].rolling(40).max().iloc[-2]
    return soporte, resistencia

# ==========================
# TENDENCIA
# ==========================
def tendencia(df):
    y = df['close'].values[-100:]
    x = np.arange(len(y))

    slope, _, _, _, _ = linregress(x, y)

    if slope > 0:
        t = "alcista"
    elif slope < 0:
        t = "bajista"
    else:
        t = "lateral"

    return t, slope

# ==========================
# RECHAZOS
# ==========================
def contar_rechazos(df, nivel, tolerancia):
    rechazos = 0

    for i in range(-15, -1):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        if abs(high - nivel) <= tolerancia or abs(low - nivel) <= tolerancia:
            rechazos += 1

    return rechazos

# ==========================
# PROMPT IA
# ==========================
def construir_prompt(ctx):
    return f"""
Eres un trader institucional experto en scalping en 5 minutos.

Devuelve SOLO JSON:

{{
"decision":"BUY|SELL|NO_TRADE",
"confidence":0-1,
"entry":0,
"sl":0,
"tp1":0,
"tp2":0,
"reason":""
}}

Contexto:

Precio actual: {ctx['precio']}

Tendencia:
- Tipo: {ctx['tendencia']}
- Fuerza: {ctx['slope']}

Soporte: {ctx['soporte']}
Resistencia: {ctx['resistencia']}

Rechazos:
- Soporte: {ctx['r_soporte']}
- Resistencia: {ctx['r_resistencia']}
- EMA20: {ctx['r_ema']}

Indicadores:
- EMA20: {ctx['ema']}
- RSI: {ctx['rsi']}
- ATR: {ctx['atr']}

Reglas:
- TP1 corto (1:1 aprox)
- TP2 libre (dejar correr tendencia)
- SL lógico (estructura)
- Evita rango sin dirección
- Prioriza rechazos múltiples
- Si no hay claridad: NO_TRADE
"""

# ==========================
# IA
# ==========================
def decision_ia(prompt):
    try:
        r = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b",
            temperature=0.2
        )

        respuesta = r.choices[0].message.content.strip()

        if "```" in respuesta:
            respuesta = respuesta.split("```")[1]

        data = json.loads(respuesta)

        return data

    except Exception as e:
        print("ERROR IA:", e)
        print("RAW:", respuesta if 'respuesta' in locals() else "N/A")
        return None

# ==========================
# ESTADO
# ==========================
posicion = None
entry = 0
sl = 0
tp1 = 0
tp1_hit = False

# ==========================
# LOOP
# ==========================
while True:
    try:
        df = obtener_velas()
        df = indicadores(df)

        precio = df['close'].iloc[-1]

        soporte, resistencia = soporte_resistencia(df)
        tend, slope = tendencia(df)

        ctx = {
            "precio": precio,
            "tendencia": tend,
            "slope": slope,
            "soporte": soporte,
            "resistencia": resistencia,
            "ema": df['ema20'].iloc[-1],
            "rsi": df['rsi'].iloc[-1],
            "atr": df['atr'].iloc[-1],
            "r_soporte": contar_rechazos(df, soporte, 50),
            "r_resistencia": contar_rechazos(df, resistencia, 50),
            "r_ema": contar_rechazos(df, df['ema20'].iloc[-1], 50)
        }

        if posicion is None:
            prompt = construir_prompt(ctx)
            data = decision_ia(prompt)

            if data:
                decision = data["decision"]
                confidence = data["confidence"]

                if decision != "NO_TRADE" and confidence >= 0.65:
                    posicion = decision
                    entry = data["entry"]
                    sl = data["sl"]
                    tp1 = data["tp1"]
                    tp1_hit = False

                    print("\nNUEVA OPERACION")
                    print(data)

        else:
            if not tp1_hit:
                if (posicion == "BUY" and precio >= tp1) or (posicion == "SELL" and precio <= tp1):
                    tp1_hit = True
                    sl = entry
                    print("TP1 alcanzado -> SL BE")

            if tp1_hit:
                if posicion == "BUY":
                    nuevo_sl = precio - (ctx["atr"] * 1.2)
                    if nuevo_sl > sl:
                        sl = nuevo_sl
                else:
                    nuevo_sl = precio + (ctx["atr"] * 1.2)
                    if nuevo_sl < sl:
                        sl = nuevo_sl

            if (posicion == "BUY" and precio <= sl) or (posicion == "SELL" and precio >= sl):
                print("CIERRE POR TRAILING / SL")
                posicion = None

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("ERROR:", e)
        time.sleep(60)
