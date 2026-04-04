# BOT TRADING V98.0 BYBIT REAL – GROQ IA (GPT-OSS-120B)
# ======================================================
# ⚠️ IA GROQ INTEGRADA: Sustituye patrones matemáticos
# por decisiones basadas en visión y contexto técnico.
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit (5m)
# ======================================================
# NOVEDADES:
# - Modelo: openai/gpt-oss-120b (Groq)
# - Sin simplificaciones: Se mantiene TODA la estructura previa.
# - Integración de indicadores: EMA 20, RSI, ATR.
# - Análisis espacial de velas mediante captura de pantalla.
# ======================================================

import os
import time
import io
import hmac
import hashlib
import requests
import json
import base64
import numpy as np
import pandas as pd
from scipy.stats import linregress
from datetime import datetime, timezone, timedelta
from PIL import Image

# Configuración crucial para Railway (Servidor sin pantalla)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURACIÓN GROQ Y GEMINI (ENTORNO)
# ======================================================
from groq import Groq
import google.generativeai as genai

# Railway inyectará las llaves desde las variables de entorno
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") # Espacio para Gemini configurado

client_groq = Groq(api_key=GROQ_API_KEY)
MODELO_GROQ = "openai/gpt-oss-120b"

# Configuración de Gemini (Si se desea usar como backup)
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

plt.rcParams['figure.figsize'] = (12, 6)

# ======================================================
# CONFIGURACIÓN GRÁFICOS Y GENERAL
# ======================================================

GRAFICO_VELAS_LIMIT = 120
SYMBOL = "BTCUSDT"
INTERVAL = "5"
LEVERAGE = 10
RISK_PER_TRADE = 0.02

# ======================================================
# VARIABLES DE ESTADO Y PAPER TRADING (PRODUCCIÓN)
# ======================================================

PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = 100.0
PAPER_POSICION_ACTIVA = None 
PAPER_PRECIO_ENTRADA = 0.0
PAPER_SL = 0.0
PAPER_TP1 = 0.0
PAPER_PNL_GLOBAL = 0.0
PAPER_TRADES_TOTALES = 0
PAPER_WIN = 0
PAPER_LOSS = 0

ultima_vela_operada = None

# ======================================================
# BLOQUE 1: OBTENCIÓN DE DATOS (BYBIT API)
# ======================================================

def obtener_velas_bybit(limit=250):
    url = f"https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        data = response.json()
        if "result" in data and "list" in data["result"]:
            kline_list = data["result"]["list"][::-1]
            df = pd.DataFrame(kline_list, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms')
            
            # --- CÁLCULO DE INDICADORES (NO SIMPLIFICADO) ---
            # ATR (14)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # RSI (14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # EMA 20
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            return df.dropna()
        return None
    except Exception as e:
        print(f"🚨 Error Bybit API: {e}")
        return None

# ======================================================
# BLOQUE 2: TELEGRAM Y COMUNICACIÓN EXTERNA
# ======================================================

def telegram_mensaje(texto):
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": texto, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def telegram_grafico(fig):
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try: requests.post(url, data={"chat_id": chat_id}, files={"photo": buf}, timeout=20)
    except: pass

# ======================================================
# BLOQUE 3: ANÁLISIS TÉCNICO (SOPORTES, TENDENCIAS)
# ======================================================

def calcular_niveles_dinamicos(df, ventana=50):
    soporte = df['low'].rolling(window=ventana).min().iloc[-1]
    resistencia = df['high'].rolling(window=ventana).max().iloc[-1]
    return soporte, resistencia

def calcular_tendencia_detallada(df, periodos=20):
    y = df['close'].tail(periodos).values
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    tendencia_str = "ALCISTA 🟢" if slope > 0 else "BAJISTA 🔴"
    return tendencia_str, slope, intercept

# ======================================================
# BLOQUE 4: MOTOR DE INTELIGENCIA ARTIFICIAL (VISION)
# ======================================================

def preparar_frame_ia(df, soporte, resistencia):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Velas Japonesas
    for i in range(len(df_plot)):
        c = 'green' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else 'red'
        ax.vlines(i, df_plot['low'].iloc[i], df_plot['high'].iloc[i], color=c, linewidth=1.5)
        ax.bar(i, abs(df_plot['close'].iloc[i] - df_plot['open'].iloc[i]), 
               bottom=min(df_plot['open'].iloc[i], df_plot['close'].iloc[i]), color=c, width=0.6)
    
    # Indicadores en Gráfica
    ax.axhline(soporte, color='blue', linestyle='--', alpha=0.5, label="Soporte")
    ax.axhline(resistencia, color='orange', linestyle='--', alpha=0.5, label="Resistencia")
    ax.plot(df_plot['ema20'].values, color='purple', alpha=0.8, label="EMA 20")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def consultar_ia_maestro(df, img_b64, soporte, resistencia, tendencia):
    ultimo = df.iloc[-1]
    
    prompt = f"""
    SISTEMA DE DECISIÓN TRADING MAESTRO V98.
    Analiza la imagen adjunta de BTC/USDT (5m) y los datos técnicos:
    
    - RSI: {ultimo['rsi']:.2f}
    - EMA 20: {ultimo['ema20']:.2f}
    - ATR: {ultimo['atr']:.2f}
    - Tendencia: {tendencia}
    - Soporte: {soporte} | Resistencia: {resistencia}
    - Precio Actual: {ultimo['close']}

    REGLAS INSTITUCIONALES (Brooks/Nison):
    1. Identifica patrones de velas (Pin Bar, Engulfing, Doji, etc.) espacialmente.
    2. Cruza con indicadores: RSI > 70 o < 30, y posición respecto a la EMA 20.
    3. Evalúa si el precio está en la 'Zona de Ejecución' (Soporte o Resistencia).
    
    Responde estrictamente en JSON:
    {{
        "decision": "Buy", "Sell" o "Hold",
        "patron_visto": "Nombre detallado del patrón",
        "analisis_espacial": "Descripción de lo que ves en las velas",
        "justificacion": ["Razón 1", "Razón 2", "Razón 3"]
    }}
    """
    
    try:
        completion = client_groq.chat.completions.create(
            model=MODELO_GROQ,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"🚨 Error Motor IA Groq: {e}")
        return {"decision": "Hold", "patron_visto": "Error API", "justificacion": [str(e)]}

# ======================================================
# BLOQUE 5: GESTIÓN DE RIESGO Y PAPER TRADING
# ======================================================

def abrir_posicion_simulada(tipo, precio, atr):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1
    PAPER_POSICION_ACTIVA = tipo
    PAPER_PRECIO_ENTRADA = precio
    distancia_riesgo = max(atr * 1.5, 80.0)
    
    if tipo == "Buy":
        PAPER_SL = precio - distancia_riesgo
        PAPER_TP1 = precio + (distancia_riesgo * 2.0)
    else:
        PAPER_SL = precio + distancia_riesgo
        PAPER_TP1 = precio - (distancia_riesgo * 2.0)
    return True

def revisar_operaciones_abiertas(df):
    global PAPER_POSICION_ACTIVA, PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_TRADES_TOTALES, PAPER_WIN, PAPER_LOSS
    if PAPER_POSICION_ACTIVA is None: return

    precio_actual = df['close'].iloc[-1]
    ha_cerrado = False
    pnl_unidades = 0

    if PAPER_POSICION_ACTIVA == "Buy":
        if precio_actual <= PAPER_SL:
            pnl_unidades = PAPER_SL - PAPER_PRECIO_ENTRADA
            ha_cerrado = True; PAPER_LOSS += 1
        elif precio_actual >= PAPER_TP1:
            pnl_unidades = PAPER_TP1 - PAPER_PRECIO_ENTRADA
            ha_cerrado = True; PAPER_WIN += 1
            
    elif PAPER_POSICION_ACTIVA == "Sell":
        if precio_actual >= PAPER_SL:
            pnl_unidades = PAPER_PRECIO_ENTRADA - PAPER_SL
            ha_cerrado = True; PAPER_LOSS += 1
        elif precio_actual <= PAPER_TP1:
            pnl_unidades = PAPER_PRECIO_ENTRADA - PAPER_TP1
            ha_cerrado = True; PAPER_WIN += 1

    if ha_cerrado:
        pnl_usd = (pnl_unidades / PAPER_PRECIO_ENTRADA) * PAPER_BALANCE * LEVERAGE
        PAPER_BALANCE += pnl_usd
        PAPER_PNL_GLOBAL += pnl_usd
        PAPER_TRADES_TOTALES += 1
        PAPER_POSICION_ACTIVA = None
        
        msg_cierre = f"🏁 <b>CIERRE DE TRADE</b>\nResultado: {'WIN 🟢' if pnl_usd > 0 else 'LOSS 🔴'}\n"
        msg_cierre += f"PnL Trade: {pnl_usd:.2f} USD\nBalance: {PAPER_BALANCE:.2f} USD\nGlobal: {PAPER_PNL_GLOBAL:.2f} USD"
        telegram_mensaje(msg_cierre)

# ======================================================
# BLOQUE 6: HEARTBEAT Y LOGS DETALLADOS
# ======================================================

def log_sistema_completo(df, tendencia, decision, patron, justificacion, soporte, resistencia):
    ultimo = df.iloc[-1]
    ahora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log_txt = f"""
╔════════════ HEARTBEAT SISTEMA V98 ════════════╗
  🕒 Tiempo: {ahora}
  📈 Tendencia: {tendencia} | RSI: {ultimo['rsi']:.1f}
  📊 Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}
  🎯 Decisión IA: {decision} | Patrón: {patron}
  🧠 Justificación: {justificacion}
  💰 PnL Global: {PAPER_PNL_GLOBAL:.2f} | Trades: {PAPER_TRADES_TOTALES}
╚═══════════════════════════════════════════════╝
    """
    print(log_txt)

# ======================================================
# BUCLE PRINCIPAL (MAIN LOOP)
# ======================================================

def ejecutar_bot_maestro():
    global ultima_vela_operada
    print(f"🔥 TRADING MAESTRO ACTIVADO | MODELO: {MODELO_GROQ}")
    telegram_mensaje(f"🚀 Bot V98 Online - Modelo {MODELO_GROQ}")

    while True:
        try:
            # A. Obtener datos Bybit
            df = obtener_velas_bybit()
            if df is None:
                print("⏳ Esperando respuesta de Bybit..."); time.sleep(30); continue

            # B. Cálculos Técnicos
            soporte_h, resistencia_h = calcular_niveles_dinamicos(df)
            tendencia, slope, intercept = calcular_tendencia_detallada(df)
            
            tiempo_vela_actual = df.index[-1]
            precio_mercado = df['close'].iloc[-1]

            # C. Gestión de posición abierta (Prioridad 1)
            revisar_operaciones_abiertas(df)

            # D. Análisis de Nueva Entrada (Prioridad 2)
            if PAPER_POSICION_ACTIVA is None and tiempo_vela_actual != ultima_vela_operada:
                
                # Generar imagen para la IA
                img_b64 = preparar_frame_ia(df, soporte_h, resistencia_h)
                
                # Consultar IA
                resultado_ia = consultar_ia_maestro(df, img_b64, soporte_h, resistencia_h, tendencia)
                
                decision = resultado_ia.get("decision", "Hold")
                patron = resultado_ia.get("patron_visto", "Ninguno")
                justificacion = ". ".join(resultado_ia.get("justificacion", []))

                # Logs detallados en consola
                log_sistema_completo(df, tendencia, decision, patron, justificacion, soporte_h, resistencia_h)

                if decision in ["Buy", "Sell"]:
                    # Regla de Zona Operativa (Espacial)
                    cerca_soporte = (precio_mercado <= soporte_h * 1.002)
                    cerca_resistencia = (precio_mercado >= resistencia_h * 0.998)
                    
                    if cerca_soporte or cerca_resistencia:
                        if abrir_posicion_simulada(decision, precio_mercado, df['atr'].iloc[-1]):
                            ultima_vela_operada = tiempo_vela_actual
                            
                            # Gráfico de Entrada para Telegram
                            fig_entrada = generar_grafico_visual(df, soporte_h, resistencia_h, tendencia, decision)
                            
                            info_msg = f"🎯 <b>NUEVA OPERACIÓN: {decision.upper()}</b>\n"
                            info_msg += f"🕯️ Patrón: {patron}\n"
                            info_msg += f"🧠 Razón: {justificacion}\n"
                            info_msg += f"📍 Precio: {precio_mercado:.2f}\n"
                            info_msg += f"🛑 SL: {PAPER_SL:.2f} | ✅ TP: {PAPER_TP1:.2f}"
                            
                            telegram_mensaje(info_msg)
                            telegram_grafico(fig_entrada)
                            plt.close(fig_entrada)
                    else:
                        print(f"⚠️ <b>Patrón detectado pero fuera de zona operativa</b>")

            # E. Control de Tiempo
            if PAPER_POSICION_ACTIVA is None:
                time.sleep(120) # Revisión cada 2 min si no hay trade
            else:
                time.sleep(30)  # Revisión cada 30 seg si hay posición activa

        except Exception as error_general:
            print(f"🚨 ERROR CRÍTICO: {error_general}")
            time.sleep(60)

# Función auxiliar para gráficos de Telegram (Mantiene el código extenso)
def generar_grafico_visual(df, soporte, resistencia, tendencia, decision):
    df_p = df.tail(80)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(df_p)):
        c = 'green' if df_p['close'].iloc[i] >= df_p['open'].iloc[i] else 'red'
        ax.vlines(i, df_p['low'].iloc[i], df_p['high'].iloc[i], color=c)
        ax.bar(i, abs(df_p['close'].iloc[i] - df_p['open'].iloc[i]), 
               bottom=min(df_p['open'].iloc[i], df_p['close'].iloc[i]), color=c)
    ax.axhline(soporte, color='blue', linestyle='--')
    ax.axhline(resistencia, color='orange', linestyle='--')
    plt.title(f"BTC 5m - Decision: {decision}")
    return fig

if __name__ == "__main__":
    ejecutar_bot_maestro()
