# BOT TRADING V97.0 BYBIT REAL – GROQ IA (RAILWAY READY)
# ======================================================
# ⚠️ IA GROQ (LLaMA 3.2 Vision) INTEGRADA: Toma de 
# decisiones basada en análisis visual y contexto.
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit (5m)
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
from datetime import datetime, timezone
from PIL import Image

# Configuración crucial para Railway (Servidor sin pantalla)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURACIÓN GROQ API
# ======================================================
from groq import Groq

# Railway inyectará la API KEY desde sus Variables de Entorno
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# Usaremos el modelo Vision de Llama 3.2 en Groq
MODELO_GROQ = "llama-3.2-90b-vision-preview"

plt.rcParams['figure.figsize'] = (12, 6)

# ======================================================
# CONFIGURACIÓN GRÁFICOS Y GENERAL
# ======================================================
GRAFICO_VELAS_LIMIT = 120
MOSTRAR_EMA20 = True

SYMBOL = "BTCUSDT"
INTERVAL = "5"  
RISK_PER_TRADE = 0.02  # 2% de riesgo
LEVERAGE = 10          # 10x de apalancamiento
SLEEP_SECONDS = 60     

MULT_SL = 1.5          
MULT_TP1 = 2.5         
MULT_TRAILING = 2.0    
PORCENTAJE_CIERRE = 0.5 

# ======================================================
# PAPER TRADING (ESTADO DE CUENTA SIMULADA)
# ======================================================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_PNL_GLOBAL = 0.0
PAPER_POSICION_ACTIVA = None
PAPER_PRECIO_ENTRADA = None
PAPER_DECISION_ACTIVA = None
PAPER_SIZE_USD = 0.0
PAPER_SIZE_BTC = 0.0
PAPER_SL = None
PAPER_TP1 = None
PAPER_PARTIAL_ACTIVADO = False
PAPER_SIZE_BTC_RESTANTE = 0.0
PAPER_TP1_EJECUTADO = False
PAPER_PNL_PARCIAL = 0.0  
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0

# CONTROL DINÁMICO DE RIESGO
MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

# ======================================================
# CREDENCIALES BYBIT Y TELEGRAM
# ======================================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL = "https://api.bybit.com"

# ======================================================
# TELEGRAM
# ======================================================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texto}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Error enviando mensaje a Telegram: {e}")

def telegram_grafico(fig):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        archivos = {'photo': buf}
        datos = {'chat_id': TELEGRAM_CHAT_ID}
        requests.post(url, files=archivos, data=datos, timeout=15)
        buf.close()
    except Exception as e:
        print(f"Error enviando foto a Telegram: {e}")

# ======================================================
# DATOS Y TÉCNICO
# ======================================================
def obtener_velas(limit=150):
    url = f"{BASE_URL}/v5/market/kline"
    params = {"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    data_json = r.json()
    data = data_json["result"]["list"][::-1]
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','turnover'])
    
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
        
    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
    df.set_index('time', inplace=True)
    return df

def calcular_indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['ema_touch'] = (df['low'] <= df['ema20']) & (df['high'] >= df['ema20'])
    return df.dropna()

def detectar_zonas_mercado(df, idx=-2, ventana_macro=120):
    df_eval = df.iloc[:idx+1]
    
    soporte_horiz = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia_horiz = df_eval['high'].rolling(40).max().iloc[-1]
    
    if len(df_eval) < ventana_macro:
        y_macro = df_eval['close'].values
    else:
        y_macro = df_eval['close'].values[-ventana_macro:]
        
    x_macro = np.arange(len(y_macro))
    slope, intercept, r_value, p_value, std_err = linregress(x_macro, y_macro)
    
    if slope > 0.01: tendencia_macro = 'ALCISTA'
    elif slope < -0.01: tendencia_macro = 'BAJISTA'
    else: tendencia_macro = 'LATERAL'
        
    linea_central = intercept + slope * x_macro
    desviacion = np.std(y_macro - linea_central)
    canal_sup = linea_central[-1] + (desviacion * 1.5)
    canal_inf = linea_central[-1] - (desviacion * 1.5)
    
    return soporte_horiz, resistencia_horiz, canal_sup, canal_inf, slope, intercept, tendencia_macro

# ======================================================
# GRÁFICA PARA GROQ IA (SIN FLECHAS) Y BASE64
# ======================================================
def generar_imagen_para_ia(df, soporte, resistencia, slope, intercept):
    df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
    x_valores = np.arange(len(df_plot))
    closes = df_plot['close'].values
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(len(df_plot)):
        color_vela = 'green' if closes[i] >= df_plot['open'].values[i] else 'red'
        ax.vlines(x_valores[i], df_plot['low'].values[i], df_plot['high'].values[i], color=color_vela, linewidth=1)
        cuerpo_y = min(df_plot['open'].values[i], closes[i])
        cuerpo_h = max(abs(closes[i] - df_plot['open'].values[i]), 0.0001)
        rect = plt.Rectangle((x_valores[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color_vela, alpha=0.9)
        ax.add_patch(rect)

    ax.axhline(soporte, color='cyan', linestyle='--', linewidth=1.5)
    ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=1.5)

    linea_tendencia = intercept + slope * x_valores
    ax.plot(x_valores, linea_tendencia, color='white', linewidth=1)

    if 'ema20' in df_plot.columns:
        ax.plot(x_valores, df_plot['ema20'].values, color='yellow', linewidth=1.5)

    ax.axis('off') 
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight')
    plt.close(fig)
    
    # Para Groq, necesitamos la imagen en base64
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    
    return base64_image

# ======================================================
# MOTOR IA GROQ (ANÁLISIS TÉCNICO Y VISUAL)
# ======================================================
def analizar_con_groq(df, base64_image, soporte, resistencia, tendencia, idx=-2):
    precio_actual = df['close'].iloc[idx]
    rsi_actual = df['rsi'].iloc[idx]
    ema_actual = df['ema20'].iloc[idx]
    
    toques_ema = df['ema_touch'].iloc[idx-10:idx+1].sum()
    ultimas_velas = df.iloc[idx-4:idx+1][['open', 'high', 'low', 'close']]
    texto_velas = ultimas_velas.to_string()

    prompt = f"""
Eres un Master Trader Institucional. Analiza la imagen del gráfico de 5 minutos proporcionada y los datos técnicos.

DATOS DE CONTEXTO:
- Activo: BTCUSDT (5 Minutos)
- Precio Actual: {precio_actual:.2f}
- Soporte Cercano: {soporte:.2f}
- Resistencia Cercana: {resistencia:.2f}
- Tendencia Macro: {tendencia}
- RSI (14): {rsi_actual:.2f}
- EMA 20: {ema_actual:.2f}
- Rechazos de EMA 20 reciente: {toques_ema}

VELAS RECIENTES:
{texto_velas}

INSTRUCCIONES:
1. Analiza espacialmente la imagen adjunta.
2. Decide si hay una oportunidad de alta probabilidad para 'Buy', 'Sell', o 'Hold'. Sé conservador.

Devuelve tu respuesta en este formato JSON exacto:
{{
  "decision": "Buy" o "Sell" o "Hold",
  "patron_detectado": "Nombre del patron",
  "razones": ["Razon 1", "Razon 2"]
}}
"""
    try:
        completion = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            # Forzamos JSON mode en Groq
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        respuesta_texto = completion.choices[0].message.content
        datos_ia = json.loads(respuesta_texto.strip())
        return datos_ia
        
    except Exception as e:
        print(f"🚨 Error procesando Groq: {e}")
        return {"decision": "Hold", "patron_detectado": "Error API", "razones": ["Error de conexión con Groq"]}

# ======================================================
# GRÁFICA FINAL PARA TELEGRAM (CON FLECHA)
# ======================================================
def generar_grafico_telegram(df, decision, soporte, resistencia, razones, patron):
    df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
    x_valores = np.arange(len(df_plot))
    closes = df_plot['close'].values
    fig, ax = plt.subplots(figsize=(14, 7))

    for i in range(len(df_plot)):
        color_vela = 'green' if closes[i] >= df_plot['open'].values[i] else 'red'
        ax.vlines(x_valores[i], df_plot['low'].values[i], df_plot['high'].values[i], color=color_vela, linewidth=1)
        cuerpo_y = min(df_plot['open'].values[i], closes[i])
        cuerpo_h = max(abs(closes[i] - df_plot['open'].values[i]), 0.0001)
        rect = plt.Rectangle((x_valores[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color_vela, alpha=0.9)
        ax.add_patch(rect)

    ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label="Soporte")
    ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label="Resistencia")

    if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
        ax.plot(x_valores, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA 20')

    entrada_x_idx = len(df_plot) - 2
    if decision == 'Buy':
        ax.scatter(entrada_x_idx, closes[-2]-50, s=300, marker='^', color='lime', edgecolors='black', zorder=5)
    else:
        ax.scatter(entrada_x_idx, closes[-2]+50, s=300, marker='v', color='red', edgecolors='black', zorder=5)

    texto_razones = "\n".join(razones)
    texto_panel = f"GROQ IA: {decision.upper()}\nPatrón: {patron}\nPrecio: {df['close'].iloc[-1]:.2f}\n\nRazonamiento IA:\n{texto_razones}"
    
    ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax.set_title(f"BOT V97.0 - Decisión IA GROQ VISION (5m)")
    ax.grid(True, alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig

# ======================================================
# MOTOR FINANCIERO Y GESTIÓN
# ======================================================
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY, PAPER_BALANCE
    hoy_utc = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy_utc:
        PAPER_CURRENT_DAY = hoy_utc
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
    
    porcentaje_drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if porcentaje_drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
        if not PAPER_STOPPED_TODAY:
            telegram_mensaje(f"🛑 PROTECCIÓN DE CAPITAL: Drawdown máximo alcanzado. Bot pausado.")
            PAPER_STOPPED_TODAY = True
        return False
    return True

def paper_abrir_posicion(decision, precio, atr):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_SIZE_BTC
    global PAPER_DECISION_ACTIVA, PAPER_TP1_EJECUTADO, PAPER_SIZE_BTC_RESTANTE

    if PAPER_POSICION_ACTIVA is not None: return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    
    if decision == "Buy":
        sl = precio - (atr * MULT_SL)
        tp1 = precio + (atr * MULT_TP1)
    else:
        sl = precio + (atr * MULT_SL)
        tp1 = precio - (atr * MULT_TP1)
    
    distancia_riesgo = abs(precio - sl)
    if distancia_riesgo == 0: return False

    size_en_dolares = min((riesgo_usd / distancia_riesgo) * precio, PAPER_BALANCE * LEVERAGE)
    
    PAPER_POSICION_ACTIVA = decision
    PAPER_DECISION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL = sl
    PAPER_TP1 = tp1
    PAPER_SIZE_BTC = size_en_dolares / precio
    PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC
    PAPER_TP1_EJECUTADO = False

    return True

def paper_revisar_sl_tp(df):
    global PAPER_SL, PAPER_TP1, PAPER_PRECIO_ENTRADA, PAPER_POSICION_ACTIVA, PAPER_DECISION_ACTIVA
    global PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_TRADES_TOTALES, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE
    global PAPER_TP1_EJECUTADO, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS

    if PAPER_POSICION_ACTIVA is None: return None

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    close = df['close'].iloc[-1]
    atr_actual = df['atr'].iloc[-1]
    
    cerrar_total = False
    motivo = None

    if PAPER_POSICION_ACTIVA == "Buy":
        if not PAPER_TP1_EJECUTADO and high >= PAPER_TP1:
            PAPER_PNL_PARCIAL = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_BALANCE += PAPER_PNL_PARCIAL
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA 
            telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even.")

        if PAPER_TP1_EJECUTADO:
            if close - (atr_actual * MULT_TRAILING) > PAPER_SL: PAPER_SL = close - (atr_actual * MULT_TRAILING)

        if low <= PAPER_SL:
            cerrar_total = True
            motivo = "Trailing Dinámico" if PAPER_TP1_EJECUTADO else "Stop Loss"

    elif PAPER_POSICION_ACTIVA == "Sell":
        if not PAPER_TP1_EJECUTADO and low <= PAPER_TP1:
            PAPER_PNL_PARCIAL = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_BALANCE += PAPER_PNL_PARCIAL
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA 
            telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even.")

        if PAPER_TP1_EJECUTADO:
            if close + (atr_actual * MULT_TRAILING) < PAPER_SL: PAPER_SL = close + (atr_actual * MULT_TRAILING)

        if high >= PAPER_SL:
            cerrar_total = True
            motivo = "Trailing Dinámico" if PAPER_TP1_EJECUTADO else "Stop Loss"

    if cerrar_total:
        pnl_final = (PAPER_SL - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - PAPER_SL) * PAPER_SIZE_BTC_RESTANTE
        PAPER_BALANCE += pnl_final
        PAPER_TRADES_TOTALES += 1
        
        pnl_total_trade = PAPER_PNL_PARCIAL + pnl_final if PAPER_TP1_EJECUTADO else pnl_final
        PAPER_WIN += 1 if pnl_total_trade > 0 else 0
        PAPER_LOSS += 1 if pnl_total_trade <= 0 else 0
        
        winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0.0
        
        telegram_mensaje(f"📤 TRADE CERRADO: {motivo}.\n💵 G/P Neta: {pnl_total_trade:.2f} USD\n📊 Balance Actual: {PAPER_BALANCE:.2f} USD\nWinrate: {winrate:.1f}%")
        
        PAPER_POSICION_ACTIVA = None
        return True

    return None

# ======================================================
# LOOP PRINCIPAL
# ======================================================
def run_bot():
    print("🤖 BOT V97.0 INICIADO: Iniciando conexión con Railway (GROQ VISION)...")
    telegram_mensaje("🤖 BOT V97.0 INICIADO: Análisis IA GROQ LLaMA 3.2 Vision en vivo desde Railway.")
    ultima_vela_operada = None

    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            idx_eval = -2
            precio_mercado = df['close'].iloc[-1] 
            tiempo_vela_cerrada = df.index[-2] 

            soporte_horiz, resistencia_horiz, canal_sup, canal_inf, slope, intercept, tendencia = detectar_zonas_mercado(df, idx_eval)
            
            if PAPER_POSICION_ACTIVA is None and ultima_vela_operada != tiempo_vela_cerrada:
                print(f"[{datetime.now(timezone.utc)}] Consultando a Groq IA...")
                
                # Obtenemos la imagen en Base64 para LLaMA Vision
                imagen_base64 = generar_imagen_para_ia(df, soporte_horiz, resistencia_horiz, slope, intercept)
                
                # Consultamos a GROQ
                respuesta_ia = analizar_con_groq(df, imagen_base64, soporte_horiz, resistencia_horiz, tendencia, idx_eval)
                
                decision = respuesta_ia.get("decision", "Hold")
                razones = respuesta_ia.get("razones", [])
                patron = respuesta_ia.get("patron_detectado", "Ninguno")

                print(f"Decisión IA: {decision} | Patrón: {patron}")

                if decision in ["Buy", "Sell"] and risk_management_check():
                    atr_entrada = df['atr'].iloc[-1]
                    if paper_abrir_posicion(decision, precio_mercado, atr_entrada):
                        ultima_vela_operada = tiempo_vela_cerrada
                        
                        texto_razones = "\n".join([f"🧠 {r}" for r in razones])
                        telegram_mensaje(f"📌 IA OPERACIÓN {decision.upper()} (5m)\n💰 Precio: {precio_mercado:.2f}\n📍 SL: {PAPER_SL:.2f} | TP1: {PAPER_TP1:.2f}\n👁️ Patrón visto: {patron}\n{texto_razones}")
                        
                        fig = generar_grafico_telegram(df, decision, soporte_horiz, resistencia_horiz, razones, patron)
                        telegram_grafico(fig)
                        plt.close(fig)

            if PAPER_POSICION_ACTIVA is not None:
                paper_revisar_sl_tp(df)

            time.sleep(SLEEP_SECONDS) 

        except Exception as e:
            print(f"🚨 ERROR CRÍTICO EN BUCLE MAIN: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
