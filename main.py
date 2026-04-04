# BOT TRADING V98.1 BYBIT REAL – GROQ IA (RAILWAY READY)
# ======================================================
# ⚠️ IA GROQ (LLaMA 3.3 Versatile): Toma de 
# decisiones basada en análisis de contexto textual detallado.
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
import textwrap  # NUEVO: Para envolver textos largos en los gráficos
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

# Cambio al modelo Llama 3.3 70b Versatile
MODELO_GROQ = "llama-3.3-70b-versatile"

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
PAPER_LAST_10_PNL = [] # Historial para el log

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
# MOTOR IA GROQ (ANÁLISIS TEXTUAL)
# ======================================================
def analizar_con_groq_texto(df, soporte, resistencia, tendencia, slope, intercept, idx=-2):
    precio_actual = df['close'].iloc[idx]
    rsi_actual = df['rsi'].iloc[idx]
    ema_actual = df['ema20'].iloc[idx]
    
    toques_ema = df['ema_touch'].iloc[idx-10:idx+1].sum()
    ultimas_velas = df.iloc[idx-6:idx+1][['open', 'high', 'low', 'close']]
    texto_velas = ultimas_velas.to_string()

    prompt = f"""
Eres un Master Trader Institucional algorítmico.
Analiza exhaustivamente el siguiente contexto de mercado en texto para tomar una decisión.

DATOS DE CONTEXTO TÉCNICO Y ESPACIAL:
- Activo: BTCUSDT (Gráfico de 5 Minutos)
- Precio Actual: {precio_actual:.2f}
- Soporte Cercano: {soporte:.2f}
- Resistencia Cercana: {resistencia:.2f}
- Tendencia Macro: {tendencia}
- Pendiente (Slope) de Tendencia: {slope:.6f}
- Intercepto de Tendencia: {intercept:.2f}
- RSI (14): {rsi_actual:.2f}
- EMA 20: {ema_actual:.2f}
- Rechazos recientes en la EMA 20: {toques_ema} veces.

DATOS DE LAS ÚLTIMAS 7 VELAS (OHLC):
{texto_velas}

INSTRUCCIONES CRÍTICAS:
1. Recrea mentalmente el gráfico visual utilizando los datos numéricos, las distancias a la EMA 20, la pendiente de la tendencia y los soportes/resistencias.
2. Identifica patrones de los 5 grupos principales. Cada patrón debe tener su contexto definido completamente para cumplir las reglas para ejecutar la orden. No basta con nombrar el patrón.
3. Decide si hay una oportunidad de alta probabilidad para 'Buy', 'Sell', o 'Hold'.
4. Si detectas un patrón válido, pero el precio se encuentra lejos de una zona clave (fuera de la zona operativa), debes marcar "fuera_de_zona": true y recomendar "Hold".

Devuelve tu respuesta en este formato JSON exacto:
{{
  "decision": "Buy" | "Sell" | "Hold",
  "patron_detectado": "Nombre completo del patrón y descripción de su contexto",
  "fuera_de_zona": true | false,
  "razones": ["Razón detallada 1", "Razón detallada 2"]
}}
"""
    try:
        completion = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        respuesta_texto = completion.choices[0].message.content
        datos_ia = json.loads(respuesta_texto.strip())
        return datos_ia
        
    except Exception as e:
        print(f"🚨 Error procesando Groq Texto: {e}")
        return {"decision": "Hold", "patron_detectado": "Error API", "fuera_de_zona": False, "razones": ["Error de conexión con Groq"]}

# ======================================================
# GRÁFICOS PARA TELEGRAM (ENTRADA Y SALIDA)
# ======================================================
def generar_grafico_telegram_entrada(df, decision, soporte, resistencia, slope, intercept, razones, patron):
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
    
    linea_tendencia = intercept + slope * x_valores
    ax.plot(x_valores, linea_tendencia, color='white', linestyle='-.', linewidth=1.5, label='Línea de Tendencia')

    if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
        ax.plot(x_valores, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA 20')

    entrada_x_idx = len(df_plot) - 2
    if decision == 'Buy':
        ax.scatter(entrada_x_idx, closes[-2]-50, s=300, marker='^', color='lime', edgecolors='black', zorder=5)
    else:
        ax.scatter(entrada_x_idx, closes[-2]+50, s=300, marker='v', color='red', edgecolors='black', zorder=5)

    # FORMATEO DE TEXTO (CORRECCIÓN TIGHT LAYOUT)
    texto_razones = "\n".join(razones)
    patron_formateado = textwrap.fill(patron, width=65)
    razones_formateadas = textwrap.fill(texto_razones, width=65)
    
    texto_panel = f"GROQ IA: {decision.upper()}\nPatrón: {patron_formateado}\nPrecio Entrada: {df['close'].iloc[-1]:.2f}\n\nRazonamiento IA:\n{razones_formateadas}"
    
    ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', edgecolor='white', alpha=0.85, label='white'))

    ax.set_title(f"BOT V98.1 - ENTRADA AL MERCADO (5m)", color='white')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='gray')
    plt.legend(loc="lower right", facecolor='black', labelcolor='white')
    
    # Manejo de la advertencia de diseño
    try:
        plt.tight_layout()
    except Exception:
        pass
        
    return fig

def generar_grafico_telegram_salida(df, posicion, precio_entrada, precio_salida, pnl, win, soporte, resistencia, slope, intercept):
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
    
    linea_tendencia = intercept + slope * x_valores
    ax.plot(x_valores, linea_tendencia, color='white', linestyle='-.', linewidth=1.5, label='Línea de Tendencia')

    if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
        ax.plot(x_valores, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA 20')

    ax.axhline(precio_entrada, color='blue', linestyle=':', linewidth=2, label="Precio Entrada")
    ax.axhline(precio_salida, color='orange', linestyle=':', linewidth=2, label="Precio Salida")

    estado = "WIN 🏆" if win else "LOSS ❌"
    texto_panel = f"RESULTADO: {estado}\nPosición: {posicion.upper()}\nPrecio Entrada: {precio_entrada:.2f}\nPrecio Salida: {precio_salida:.2f}\nPnL del Trade: {pnl:.2f} USD"
    
    ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', edgecolor='white', alpha=0.85))

    ax.set_title(f"BOT V98.1 - SALIDA DEL MERCADO (5m)", color='white')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='gray')
    plt.legend(loc="lower right", facecolor='black', labelcolor='white')
    
    try:
        plt.tight_layout()
    except Exception:
        pass
        
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

def paper_revisar_sl_tp(df, soporte, resistencia, slope, intercept):
    global PAPER_SL, PAPER_TP1, PAPER_PRECIO_ENTRADA, PAPER_POSICION_ACTIVA, PAPER_DECISION_ACTIVA
    global PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_TRADES_TOTALES, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE
    global PAPER_TP1_EJECUTADO, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS, PAPER_LAST_10_PNL

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
        precio_salida = PAPER_SL
        pnl_final = (precio_salida - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - precio_salida) * PAPER_SIZE_BTC_RESTANTE
        PAPER_BALANCE += pnl_final
        PAPER_TRADES_TOTALES += 1
        
        pnl_total_trade = PAPER_PNL_PARCIAL + pnl_final if PAPER_TP1_EJECUTADO else pnl_final
        win_status = pnl_total_trade > 0
        PAPER_WIN += 1 if win_status else 0
        PAPER_LOSS += 1 if not win_status else 0
        
        PAPER_LAST_10_PNL.append(pnl_total_trade)
        if len(PAPER_LAST_10_PNL) > 10:
            PAPER_LAST_10_PNL.pop(0)
            
        winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0.0
        
        telegram_mensaje(f"📤 TRADE CERRADO: {motivo}.\n💵 G/P Neta: {pnl_total_trade:.2f} USD\n📊 Balance Actual: {PAPER_BALANCE:.2f} USD\nWinrate: {winrate:.1f}%")
        
        fig_salida = generar_grafico_telegram_salida(df, PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, precio_salida, pnl_total_trade, win_status, soporte, resistencia, slope, intercept)
        telegram_grafico(fig_salida)
        plt.close(fig_salida)
        
        PAPER_POSICION_ACTIVA = None
        return True

    return None

# ======================================================
# LOOP PRINCIPAL
# ======================================================
def run_bot():
    print("🤖 BOT V98.1 INICIADO: LLaMA 3.3 70b Versatile - Análisis Contextual Completo...")
    telegram_mensaje("🤖 BOT V98.1 INICIADO: Análisis IA GROQ Textual en vivo desde Railway.")
    ultima_vela_operada = None

    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            idx_eval = -2
            precio_mercado = df['close'].iloc[-1] 
            tiempo_vela_cerrada = df.index[-2] 

            soporte_horiz, resistencia_horiz, canal_sup, canal_inf, slope, intercept, tendencia = detectar_zonas_mercado(df, idx_eval)
            
            # LOG DETALLADO (Heartbeat)
            tendencia_str = "BULLISH" if tendencia == "ALCISTA" else "BEARISH" if tendencia == "BAJISTA" else "LATERAL"
            pnl_10 = sum(PAPER_LAST_10_PNL) if PAPER_LAST_10_PNL else 0.0
            print(f"\n💓 [HEARTBEAT] Mercado: {tendencia_str} | PnL Global (10 trades): {pnl_10:.2f} USD | Trades: {PAPER_TRADES_TOTALES}")
            print(f"📊 Soporte: {soporte_horiz:.2f} | Resistencia: {resistencia_horiz:.2f} | Precio: {precio_mercado:.2f}")

            if PAPER_POSICION_ACTIVA is None and ultima_vela_operada != tiempo_vela_cerrada:
                print(f"[{datetime.now(timezone.utc)}] Enviando contexto textual exhaustivo a Groq IA...")
                
                # Consultamos a GROQ enviando todo el contexto en texto
                respuesta_ia = analizar_con_groq_texto(df, soporte_horiz, resistencia_horiz, tendencia, slope, intercept, idx_eval)
                
                decision = respuesta_ia.get("decision", "Hold")
                razones = respuesta_ia.get("razones", [])
                patron = respuesta_ia.get("patron_detectado", "Ninguno")
                fuera_de_zona = respuesta_ia.get("fuera_de_zona", False)

                if decision == "Hold" and fuera_de_zona and patron != "Ninguno" and patron != "":
                    print("⚠️ Patrón detectado pero fuera de zona operativa")

                motivo_rechazo = razones[0] if razones else 'Sin patrón claro o fuera de zona'
                if decision == 'Hold':
                    print(f"❌ Trade Rechazado. Razón IA: {motivo_rechazo} | Patrón: {patron}")
                else:
                    print(f"✅ Trade Aprobado: {decision} | Patrón: {patron}")

                if decision in ["Buy", "Sell"] and risk_management_check():
                    atr_entrada = df['atr'].iloc[-1]
                    if paper_abrir_posicion(decision, precio_mercado, atr_entrada):
                        ultima_vela_operada = tiempo_vela_cerrada
                        
                        texto_razones = "\n".join([f"🧠 {r}" for r in razones])
                        telegram_mensaje(f"📌 IA OPERACIÓN {decision.upper()} (5m)\n💰 Precio: {precio_mercado:.2f}\n📍 SL: {PAPER_SL:.2f} | TP1: {PAPER_TP1:.2f}\n👁️ Patrón visto: {patron}\n{texto_razones}")
                        
                        fig = generar_grafico_telegram_entrada(df, decision, soporte_horiz, resistencia_horiz, slope, intercept, razones, patron)
                        telegram_grafico(fig)
                        plt.close(fig)

            if PAPER_POSICION_ACTIVA is not None:
                paper_revisar_sl_tp(df, soporte_horiz, resistencia_horiz, slope, intercept)

            time.sleep(SLEEP_SECONDS) 

        except Exception as e:
            print(f"🚨 ERROR CRÍTICO EN BUCLE MAIN: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
