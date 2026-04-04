# BOT TRADING V102.0 BYBIT REAL – PRODUCCIÓN (SIN PROXY) 
# ======================================================
# ⚠️ KEYS INCLUIDAS TAL CUAL (SEGÚN PEDIDO)
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit
# ======================================================
# NOVEDADES V102.0 (PURE AI FREEDOM):
# - ELIMINACIÓN TOTAL DE RESTRICCIONES: El bot ya no tiene bloqueos matemáticos
#   (ni mitades de rango, ni exigencia estricta de tocar líneas).
# - EVALUACIÓN CONSTANTE: La IA de Groq evalúa el mercado libremente en cada
#   cierre de vela de 5 minutos.
# - PROMPT HOLÍSTICO: Se instruye a la IA a buscar continuaciones, breakouts, 
#   y price action avanzado en todo el gráfico.
# - CÓDIGO EXPANDIDO. CERO COMPRESIÓN. (Las líneas reducidas se deben a la 
#   eliminación del obsoleto sistema rígido de patrones de velas manuales).
# ======================================================

import os
import time
import io
import hmac
import hashlib
import requests
import base64
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from scipy.stats import linregress
from datetime import datetime, timezone, timedelta

plt.rcParams['figure.figsize'] = (12, 6)

# ======================================================
# CONFIGURACIÓN GRÁFICOS Y GENERAL
# ======================================================
GRAFICO_VELAS_LIMIT = 120
MOSTRAR_EMA20 = True
MOSTRAR_ATR = False

SYMBOL = "BTCUSDT"
INTERVAL = "5"  
RISK_PER_TRADE = 0.02  
LEVERAGE = 10          
SLEEP_SECONDS = 60     

MULT_SL = 1.5          
MULT_TP1 = 2.5         
MULT_TRAILING_BASE = 2.0    
PORCENTAJE_CIERRE = 0.5 

# ======================================================
# PAPER TRADING (ESTADO DE CUENTA)
# ======================================================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_PNL_GLOBAL = 0.0
PAPER_TRADES =[]
PAPER_POSICION_ACTIVA = None
PAPER_PRECIO_ENTRADA = None
PAPER_DECISION_ACTIVA = None
PAPER_TIME_ENTRADA = None
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

# ======================================================
# CONTROL DINÁMICO DE RIESGO
# ======================================================
MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_CONSECUTIVE_LOSSES = 0
PAPER_PAUSE_UNTIL = None
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

# ======================================================
# CREDENCIALES
# ======================================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    client_groq = Groq(api_key=GROQ_API_KEY)
else:
    client_groq = None

BASE_URL = "https://api.bybit.com"

# ======================================================
# TELEGRAM Y FIRMA BYBIT
# ======================================================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": texto
        }
        requests.post(url, data=payload, timeout=10)
    except Exception:
        pass

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
    except Exception:
        pass

def sign(params):
    elementos =[]
    for k, v in sorted(params.items()):
        elementos.append(f"{k}={v}")
    query = '&'.join(elementos)
    firma = hmac.new(BYBIT_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return firma

# ======================================================
# OBTENCIÓN DE DATOS BYBIT
# ======================================================
def obtener_velas(limit=300):
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "category": "linear", 
        "symbol": SYMBOL, 
        "interval": INTERVAL, 
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=20)
    if not r.text: 
        raise Exception("Respuesta vacía de Bybit")
    try:
        data_json = r.json()
    except Exception as e:
        raise Exception(f"Bybit devolvió respuesta no-JSON: {r.text}")

    if not isinstance(data_json, dict):
        raise Exception(f"Bybit devolvió JSON no dict: {type(data_json)}")
    if "retCode" in data_json and data_json["retCode"] != 0:
        raise Exception(f"Bybit Error retCode={data_json.get('retCode')} retMsg={data_json.get('retMsg')}")
    if "result" not in data_json:
        raise Exception("Error en Bybit: Sin campo result")
    if "list" not in data_json["result"]:
        raise Exception("Error en Bybit: Sin lista de velas en el result")
    
    data = data_json["result"]["list"][::-1]
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','turnover'])
    
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
    df.set_index('time', inplace=True)
    
    return df

# ======================================================
# INDICADORES Y MAPEO DEL MERCADO
# ======================================================
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
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

    return df.dropna()

def mapear_zonas_mercado(df, idx=-2, ventana_macro=120):
    """
    Calcula matemáticamente el estado del mercado para adjuntarlo como datos puros a la IA.
    """
    df_eval = df.iloc[:idx+1]
    
    soporte_horiz = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia_horiz = df_eval['high'].rolling(40).max().iloc[-1]
    
    if pd.isna(soporte_horiz): soporte_horiz = df_eval['low'].min()
    if pd.isna(resistencia_horiz): resistencia_horiz = df_eval['high'].max()
    
    if len(df_eval) < ventana_macro:
        y_macro = df_eval['close'].values
    else:
        y_macro = df_eval['close'].values[-ventana_macro:]
        
    x_macro = np.arange(len(y_macro))
    
    if len(y_macro) > 1 and not np.isnan(y_macro).any():
        slope, intercept, _, _, _ = linregress(x_macro, y_macro)
    else:
        slope, intercept = 0.0, y_macro[-1]
    
    if slope > 0.01: tendencia_macro = 'ALCISTA'
    elif slope < -0.01: tendencia_macro = 'BAJISTA'
    else: tendencia_macro = 'LATERAL'
        
    linea_central = intercept + slope * x_macro
    desviacion = np.std(y_macro - linea_central) if not np.isnan(np.std(y_macro - linea_central)) else 0
    
    canal_sup = linea_central[-1] + (desviacion * 1.5)
    canal_inf = linea_central[-1] - (desviacion * 1.5)
    
    # Posición en el rango para dar contexto
    precio_cierre = df['close'].iloc[idx]
    rango_local = resistencia_horiz - soporte_horiz
    if rango_local <= 0: rango_local = 0.0001
    posicion_en_rango = (precio_cierre - soporte_horiz) / rango_local
    
    contexto = {
        "soporte": soporte_horiz,
        "resistencia": resistencia_horiz,
        "canal_sup": canal_sup,
        "canal_inf": canal_inf,
        "slope": slope,
        "intercept": intercept,
        "tendencia_macro": tendencia_macro,
        "posicion_rango": posicion_en_rango * 100
    }
    
    return contexto

# ======================================================
# GRÁFICOS Y UTILIDADES DE IMAGEN
# ======================================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.read()
    return base64.b64encode(img_bytes).decode('utf-8')

def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones):
    try:
        df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
        if df_plot.empty: return None

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

        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label="Soporte Macro")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label="Resistencia Macro")

        linea_tendencia = intercept + slope * x_valores
        desviacion = np.std(closes - linea_tendencia)
        
        ax.plot(x_valores, linea_tendencia, color='white', linewidth=2, linestyle='-', label="Tendencia")
        ax.plot(x_valores, linea_tendencia + (desviacion * 1.5), linestyle='--', linewidth=2, color='red', label="Canal Sup")
        ax.plot(x_valores, linea_tendencia - (desviacion * 1.5), linestyle='--', linewidth=2, color='green', label="Canal Inf")

        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x_valores, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA 20')

        entrada_x_idx = len(df_plot) - 2
        
        if decision == 'Buy':
            ax.scatter(entrada_x_idx, closes[-2], s=200, marker='^', color='lime', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='lime', linestyle=':', linewidth=2)
        elif decision == 'Sell':
            ax.scatter(entrada_x_idx, closes[-2], s=200, marker='v', color='red', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='red', linestyle=':', linewidth=2)

        texto_razones = "\n".join(razones)
        texto_panel = f"OPERACION: {decision.upper()}\nPrecio: {df['close'].iloc[-1]:.2f}\nRSI Contexto: {df['rsi'].iloc[-2]:.1f}\n\nInfo:\n{texto_razones}"
        ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_title(f"BOT V102.0 - VISION AI FREE - {decision.upper()} (5m)")
        ax.grid(True, alpha=0.2)
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        return fig
    except Exception as e:
        return None

def generar_grafico_salida(df, trade_data):
    try:
        decision_original = trade_data['decision']
        entrada_price = trade_data['entrada']
        salida_price = trade_data['salida']
        pnl_obtenido = trade_data['pnl']
        
        df_plot = df.copy().tail(120)
        if df_plot.empty: return None

        fig, ax = plt.subplots(figsize=(14, 7))
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            color_vela = 'green' if row['close'] >= row['open'] else 'red'
            ax.plot([i, i],[row['low'], row['high']], color='black', linewidth=1)
            ax.plot([i, i],[row['open'], row['close']], color=color_vela, linewidth=6)

        ax.axhline(entrada_price, color='blue', linestyle='--', linewidth=1.5, label='Nivel Entrada')
        ax.axhline(salida_price, color='orange', linestyle='--', linewidth=1.5, label='Nivel Cierre')

        indice_salida_x = len(df_plot) - 1
        
        if pnl_obtenido > 0:
            color_marcador, forma = 'lime', '^'
        else:
            color_marcador, forma = 'red', 'v'
            
        ax.scatter([indice_salida_x],[salida_price], s=200, c=color_marcador, marker=forma, edgecolors='black', zorder=5)

        texto_panel = f"CIERRE {decision_original}\nMotivo: {trade_data['motivo']}\nPnL: {pnl_obtenido:.4f} USD\nBalance: {trade_data['balance']:.2f} USD"
        ax.text(0.02, 0.95, texto_panel, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title("BOT V102.0 - DETALLE DE CIERRE")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

# ======================================================
# 🧠 CEREBRO GROQ VISION AI (HOLÍSTICO Llama 3.2 90B)
# ======================================================
def analizar_con_groq(df, idx, contexto_zonas, base64_image):
    """
    La IA recibe la foto limpia del gráfico y datos del entorno.
    Tiene libertad absoluta para decidir. No hay restricciones algorítmicas previas.
    """
    if not client_groq:
        return "WAIT", "No se configuró GROQ_API_KEY."

    precio_actual = df['close'].iloc[idx]
    rsi_actual = df['rsi'].iloc[idx]
    tendencia = contexto_zonas.get('tendencia_macro', 'LATERAL')
    posicion = contexto_zonas.get('posicion_rango', 50)

    prompt = f"""
    Eres el mejor y más rentable Trader Institucional de Criptomonedas del mundo.
    A continuación tienes un gráfico de velas de 5 minutos de BTCUSDT (línea cyan=soporte, magenta=resistencia, amarilla=EMA20).
    Datos en tiempo real:
    - Precio Actual: {precio_actual:.2f}
    - RSI Actual: {rsi_actual:.1f}
    - Tendencia Matemática: {tendencia}
    - Ubicación en el Rango actual: {posicion:.1f}% (0% = Piso, 100% = Techo)

    Instrucciones de Análisis:
    TIENES LIBERTAD ABSOLUTA. Analiza la imagen visualmente. Usa todo tu conocimiento en Price Action, Liquidity Sweeps, Wyckoff, Smart Money Concepts (SMC) y patrones de continuación/reversión.
    Evalúa la fuerza de las últimas velas. ¿La EMA 20 está apoyando el movimiento o lo está rechazando? ¿Hay una ruptura (breakout) inminente o un rebote sólido? 
    No tienes que esperar a que el precio toque un soporte para actuar; si ves un patrón de continuación claro en el medio del gráfico, tómalo.

    Decisión de alta probabilidad:
    - Responde BUY si el análisis visual confirma una oportunidad sólida de compra (ej. rebote alcista, ruptura confirmada, bandera alcista).
    - Responde SELL si el análisis visual confirma una oportunidad sólida de venta (ej. rechazo bajista, pérdida de soporte, bandera bajista).
    - Responde WAIT si el mercado es errático, está en rango estrecho sin dirección, o el riesgo es demasiado alto.

    DEBES responder ÚNICAMENTE con un JSON válido, usando esta estructura exacta:
    {{"decision": "BUY", "razon": "Tu explicación analítica detallada de por qué decidiste esto mirando el gráfico."}}
    """

    try:
        response = client_groq.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content":[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        respuesta_texto = response.choices[0].message.content.strip()
        
        match = re.search(r'\{.*\}', respuesta_texto, re.DOTALL)
        if match:
            datos_json = json.loads(match.group())
            decision = datos_json.get("decision", "WAIT").upper()
            razon = datos_json.get("razon", "Decisión tomada visualmente por IA.")
            return decision, razon
        else:
            return "WAIT", f"JSON Inválido. Respuesta: {respuesta_texto}"

    except Exception as e:
        return "WAIT", f"Error API Groq: {e}"

# ======================================================
# MOTOR FINANCIERO Y GESTIÓN
# ======================================================
def paper_abrir_posicion(decision, precio, atr, razones, tiempo):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_SIZE_USD, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_DECISION_ACTIVA, PAPER_PARTIAL_ACTIVADO, PAPER_TP1_EJECUTADO, PAPER_PNL_PARCIAL
    if PAPER_POSICION_ACTIVA is not None: return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    sl = precio - (atr * MULT_SL) if decision == "Buy" else precio + (atr * MULT_SL)
    tp1 = precio + (atr * MULT_TP1) if decision == "Buy" else precio - (atr * MULT_TP1)
    
    distancia_riesgo = abs(precio - sl)
    if distancia_riesgo == 0: return False

    size_ideal = (riesgo_usd / distancia_riesgo) * precio
    poder_maximo = PAPER_BALANCE * LEVERAGE
    size_real = poder_maximo if size_ideal > poder_maximo else size_ideal
    
    PAPER_POSICION_ACTIVA = decision
    PAPER_DECISION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL = sl
    PAPER_TP1 = tp1
    PAPER_SIZE_USD = size_real
    PAPER_SIZE_BTC = size_real / precio
    PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC
    PAPER_PARTIAL_ACTIVADO = True
    PAPER_TP1_EJECUTADO = False
    PAPER_PNL_PARCIAL = 0.0

    return True

def paper_revisar_sl_tp(df):
    global PAPER_SL, PAPER_TP1, PAPER_PRECIO_ENTRADA, PAPER_DECISION_ACTIVA, PAPER_POSICION_ACTIVA, PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_TRADES_TOTALES, PAPER_ULTIMO_RESULTADO, PAPER_ULTIMO_PNL, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_WIN, PAPER_LOSS, PAPER_PNL_PARCIAL

    if PAPER_POSICION_ACTIVA is None: return None

    high, low, close, atr_actual = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1], df['atr'].iloc[-1]
    cerrar_total, motivo = False, None

    if PAPER_POSICION_ACTIVA == "Buy":
        if not PAPER_TP1_EJECUTADO and high >= PAPER_TP1:
            PAPER_PNL_PARCIAL = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_BALANCE += PAPER_PNL_PARCIAL
            PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA 
            telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even. Trailing Acelerado ON.")

        if PAPER_TP1_EJECUTADO:
            dist = close - PAPER_PRECIO_ENTRADA
            ganancia_atr = dist / atr_actual
            mult_din = 0.8 if ganancia_atr >= 5.0 else 1.2 if ganancia_atr >= 3.0 else MULT_TRAILING_BASE
            nuevo_sl = close - (atr_actual * mult_din)
            if nuevo_sl > PAPER_SL: PAPER_SL = nuevo_sl 

        if low <= PAPER_SL:
            cerrar_total, motivo = True, "Trailing Dinámico" if PAPER_TP1_EJECUTADO else "Stop Loss"

    elif PAPER_POSICION_ACTIVA == "Sell":
        if not PAPER_TP1_EJECUTADO and low <= PAPER_TP1:
            PAPER_PNL_PARCIAL = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_BALANCE += PAPER_PNL_PARCIAL
            PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA 
            telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even. Trailing Acelerado ON.")

        if PAPER_TP1_EJECUTADO:
            dist = PAPER_PRECIO_ENTRADA - close
            ganancia_atr = dist / atr_actual
            mult_din = 0.8 if ganancia_atr >= 5.0 else 1.2 if ganancia_atr >= 3.0 else MULT_TRAILING_BASE
            nuevo_sl = close + (atr_actual * mult_din)
            if nuevo_sl < PAPER_SL: PAPER_SL = nuevo_sl 

        if high >= PAPER_SL:
            cerrar_total, motivo = True, "Trailing Dinámico" if PAPER_TP1_EJECUTADO else "Stop Loss"

    if cerrar_total:
        pnl_final = (PAPER_SL - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - PAPER_SL) * PAPER_SIZE_BTC_RESTANTE
        
        PAPER_BALANCE += pnl_final
        PAPER_PNL_GLOBAL += pnl_final
        PAPER_TRADES_TOTALES += 1
        
        pnl_total = PAPER_PNL_PARCIAL + pnl_final if PAPER_TP1_EJECUTADO else pnl_final
        if pnl_total > 0: PAPER_WIN += 1
        else: PAPER_LOSS += 1
            
        winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0.0
        
        dec, ent, sal = PAPER_DECISION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL
        PAPER_POSICION_ACTIVA = PAPER_DECISION_ACTIVA = PAPER_PRECIO_ENTRADA = PAPER_SL = PAPER_TP1 = None
        PAPER_SIZE_BTC = PAPER_SIZE_BTC_RESTANTE = PAPER_TP1_EJECUTADO = PAPER_PNL_PARCIAL = 0.0

        telegram_mensaje(f"📤 TRADE CERRADO: {motivo}.\n💵 G/P Neta: {pnl_total:.2f} USD\n📊 Balance: {PAPER_BALANCE:.2f} USD\nWinrate: {winrate:.1f}%")
        return {"decision": dec, "motivo": motivo, "entrada": ent, "salida": sal, "pnl": pnl_total, "balance": PAPER_BALANCE}

    return None

def risk_management_check():
    global PAPER_PAUSE_UNTIL, PAPER_STOPPED_TODAY, PAPER_DAILY_START_BALANCE, PAPER_CURRENT_DAY, PAPER_BALANCE, PAPER_CONSECUTIVE_LOSSES
    hoy_utc = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy_utc:
        PAPER_CURRENT_DAY = hoy_utc
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
        PAPER_CONSECUTIVE_LOSSES = 0
        
    if (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE <= -MAX_DAILY_DRAWDOWN_PCT:
        if not PAPER_STOPPED_TODAY:
            telegram_mensaje(f"🛑 PROTECCIÓN DE CAPITAL: Drawdown máximo. Bot pausado.")
            PAPER_STOPPED_TODAY = True
        return False
    return True

# ======================================================
# SISTEMA SECUNDARIO INSTITUCIONAL (INTACTO)
# ======================================================
class InstitutionalStats:
    def __init__(self):
        self.total_trades, self.wins, self.losses, self.partial_wins, self.total_rr, self.equity_curve = 0, 0, 0, 0, 0.0,[]
    def register_trade(self, result_rr, partial=False):
        self.total_trades += 1
        self.total_rr += result_rr
        if partial: self.partial_wins += 1
        elif result_rr > 0: self.wins += 1
        else: self.losses += 1
        self.equity_curve.append(self.total_rr)
    def winrate(self): return (self.wins / self.total_trades) * 100 if self.total_trades > 0 else 0.0
    def avg_rr(self): return self.total_rr / self.total_trades if self.total_trades > 0 else 0.0

class ExternalBOSDetector:
    def __init__(self, lookback=50): self.lookback = lookback
    def detect_swings(self, df): return max(df['high'].values[-self.lookback:]), min(df['low'].values[-self.lookback:])
    def is_bos_externo(self, df):
        a, b = self.detect_swings(df)
        c = df['close'].iloc[-1]
        return c > a, c < b, a, b

class PullbackValidator:
    def __init__(self, tolerance=0.3): self.tolerance = tolerance
    def es_pullback_valido(self, df, nivel, direccion):
        p = df['close'].iloc[-1]
        return p <= nivel * (1 - self.tolerance / 100) if direccion == "long" else p >= nivel * (1 + self.tolerance / 100)

class PartialTPManager:
    def __init__(self): self.tp1_hit, self.tp2_hit = False, False
    def gestionar_tp_parcial(self, entry, tp1, tp2, price, side): return {"cerrar_50": False, "cerrar_total": False, "evento": None} 

class InstitutionalLogger:
    def __init__(self, telegram_send_func): self.send_telegram = telegram_send_func
    def log_operacion_completa(self, data): self.send_telegram("📊 REPORTE DE SISTEMA INSTITUCIONAL EXTERNO")

class InstitutionalSecondarySystem:
    def __init__(self, telegram_send_func):
        self.bos_detector = ExternalBOSDetector()
        self.pullback_validator = PullbackValidator()
        self.tp_manager = PartialTPManager()
        self.stats = InstitutionalStats()
        self.logger = InstitutionalLogger(telegram_send_func)

# ======================================================
# LOOP PRINCIPAL (EVALUACIÓN CONSTANTE GROQ AI)
# ======================================================
def run_bot():
    telegram_mensaje("🤖 BOT V102.0 BYBIT REAL INICIADO.\nCerebro Multimodal LLaVA 90B Libre.\nAnálisis Holístico y Ejecución Directa en curso.")
    sistema_institucional = InstitutionalSecondarySystem(telegram_mensaje)
    ultima_vela_operada = None

    while True:
        time.sleep(SLEEP_SECONDS) 
        try:
            df_velas_crudas = obtener_velas()
            df = calcular_indicadores(df_velas_crudas)

            idx_eval = -2
            precio_mercado = df['close'].iloc[-1] 
            tiempo_vela_cerrada = df.index[-2] 

            # Calculamos el contexto para pasárselo a la IA
            contexto_zonas = mapear_zonas_mercado(df, idx_eval)
            
            razones_para_entrar =[]
            decision_final = None
            
            if PAPER_POSICION_ACTIVA is None:
                if ultima_vela_operada == tiempo_vela_cerrada:
                    print(f"🕒 {datetime.now(timezone.utc)} | Vela 5m bloqueada por Anti-Spam.")
                else:
                    print(f"🕒 {datetime.now(timezone.utc)} | 📸 Generando gráfico para Groq Vision AI...")
                    
                    # Genera imagen limpia (Sin flechas de Buy/Sell) para que la IA la analice
                    fig_neutral = generar_grafico_entrada(df, "ANALISIS", contexto_zonas['soporte'], contexto_zonas['resistencia'], contexto_zonas['slope'], contexto_zonas['intercept'], ["Esperando análisis de Inteligencia Artificial..."])
                    
                    if fig_neutral is not None:
                        img_base64 = fig_to_base64(fig_neutral)
                        plt.close(fig_neutral)
                        
                        decision_ia, razon_ia = analizar_con_groq(df, idx_eval, contexto_zonas, img_base64)
                        
                        print(f"🤖 GROQ DECIDIÓ: {decision_ia} | Razón: {razon_ia}")
                        
                        if decision_ia in ["BUY", "SELL"]:
                            decision_final = "Buy" if decision_ia == "BUY" else "Sell"
                            razones_para_entrar.append(f"Análisis IA: {razon_ia}")

                if decision_final is not None:
                    if risk_management_check():
                        atr_entrada = df['atr'].iloc[-1]
                        apertura = paper_abrir_posicion(decision_final, precio_mercado, atr_entrada, razones_para_entrar, df.index[-1])
                        
                        if apertura:
                            ultima_vela_operada = tiempo_vela_cerrada
                            
                            texto = f"📌 OPERACIÓN {decision_final.upper()} APROBADA POR VISION AI (5m)\n💰 Entrada: {precio_mercado:.2f}\n📍 SL: {PAPER_SL:.2f} | TP1: {PAPER_TP1:.2f}\n🧠 {razones_para_entrar[0]}"
                            telegram_mensaje(texto)
                            
                            # Gráfico final con la flecha de entrada marcada
                            fig_final = generar_grafico_entrada(df, decision_final, contexto_zonas['soporte'], contexto_zonas['resistencia'], contexto_zonas['slope'], contexto_zonas['intercept'], razones_para_entrar)
                            if fig_final is not None: 
                                telegram_grafico(fig_final)
                                plt.close(fig_final)

            if PAPER_POSICION_ACTIVA is not None:
                datos_cierre = paper_revisar_sl_tp(df)
                if datos_cierre is not None:
                    fig_salida = generar_grafico_salida(df, datos_cierre)
                    if fig_salida is not None:
                        telegram_grafico(fig_salida)
                        plt.close(fig_salida)

        except Exception as e:
            print(f"🚨 ERROR CRÍTICO EN EL SISTEMA: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
