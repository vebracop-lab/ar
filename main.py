# =====================================================================
# BOT TRADING MAESTRO V102.0 - SISTEMA DE AUDITORÍA ESTRUCTURAL COMPLETO
# =====================================================================
# ⚠️ ADVERTENCIA: PROHIBIDA CUALQUIER FORMA DE SIMPLIFICACIÓN.
# ⚠️ ESTE CÓDIGO INCLUYE LÓGICA DE DETECCIÓN DE ABSORCIÓN MULTI-VELA.
# ⚠️ DISEÑADO PARA: BTCUSDT - TEMPORALIDAD 5 MINUTOS.
# =====================================================================

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
import textwrap
from scipy.stats import linregress
from datetime import datetime, timezone
from PIL import Image

# Configuración de Matplotlib para servidores sin interfaz gráfica
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ======================================================
# 1. CONFIGURACIÓN DE APIS Y CLIENTES
# ======================================================
from groq import Groq

# Carga de credenciales desde el entorno
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)
MODELO_GROQ = "llama-3.3-70b-versatile"

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL = "https://api.bybit.com"

# ======================================================
# 2. CONSTANTES DE ESTRATEGIA (MÉTRICAS NISON/BROOKS)
# ======================================================
SYMBOL = "BTCUSDT"
INTERVAL = "5"
LEVERAGE = 10
RISK_PER_TRADE = 0.02
SLEEP_SECONDS = 60

# Configuración de Risk Management
MULT_SL = 1.6
MULT_TP1 = 2.8
MULT_BE = 1.1 # Mover a Break Even tras 1.1 ATR de profit
PORCENTAJE_CIERRE_PARCIAL = 0.5

# ======================================================
# 3. BASE DE DATOS TEMPORAL (LOGS Y PNL)
# ======================================================
PAPER_BALANCE = 100.0
PAPER_POSICION_ACTIVA = None # 'Buy', 'Sell', None
PAPER_PRECIO_ENTRADA = 0.0
PAPER_SL = 0.0
PAPER_TP1 = 0.0
PAPER_TP1_DONE = False
PAPER_SIZE_BTC = 0.0
PAPER_TRADES_COUNT = 0
PAPER_WIN_COUNT = 0
PAPER_LOSS_COUNT = 0
PAPER_PNL_HISTORY = [] # Almacena los últimos 10 PnL

# ======================================================
# 4. MÓDULO DE COMUNICACIÓN (TELEGRAM)
# ======================================================
def send_telegram_msg(msg):
    if not TELEGRAM_TOKEN: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"Error Telegram Msg: {e}")

def send_telegram_chart(fig):
    if not TELEGRAM_TOKEN: return
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': buf}
        data = {'chat_id': TELEGRAM_CHAT_ID}
        requests.post(url, files=files, data=data, timeout=20)
        buf.close()
    except Exception as e:
        print(f"Error Telegram Chart: {e}")

# ======================================================
# 5. MOTOR DE DATOS: ANATOMÍA DE VELAS Y EMA
# ======================================================
def get_ohlcv_data(limit=220):
    endpoint = f"{BASE_URL}/v5/market/kline"
    params = {"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    try:
        resp = requests.get(endpoint, params=params, timeout=15).json()
        raw = resp["result"]["list"][::-1]
        df = pd.DataFrame(raw, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"Error Fetching Data: {e}")
        return pd.DataFrame()

def apply_technical_logic(df):
    # EMA 20 - El filtro de tendencia e impulso
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # ATR - Volatilidad para gestión de stops
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift()).abs()
    l_pc = (df['low'] - df['close'].shift()).abs()
    df['atr'] = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1).rolling(14).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/loss)))

    # --- ANÁLISIS DE MECHAS (DETECCIÓN DE ABSORCIÓN) ---
    df['rango'] = df['high'] - df['low']
    df['mecha_sup'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['mecha_inf'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Porcentajes de mecha sobre el rango total (Vital para la IA)
    df['pct_mecha_sup'] = (df['mecha_sup'] / df['rango'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    df['pct_mecha_inf'] = (df['mecha_inf'] / df['rango'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    # --- LÓGICA DE CRUCES ---
    df['cruce_bajista_ema'] = (df['close'] < df['ema20']) & (df['close'].shift(1) > df['ema20'])
    df['cruce_alcista_ema'] = (df['close'] > df['ema20']) & (df['close'].shift(1) < df['ema20'])
    df['cierre_bajo_ema'] = df['close'] < df['ema20']

    return df.dropna()

def extract_geometry_context(df):
    # Zonas de Resistencia y Soporte en 60 periodos
    res_line = df['high'].rolling(60).max().iloc[-1]
    sup_line = df['low'].rolling(60).min().iloc[-1]
    
    # Tendencia por regresión lineal (Últimas 40 velas)
    y = df['close'].tail(40).values
    x = np.arange(len(y))
    slope, intercept, r_val, p_val, std_err = linregress(x, y)
    angulo = np.degrees(np.arctan(slope))
    
    # Detección de "Precio en Techo"
    dist_a_res = res_line - df['close'].iloc[-1]
    umbral = df['close'].iloc[-1] * 0.001
    en_resistencia = dist_a_res < umbral
    
    return {
        "resistencia": res_line,
        "soporte": sup_line,
        "angulo": angulo,
        "en_resistencia": en_resistencia,
        "distancia_res": dist_a_res
    }

# ======================================================
# 6. MÓDULO IA: ANALIZADOR DE SECUENCIAS ANATÓMICAS
# ======================================================
def analyze_with_groq_v102(df, geo):
    # Extraemos un historial de 15 velas para que la IA vea la película, no la foto
    history = df.tail(15)
    
    candle_report = ""
    for i, (t, r) in enumerate(history.iterrows()):
        tipo = "VERDE" if r['close'] >= r['open'] else "ROJA"
        pos = "BAJO EMA" if r['close'] < r['ema20'] else "SOBRE EMA"
        candle_report += (f"T-{14-i}: {tipo} | Cierre: {r['close']} | "
                         f"Mecha Sup: {r['pct_mecha_sup']:.1f}% | "
                         f"Mecha Inf: {r['pct_mecha_inf']:.1f}% | {pos}\n")

    prompt = f"""
SISTEMA DE ANÁLISIS CUÁNTICO V102 - BTCUSDT 5M
No permitas un Long si hay señales de agotamiento masivo.

CONTEXTO GEOMÉTRICO:
- Precio Actual: {df['close'].iloc[-1]}
- Resistencia Institucional (Línea Morada): {geo['resistencia']:.2f}
- Soporte Base: {geo['soporte']:.2f}
- Ángulo de Tendencia: {geo['angulo']:.2f}°
- ¿Está el precio en la zona de resistencia?: {"SÍ" if geo['en_resistencia'] else "NO"}

ESTADO DE LA EMA 20:
- ¿Pérdida de EMA confirmada en la última vela?: {"SÍ, CRUCE BAJISTA" if df['cruce_bajista_ema'].iloc[-1] else "No"}
- ¿Cierre actual por debajo de la EMA?: {"SÍ" if df['cierre_bajo_ema'].iloc[-1] else "No"}

HISTORIAL DE ACCIÓN DE PRECIO (Últimas 15 velas):
{candle_report}

INSTRUCCIONES DE VEREDICTO:
1. Si el precio ha estado "golpeando" la resistencia morada ({geo['resistencia']}) y muestra múltiples velas con mecha superior > 40%, hay absorción.
2. Si tras este rechazo, el precio cruza hacia ABAJO la EMA 20, es un SHORT (Sell) inmediato.
3. Ignora cualquier señal de compra (Buy) si el precio está por debajo de la EMA 20 en este contexto de rechazo.

Responde ÚNICAMENTE en JSON:
{{
  "decision": "Buy" | "Sell" | "Hold",
  "patron": "Nombre técnico (ej. Shooting Star Cluster + EMA Cross)",
  "analisis_detallado": "Explicación de por qué el precio perdió el impulso alcista",
  "razones": ["Razón 1", "Razón 2", "Razón 3"]
}}
"""
    try:
        resp = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"decision": "Hold", "razones": [f"Error IA: {e}"]}

# ======================================================
# 7. GRÁFICOS Y GESTIÓN DE LOGS
# ======================================================
def generate_v102_chart(df, geo, res_ia):
    df_p = df.tail(100)
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(df_p))
    
    # Dibujo de velas institucionales
    for i in range(len(df_p)):
        c = 'lime' if df_p['close'].iloc[i] >= df_p['open'].iloc[i] else 'red'
        ax.vlines(x[i], df_p['low'].iloc[i], df_p['high'].iloc[i], color=c, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.3, min(df_p['open'].iloc[i], df_p['close'].iloc[i])), 0.6, abs(df_p['close'].iloc[i]-df_p['open'].iloc[i]), color=c))

    # Elementos técnicos
    ax.axhline(geo['resistencia'], color='purple', linestyle='--', linewidth=2, label='Línea Morada (RES)')
    ax.axhline(geo['soporte'], color='blue', linestyle='--', linewidth=2, label='Soporte')
    ax.plot(x, df_p['ema20'].values, color='yellow', linewidth=2.5, label='EMA 20')
    
    # Cuadro de texto
    info = (f"DECISIÓN: {res_ia['decision'].upper()}\n"
            f"PATRÓN: {res_ia.get('patron')}\n"
            f"ANÁLISIS: {textwrap.fill(res_ia.get('analisis_detallado', ''), 65)}")
    
    ax.text(0.01, 0.98, info, transform=ax.transAxes, color='white', verticalalignment='top', 
            bbox=dict(facecolor='black', edgecolor='purple', alpha=0.9, boxstyle='round,pad=1'))

    ax.set_facecolor('#0d0d0d'); fig.patch.set_facecolor('#0d0d0d')
    ax.tick_params(colors='white'); ax.grid(alpha=0.05)
    plt.legend(loc='lower left')
    return fig

# ======================================================
# 8. SISTEMA DE GESTIÓN DE POSICIONES (REGLAS CLARAS)
# ======================================================
def manage_active_trade(df):
    global PAPER_POSICION_ACTIVA, PAPER_BALANCE, PAPER_TP1_DONE, PAPER_TRADES_COUNT, PAPER_WIN_COUNT, PAPER_LOSS_COUNT, PAPER_PNL_HISTORY, PAPER_SL
    
    if PAPER_POSICION_ACTIVA is None: return

    cur = df.iloc[-1]
    pnl = 0
    closed = False

    if PAPER_POSICION_ACTIVA == "Buy":
        # TP1 y Mover SL a Break Even
        if not PAPER_TP1_DONE and cur['high'] >= PAPER_TP1:
            PAPER_BALANCE += (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_PARCIAL)
            PAPER_TP1_DONE = True
            PAPER_SL = PAPER_PRECIO_ENTRADA # Protección
            send_telegram_msg("🎯 *TP1 Long Alcanzado.* 50% cerrado, SL a BE.")
        
        # Stop Loss o Cierre Total
        if cur['low'] <= PAPER_SL:
            pnl = (PAPER_SL - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_DONE else 1.0))
            closed = True

    elif PAPER_POSICION_ACTIVA == "Sell":
        if not PAPER_TP1_DONE and cur['low'] <= PAPER_TP1:
            PAPER_BALANCE += (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_PARCIAL)
            PAPER_TP1_DONE = True
            PAPER_SL = PAPER_PRECIO_ENTRADA
            send_telegram_msg("🎯 *TP1 Short Alcanzado.* 50% cerrado, SL a BE.")
            
        if cur['high'] >= PAPER_SL:
            pnl = (PAPER_PRECIO_ENTRADA - PAPER_SL) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_DONE else 1.0))
            closed = True

    if closed:
        PAPER_BALANCE += pnl
        PAPER_TRADES_COUNT += 1
        if pnl > 0: PAPER_WIN_COUNT += 1
        else: PAPER_LOSS_COUNT += 1
        PAPER_PNL_HISTORY.append(pnl)
        
        send_telegram_msg(f"📤 *POSICIÓN CERRADA*\n💰 PnL: {pnl:.2f} USD\n📉 Balance: {PAPER_BALANCE:.2f}")
        PAPER_POSICION_ACTIVA = None

# ======================================================
# 9. BUCLE MAESTRO DE EJECUCIÓN (SIN RECORTES)
# ======================================================
def bot_heartbeat():
    print("💎 BOT V102.0 INICIADO - PROTOCOLO DE PRECISIÓN ABSOLUTA")
    send_telegram_msg("🚀 *TRADING MAESTRO V102.0 ONLINE*\nAnalizador de Secuencia de Mechas y Acción de Precio activado.")
    
    last_candle_time = None

    while True:
        try:
            # 1. Obtener y procesar
            df = apply_technical_logic(get_ohlcv_data())
            if df.empty: continue
            
            geo = extract_geometry_context(df)
            pnl_10 = sum(PAPER_PNL_HISTORY[-10:]) if PAPER_PNL_HISTORY else 0.0
            
            # --- LOG DE AUDITORÍA (Lo que pediste) ---
            print(f"\n💓 [{datetime.now().strftime('%H:%M:%S')}] HEARTBEAT")
            print(f"   Precio: {df['close'].iloc[-1]} | EMA20: {df['ema20'].iloc[-1]:.2f} | RSI: {df['rsi'].iloc[-1]:.1f}")
            print(f"   Ángulo: {geo['angulo']:.2f}° | PnL 10 Trades: {pnl_10:.2f} | Total: {PAPER_TRADES_COUNT}")
            print(f"   Niveles: RES {geo['resistencia']:.2f} | SUP {geo['soporte']:.2f}")
            print(f"   Tendencia: {'ALCISTA' if geo['angulo'] > 0 else 'BAJISTA'}")

            # 2. Análisis de Entrada
            if PAPER_POSICION_ACTIVA is None and last_candle_time != df.index[-2]:
                print("🔍 Consultando al Cerebro Groq...")
                ia_res = analyze_with_groq_v102(df, geo)
                
                decision = ia_res.get("decision", "Hold")
                if decision in ["Buy", "Sell"]:
                    # Apertura de posición real
                    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_SIZE_BTC, PAPER_TP1_DONE
                    p_in = df['close'].iloc[-1]
                    a_in = df['atr'].iloc[-1]
                    
                    stop_dist = a_in * MULT_SL
                    PAPER_POSICION_ACTIVA = decision
                    PAPER_PRECIO_ENTRADA = p_in
                    PAPER_TP1_DONE = False
                    
                    if decision == "Buy":
                        PAPER_SL = p_in - stop_dist
                        PAPER_TP1 = p_in + (a_in * MULT_TP1)
                    else:
                        PAPER_SL = p_in + stop_dist
                        PAPER_TP1 = p_in - (a_in * MULT_TP1)
                    
                    PAPER_SIZE_BTC = (PAPER_BALANCE * RISK_PER_TRADE) / stop_dist
                    
                    send_telegram_msg(f"🔔 *ORDEN {decision.upper()}*\n📍 Patrón: {ia_res.get('patron')}")
                    chart_fig = generate_v102_chart(df, geo, ia_res)
                    send_telegram_chart(chart_fig)
                    plt.close(chart_fig)
                    
                    last_candle_time = df.index[-2]

            # 3. Gestión de Posiciones
            manage_active_trade(df)
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"🚨 ERROR CRÍTICO: {e}")
            time.sleep(30)
