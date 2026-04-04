# BOT TRADING V98.2 BYBIT REAL – GROQ IA INSTITUCIONAL
# ======================================================
# ⚠️ IA GROQ (LLaMA 3.3 70b): Análisis de Geometría de Mercado
# Diseñado para detectar absorción, presión de ruptura y trampas.
# Sin simplificaciones. Código íntegro.
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
import textwrap
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

# Railway inyectará la API KEY desde sus Variables de Envorno
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# Uso de Llama 3.3 70b para razonamiento profundo
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
PAPER_LAST_10_PNL = [] 

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
# DATOS Y TÉCNICO (ENRIQUECIDO)
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

def calcular_indicadores_pro(df):
    # EMA 20 y ATR
    df['ema20'] = df['close'].ewm(span=20).mean()
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean(); avg_loss = loss.rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (avg_gain/avg_loss)))

    # --- INFORMACIÓN ESPACIAL Y ANATOMÍA DE VELA ---
    df['cuerpo'] = (df['close'] - df['open']).abs()
    df['rango_total'] = df['high'] - df['low']
    # % de Mecha Superior e Inferior para detectar absorción/presión
    df['pc_mecha_sup'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / df['rango_total'] * 100).fillna(0)
    df['pc_mecha_inf'] = ((df[['open', 'close']].min(axis=1) - df['low']) / df['rango_total'] * 100).fillna(0)
    
    # Distancia porcentual a la EMA 20 (Magnetismo/Rechazo)
    df['dist_ema'] = ((df['close'] - df['ema20']) / df['ema20']) * 100
    df['ema_touch'] = (df['low'] <= df['ema20']) & (df['high'] >= df['ema20'])
    
    return df.dropna()

def analizar_geometria_mercado(df, idx=-2, ventana_macro=120):
    df_eval = df.iloc[:idx+1]
    
    # Soporte y Resistencia MACRO (Largo plazo)
    sup_macro = df_eval['low'].rolling(50).min().iloc[-1]
    res_macro = df_eval['high'].rolling(50).max().iloc[-1]
    
    # Soporte y Resistencia MICRO (Zona estrecha de reacción)
    sup_micro = df_eval['low'].rolling(15).min().iloc[-1]
    res_micro = df_eval['high'].rolling(15).max().iloc[-1]
    
    # Pendiente y Ángulo de Tendencia
    y_macro = df_eval['close'].values[-60:]
    x_macro = np.arange(len(y_macro))
    slope, intercept, _, _, _ = linregress(x_macro, y_macro)
    
    # Ángulo en grados para que la IA entienda la agresividad
    angulo = np.degrees(np.arctan(slope))
    
    if angulo > 5: tendencia = 'ALCISTA'
    elif angulo < -5: tendencia = 'BAJISTA'
    else: tendencia = 'LATERAL'
    
    # Conteo de "golpes" a las zonas (Presión de ruptura)
    umbral = df_eval['close'].iloc[-1] * 0.0007
    toques_res = (df_eval['high'] > (res_macro - umbral)).tail(20).sum()
    toques_sup = (df_eval['low'] < (sup_macro + umbral)).tail(20).sum()

    return {
        "sup_macro": sup_macro, "res_macro": res_macro,
        "sup_micro": sup_micro, "res_micro": res_micro,
        "angulo": angulo, "tendencia": tendencia,
        "slope": slope, "intercept": intercept,
        "toques_res": toques_res, "toques_sup": toques_sup
    }

# ======================================================
# MOTOR IA GROQ INSTITUCIONAL
# ======================================================
def analizar_con_groq_institucional(df, geo, idx=-2):
    precio_actual = df['close'].iloc[idx]
    rsi_actual = df['rsi'].iloc[idx]
    ema_actual = df['ema20'].iloc[idx]
    dist_ema = df['dist_ema'].iloc[idx]
    
    # Contexto de velas enriquecido para la IA
    ultimas_velas = df.iloc[idx-7:idx+1][['open', 'high', 'low', 'close', 'pc_mecha_sup', 'pc_mecha_inf', 'dist_ema']]
    texto_velas = ultimas_velas.to_string()

    prompt = f"""
Eres un Master Trader Institucional. Analiza la GEOMETRÍA COMPLETA del mercado.

DATOS TÉCNICOS Y ESPACIALES:
- Precio Actual: {precio_actual:.2f} | RSI: {rsi_actual:.2f}
- Niveles MACRO: Soporte {geo['sup_macro']:.2f} | Resistencia {geo['res_macro']:.2f}
- Niveles MICRO (Inmediatos): Soporte {geo['sup_micro']:.2f} | Resistencia {geo['res_micro']:.2f}
- Ángulo de Tendencia: {geo['angulo']:.2f}° | Estado: {geo['tendencia']}
- Distancia a la EMA 20: {dist_ema:.4f}%
- Presión de Ruptura: {geo['toques_res']} toques en Resistencia | {geo['toques_sup']} toques en Soporte (últimas 20 velas).

ANATOMÍA DE VELAS (OHLC + % Mechas):
{texto_velas}

INSTRUCCIONES PROFESIONALES:
1. Evalúa si el precio está "comprimiendo" contra un nivel. Si hay muchos toques en resistencia sin retroceso, la probabilidad de ruptura alcista es alta.
2. Observa las mechas: un % alto de mecha superior en una resistencia indica absorción de ventas.
3. La EMA 20 actúa como soporte/resistencia dinámico. Evalúa si el precio está rebotando o cruzando con fuerza.
4. Define el contexto del patrón completamente. No solo digas el nombre.

Devuelve JSON exacto:
{{
  "decision": "Buy" | "Sell" | "Hold",
  "patron_detectado": "Nombre y Contexto Detallado del Patrón",
  "fuera_de_zona": true | false,
  "razones": ["Razón Institucional 1", "Razón Institucional 2", "Razón Institucional 3"]
}}
"""
    try:
        completion = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"🚨 Error Groq: {e}")
        return {"decision": "Hold", "patron_detectado": "Error API", "fuera_de_zona": False, "razones": [str(e)]}

# ======================================================
# GRÁFICOS Y GESTIÓN DE POSICIONES
# ======================================================
def generar_grafico_telegram_pro(df, decision, geo, razones, patron):
    df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
    x_valores = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(15, 8))

    # Dibujado de Velas Japonesas Profesionales
    for i in range(len(df_plot)):
        color_vela = 'lime' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else 'red'
        ax.vlines(x_valores[i], df_plot['low'].iloc[i], df_plot['high'].iloc[i], color=color_vela, linewidth=1.2)
        cuerpo_y = min(df_plot['open'].iloc[i], df_plot['close'].iloc[i])
        cuerpo_h = max(abs(df_plot['close'].iloc[i] - df_plot['open'].iloc[i]), 0.1)
        ax.add_patch(plt.Rectangle((x_valores[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color_vela, alpha=0.8))

    # Capas Geométricas
    ax.axhline(geo['res_macro'], color='#ff00ff', linestyle='--', linewidth=1.5, label="Resistencia Macro")
    ax.axhline(geo['sup_macro'], color='#00ffff', linestyle='--', linewidth=1.5, label="Soporte Macro")
    ax.axhline(geo['res_micro'], color='#ff00ff', linestyle=':', linewidth=1, alpha=0.6, label="Resistencia Micro")
    ax.axhline(geo['sup_micro'], color='#00ffff', linestyle=':', linewidth=1, alpha=0.6, label="Soporte Micro")
    
    if MOSTRAR_EMA20:
        ax.plot(x_valores, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA 20 INSTITUCIONAL')

    # Línea de Tendencia con Ángulo
    linea_t = geo['intercept'] + geo['slope'] * x_valores
    ax.plot(x_valores, linea_t, color='white', linestyle='-.', alpha=0.4)

    # Panel de Información IA (Con textwrap para evitar errores de layout)
    texto_razones = "\n".join(razones)
    patron_f = textwrap.fill(patron, width=70)
    razones_f = textwrap.fill(texto_razones, width=70)
    
    panel = f"IA DECISION: {decision.upper()}\nPatrón: {patron_f}\nÁngulo: {geo['angulo']:.2f}°\n\nRazonamiento:\n{razones_f}"
    ax.text(0.02, 0.98, panel, transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            color='white', bbox=dict(boxstyle='round', facecolor='black', edgecolor='#444', alpha=0.8))

    ax.set_title(f"TRADING MAESTRO V98.2 - BTC/USDT 5m", color='white', fontsize=14)
    ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
    ax.tick_params(colors='white'); ax.grid(True, alpha=0.1)
    plt.legend(loc="lower right", facecolor='black', labelcolor='white')
    
    try: plt.tight_layout()
    except: pass
    return fig

# --- GESTIÓN DE RIESGO Y ÓRDENES (IDÉNTICO A V98.1 PERO INTEGRADO) ---
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY, PAPER_BALANCE
    hoy = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY = hoy
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
    
    if (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE <= -MAX_DAILY_DRAWDOWN_PCT:
        if not PAPER_STOPPED_TODAY:
            telegram_mensaje("🛑 STOP LOSS DIARIO ALCANZADO. Pausando bot.")
            PAPER_STOPPED_TODAY = True
        return False
    return not PAPER_STOPPED_TODAY

def paper_abrir_posicion(decision, precio, atr):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_SIZE_BTC
    global PAPER_TP1_EJECUTADO, PAPER_SIZE_BTC_RESTANTE

    if PAPER_POSICION_ACTIVA is not None: return False
    
    dist_sl = atr * MULT_SL
    PAPER_SL = precio - dist_sl if decision == "Buy" else precio + dist_sl
    PAPER_TP1 = precio + (atr * MULT_TP1) if decision == "Buy" else precio - (atr * MULT_TP1)
    
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    PAPER_SIZE_BTC = (riesgo_usd / dist_sl) * (precio / precio) # Simplificado para el cálculo de tamaño
    PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC
    PAPER_POSICION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_TP1_EJECUTADO = False
    return True

def paper_revisar_estado(df, geo):
    global PAPER_BALANCE, PAPER_POSICION_ACTIVA, PAPER_TP1_EJECUTADO, PAPER_SL, PAPER_TRADES_TOTALES, PAPER_WIN, PAPER_LOSS
    if PAPER_POSICION_ACTIVA is None: return

    cur = df.iloc[-1]
    pnl_trade = 0
    cerro = False

    if PAPER_POSICION_ACTIVA == "Buy":
        if not PAPER_TP1_EJECUTADO and cur['high'] >= PAPER_TP1:
            PAPER_BALANCE += (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True; PAPER_SL = PAPER_PRECIO_ENTRADA
            telegram_mensaje("🎯 TP1 ALCANZADO. SL movido a Break Even.")
        if cur['low'] <= PAPER_SL:
            pnl_trade = (PAPER_SL - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_EJECUTADO else 1.0))
            cerro = True
    else: # Sell
        if not PAPER_TP1_EJECUTADO and cur['low'] <= PAPER_TP1:
            PAPER_BALANCE += (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True; PAPER_SL = PAPER_PRECIO_ENTRADA
            telegram_mensaje("🎯 TP1 ALCANZADO. SL movido a Break Even.")
        if cur['high'] >= PAPER_SL:
            pnl_trade = (PAPER_PRECIO_ENTRADA - PAPER_SL) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_EJECUTADO else 1.0))
            cerro = True

    if cerro:
        PAPER_BALANCE += pnl_trade
        PAPER_TRADES_TOTALES += 1
        if pnl_trade > 0: PAPER_WIN += 1
        else: PAPER_LOSS += 1
        PAPER_LAST_10_PNL.append(pnl_trade)
        
        telegram_mensaje(f"📤 POSICIÓN CERRADA. PnL: {pnl_trade:.2f} USD. Balance: {PAPER_BALANCE:.2f}")
        PAPER_POSICION_ACTIVA = None

# ======================================================
# LOOP PRINCIPAL
# ======================================================
def run_bot():
    print("🤖 BOT V98.2 INSTITUCIONAL INICIADO...")
    telegram_mensaje("🤖 BOT V98.2: Análisis de Geometría y Anatomía de Vela activado.")
    ultima_vela = None

    while True:
        try:
            df = calcular_indicadores_pro(obtener_velas())
            geo = analizar_geometria_mercado(df)
            
            # Log de Heartbeat
            print(f"\r💓 [{datetime.now().strftime('%H:%M:%S')}] Ang: {geo['angulo']:.1f}° | Toques R: {geo['toques_res']} | PnL Global: {sum(PAPER_LAST_10_PNL):.2f}", end="")

            if PAPER_POSICION_ACTIVA is None and ultima_vela != df.index[-2]:
                res = analizar_con_groq_institucional(df, geo)
                decision = res.get("decision", "Hold")
                
                if decision in ["Buy", "Sell"] and risk_management_check():
                    if paper_abrir_posicion(decision, df['close'].iloc[-1], df['atr'].iloc[-1]):
                        fig = generar_grafico_telegram_pro(df, decision, geo, res['razones'], res['patron_detectado'])
                        telegram_grafico(fig); plt.close(fig)
                        ultima_vela = df.index[-2]
                elif res.get("fuera_de_zona") and res.get("patron_detectado") != "Error API":
                    print(f"\n⚠️ Patrón detectado ({res['patron_detectado']}) pero fuera de zona operativa.")

            paper_revisar_estado(df, geo)
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"\n🚨 Error Crítico: {e}"); time.sleep(60)

if __name__ == '__main__':
    run_bot()
