# BOT TRADING V99.0 ULTRA-DETALLADO - ACCIÓN DE PRECIO INSTITUCIONAL
# =====================================================================
# ⚠️ PROTOTIPO DE ALTA PRECISIÓN: ANTI-ALUCINACIÓN DE IA
# Prohibida la simplificación. Este código es extenso por diseño.
# Incluye: Detección de mechas, Cruces de EMA, Gestión de PnL y Logs.
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

# Configuración para servidores (Railway)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURACIÓN DE APIS Y MODELO
# ======================================================
from groq import Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)
MODELO_GROQ = "llama-3.3-70b-versatile"

# ======================================================
# PARÁMETROS TÉCNICOS GLOBALES
# ======================================================
SYMBOL = "BTCUSDT"
INTERVAL = "5"
LEVERAGE = 10
RISK_PER_TRADE = 0.02
SLEEP_SECONDS = 60

# Configuración de Estrategia
MULT_SL = 1.5
MULT_TP1 = 2.5
PORCENTAJE_CIERRE_PARCIAL = 0.5

# ======================================================
# ESTADO DEL BOT (CONTADORES Y PNL)
# ======================================================
PAPER_BALANCE = 100.0
PAPER_POSICION_ACTIVA = None  # None, 'Buy', 'Sell'
PAPER_PRECIO_ENTRADA = 0.0
PAPER_SL = 0.0
PAPER_TP1 = 0.0
PAPER_TP1_HECHO = False
PAPER_SIZE_BTC = 0.0
PAPER_TRADES_TOTALES = 0
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_LAST_10_PNL = []

# ======================================================
# CREDENCIALES
# ======================================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ======================================================
# UTILIDADES DE RED (TELEGRAM)
# ======================================================
def enviar_telegram_texto(mensaje):
    if not TELEGRAM_TOKEN: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": mensaje}, timeout=10)
    except: pass

def enviar_telegram_foto(fig):
    if not TELEGRAM_TOKEN: return
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(url, files={'photo': buf}, data={'chat_id': TELEGRAM_CHAT_ID}, timeout=20)
        buf.close()
    except: pass

# ======================================================
# MOTOR DE DATOS: EXTRACCIÓN Y CÁLCULO DE "OJOS"
# ======================================================
def obtener_datos_mercado(limit=200):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=20).json()
        raw_data = r["result"]["list"][::-1]
        df = pd.DataFrame(raw_data, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"Error obteniendo velas: {e}")
        return pd.DataFrame()

def inyectar_indicadores_institucionales(df):
    # 1. EMA 20 (El corazón de la pérdida de impulso)
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # 2. ATR para volatilidad
    high_low = df['high'] - df['low']
    high_pc = (df['high'] - df['close'].shift()).abs()
    low_pc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # 3. RSI para sobrecompra/venta
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))

    # 4. ANATOMÍA DE VELA (Price Action Puro)
    df['rango'] = df['high'] - df['low']
    df['cuerpo'] = (df['close'] - df['open']).abs()
    # Mechas (Absorción)
    df['mecha_sup'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['mecha_inf'] = df[['open', 'close']].min(axis=1) - df['low']
    # % de mecha respecto al rango total (Vital para detectar muros)
    df['pct_mecha_sup'] = (df['mecha_sup'] / df['rango'] * 100).fillna(0)
    df['pct_mecha_inf'] = (df['mecha_inf'] / df['rango'] * 100).fillna(0)

    # 5. ESTADO RESPECTO A EMA (Cruces y Tendencia)
    df['cierre_bajo_ema'] = df['close'] < df['ema20']
    df['cierre_sobre_ema'] = df['close'] > df['ema20']
    # Detectar el momento exacto del cruce
    df['cruce_bajista'] = (df['close'] < df['ema20']) & (df['close'].shift(1) > df['ema20'])
    df['cruce_alcista'] = (df['close'] > df['ema20']) & (df['close'].shift(1) < df['ema20'])
    
    return df.dropna()

def analizar_geometria_espacial(df):
    # Niveles de soporte y resistencia dinámicos
    resistencia = df['high'].rolling(50).max().iloc[-1]
    soporte = df['low'].rolling(50).min().iloc[-1]
    
    # Regresión lineal para ángulo de tendencia
    y = df['close'].tail(40).values
    x = np.arange(len(y))
    slope, intercept, r_value, _, _ = linregress(x, y)
    angulo = np.degrees(np.arctan(slope))
    
    # Análisis de "golpes" a la resistencia
    umbral = df['close'].iloc[-1] * 0.0006
    toques_res = (df['high'] > (resistencia - umbral)).tail(20).sum()
    
    return {
        "resistencia": resistencia,
        "soporte": soporte,
        "angulo": angulo,
        "slope": slope,
        "intercept": intercept,
        "toques_res": toques_res
    }

# ======================================================
# EL CEREBRO: PROMPT SIN FILTROS NI SIMPLIFICACIONES
# ======================================================
def analizar_con_groq_v99(df, geo):
    vela = df.iloc[-1]
    # Enviamos un bloque de velas para que la IA vea la "pérdida de impulso"
    historial = df.tail(8)
    
    cuerpo_texto_velas = ""
    for i, (t, r) in enumerate(historial.iterrows()):
        pos = "BAJO EMA" if r['close'] < r['ema20'] else "SOBRE EMA"
        cuerpo_texto_velas += f"T-{7-i}: C={r['close']}, MechaSup={r['pct_mecha_sup']:.1f}%, MechaInf={r['pct_mecha_inf']:.1f}%, {pos}\n"

    prompt = f"""
INSTRUCCIÓN DE TRADER QUANT SENIOR.
No falles. Analiza por qué el precio está rechazando zonas.

SITUACIÓN ACTUAL:
- Precio: {vela['close']} | EMA 20: {vela['ema20']:.2f}
- Resistencia Clave (Línea Morada): {geo['resistencia']:.2f}
- Soporte Base: {geo['soporte']:.2f}
- Ángulo de Tendencia: {geo['angulo']:.2f}°
- Toques en Resistencia: {geo['toques_res']} (Indica si el precio está 'pegado' al techo).

ESTADO DE CRUCE:
- ¿Cierre por debajo de EMA 20?: {"SÍ (BAJISTA)" if vela['cierre_bajo_ema'] else "NO"}
- ¿Cruce bajista reciente?: {"ALERTA: ACABA DE CRUZAR HACIA ABAJO" if vela['cruce_bajista'] else "No"}

ANATOMÍA DE LAS ÚLTIMAS 8 VELAS:
{cuerpo_texto_velas}

REGLAS INSTITUCIONALES PARA EL VEREDICTO:
1. Si el precio alcanzó la resistencia ({geo['resistencia']}) y no logró romperla con volumen/fuerza...
2. Y si el precio ACABA DE CRUZAR o cerrar por debajo de la EMA 20...
3. Y si ves mechas superiores largas (>40%) en las velas recientes...
ENTONCES: El impulso alcista ha muerto. Busca un SHORT (Sell). No abras Long (Buy) bajo ninguna circunstancia en esta configuración.

Responde ÚNICAMENTE en formato JSON:
{{
  "decision": "Buy" | "Sell" | "Hold",
  "patron_detectado": "Nombre detallado del patrón de Price Action",
  "razones": ["Razón 1: Estructura", "Razón 2: EMA/Momentum", "Razón 3: Absorción"],
  "fuera_de_zona": false
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
        return {"decision": "Hold", "patron_detectado": f"Error: {e}", "razones": []}

# ======================================================
# GRÁFICOS Y LOGS (EL CORAZÓN DEL MONITOREO)
# ======================================================
def generar_grafico_detallado(df, geo, res_ia, decision):
    df_p = df.tail(100)
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(df_p))
    
    # Velas Japonesas
    for i in range(len(df_p)):
        c = 'lime' if df_p['close'].iloc[i] >= df_p['open'].iloc[i] else 'red'
        ax.vlines(x[i], df_p['low'].iloc[i], df_p['high'].iloc[i], color=c, linewidth=1)
        ax.add_patch(plt.Rectangle((x[i]-0.3, min(df_p['open'].iloc[i], df_p['close'].iloc[i])), 0.6, abs(df_p['open'].iloc[i]-df_p['close'].iloc[i]), color=c))

    # Líneas de Estrategia
    ax.axhline(geo['resistencia'], color='purple', linewidth=2, linestyle='--', label='Resistencia Morada')
    ax.axhline(geo['soporte'], color='blue', linewidth=2, linestyle='--', label='Soporte Base')
    ax.plot(x, df_p['ema20'].values, color='yellow', linewidth=2, label='EMA 20')
    
    # Etiquetado de decisión
    razones_txt = "\n".join([textwrap.fill(r, 60) for r in res_ia.get('razones', [])])
    info_box = f"DECISIÓN: {decision.upper()}\nPATRÓN: {res_ia.get('patron_detectado')}\n\n{razones_txt}"
    ax.text(0.02, 0.95, info_box, transform=ax.transAxes, color='white', verticalalignment='top', bbox=dict(facecolor='black', alpha=0.8, edgecolor='purple'))

    ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
    ax.tick_params(colors='white'); ax.grid(alpha=0.1)
    plt.legend()
    return fig

# ======================================================
# GESTIÓN DE POSICIONES (SIMULACIÓN REALISTA)
# ======================================================
def gestionar_posicion(df):
    global PAPER_POSICION_ACTIVA, PAPER_BALANCE, PAPER_TP1_HECHO, PAPER_TRADES_TOTALES, PAPER_WIN, PAPER_LOSS
    if PAPER_POSICION_ACTIVA is None: return

    vela = df.iloc[-1]
    precio_actual = vela['close']
    pnl_final = 0
    cerro = False

    if PAPER_POSICION_ACTIVA == "Buy":
        if not PAPER_TP1_HECHO and vela['high'] >= PAPER_TP1:
            PAPER_BALANCE += (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * 0.5)
            PAPER_TP1_HECHO = True
            enviar_telegram_texto("🎯 TP1 ALCANZADO (Long). Cerrado 50%.")
        if vela['low'] <= PAPER_SL:
            pnl_final = (PAPER_SL - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_HECHO else 1.0))
            cerro = True
    elif PAPER_POSICION_ACTIVA == "Sell":
        if not PAPER_TP1_HECHO and vela['low'] <= PAPER_TP1:
            PAPER_BALANCE += (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * 0.5)
            PAPER_TP1_HECHO = True
            enviar_telegram_texto("🎯 TP1 ALCANZADO (Short). Cerrado 50%.")
        if vela['high'] >= PAPER_SL:
            pnl_final = (PAPER_PRECIO_ENTRADA - PAPER_SL) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_HECHO else 1.0))
            cerro = True

    if cerro:
        PAPER_BALANCE += pnl_final
        PAPER_TRADES_TOTALES += 1
        if pnl_final > 0: PAPER_WIN += 1
        else: PAPER_LOSS += 1
        PAPER_LAST_10_PNL.append(pnl_final)
        enviar_telegram_texto(f"📤 POSICIÓN CERRADA. PnL: {pnl_final:.2f} USD. Balance: {PAPER_BALANCE:.2f}")
        PAPER_POSICION_ACTIVA = None

# ======================================================
# BUCLE PRINCIPAL (DETALLADO)
# ======================================================
def ejecutar_bot():
    print("🚀 BOT V99.0 ACTIVADO - ESCANEO DE PRECISIÓN")
    enviar_telegram_texto("🚀 BOT V99.0: Modo Anti-Alucinación y Acción de Precio activado.")
    ultima_vela_t = None

    while True:
        try:
            df = inyectar_indicadores_institucionales(obtener_datos_mercado())
            if df.empty: continue
            
            geo = analizar_geometria_espacial(df)
            pnl_10 = sum(PAPER_LAST_10_PNL[-10:]) if PAPER_LAST_10_PNL else 0.0
            
            # LOG DE HEARTBEAT (Lo que pediste)
            print(f"\n💎 [{datetime.now().strftime('%H:%M:%S')}] PnL 10: {pnl_10:.2f} | Trades: {PAPER_TRADES_TOTALES}")
            print(f"   Precio: {df['close'].iloc[-1]} | EMA20: {df['ema20'].iloc[-1]:.2f} | Ángulo: {geo['angulo']:.1f}°")
            print(f"   Tendencia: {'BULLISH' if geo['angulo'] > 0 else 'BEARISH'} | RSI: {df['rsi'].iloc[-1]:.1f}")

            # Solo analizar al cerrar una vela
            if PAPER_POSICION_ACTIVA is None and ultima_vela_t != df.index[-2]:
                print("🔍 Consultando al experto Groq...")
                res = analizar_con_groq_v99(df, geo)
                decision = res.get("decision", "Hold")
                
                if decision in ["Buy", "Sell"]:
                    # Lógica de apertura
                    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_SIZE_BTC, PAPER_TP1_HECHO
                    precio = df['close'].iloc[-1]
                    atr = df['atr'].iloc[-1]
                    
                    PAPER_POSICION_ACTIVA = decision
                    PAPER_PRECIO_ENTRADA = precio
                    PAPER_TP1_HECHO = False
                    
                    dist_atr = atr * MULT_SL
                    if decision == "Buy":
                        PAPER_SL = precio - dist_atr
                        PAPER_TP1 = precio + (atr * MULT_TP1)
                    else:
                        PAPER_SL = precio + dist_atr
                        PAPER_TP1 = precio - (atr * MULT_TP1)
                        
                    PAPER_SIZE_BTC = (PAPER_BALANCE * RISK_PER_TRADE) / dist_atr
                    
                    enviar_telegram_texto(f"🔔 ENTRADA {decision.upper()}\nPatrón: {res['patron_detectado']}\nPrecio: {precio}")
                    fig = generar_grafico_detallado(df, geo, res, decision)
                    enviar_telegram_foto(fig)
                    plt.close(fig)
                    ultima_vela_t = df.index[-2]
                
                elif res.get("fuera_de_zona"):
                    print("⚠️ Patrón detectado pero fuera de zona operativa.")

            gestionar_posicion(df)
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"🚨 Error crítico: {e}"); time.sleep(30)

if __name__ == "__main__":
    ejecutar_bot()
