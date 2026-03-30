# BOT TRADING V93.2 BYBIT REAL – PRODUCCIÓN (SIN PROXY) 
# ======================================================
# ⚠️ KEYS INCLUIDAS TAL CUAL (SEGÚN PEDIDO)
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit
# ======================================================
# NOVEDADES V93.2 (ULTIMATE CONTEXT & FIX CRÍTICO):
# - FIX: Solucionado NameError unificando la detección de zonas en una sola función.
# - INYECCIÓN DE CONTEXTO: Todos los 12 patrones exigen OBLIGATORIAMENTE su pullback 
#   previo (micro-tendencia) y su zona de impacto (S/R, EMA20 o Canal).
# - Simulación financiera estricta: Apalancamiento 10x y Riesgo 2%. Temporalidad 5m.
# - CÓDIGO 100% EXPANDIDO SIN RECORTES.
# ======================================================

import os
import time
import io
import hmac
import hashlib
import requests
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

def cerca_de_nivel(precio, nivel, margen):
    distancia = abs(precio - nivel)
    if distancia <= margen:
        return True
    else:
        return False

SYMBOL = "BTCUSDT"
INTERVAL = "5"  
RISK_PER_TRADE = 0.02  # 2% de riesgo por trade ($2.00 USD)
LEVERAGE = 10          # 10x de apalancamiento
SLEEP_SECONDS = 60     

# MULTIPLICADORES DE RIESGO/BENEFICIO (EQUILIBRADO)
MULT_SL = 1.5          
MULT_TP1 = 2.5         
MULT_TRAILING = 2.0    
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
PAPER_MAX_DRAWDOWN = 0.0
PAPER_BALANCE_MAX = PAPER_BALANCE_INICIAL

# ======================================================
# CONTROL DINÁMICO DE RIESGO
# ======================================================
MAX_CONSECUTIVE_LOSSES = 6
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

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise Exception("❌ BYBIT_API_KEY o BYBIT_API_SECRET no configuradas en el entorno")

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
    except Exception as e:
        pass

# ======================================================
# FIRMA BYBIT
# ======================================================
def sign(params):
    elementos =[]
    for k, v in sorted(params.items()):
        elementos.append(f"{k}={v}")
    query = '&'.join(elementos)
    firma = hmac.new(BYBIT_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return firma

# ======================================================
# BYBIT DATA
# ======================================================
def obtener_velas(limit=300):
    url = f"{BASE_URL}/v5/market/kline"
    params = {"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
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
# INDICADORES Y RSI
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
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df.dropna()

def detectar_zonas_mercado(df, idx=-2, ventana_macro=120):
    """
    V93.2: Función unificada que calcula TODAS las zonas clave a la vez para evitar NameErrors.
    Devuelve Soportes Horizontales, Canales Dinámicos y Tendencia Macro.
    """
    df_eval = df.iloc[:idx+1]
    
    # Soportes Horizontales (40 velas = 200 min)
    soporte_horiz = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia_horiz = df_eval['high'].rolling(40).max().iloc[-1]
    
    # Tendencia Macro y Canal de Regresión Lineal
    if len(df_eval) < ventana_macro:
        y_macro = df_eval['close'].values
    else:
        y_macro = df_eval['close'].values[-ventana_macro:]
        
    x_macro = np.arange(len(y_macro))
    slope, intercept, r_value, p_value, std_err = linregress(x_macro, y_macro)
    
    if slope > 0.01: tendencia_macro = '📈 ALCISTA'
    elif slope < -0.01: tendencia_macro = '📉 BAJISTA'
    else: tendencia_macro = '➡️ LATERAL'
        
    linea_central = intercept + slope * x_macro
    residuos = y_macro - linea_central
    desviacion = np.std(residuos)
    canal_sup = linea_central[-1] + (desviacion * 1.5)
    canal_inf = linea_central[-1] - (desviacion * 1.5)
    
    return soporte_horiz, resistencia_horiz, canal_sup, canal_inf, slope, intercept, tendencia_macro


# ======================================================
# 🕯️ ARSENAL NISON (SÚPER HUMANIZADO Y CONTEXTUALIZADO)
# ======================================================

def calcular_cuerpo_mechas(row):
    cuerpo = abs(row['close'] - row['open'])
    alto = row['high']
    bajo = row['low']
    rango = alto - bajo
    if rango == 0: rango = 0.0001
    top = max(row['open'], row['close'])
    bottom = min(row['open'], row['close'])
    mecha_sup = alto - top
    mecha_inf = bottom - bajo
    return cuerpo, mecha_sup, mecha_inf, rango, top, bottom

def tendencia_previa_micro(df, idx, velas=6):
    """Evalúa un verdadero pullback. ¿El precio actual es más bajo que hace 30 minutos?"""
    if idx - velas < 0: return "neutral"
    precio_inicio = df['close'].iloc[idx-velas]
    precio_fin = df['close'].iloc[idx-1]
    
    if precio_fin < precio_inicio: return "bajista"
    elif precio_fin > precio_inicio: return "alcista"
    else: return "lateral"

def cierre_fuerte(row, direccion):
    cuerpo, mecha_sup, mecha_inf, rango, top, bottom = calcular_cuerpo_mechas(row)
    if direccion == "alcista":
        return row['close'] >= (row['low'] + (rango * 0.55))
    else:
        return row['close'] <= (row['high'] - (rango * 0.55))

# --- 1. HAMMER ---
def es_hammer_nison(df, idx, micro_tendencia):
    # CONTEXTO INYECTADO: Si no hay caída previa, no es un martillo.
    if micro_tendencia != "bajista": return False 
    
    vela_martillo, vela_confirmacion = df.iloc[idx-1], df.iloc[idx]   
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela_martillo)
    if cuerpo == 0: cuerpo = 0.0001
    
    forma_valida = (m_inf >= 1.2 * cuerpo) and (m_sup <= m_inf * 0.8) 
    confirmacion_alcista = vela_confirmacion['close'] > vela_confirmacion['open'] 
    
    return forma_valida and confirmacion_alcista

# --- 2. SHOOTING STAR ---
def es_shooting_star_nison(df, idx, micro_tendencia):
    if micro_tendencia != "alcista": return False
    
    vela_estrella, vela_confirmacion = df.iloc[idx-1], df.iloc[idx]
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela_estrella)
    if cuerpo == 0: cuerpo = 0.0001
    
    forma_valida = (m_sup >= 1.2 * cuerpo) and (m_inf <= m_sup * 0.8)
    confirmacion_bajista = vela_confirmacion['close'] < vela_confirmacion['open'] 
    
    return forma_valida and confirmacion_bajista

# --- 3. BULLISH ENGULFING ---
def es_bullish_engulfing_nison(df, idx, micro_tendencia):
    if micro_tendencia != "bajista": return False
    vela_previa, vela_actual = df.iloc[idx-1], df.iloc[idx]
    
    if not (vela_previa['close'] <= vela_previa['open'] and vela_actual['close'] > vela_actual['open']): return False
    
    return (vela_actual['close'] > vela_previa['open']) and cierre_fuerte(vela_actual, "alcista")

# --- 4. BEARISH ENGULFING ---
def es_bearish_engulfing_nison(df, idx, micro_tendencia):
    if micro_tendencia != "alcista": return False
    vela_previa, vela_actual = df.iloc[idx-1], df.iloc[idx]
    
    if not (vela_previa['close'] >= vela_previa['open'] and vela_actual['close'] < vela_actual['open']): return False
    
    return (vela_actual['close'] < vela_previa['open']) and cierre_fuerte(vela_actual, "bajista")

# --- 5. PIERCING PATTERN ---
def es_piercing_nison(df, idx, micro_tendencia):
    if micro_tendencia != "bajista": return False
    vela_previa, vela_actual = df.iloc[idx-1], df.iloc[idx]
    
    if not (vela_previa['close'] < vela_previa['open'] and vela_actual['close'] > vela_actual['open']): return False
        
    rango_cuerpo_rojo = vela_previa['open'] - vela_previa['close']
    umbral_penetracion = vela_previa['close'] + (rango_cuerpo_rojo * 0.35) 
    
    condicion_penetra = vela_actual['close'] >= umbral_penetracion
    condicion_apertura_baja = vela_actual['open'] <= vela_previa['close'] + (rango_cuerpo_rojo * 0.3) 
    condicion_no_envuelve = vela_actual['close'] <= vela_previa['open']
    
    return condicion_apertura_baja and condicion_penetra and condicion_no_envuelve and cierre_fuerte(vela_actual, "alcista")

# --- 6. DARK CLOUD COVER ---
def es_dark_cloud_nison(df, idx, micro_tendencia):
    if micro_tendencia != "alcista": return False
    vela_previa, vela_actual = df.iloc[idx-1], df.iloc[idx]
    
    if not (vela_previa['close'] > vela_previa['open'] and vela_actual['close'] < vela_actual['open']): return False
    
    rango_cuerpo_verde = vela_previa['close'] - vela_previa['open']
    umbral_penetracion = vela_previa['close'] - (rango_cuerpo_verde * 0.35)
    
    condicion_penetra = vela_actual['close'] <= umbral_penetracion
    condicion_apertura_alta = vela_actual['open'] >= vela_previa['close'] - (rango_cuerpo_verde * 0.3)
    condicion_no_envuelve = vela_actual['close'] >= vela_previa['open']
    
    return condicion_apertura_alta and condicion_penetra and condicion_no_envuelve and cierre_fuerte(vela_actual, "bajista")

# --- 7. MORNING STAR ---
def es_morning_star_nison(df, idx, micro_tendencia):
    if micro_tendencia != "bajista": return False
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    if not (c1['close'] < c1['open'] and c3['close'] > c3['open']): return False
    c1_body, _, _, _, _, _ = calcular_cuerpo_mechas(c1)
    c2_body, _, _, _, _, _ = calcular_cuerpo_mechas(c2)
    
    condicion_cuerpo_flexible = c2_body <= (c1_body * 0.8) 
    umbral_penetracion = c1['close'] + (c1_body * 0.20)
    condicion_penetracion = c3['close'] >= umbral_penetracion
    
    return condicion_cuerpo_flexible and condicion_penetracion and cierre_fuerte(c3, "alcista")

# --- 8. EVENING STAR ---
def es_evening_star_nison(df, idx, micro_tendencia):
    if micro_tendencia != "alcista": return False
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    if not (c1['close'] > c1['open'] and c3['close'] < c3['open']): return False
    c1_body, _, _, _, _, _ = calcular_cuerpo_mechas(c1)
    c2_body, _, _, _, _, _ = calcular_cuerpo_mechas(c2)
    
    condicion_cuerpo_flexible = c2_body <= (c1_body * 0.8)
    umbral_penetracion = c1['close'] - (c1_body * 0.20)
    condicion_penetracion = c3['close'] <= umbral_penetracion
    
    return condicion_cuerpo_flexible and condicion_penetracion and cierre_fuerte(c3, "bajista")

# --- 9. TWEEZER BOTTOMS ---
def es_tweezer_bottom_nison(df, idx, micro_tendencia):
    if micro_tendencia != "bajista": return False
    c1, c2 = df.iloc[idx-1], df.iloc[idx]
    tolerancia = df['atr'].iloc[idx] * 0.50 
    
    mismos_minimos = abs(c2['low'] - c1['low']) <= tolerancia
    es_c2_verde = c2['close'] > c2['open']
    
    return mismos_minimos and es_c2_verde and cierre_fuerte(c2, "alcista")

# --- 10. TWEEZER TOPS ---
def es_tweezer_top_nison(df, idx, micro_tendencia):
    if micro_tendencia != "alcista": return False
    c1, c2 = df.iloc[idx-1], df.iloc[idx]
    tolerancia = df['atr'].iloc[idx] * 0.50
    
    mismos_maximos = abs(c2['high'] - c1['high']) <= tolerancia
    es_c2_roja = c2['close'] < c2['open']
    
    return mismos_maximos and es_c2_roja and cierre_fuerte(c2, "bajista")

# --- 11. THREE WHITE SOLDIERS ---
def es_three_white_soldiers(df, idx, micro_tendencia, rsi_actual):
    if micro_tendencia != "bajista" or rsi_actual > 75: return False
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    if es_c1_verde := c1['close'] > c1['open']:
        if es_c2_verde := c2['close'] > c2['open']:
            if es_c3_verde := c3['close'] > c3['open']:
                subida_escalonada = (c2['close'] > c1['close']) and (c3['close'] > c2['close'])
                return subida_escalonada and cierre_fuerte(c2, "alcista") and cierre_fuerte(c3, "alcista")
    return False

# --- 12. THREE BLACK CROWS ---
def es_three_black_crows(df, idx, micro_tendencia, rsi_actual):
    if micro_tendencia != "alcista" or rsi_actual < 25: return False
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    if es_c1_roja := c1['close'] < c1['open']:
        if es_c2_roja := c2['close'] < c2['open']:
            if es_c3_roja := c3['close'] < c3['open']:
                bajada_escalonada = (c2['close'] < c1['close']) and (c3['close'] < c2['close'])
                return bajada_escalonada and cierre_fuerte(c2, "bajista") and cierre_fuerte(c3, "bajista")
    return False

# === DETECTOR MAESTRO NISON MULTI-ZONA ===
def detectar_patron_nison(df, sop_horiz, res_horiz, canal_sup, canal_inf, idx=-2):
    if len(df) < 15: 
        return False, None, None, {}
        
    atr_actual = df['atr'].iloc[idx]
    rsi_actual = df['rsi'].iloc[idx]
    ema20_actual = df['ema20'].iloc[idx]
    precio_cierre = df['close'].iloc[idx]
    
    min_patron = df['low'].iloc[idx-2 : idx+1].min()
    max_patron = df['high'].iloc[idx-2 : idx+1].max()
    
    # 1. ESTADO DE LA TENDENCIA (OBLIGATORIO PARA NISON)
    micro_tendencia = tendencia_previa_micro(df, idx)
    
    # 2. EVALUACIÓN DE ZONAS (SOPORTES HORIZONTALES, CANAL Y EMA)
    tolerancia_zona = max(atr_actual * 1.5, 80)
    
    toca_sop_horiz = cerca_de_nivel(min_patron, sop_horiz, tolerancia_zona) 
    toca_res_horiz = cerca_de_nivel(max_patron, res_horiz, tolerancia_zona)
    
    toca_sop_dinamico = cerca_de_nivel(min_patron, canal_inf, tolerancia_zona)
    toca_res_dinamica = cerca_de_nivel(max_patron, canal_sup, tolerancia_zona)
    
    toca_ema20_sop = cerca_de_nivel(min_patron, ema20_actual, tolerancia_zona) and (precio_cierre > ema20_actual)
    toca_ema20_res = cerca_de_nivel(max_patron, ema20_actual, tolerancia_zona) and (precio_cierre < ema20_actual)
    
    en_zona_compra = toca_sop_horiz or toca_sop_dinamico or toca_ema20_sop
    en_zona_venta = toca_res_horiz or toca_res_dinamica or toca_ema20_res
    
    zonas_compra =[]
    if toca_sop_horiz: zonas_compra.append("Soporte Horiz")
    if toca_sop_dinamico: zonas_compra.append("Canal Inferior")
    if toca_ema20_sop: zonas_compra.append("EMA 20")
        
    zonas_venta =[]
    if toca_res_horiz: zonas_venta.append("Resistencia Horiz")
    if toca_res_dinamica: zonas_venta.append("Canal Superior")
    if toca_ema20_res: zonas_venta.append("EMA 20")
    
    log_zonas = {
        "tolerancia": tolerancia_zona,
        "en_soporte": en_zona_compra,
        "en_resistencia": en_zona_venta,
        "zonas_compra": ", ".join(zonas_compra) if zonas_compra else "Ninguna",
        "zonas_venta": ", ".join(zonas_venta) if zonas_venta else "Ninguna",
        "micro_tendencia": micro_tendencia
    }
    
    # --- EVALUACIÓN FINAL: ZONA + PATRÓN (CON MICRO-TENDENCIA INYECTADA) ---
    if en_zona_compra:
        if es_hammer_nison(df, idx, micro_tendencia): return True, "Buy", "Nison Hammer", log_zonas
        if es_bullish_engulfing_nison(df, idx, micro_tendencia): return True, "Buy", "Nison Bullish Engulfing", log_zonas
        if es_piercing_nison(df, idx, micro_tendencia): return True, "Buy", "Nison Piercing Pattern", log_zonas
        if es_morning_star_nison(df, idx, micro_tendencia): return True, "Buy", "Nison Morning Star", log_zonas
        if es_tweezer_bottom_nison(df, idx, micro_tendencia): return True, "Buy", "Nison Tweezer Bottoms", log_zonas
        if es_three_white_soldiers(df, idx, micro_tendencia, rsi_actual): return True, "Buy", "Three White Soldiers", log_zonas

    if en_zona_venta:
        if es_shooting_star_nison(df, idx, micro_tendencia): return True, "Sell", "Nison Shooting Star", log_zonas
        if es_bearish_engulfing_nison(df, idx, micro_tendencia): return True, "Sell", "Nison Bearish Engulfing", log_zonas
        if es_dark_cloud_nison(df, idx, micro_tendencia): return True, "Sell", "Nison Dark Cloud Cover", log_zonas
        if es_evening_star_nison(df, idx, micro_tendencia): return True, "Sell", "Nison Evening Star", log_zonas
        if es_tweezer_top_nison(df, idx, micro_tendencia): return True, "Sell", "Nison Tweezer Tops", log_zonas
        if es_three_black_crows(df, idx, micro_tendencia, rsi_actual): return True, "Sell", "Three Black Crows", log_zonas

    return False, None, None, log_zonas

    # ----------------------- INICIO DE LA PARTE 2 -----------------------

# ======================================================
# GRÁFICOS MATPLOTLIB (SIN EMOJIS PARA EVITAR WARNINGS)
# ======================================================

def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones):
    try:
        df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
        if df_plot.empty:
            return None

        times = df_plot.index
        opens = df_plot['open'].values
        highs = df_plot['high'].values
        lows = df_plot['low'].values
        closes = df_plot['close'].values
        
        x_valores = np.arange(len(df_plot))
        fig, ax = plt.subplots(figsize=(14, 7))

        for i in range(len(df_plot)):
            color_vela = 'green' if closes[i] >= opens[i] else 'red'
            ax.vlines(x_valores[i], lows[i], highs[i], color=color_vela, linewidth=1)
            cuerpo_y = min(opens[i], closes[i])
            cuerpo_h = max(abs(closes[i] - opens[i]), 0.0001)
            ax.add_patch(plt.Rectangle((x_valores[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color_vela, alpha=0.9))

        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")

        linea_tendencia = intercept + slope * x_valores
        desviacion = np.std(closes - linea_tendencia)
        
        ax.plot(x_valores, linea_tendencia, color='white', linewidth=2, linestyle='-', label="Tendencia Macro")
        ax.plot(x_valores, linea_tendencia + (desviacion * 1.5), linestyle='--', linewidth=2, color='red', label='Canal Sup')
        ax.plot(x_valores, linea_tendencia - (desviacion * 1.5), linestyle='--', linewidth=2, color='green', label='Canal Inf')
        
        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x_valores, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA 20')

        entrada_x_idx = len(df_plot) - 2
        entrada_precio_final = closes[-2]
        
        if decision == 'Buy':
            ax.scatter(entrada_x_idx, entrada_precio_final, s=200, marker='^', color='lime', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='lime', linestyle=':', linewidth=2)
        else:
            ax.scatter(entrada_x_idx, entrada_precio_final, s=200, marker='v', color='red', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='red', linestyle=':', linewidth=2)

        texto_razones = "\n".join(razones)
        texto_panel = f"OPERACION: {decision.upper()}\nPrecio: {df['close'].iloc[-1]:.2f}\nRSI Contexto: {df['rsi'].iloc[-2]:.1f}\n\nRazones:\n{texto_razones}"
        ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_title(f"BOT V93.2 - BTCUSDT - Entrada {decision} Multi-Zona (5m)")
        ax.grid(True, alpha=0.2)
        plt.legend(loc="lower right")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"🚨 ERROR GRAFICO ENTRADA: {e}")
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

        ax.axhline(entrada_price, color='blue', linestyle='--', linewidth=1.5, label='Nivel Entrada Original')
        ax.axhline(salida_price, color='orange', linestyle='--', linewidth=1.5, label='Nivel Cierre Definitivo')

        indice_salida_x = len(df_plot) - 1
        color_marcador, forma_marcador, texto_resultado = ('lime', '^', "GANADA (+)") if pnl_obtenido > 0 else ('red', 'v', "PERDIDA (-)")
        ax.scatter([indice_salida_x],[salida_price], s=200, c=color_marcador, marker=forma_marcador, edgecolors='black', zorder=5)

        texto_panel_salida = f"CIERRE DE OPERACION {decision_original}\nMotivo: {trade_data['motivo']}\nResultado PnL: {pnl_obtenido:.4f} USD\nNuevo Balance: {trade_data['balance']:.2f} USD"
        ax.text(0.02, 0.95, texto_panel_salida, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title(f"BOT V93.2 - DETALLE DE CIERRE - {texto_resultado}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"🚨 ERROR GRAFICO SALIDA: {e}")
        return None

def log_colab(df, tendencia, slope, soporte, resistencia, decision, razones, log_zonas, idx=-2):
    ahora = datetime.now(timezone.utc)
    print("="*100)
    print(f"🕒 {ahora} | 💰 Precio: {df['close'].iloc[idx]:.2f}")
    if log_zonas:
        print(f"🔎 Estado Zona COMPRA: {log_zonas.get('zonas_compra')} | Validada: {log_zonas.get('en_soporte')}")
        print(f"🔎 Estado Zona VENTA:  {log_zonas.get('zonas_venta')} | Validada: {log_zonas.get('en_resistencia')}")
        print(f"🔎 Micro Tendencia Real: {log_zonas.get('micro_tendencia').upper()}")
    print(f"🎯 DECISIÓN TOMADA: {decision.upper() if decision else 'NO TRADE'}")
    for razon in razones: print(f"🧠 Lógica: {razon}")
    print("="*100)

# ======================================================
# MOTOR FINANCIERO Y GESTIÓN 
# ======================================================

def paper_abrir_posicion(decision, precio, atr, razones, tiempo):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_TP2, PAPER_SIZE_USD, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_DECISION_ACTIVA, PAPER_PARTIAL_ACTIVADO, PAPER_TP1_EJECUTADO, PAPER_PNL_PARCIAL
    if PAPER_POSICION_ACTIVA is not None: return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    sl = precio - (atr * MULT_SL) if decision == "Buy" else precio + (atr * MULT_SL)
    tp1 = precio + (atr * MULT_TP1) if decision == "Buy" else precio - (atr * MULT_TP1)
    
    distancia_riesgo = abs(precio - sl)
    if distancia_riesgo == 0: return False

    size_en_dolares_ideal = (riesgo_usd / distancia_riesgo) * precio
    poder_de_compra_maximo = PAPER_BALANCE * LEVERAGE
    
    size_en_dolares_real = poder_de_compra_maximo if size_en_dolares_ideal > poder_de_compra_maximo else size_en_dolares_ideal
    size_en_cripto_real = size_en_dolares_real / precio
    
    PAPER_POSICION_ACTIVA = decision
    PAPER_DECISION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL = sl
    PAPER_TP1 = tp1
    PAPER_TP2 = None  
    PAPER_SIZE_USD = size_en_dolares_real
    PAPER_SIZE_BTC = size_en_cripto_real
    PAPER_SIZE_BTC_RESTANTE = size_en_cripto_real
    PAPER_PARTIAL_ACTIVADO = True
    PAPER_TP1_EJECUTADO = False
    PAPER_PNL_PARCIAL = 0.0

    print(f"💰 FINANZAS: Margen Usado: {(size_en_dolares_real / LEVERAGE):.2f} USD | Posición Total: {PAPER_SIZE_USD:.2f} USD | Apalancamiento: {LEVERAGE}x")
    return True

def paper_revisar_sl_tp(df):
    global PAPER_SL, PAPER_TP1, PAPER_PRECIO_ENTRADA, PAPER_DECISION_ACTIVA, PAPER_POSICION_ACTIVA, PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_TRADES_TOTALES, PAPER_ULTIMO_RESULTADO, PAPER_ULTIMO_PNL, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_WIN, PAPER_LOSS, PAPER_PNL_PARCIAL
    if PAPER_POSICION_ACTIVA is None: return None

    high, low, close, atr_actual = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1], df['atr'].iloc[-1]
    cerrar_total, motivo = False, None

    if PAPER_POSICION_ACTIVA == "Buy":
        if PAPER_TP1_EJECUTADO == False and high >= PAPER_TP1:
            PAPER_PNL_PARCIAL = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_BALANCE += PAPER_PNL_PARCIAL
            PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA 
            telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). 50% cerrado, SL a Break Even. Iniciando Trailing...")

        if PAPER_TP1_EJECUTADO == True:
            nuevo_sl_dinamico = close - (atr_actual * MULT_TRAILING)
            if nuevo_sl_dinamico > PAPER_SL: PAPER_SL = nuevo_sl_dinamico 

        if low <= PAPER_SL:
            cerrar_total = True
            motivo = "Trailing Stop Dinámico" if PAPER_TP1_EJECUTADO else "Stop Loss"

    elif PAPER_POSICION_ACTIVA == "Sell":
        if PAPER_TP1_EJECUTADO == False and low <= PAPER_TP1:
            PAPER_PNL_PARCIAL = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_BALANCE += PAPER_PNL_PARCIAL
            PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA 
            telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). 50% cerrado, SL a Break Even. Iniciando Trailing...")

        if PAPER_TP1_EJECUTADO == True:
            nuevo_sl_dinamico = close + (atr_actual * MULT_TRAILING)
            if nuevo_sl_dinamico < PAPER_SL: PAPER_SL = nuevo_sl_dinamico 

        if high >= PAPER_SL:
            cerrar_total = True
            motivo = "Trailing Stop Dinámico" if PAPER_TP1_EJECUTADO else "Stop Loss"

    if cerrar_total == True:
        pnl_final = (PAPER_SL - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - PAPER_SL) * PAPER_SIZE_BTC_RESTANTE
        dec, ent, sal = PAPER_DECISION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL
        
        PAPER_BALANCE += pnl_final
        PAPER_PNL_GLOBAL += pnl_final
        PAPER_TRADES_TOTALES += 1
        
        pnl_total_trade = PAPER_PNL_PARCIAL + pnl_final if PAPER_TP1_EJECUTADO else pnl_final
        if pnl_total_trade > 0: PAPER_WIN += 1
        else: PAPER_LOSS += 1
            
        winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100
        crecimiento_pct = ((PAPER_BALANCE - PAPER_BALANCE_INICIAL) / PAPER_BALANCE_INICIAL) * 100
        
        PAPER_POSICION_ACTIVA = PAPER_DECISION_ACTIVA = PAPER_PRECIO_ENTRADA = PAPER_SL = PAPER_TP1 = None
        PAPER_SIZE_BTC = PAPER_SIZE_BTC_RESTANTE = PAPER_TP1_EJECUTADO = PAPER_PNL_PARCIAL = 0.0

        texto_cierre = f"📤 TRADE CERRADO: Salida por {motivo}.\n💵 G/P Neta: {pnl_total_trade:.2f} USD\n\n📊 ESTADO DE LA CUENTA\nBalance Actual: {PAPER_BALANCE:.2f} USD\nROI: {crecimiento_pct:.2f}%\nWinrate: {winrate:.1f}% ({PAPER_WIN}W / {PAPER_LOSS}L)"
        telegram_mensaje(texto_cierre)
        
        return {"decision": dec, "motivo": motivo, "entrada": ent, "salida": sal, "pnl": pnl_total_trade, "balance": PAPER_BALANCE}

    return None

def risk_management_check():
    global PAPER_PAUSE_UNTIL, PAPER_STOPPED_TODAY, PAPER_DAILY_START_BALANCE, PAPER_CURRENT_DAY, PAPER_BALANCE, PAPER_CONSECUTIVE_LOSSES
    hoy_utc = datetime.now(timezone.utc).date()
    
    if PAPER_CURRENT_DAY != hoy_utc:
        PAPER_CURRENT_DAY = hoy_utc
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
        PAPER_CONSECUTIVE_LOSSES = 0
        
    porcentaje_drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if porcentaje_drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
        if PAPER_STOPPED_TODAY == False:
            telegram_mensaje(f"🛑 PROTECCIÓN DE CAPITAL ACTIVADA: Se alcanzó un Drawdown de {porcentaje_drawdown*100:.2f}%. Bot pausado hasta mañana.")
            PAPER_STOPPED_TODAY = True
        return False
    return True

# ======================================================
# SISTEMA SECUNDARIO INSTITUCIONAL
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
    def __init__(self, lookback=50):
        self.lookback = lookback
    def detect_swings(self, df):
        return max(df['high'].values[-self.lookback:]), min(df['low'].values[-self.lookback:])
    def is_bos_externo(self, df):
        alto, bajo = self.detect_swings(df)
        cierre = df['close'].iloc[-1]
        return cierre > alto, cierre < bajo, alto, bajo

class PullbackValidator:
    def __init__(self, tolerance=0.3):
        self.tolerance = tolerance
    def es_pullback_valido(self, df, nivel_estructura, direccion):
        precio = df['close'].iloc[-1]
        return precio <= nivel_estructura * (1 - self.tolerance / 100) if direccion == "long" else precio >= nivel_estructura * (1 + self.tolerance / 100)

class PartialTPManager:
    def __init__(self):
        self.tp1_hit, self.tp2_hit = False, False
    def gestionar_tp_parcial(self, entry, tp1, tp2, price, side):
        return {"cerrar_50": False, "cerrar_total": False, "evento": None} # Placeholder mantenido para estructura institucional paralela.

class InstitutionalLogger:
    def __init__(self, telegram_send_func):
        self.send_telegram = telegram_send_func
    def log_operacion_completa(self, data):
        self.send_telegram("📊 REPORTE DE SISTEMA INSTITUCIONAL EXTERNO")

class InstitutionalSecondarySystem:
    def __init__(self, telegram_send_func):
        self.bos_detector = ExternalBOSDetector()
        self.pullback_validator = PullbackValidator()
        self.tp_manager = PartialTPManager()
        self.stats = InstitutionalStats()
        self.logger = InstitutionalLogger(telegram_send_func)

# ======================================================
# LOOP PRINCIPAL Y ANTI-SPAM
# ======================================================

def run_bot():
    mensaje_inicio = "🤖 BOT V93.2 BYBIT REAL INICIADO.\nConfiguración 5m MULTI-ZONE.\nContexto Inyectado Estrictamente en Patrones."
    telegram_mensaje(mensaje_inicio)

    sistema_institucional = InstitutionalSecondarySystem(telegram_mensaje)
    ultima_vela_operada = None

    while True:
        time.sleep(SLEEP_SECONDS) 
        
        try:
            df_velas_crudas = obtener_velas()
            df = calcular_indicadores(df_velas_crudas)

            idx_eval = -2
            precio_mercado_actual = df['close'].iloc[-1] 
            tiempo_vela_cerrada = df.index[-2] 

            # DETECCIÓN DE ZONAS Y MACRO
            soporte_horiz, resistencia_horiz, canal_sup, canal_inf, slope, intercept, tendencia_macro = detectar_zonas_mercado(df, idx_eval)
            
            razones_para_entrar =[]
            
            if PAPER_POSICION_ACTIVA is None:
                # DETECTOR NISON (EVALÚA MICRO-TENDENCIA INTERNAMENTE)
                patron_detectado, decision_final, nombre_patron, log_zonas = detectar_patron_nison(df, soporte_horiz, resistencia_horiz, canal_sup, canal_inf, idx=idx_eval)

                if ultima_vela_operada == tiempo_vela_cerrada:
                    decision_final = None
                    if patron_detectado:
                        lista_log =[f"Patrón {nombre_patron} bloqueado (Anti-Spam de vela 5m)"]
                        patron_detectado = False
                else:
                    if patron_detectado == True:
                        lista_log = [nombre_patron]
                    else:
                        lista_log =["Buscando patrón Nison CERRADO válido..."]
                    
                log_colab(df, tendencia_macro, slope, soporte_horiz, resistencia_horiz, decision_final, lista_log, log_zonas, idx=idx_eval)

                if decision_final is not None:
                    zonas_activadas = log_zonas.get('zonas_compra') if decision_final == "Buy" else log_zonas.get('zonas_venta')
                    razones_para_entrar.append(f"Arquitectura Confirmada: {nombre_patron}")
                    razones_para_entrar.append(f"Rebote Confirmado en: {zonas_activadas}")
                    
                    riesgo_valido = risk_management_check()
                    
                    if riesgo_valido == True:
                        atr_entrada = df['atr'].iloc[-1]
                        apertura_exitosa = paper_abrir_posicion(decision_final, precio_mercado_actual, atr_entrada, razones_para_entrar, df.index[-1])
                        
                        if apertura_exitosa == True:
                            ultima_vela_operada = tiempo_vela_cerrada
                            
                            texto_entrada = f"📌 SE HA INICIADO UNA OPERACIÓN {decision_final.upper()} (5m)\n"
                            texto_entrada += f"💰 Nivel de Entrada: {precio_mercado_actual:.2f}\n"
                            texto_entrada += f"📍 SL Inicial: {PAPER_SL:.2f} | TP1 Objetivo: {PAPER_TP1:.2f}\n"
                            
                            margen_inversion = PAPER_SIZE_USD / LEVERAGE
                            texto_entrada += f"💼 Margen Usado: {margen_inversion:.2f} USD ({LEVERAGE}x)\n\n"
                            
                            razones_unidas = '\n'.join(razones_para_entrar)
                            texto_entrada += f"🧠 Justificación Analítica:\n{razones_unidas}"
                            
                            telegram_mensaje(texto_entrada)
                            
                            figura_generada = generar_grafico_entrada(df, decision_final, soporte_horiz, resistencia_horiz, slope, intercept, razones_para_entrar)
                            
                            if figura_generada is not None:
                                telegram_grafico(figura_generada)
                                plt.close(figura_generada)

            # GESTIÓN CONTINUA DEL TRADE
            if PAPER_POSICION_ACTIVA is not None:
                datos_del_cierre = paper_revisar_sl_tp(df)
                
                if datos_del_cierre is not None:
                    figura_de_salida = generar_grafico_salida(df, datos_del_cierre)
                    
                    if figura_de_salida is not None:
                        telegram_grafico(figura_de_salida)
                        plt.close(figura_de_salida)

        except Exception as error_general:
            texto_error = f"🚨 ERROR CRÍTICO EN EL SISTEMA: {error_general}"
            print(texto_error)
            time.sleep(60)

if __name__ == '__main__':
    run_bot()



# ----------------------- FIN DE LA PARTE 1 -----------------------
