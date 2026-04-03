# BOT TRADING V99.0 BYBIT REAL – PRODUCCIÓN (SIN PROXY) 
# ======================================================
# ⚠️ KEYS INCLUIDAS TAL CUAL (SEGÚN PEDIDO)
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit
# ======================================================
# NOVEDADES V99.0 (POLARITY & ACCELERATED TRAILING):
# - PRINCIPIO DE POLARIDAD: Detecta Soportes convertidos en Resistencias y viceversa.
# - TRAILING ACELERADO: El Stop Loss Dinámico se "aprieta" conforme las ganancias 
#   aumentan (+3 ATR, +5 ATR) para asfixiar al precio y maximizar el TP2 infinito.
# - Inteligencia espacial, EMA Rejection y Canales Dinámicos activos.
# - CÓDIGO 100% EXPANDIDO. CERO COMPRESIÓN.
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
RISK_PER_TRADE = 0.02  
LEVERAGE = 10          
SLEEP_SECONDS = 60     

# MULTIPLICADORES DE RIESGO/BENEFICIO 
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
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texto}
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
# INDICADORES Y ZONAS MULTIPLES (CON POLARIDAD)
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

def detectar_zonas_mercado(df, idx=-2, ventana_macro=120):
    df_eval = df.iloc[:idx+1]
    
    # Soportes y Resistencias Inmediatos (40 velas = 3.3 horas en 5m)
    soporte_horiz = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia_horiz = df_eval['high'].rolling(40).max().iloc[-1]
    
    if pd.isna(soporte_horiz): 
        soporte_horiz = df_eval['low'].min()
    if pd.isna(resistencia_horiz): 
        resistencia_horiz = df_eval['high'].max()
        
    # PRINCIPIO DE POLARIDAD (Histórico de 40 a 80 velas atrás)
    if len(df_eval) >= 80:
        max_previo = df_eval['high'].iloc[-80:-40].max()
        min_previo = df_eval['low'].iloc[-80:-40].min()
    else:
        max_previo = resistencia_horiz
        min_previo = soporte_horiz
    
    # Tendencia Macro y Canal Dinámico
    if len(df_eval) < ventana_macro:
        y_macro = df_eval['close'].values
    else:
        y_macro = df_eval['close'].values[-ventana_macro:]
        
    x_macro = np.arange(len(y_macro))
    
    if len(y_macro) > 1 and not np.isnan(y_macro).any():
        slope, intercept, r_value, p_value, std_err = linregress(x_macro, y_macro)
    else:
        slope = 0.0
        intercept = y_macro[-1]
    
    if slope > 0.01: 
        tendencia_macro = '📈 ALCISTA'
    elif slope < -0.01: 
        tendencia_macro = '📉 BAJISTA'
    else: 
        tendencia_macro = '➡️ LATERAL'
        
    linea_central = intercept + slope * x_macro
    desviacion = np.std(y_macro - linea_central) if not np.isnan(np.std(y_macro - linea_central)) else 0
    
    canal_sup = linea_central[-1] + (desviacion * 1.5)
    canal_inf = linea_central[-1] - (desviacion * 1.5)
    
    return soporte_horiz, resistencia_horiz, canal_sup, canal_inf, slope, intercept, tendencia_macro, max_previo, min_previo

def calcular_cuerpo_mechas(row):
    cuerpo = abs(row['close'] - row['open'])
    if cuerpo == 0:
        cuerpo = 0.0001
        
    alto = row['high']
    bajo = row['low']
    rango = alto - bajo
    
    if rango == 0:
        rango = 0.0001
        
    top = max(row['open'], row['close'])
    bottom = min(row['open'], row['close'])
    
    mecha_sup = alto - top
    mecha_inf = bottom - bajo
    
    return cuerpo, mecha_sup, mecha_inf, rango, top, bottom

def cierre_fuerte(row, direccion):
    cuerpo, mecha_sup, mecha_inf, rango, top, bottom = calcular_cuerpo_mechas(row)
    if direccion == "alcista":
        if row['close'] >= (row['low'] + (rango * 0.45)):
            return True
        else:
            return False
    else:
        if row['close'] <= (row['high'] - (rango * 0.45)):
            return True
        else:
# ======================================================
# 🕯️ ARSENAL NISON (SIN BLOQUEOS / 100% EXPANDIDO)
# ======================================================

def es_hammer_nison(df, idx):
    vela = df.iloc[idx] 
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela)
    
    if m_inf >= (1.5 * cuerpo):
        condicion_mecha_larga = True
    else:
        condicion_mecha_larga = False
        
    if m_sup <= (m_inf * 0.5):
        condicion_sin_mecha_arriba = True
    else:
        condicion_sin_mecha_arriba = False
        
    if condicion_mecha_larga and condicion_sin_mecha_arriba:
        return True
    return False

def es_shooting_star_nison(df, idx):
    vela = df.iloc[idx]
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela)
    
    if m_sup >= (1.5 * cuerpo):
        condicion_mecha_larga = True
    else:
        condicion_mecha_larga = False
        
    if m_inf <= (m_sup * 0.5):
        condicion_sin_mecha_abajo = True
    else:
        condicion_sin_mecha_abajo = False
        
    if condicion_mecha_larga and condicion_sin_mecha_abajo:
        return True
    return False

def es_bullish_engulfing_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if prev['close'] <= prev['open']:
        es_prev_roja = True
    else:
        es_prev_roja = False
        
    if curr['close'] > curr['open']:
        es_curr_verde = True
    else:
        es_curr_verde = False
    
    if not (es_prev_roja and es_curr_verde): 
        return False
        
    if curr['close'] >= prev['open']:
        condicion_envuelve = True
    else:
        condicion_envuelve = False
        
    if condicion_envuelve and cierre_fuerte(curr, "alcista"):
        return True
    return False

def es_bearish_engulfing_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if prev['close'] >= prev['open']:
        es_prev_verde = True
    else:
        es_prev_verde = False
        
    if curr['close'] < curr['open']:
        es_curr_roja = True
    else:
        es_curr_roja = False
    
    if not (es_prev_verde and es_curr_roja): 
        return False
        
    if curr['close'] <= prev['open']:
        condicion_envuelve = True
    else:
        condicion_envuelve = False
        
    if condicion_envuelve and cierre_fuerte(curr, "bajista"):
        return True
    return False

def es_piercing_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if prev['close'] < prev['open']:
        es_prev_roja = True
    else:
        es_prev_roja = False
        
    if curr['close'] > curr['open']:
        es_curr_verde = True
    else:
        es_curr_verde = False
        
    if not (es_prev_roja and es_curr_verde): 
        return False
        
    rango_rojo = prev['open'] - prev['close']
    umbral_penetracion = prev['close'] + (rango_rojo * 0.20)
    
    if curr['close'] >= umbral_penetracion:
        condicion_penetra = True
    else:
        condicion_penetra = False
        
    if curr['close'] <= prev['open']:
        condicion_no_envuelve = True
    else:
        condicion_no_envuelve = False
    
    if condicion_penetra and condicion_no_envuelve and cierre_fuerte(curr, "alcista"):
        return True
    return False

def es_dark_cloud_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if prev['close'] > prev['open']:
        es_prev_verde = True
    else:
        es_prev_verde = False
        
    if curr['close'] < curr['open']:
        es_curr_roja = True
    else:
        es_curr_roja = False
        
    if not (es_prev_verde and es_curr_roja): 
        return False
        
    rango_verde = prev['close'] - prev['open']
    umbral_penetracion = prev['close'] - (rango_verde * 0.20)
    
    if curr['close'] <= umbral_penetracion:
        condicion_penetra = True
    else:
        condicion_penetra = False
        
    if curr['close'] >= prev['open']:
        condicion_no_envuelve = True
    else:
        condicion_no_envuelve = False
    
    if condicion_penetra and condicion_no_envuelve and cierre_fuerte(curr, "bajista"):
        return True
    return False

def es_morning_star_nison(df, idx):
    if idx < 2: 
        return False
        
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    if c1['close'] < c1['open']:
        es_c1_roja = True
    else:
        es_c1_roja = False
        
    if c3['close'] > c3['open']:
        es_c3_verde = True
    else:
        es_c3_verde = False
        
    if not (es_c1_roja and es_c3_verde): 
        return False
        
    c1_body, _, _, _, _, _ = calcular_cuerpo_mechas(c1)
    c2_body, _, _, _, _, _ = calcular_cuerpo_mechas(c2)
    
    if c2_body <= (c1_body * 0.70):
        condicion_cuerpo = True
    else:
        condicion_cuerpo = False
        
    umbral_penetracion = c1['close'] + (c1_body * 0.20)
    if c3['close'] >= umbral_penetracion:
        condicion_penetra = True
    else:
        condicion_penetra = False
        
    if condicion_cuerpo and condicion_penetra and cierre_fuerte(c3, "alcista"):
        return True
    return False

def es_evening_star_nison(df, idx):
    if idx < 2: 
        return False
        
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    if c1['close'] > c1['open']:
        es_c1_verde = True
    else:
        es_c1_verde = False
        
    if c3['close'] < c3['open']:
        es_c3_roja = True
    else:
        es_c3_roja = False
        
    if not (es_c1_verde and es_c3_roja): 
        return False
        
    c1_body, _, _, _, _, _ = calcular_cuerpo_mechas(c1)
    c2_body, _, _, _, _, _ = calcular_cuerpo_mechas(c2)
    
    if c2_body <= (c1_body * 0.70):
        condicion_cuerpo = True
    else:
        condicion_cuerpo = False
        
    umbral_penetracion = c1['close'] - (c1_body * 0.20)
    if c3['close'] <= umbral_penetracion:
        condicion_penetra = True
    else:
        condicion_penetra = False
        
    if condicion_cuerpo and condicion_penetra and cierre_fuerte(c3, "bajista"):
        return True
    return False

def es_tweezer_bottom_nison(df, idx):
    c1 = df.iloc[idx-1]
    c2 = df.iloc[idx]
    
    tolerancia = df['atr'].iloc[idx] * 0.60 
    
    if abs(c2['low'] - c1['low']) <= tolerancia:
        mismos_minimos = True
    else:
        mismos_minimos = False
        
    if c2['close'] > c2['open']:
        es_c2_verde = True
    else:
        es_c2_verde = False
    
    if mismos_minimos and es_c2_verde and cierre_fuerte(c2, "alcista"):
        return True
    return False

def es_tweezer_top_nison(df, idx):
    c1 = df.iloc[idx-1]
    c2 = df.iloc[idx]
    
    tolerancia = df['atr'].iloc[idx] * 0.60
    
    if abs(c2['high'] - c1['high']) <= tolerancia:
        mismos_maximos = True
    else:
        mismos_maximos = False
        
    if c2['close'] < c2['open']:
        es_c2_roja = True
    else:
        es_c2_roja = False
    
    if mismos_maximos and es_c2_roja and cierre_fuerte(c2, "bajista"):
        return True
    return False

def es_three_white_soldiers(df, idx, rsi_actual):
    if rsi_actual > 75: 
        return False 
    if idx < 2: 
        return False
        
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    if c1['close'] > c1['open']:
        es_c1_verde = True
    else:
        es_c1_verde = False
        
    if c2['close'] > c2['open']:
        es_c2_verde = True
    else:
        es_c2_verde = False
        
    if c3['close'] > c3['open']:
        es_c3_verde = True
    else:
        es_c3_verde = False
    
    if es_c1_verde and es_c2_verde and es_c3_verde:
        if (c2['close'] > c1['close']) and (c3['close'] > c2['close']):
            subida_escalonada = True
        else:
            subida_escalonada = False
            
        if subida_escalonada and cierre_fuerte(c2, "alcista") and cierre_fuerte(c3, "alcista"):
            return True
    return False

def es_three_black_crows(df, idx, rsi_actual):
    if rsi_actual < 25: 
        return False 
    if idx < 2: 
        return False
        
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    if c1['close'] < c1['open']:
        es_c1_roja = True
    else:
        es_c1_roja = False
        
    if c2['close'] < c2['open']:
        es_c2_roja = True
    else:
        es_c2_roja = False
        
    if c3['close'] < c3['open']:
        es_c3_roja = True
    else:
        es_c3_roja = False
    
    if es_c1_roja and es_c2_roja and es_c3_roja:
        if (c2['close'] < c1['close']) and (c3['close'] < c2['close']):
            bajada_escalonada = True
        else:
            bajada_escalonada = False
            
        if bajada_escalonada and cierre_fuerte(c2, "bajista") and cierre_fuerte(c3, "bajista"):
            return True
    return False

    # === DETECTOR MAESTRO NISON MULTI-ZONA (CON POLARIDAD) ===
def detectar_patron_nison(df, sop_horiz, res_horiz, canal_sup, canal_inf, tendencia_macro, max_previo, min_previo, idx=-2):
    if len(df) < 15: 
        return False, None, None, {}
        
    atr_actual = df['atr'].iloc[idx]
    rsi_actual = df['rsi'].iloc[idx]
    ema20_actual = df['ema20'].iloc[idx]
    precio_cierre = df['close'].iloc[idx]
    
    min_patron = df['low'].iloc[idx]
    max_patron = df['high'].iloc[idx]
    
    # ----------------------------------------------------
    # 🧠 1. INTELIGENCIA ESPACIAL: Rango Local y Posición
    # ----------------------------------------------------
    rango_local = res_horiz - sop_horiz
    if rango_local <= 0: 
        rango_local = 0.0001
    
    posicion_en_rango = (precio_cierre - sop_horiz) / rango_local
    
    tolerancia_zona = atr_actual * 1.5
    
    # ----------------------------------------------------
    # 🧠 2. ANÁLISIS DE RECHAZO DE MEDIA MÓVIL (EMA 20)
    # ----------------------------------------------------
    toques_techo_ema = 0
    toques_suelo_ema = 0
    
    for i in range(idx-3, idx+1):
        h = df['high'].iloc[i]
        l = df['low'].iloc[i]
        c = df['close'].iloc[i]
        ema = df['ema20'].iloc[i]
        
        if h >= (ema - atr_actual*0.1) and c <= ema:
            toques_techo_ema += 1
            
        if l <= (ema + atr_actual*0.1) and c >= ema:
            toques_suelo_ema += 1

    if toques_techo_ema >= 1 and precio_cierre < ema20_actual:
        rechazo_bajista_ema = True
    else:
        rechazo_bajista_ema = False
        
    if toques_suelo_ema >= 1 and precio_cierre > ema20_actual:
        rechazo_alcista_ema = True
    else:
        rechazo_alcista_ema = False
    
    # ----------------------------------------------------
    # 🧠 3. EVALUACIÓN DE ZONAS ESTÁTICAS, DINÁMICAS Y POLARIDAD
    # ----------------------------------------------------
    if cerca_de_nivel(min_patron, sop_horiz, tolerancia_zona) and posicion_en_rango < 0.5:
        toca_sop_horiz = True
    else:
        toca_sop_horiz = False
        
    if cerca_de_nivel(max_patron, res_horiz, tolerancia_zona) and posicion_en_rango > 0.5:
        toca_res_horiz = True
    else:
        toca_res_horiz = False
        
    if cerca_de_nivel(min_patron, max_previo, tolerancia_zona) and precio_cierre >= max_previo:
        toca_polaridad_sop = True
    else:
        toca_polaridad_sop = False
        
    if cerca_de_nivel(max_patron, min_previo, tolerancia_zona) and precio_cierre <= min_previo:
        toca_polaridad_res = True
    else:
        toca_polaridad_res = False
    
    toca_sop_dinamico = cerca_de_nivel(min_patron, canal_inf, tolerancia_zona)
    toca_res_dinamica = cerca_de_nivel(max_patron, canal_sup, tolerancia_zona)
    
    en_zona_compra = toca_sop_horiz or toca_sop_dinamico or rechazo_alcista_ema or toca_polaridad_sop
    en_zona_venta = toca_res_horiz or toca_res_dinamica or rechazo_bajista_ema or toca_polaridad_res

    # ----------------------------------------------------
    # 🚫 CANDADOS DE CONTEXTO SUPREMO 🚫
    # ----------------------------------------------------
    if rechazo_bajista_ema:
        en_zona_compra = False
        
    if rechazo_alcista_ema:
        en_zona_venta = False

    if tendencia_macro == '📉 BAJISTA':
        en_zona_compra = False
    elif tendencia_macro == '📈 ALCISTA':
        en_zona_venta = False

    if posicion_en_rango >= 0.50 and rechazo_bajista_ema:
        en_zona_venta = True
        
    if posicion_en_rango <= 0.50 and rechazo_alcista_ema:
        en_zona_compra = True

    # REGISTRO PARA LOGS
    zonas_compra =[]
    if toca_sop_horiz: 
        zonas_compra.append("Soporte Horiz")
    if toca_sop_dinamico: 
        zonas_compra.append("Canal Inf")
    if rechazo_alcista_ema: 
        zonas_compra.append("EMA 20 (Soporte)")
    if toca_polaridad_sop:
        zonas_compra.append("Polaridad (Techo roto)")
        
    zonas_venta =[]
    if toca_res_horiz: 
        zonas_venta.append("Resistencia Horiz")
    if toca_res_dinamica: 
        zonas_venta.append("Canal Sup")
    if rechazo_bajista_ema: 
        zonas_venta.append("EMA 20 (Resistencia)")
    if toca_polaridad_res:
        zonas_venta.append("Polaridad (Suelo roto)")
        
    if len(zonas_compra) > 0:
        texto_zonas_compra = ", ".join(zonas_compra)
    else:
        texto_zonas_compra = "Ninguna"
        
    if len(zonas_venta) > 0:
        texto_zonas_venta = ", ".join(zonas_venta)
    else:
        texto_zonas_venta = "Ninguna"
    
    log_zonas = {
        "tolerancia": tolerancia_zona, 
        "en_soporte": en_zona_compra, 
        "en_resistencia": en_zona_venta,
        "zonas_compra": texto_zonas_compra,
        "zonas_venta": texto_zonas_venta,
        "posicion_rango": posicion_en_rango
    }
    
    if en_zona_compra:
        if es_hammer_nison(df, idx): return True, "Buy", "Nison Hammer", log_zonas
        if es_bullish_engulfing_nison(df, idx): return True, "Buy", "Nison Bullish Engulfing", log_zonas
        if es_piercing_nison(df, idx): return True, "Buy", "Nison Piercing Pattern", log_zonas
        if es_morning_star_nison(df, idx): return True, "Buy", "Nison Morning Star", log_zonas
        if es_tweezer_bottom_nison(df, idx): return True, "Buy", "Nison Tweezer Bottoms", log_zonas
        if es_three_white_soldiers(df, idx, rsi_actual): return True, "Buy", "Three White Soldiers", log_zonas

    if en_zona_venta:
        if es_shooting_star_nison(df, idx): return True, "Sell", "Nison Shooting Star", log_zonas
        if es_bearish_engulfing_nison(df, idx): return True, "Sell", "Nison Bearish Engulfing", log_zonas
        if es_dark_cloud_nison(df, idx): return True, "Sell", "Nison Dark Cloud Cover", log_zonas
        if es_evening_star_nison(df, idx): return True, "Sell", "Nison Evening Star", log_zonas
        if es_tweezer_top_nison(df, idx): return True, "Sell", "Nison Tweezer Tops", log_zonas
        if es_three_black_crows(df, idx, rsi_actual): return True, "Sell", "Three Black Crows", log_zonas

    return False, None, None, log_zonas

# ======================================================
# GRÁFICOS MATPLOTLIB
# ======================================================
def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones):
    try:
        df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
        if df_plot.empty: 
            return None

        x_valores = np.arange(len(df_plot))
        closes = df_plot['close'].values
        fig, ax = plt.subplots(figsize=(14, 7))

        for i in range(len(df_plot)):
            if closes[i] >= df_plot['open'].values[i]:
                color_vela = 'green'
            else:
                color_vela = 'red'
                
            ax.vlines(x_valores[i], df_plot['low'].values[i], df_plot['high'].values[i], color=color_vela, linewidth=1)
            
            cuerpo_y = min(df_plot['open'].values[i], closes[i])
            cuerpo_h = max(abs(closes[i] - df_plot['open'].values[i]), 0.0001)
            
            rect = plt.Rectangle((x_valores[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color_vela, alpha=0.9)
            ax.add_patch(rect)

        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label="Soporte Horiz")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label="Resistencia Horiz")

        linea_tendencia = intercept + slope * x_valores
        desviacion = np.std(closes - linea_tendencia)
        
        ax.plot(x_valores, linea_tendencia, color='white', linewidth=2, linestyle='-', label="Tendencia Macro")
        ax.plot(x_valores, linea_tendencia + (desviacion * 1.5), linestyle='--', linewidth=2, color='red', label="Canal Sup")
        ax.plot(x_valores, linea_tendencia - (desviacion * 1.5), linestyle='--', linewidth=2, color='green', label="Canal Inf")

        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x_valores, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA 20')

        entrada_x_idx = len(df_plot) - 2
        
        if decision == 'Buy':
            ax.scatter(entrada_x_idx, closes[-2], s=200, marker='^', color='lime', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='lime', linestyle=':', linewidth=2)
        else:
            ax.scatter(entrada_x_idx, closes[-2], s=200, marker='v', color='red', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='red', linestyle=':', linewidth=2)

        texto_razones = "\n".join(razones)
        texto_panel = f"OPERACION: {decision.upper()}\nPrecio: {df['close'].iloc[-1]:.2f}\nRSI Contexto: {df['rsi'].iloc[-2]:.1f}\n\nRazones:\n{texto_razones}"
        ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_title(f"BOT V99.0 - Entrada {decision} Espacial (5m)")
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
        if df_plot.empty: 
            return None

        fig, ax = plt.subplots(figsize=(14, 7))
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            if row['close'] >= row['open']:
                color_vela = 'green'
            else:
                color_vela = 'red'
                
            ax.plot([i, i],[row['low'], row['high']], color='black', linewidth=1)
            ax.plot([i, i],[row['open'], row['close']], color=color_vela, linewidth=6)

        ax.axhline(entrada_price, color='blue', linestyle='--', linewidth=1.5, label='Nivel Entrada')
        ax.axhline(salida_price, color='orange', linestyle='--', linewidth=1.5, label='Nivel Cierre')

        indice_salida_x = len(df_plot) - 1
        
        if pnl_obtenido > 0:
            color_marcador = 'lime'
            forma = '^'
        else:
            color_marcador = 'red'
            forma = 'v'
            
        ax.scatter([indice_salida_x],[salida_price], s=200, c=color_marcador, marker=forma, edgecolors='black', zorder=5)

        texto_panel = f"CIERRE {decision_original}\nMotivo: {trade_data['motivo']}\nPnL: {pnl_obtenido:.4f} USD\nBalance: {trade_data['balance']:.2f} USD"
        ax.text(0.02, 0.95, texto_panel, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title("BOT V99.0 - DETALLE DE CIERRE ACELERADO")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

    def log_colab(df, tendencia, slope, soporte, resistencia, decision, razones, log_zonas, idx=-2):
    ahora = datetime.now(timezone.utc)
    print("="*100)
    print(f"🕒 {ahora} | 💰 Precio Cerrado Evaluado: {df['close'].iloc[idx]:.2f}")
    if log_zonas:
        pos_rango = log_zonas.get('posicion_rango', 0)
        print(f"🏢 Posición en el Rango: {pos_rango*100:.1f}% (0%=Suelo, 100%=Techo)")
        print(f"🔎 Zonas de COMPRA Válidas: {log_zonas.get('zonas_compra')}")
        print(f"🔎 Zonas de VENTA Válidas: {log_zonas.get('zonas_venta')}")
    
    if decision:
        print(f"🎯 DECISIÓN TOMADA: {decision.upper()}")
    else:
        print(f"🎯 DECISIÓN TOMADA: MANTENER AL MARGEN (NO TRADE)")
        
    for razon in razones: 
        print(f"🧠 {razon}")
    print("="*100)

# ======================================================
# MOTOR FINANCIERO Y GESTIÓN (TRAILING ACELERADO V99)
# ======================================================
def paper_abrir_posicion(decision, precio, atr, razones, tiempo):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_SIZE_USD, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_DECISION_ACTIVA, PAPER_PARTIAL_ACTIVADO, PAPER_TP1_EJECUTADO, PAPER_PNL_PARCIAL
    if PAPER_POSICION_ACTIVA is not None: 
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    
    if decision == "Buy":
        sl = precio - (atr * MULT_SL)
        tp1 = precio + (atr * MULT_TP1)
    elif decision == "Sell":
        sl = precio + (atr * MULT_SL)
        tp1 = precio - (atr * MULT_TP1)
    else:
        return False
    
    distancia_riesgo = abs(precio - sl)
    if distancia_riesgo == 0: 
        return False

    size_en_dolares_ideal = (riesgo_usd / distancia_riesgo) * precio
    poder_maximo = PAPER_BALANCE * LEVERAGE
    
    if size_en_dolares_ideal > poder_maximo:
        size_real = poder_maximo
    else:
        size_real = size_en_dolares_ideal
        
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

    if PAPER_POSICION_ACTIVA is None: 
        return None

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    close = df['close'].iloc[-1]
    atr_actual = df['atr'].iloc[-1]
    
    cerrar_total = False
    motivo = None

    # ==========================
    # LÓGICA DE COMPRA (LONG)
    # ==========================
    if PAPER_POSICION_ACTIVA == "Buy":
        if PAPER_TP1_EJECUTADO == False:
            if high >= PAPER_TP1:
                PAPER_PNL_PARCIAL = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
                PAPER_BALANCE += PAPER_PNL_PARCIAL
                PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
                PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
                PAPER_TP1_EJECUTADO = True
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even. Iniciando Trailing Acelerado...")

        if PAPER_TP1_EJECUTADO == True:
            # ACELERADOR DINÁMICO DE TRAILING
            distancia_a_favor = close - PAPER_PRECIO_ENTRADA
            atrs_ganados = distancia_a_favor / atr_actual
            
            if atrs_ganados >= 5.0:
                multiplicador_dinamico = 0.8  # Aprieta el SL para proteger mega rally
            elif atrs_ganados >= 3.0:
                multiplicador_dinamico = 1.2  # Ajuste medio
            else:
                multiplicador_dinamico = MULT_TRAILING_BASE
                
            nuevo_sl = close - (atr_actual * multiplicador_dinamico)
            
            if nuevo_sl > PAPER_SL: 
                PAPER_SL = nuevo_sl 

        if low <= PAPER_SL:
            cerrar_total = True
            if PAPER_TP1_EJECUTADO == True:
                motivo = "Trailing Acelerado Dinámico" 
            else:
                motivo = "Stop Loss"

    # ==========================
    # LÓGICA DE VENTA (SHORT)
    # ==========================
    elif PAPER_POSICION_ACTIVA == "Sell":
        if PAPER_TP1_EJECUTADO == False:
            if low <= PAPER_TP1:
                PAPER_PNL_PARCIAL = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
                PAPER_BALANCE += PAPER_PNL_PARCIAL
                PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
                PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
                PAPER_TP1_EJECUTADO = True
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even. Iniciando Trailing Acelerado...")

        if PAPER_TP1_EJECUTADO == True:
            # ACELERADOR DINÁMICO DE TRAILING
            distancia_a_favor = PAPER_PRECIO_ENTRADA - close
            atrs_ganados = distancia_a_favor / atr_actual
            
            if atrs_ganados >= 5.0:
                multiplicador_dinamico = 0.8  
            elif atrs_ganados >= 3.0:
                multiplicador_dinamico = 1.2  
            else:
                multiplicador_dinamico = MULT_TRAILING_BASE
                
            nuevo_sl = close + (atr_actual * multiplicador_dinamico)
            
            if nuevo_sl < PAPER_SL: 
                PAPER_SL = nuevo_sl 

        if high >= PAPER_SL:
            cerrar_total = True
            if PAPER_TP1_EJECUTADO == True:
                motivo = "Trailing Acelerado Dinámico" 
            else:
                motivo = "Stop Loss"

    if cerrar_total == True:
        if PAPER_POSICION_ACTIVA == "Buy":
            pnl_final = (PAPER_SL - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE 
        else:
            pnl_final = (PAPER_PRECIO_ENTRADA - PAPER_SL) * PAPER_SIZE_BTC_RESTANTE
        
        PAPER_BALANCE += pnl_final
        PAPER_PNL_GLOBAL += pnl_final
        PAPER_TRADES_TOTALES += 1
        
        if PAPER_TP1_EJECUTADO == True:
            pnl_total_trade = PAPER_PNL_PARCIAL + pnl_final 
        else:
            pnl_total_trade = pnl_final
            
        if pnl_total_trade > 0: 
            PAPER_WIN += 1
        else: 
            PAPER_LOSS += 1
            
        if PAPER_TRADES_TOTALES > 0:
            winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100
        else:
            winrate = 0.0
            
        crecimiento_pct = ((PAPER_BALANCE - PAPER_BALANCE_INICIAL) / PAPER_BALANCE_INICIAL) * 100
        
        dec = PAPER_DECISION_ACTIVA
        ent = PAPER_PRECIO_ENTRADA
        sal = PAPER_SL
        
        PAPER_POSICION_ACTIVA = None
        PAPER_DECISION_ACTIVA = None
        PAPER_PRECIO_ENTRADA = None
        PAPER_SL = None
        PAPER_TP1 = None
        PAPER_SIZE_BTC = 0.0
        PAPER_SIZE_BTC_RESTANTE = 0.0
        PAPER_TP1_EJECUTADO = False
        PAPER_PNL_PARCIAL = 0.0

        telegram_mensaje(f"📤 TRADE CERRADO: Salida por {motivo}.\n💵 G/P Neta: {pnl_total_trade:.2f} USD\n📊 Balance Actual: {PAPER_BALANCE:.2f} USD\nWinrate: {winrate:.1f}%")
        return {"decision": dec, "motivo": motivo, "entrada": ent, "salida": sal, "pnl": pnl_total_trade, "balance": PAPER_BALANCE}

    return None

def risk_management_check():
    global PAPER_PAUSE_UNTIL
    global PAPER_STOPPED_TODAY
    global PAPER_DAILY_START_BALANCE
    global PAPER_CURRENT_DAY
    global PAPER_BALANCE
    global PAPER_CONSECUTIVE_LOSSES
    
    hoy_utc = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy_utc:
        PAPER_CURRENT_DAY = hoy_utc
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
        PAPER_CONSECUTIVE_LOSSES = 0
        
    porcentaje_drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if porcentaje_drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
        if PAPER_STOPPED_TODAY == False:
            telegram_mensaje(f"🛑 PROTECCIÓN DE CAPITAL: Drawdown máximo alcanzado. Bot pausado.")
            PAPER_STOPPED_TODAY = True
        return False
    return True

# ======================================================
# SISTEMA SECUNDARIO INSTITUCIONAL (PARALELO)
# ======================================================
class InstitutionalStats:
    def __init__(self):
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.partial_wins = 0
        self.total_rr = 0.0
        self.equity_curve = []
        self.trade_log =[]

    def register_trade(self, result_rr, partial=False):
        self.total_trades += 1
        self.total_rr += result_rr
        if partial: 
            self.partial_wins += 1
        elif result_rr > 0: 
            self.wins += 1
        else: 
            self.losses += 1
        self.equity_curve.append(self.total_rr)

    def winrate(self):
        if self.total_trades > 0:
            return (self.wins / self.total_trades) * 100 
        else:
            return 0.0

    def avg_rr(self):
        if self.total_trades > 0:
            return self.total_rr / self.total_trades 
        else:
            return 0.0

class ExternalBOSDetector:
    def __init__(self, lookback=50):
        self.lookback = lookback

    def detect_swings(self, df):
        altos = df['high'].values
        bajos = df['low'].values
        swing_alto_reciente = max(altos[-self.lookback:])
        swing_bajo_reciente = min(bajos[-self.lookback:])
        return swing_alto_reciente, swing_bajo_reciente

    def is_bos_externo(self, df):
        alto_est, bajo_est = self.detect_swings(df)
        cierre = df['close'].iloc[-1]
        
        if cierre > alto_est:
            rup_alza = True
        else:
            rup_alza = False
            
        if cierre < bajo_est:
            rup_baja = True
        else:
            rup_baja = False
            
        return rup_alza, rup_baja, alto_est, bajo_est

class PullbackValidator:
    def __init__(self, tolerance=0.3):
        self.tolerance = tolerance

    def es_pullback_valido(self, df, nivel, direccion):
        precio = df['close'].iloc[-1]
        if direccion == "long":
            if precio <= nivel * (1 - self.tolerance / 100):
                return True
            return False
        if direccion == "short":
            if precio >= nivel * (1 + self.tolerance / 100):
                return True
            return False
        return False

class PartialTPManager:
    def __init__(self):
        self.tp1_hit = False
        self.tp2_hit = False
        
    def gestionar_tp_parcial(self, entry, tp1, tp2, price, side):
        return {"cerrar_50": False, "cerrar_total": False, "evento": None} 

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
# LOOP PRINCIPAL 
# ======================================================
def run_bot():
    telegram_mensaje("🤖 BOT V99.0 BYBIT REAL INICIADO.\nPolaridad Activa y Trailing Dinámico Acelerado.")
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

            # Detección integral con POLARIDAD V99.0
            soporte_horiz, resistencia_horiz, canal_sup, canal_inf, slope, intercept, tendencia_macro, max_previo, min_previo = detectar_zonas_mercado(df, idx_eval)
            
            razones_para_entrar =[]
            
            if PAPER_POSICION_ACTIVA is None:
                patron_detectado, decision_final, nombre_patron, log_zonas = detectar_patron_nison(df, soporte_horiz, resistencia_horiz, canal_sup, canal_inf, tendencia_macro, max_previo, min_previo, idx=idx_eval)

                if ultima_vela_operada == tiempo_vela_cerrada:
                    decision_final = None
                    if patron_detectado:
                        lista_log =[f"Patrón {nombre_patron} bloqueado (Anti-Spam de vela)"]
                        patron_detectado = False
                    else:
                        lista_log =["Buscando patrón Nison CERRADO válido..."]
                else:
                    if patron_detectado == True:
                        lista_log = [nombre_patron]
                    else:
                        lista_log =["Buscando patrón Nison CERRADO válido..."]
                    
                log_colab(df, tendencia_macro, slope, soporte_horiz, resistencia_horiz, decision_final, lista_log, log_zonas, idx_eval)

                if decision_final is not None:
                    if decision_final == "Buy":
                        zonas_activadas = log_zonas.get('zonas_compra')
                    else:
                        zonas_activadas = log_zonas.get('zonas_venta')
                        
                    razones_para_entrar.append(f"Patrón Validado: {nombre_patron}")
                    razones_para_entrar.append(f"Zonas de Rechazo: {zonas_activadas}")
                    
                    if risk_management_check():
                        apertura = paper_abrir_posicion(decision_final, precio_mercado, df['atr'].iloc[-1], razones_para_entrar, df.index[-1])
                        if apertura:
                            ultima_vela_operada = tiempo_vela_cerrada
                            texto = f"📌 OPERACIÓN {decision_final.upper()} (5m)\n💰 Entrada: {precio_mercado:.2f}\n📍 SL: {PAPER_SL:.2f} | TP1: {PAPER_TP1:.2f}\n🧠 {', '.join(razones_para_entrar)}"
                            telegram_mensaje(texto)
                            
                            fig = generar_grafico_entrada(df, decision_final, soporte_horiz, resistencia_horiz, slope, intercept, razones_para_entrar)
                            if fig is not None: 
                                telegram_grafico(fig)
                                plt.close(fig)

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
            return False
