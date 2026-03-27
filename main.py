# BOT TRADING V91.7 BYBIT REAL – PRODUCCIÓN (SIN PROXY) 
# ======================================================
# ⚠️ KEYS INCLUIDAS TAL CUAL (SEGÚN PEDIDO)
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit
# ======================================================
# NOVEDADES V91.7 (SNIPER ZONES):
# - FIX: Soportes y Resistencias mucho más estrictos y pegados al nivel.
# - Tolerancia reducida a 1.5x ATR o un mínimo de 80 USD.
# - El bot solo operará patrones que "besen" la línea de S/R real.
# - Simulación financiera estricta: Apalancamiento 10x y Riesgo 2%.
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
MARGEN_NIVEL_BASE = 80  # Mínimo de dólares de tolerancia para S/R ajustado

def cerca_de_nivel(precio, nivel, margen):
    distancia = abs(precio - nivel)
    if distancia <= margen:
        return True
    else:
        return False

SYMBOL = "BTCUSDT"
INTERVAL = "1"
# Ajustes Financieros Realistas para cuenta de $100
RISK_PER_TRADE = 0.02  # 2% de riesgo por trade ($2.00 USD)
LEVERAGE = 10          # 10x de apalancamiento (Poder de compra de $1000)
SLEEP_SECONDS = 60

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
PAPER_TP = None
PAPER_TP1 = None
PAPER_TP2 = None
PAPER_PARTIAL_ACTIVADO = False
PAPER_SIZE_BTC_RESTANTE = 0.0
PAPER_TP1_EJECUTADO = False
PAPER_PNL_PARCIAL = 0.0  
PAPER_ULTIMO_RESULTADO = None
PAPER_ULTIMO_PNL = 0.0
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
PAPER_MAX_DRAWDOWN = 0.0
PAPER_BALANCE_MAX = PAPER_BALANCE_INICIAL

# ======================================================
# CONTROL DINÁMICO DE RIESGO
# ======================================================
MAX_CONSECUTIVE_LOSSES = 6
PAUSE_AFTER_LOSSES_SECONDS = 60 * 60 * 2
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
        payload = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": texto
        }
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

    df_limpio = df.dropna()
    return df_limpio

def detectar_soportes_resistencias(df, idx=-2):
    df_eval = df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    return soporte, resistencia

def detectar_tendencia_macro(df, idx=-2, ventana=120):
    df_eval = df.iloc[:idx+1]
    if len(df_eval) < ventana:
        y = df_eval['close'].values
    else:
        y = df_eval['close'].values[-ventana:]
        
    x = np.arange(len(y))
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    if slope > 0.01: 
        tendencia = '📈 ALCISTA'
    elif slope < -0.01: 
        tendencia = '📉 BAJISTA'
    else: 
        tendencia = '➡️ LATERAL'
        
    return slope, intercept, tendencia


# ======================================================
# 🕯️ ARSENAL NISON (SÚPER HUMANIZADO)
# ======================================================

def calcular_cuerpo_mechas(row):
    cuerpo = abs(row['close'] - row['open'])
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

def tendencia_previa_micro(df, idx, velas=6):
    if idx - velas < 0: 
        return "neutral"
        
    precio_inicio = df['close'].iloc[idx-velas]
    precio_fin = df['close'].iloc[idx-1]
    
    if precio_fin < precio_inicio:
        return "bajista"
    elif precio_fin > precio_inicio:
        return "alcista"
    else:
        return "lateral"

def cierre_fuerte(row, direccion):
    cuerpo, mecha_sup, mecha_inf, rango, top, bottom = calcular_cuerpo_mechas(row)
    
    if direccion == "alcista":
        return row['close'] >= (row['low'] + (rango * 0.55))
    else:
        return row['close'] <= (row['high'] - (rango * 0.55))

# --- 1. HAMMER ---
def es_hammer_nison(df, idx):
    vela_martillo = df.iloc[idx-1] 
    vela_confirmacion = df.iloc[idx]   
    
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela_martillo)
    if cuerpo == 0: cuerpo = 0.0001
    
    condicion_mecha_larga = m_inf >= (1.2 * cuerpo)
    condicion_sin_mecha_arriba = m_sup <= (m_inf * 0.8) 
    
    forma_valida = condicion_mecha_larga and condicion_sin_mecha_arriba
    
    confirmacion_alcista = vela_confirmacion['close'] > vela_confirmacion['open'] 
    
    if forma_valida and confirmacion_alcista:
        return True
    return False

# --- 2. SHOOTING STAR ---
def es_shooting_star_nison(df, idx):
    vela_estrella = df.iloc[idx-1]
    vela_confirmacion = df.iloc[idx]
    
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela_estrella)
    if cuerpo == 0: cuerpo = 0.0001
    
    condicion_mecha_larga = m_sup >= (1.2 * cuerpo)
    condicion_sin_mecha_abajo = m_inf <= (m_sup * 0.8)
    
    forma_valida = condicion_mecha_larga and condicion_sin_mecha_abajo
    
    confirmacion_bajista = vela_confirmacion['close'] < vela_confirmacion['open'] 
    
    if forma_valida and confirmacion_bajista:
        return True
    return False

# --- 3. BULLISH ENGULFING ---
def es_bullish_engulfing_nison(df, idx):
    vela_previa = df.iloc[idx-1]
    vela_actual = df.iloc[idx]
    
    es_previa_roja = vela_previa['close'] <= vela_previa['open']
    es_actual_verde = vela_actual['close'] > vela_actual['open']
    
    if not (es_previa_roja and es_actual_verde):
        return False
    
    condicion_envuelve = vela_actual['close'] > vela_previa['open']
    
    if condicion_envuelve and cierre_fuerte(vela_actual, "alcista"):
        return True
    return False

# --- 4. BEARISH ENGULFING ---
def es_bearish_engulfing_nison(df, idx):
    vela_previa = df.iloc[idx-1]
    vela_actual = df.iloc[idx]
    
    es_previa_verde = vela_previa['close'] >= vela_previa['open']
    es_actual_roja = vela_actual['close'] < vela_actual['open']
    
    if not (es_previa_verde and es_actual_roja):
        return False
    
    condicion_envuelve = vela_actual['close'] < vela_previa['open']
    
    if condicion_envuelve and cierre_fuerte(vela_actual, "bajista"):
        return True
    return False

# --- 5. PIERCING PATTERN ---
def es_piercing_nison(df, idx):
    vela_previa = df.iloc[idx-1]
    vela_actual = df.iloc[idx]
    
    es_previa_roja = vela_previa['close'] < vela_previa['open']
    es_actual_verde = vela_actual['close'] > vela_actual['open']
    
    if not (es_previa_roja and es_actual_verde):
        return False
        
    rango_cuerpo_rojo = vela_previa['open'] - vela_previa['close']
    umbral_penetracion = vela_previa['close'] + (rango_cuerpo_rojo * 0.35) 
    
    condicion_penetra = vela_actual['close'] >= umbral_penetracion
    condicion_apertura_baja = vela_actual['open'] <= vela_previa['close'] + (rango_cuerpo_rojo * 0.3) 
    condicion_no_envuelve = vela_actual['close'] <= vela_previa['open']
    
    if condicion_apertura_baja and condicion_penetra and condicion_no_envuelve and cierre_fuerte(vela_actual, "alcista"):
        return True
    return False

# --- 6. DARK CLOUD COVER ---
def es_dark_cloud_nison(df, idx):
    vela_previa = df.iloc[idx-1]
    vela_actual = df.iloc[idx]
    
    es_previa_verde = vela_previa['close'] > vela_previa['open']
    es_actual_roja = vela_actual['close'] < vela_actual['open']
    
    if not (es_previa_verde and es_actual_roja):
        return False
    
    rango_cuerpo_verde = vela_previa['close'] - vela_previa['open']
    umbral_penetracion = vela_previa['close'] - (rango_cuerpo_verde * 0.35)
    
    condicion_penetra = vela_actual['close'] <= umbral_penetracion
    condicion_apertura_alta = vela_actual['open'] >= vela_previa['close'] - (rango_cuerpo_verde * 0.3)
    condicion_no_envuelve = vela_actual['close'] >= vela_previa['open']
    
    if condicion_apertura_alta and condicion_penetra and condicion_no_envuelve and cierre_fuerte(vela_actual, "bajista"):
        return True
    return False

# --- 7. MORNING STAR ---
def es_morning_star_nison(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    es_c1_roja = c1['close'] < c1['open']
    es_c3_verde = c3['close'] > c3['open']
    if not (es_c1_roja and es_c3_verde): return False
    
    c1_body, _, _, _, _, _ = calcular_cuerpo_mechas(c1)
    c2_body, _, _, _, _, _ = calcular_cuerpo_mechas(c2)
    
    condicion_cuerpo_flexible = c2_body <= (c1_body * 0.8) 
    
    umbral_penetracion = c1['close'] + (c1_body * 0.20)
    condicion_penetracion = c3['close'] >= umbral_penetracion
    
    if condicion_cuerpo_flexible and condicion_penetracion and cierre_fuerte(c3, "alcista"):
        return True
    return False

# --- 8. EVENING STAR ---
def es_evening_star_nison(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    es_c1_verde = c1['close'] > c1['open']
    es_c3_roja = c3['close'] < c3['open']
    if not (es_c1_verde and es_c3_roja): return False
    
    c1_body, _, _, _, _, _ = calcular_cuerpo_mechas(c1)
    c2_body, _, _, _, _, _ = calcular_cuerpo_mechas(c2)
    
    condicion_cuerpo_flexible = c2_body <= (c1_body * 0.8)
    
    umbral_penetracion = c1['close'] - (c1_body * 0.20)
    condicion_penetracion = c3['close'] <= umbral_penetracion
    
    if condicion_cuerpo_flexible and condicion_penetracion and cierre_fuerte(c3, "bajista"):
        return True
    return False

# --- 9. TWEEZER BOTTOMS ---
def es_tweezer_bottom_nison(df, idx):
    c1, c2 = df.iloc[idx-1], df.iloc[idx]
    
    atr_actual = df['atr'].iloc[idx]
    tolerancia = atr_actual * 0.50 
    
    mismos_minimos = abs(c2['low'] - c1['low']) <= tolerancia
    es_c2_verde = c2['close'] > c2['open']
    
    if mismos_minimos and es_c2_verde and cierre_fuerte(c2, "alcista"):
        return True
    return False

# --- 10. TWEEZER TOPS ---
def es_tweezer_top_nison(df, idx):
    c1, c2 = df.iloc[idx-1], df.iloc[idx]
    
    atr_actual = df['atr'].iloc[idx]
    tolerancia = atr_actual * 0.50
    
    mismos_maximos = abs(c2['high'] - c1['high']) <= tolerancia
    es_c2_roja = c2['close'] < c2['open']
    
    if mismos_maximos and es_c2_roja and cierre_fuerte(c2, "bajista"):
        return True
    return False

# --- 11. THREE WHITE SOLDIERS ---
def es_three_white_soldiers(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    rsi_actual = df['rsi'].iloc[idx]
    if rsi_actual > 75: return False
    
    es_c1_verde = c1['close'] > c1['open']
    es_c2_verde = c2['close'] > c2['open']
    es_c3_verde = c3['close'] > c3['open']
    
    if es_c1_verde and es_c2_verde and es_c3_verde:
        subida_escalonada = (c2['close'] > c1['close']) and (c3['close'] > c2['close'])
        if subida_escalonada and cierre_fuerte(c2, "alcista") and cierre_fuerte(c3, "alcista"):
            return True
    return False

# --- 12. THREE BLACK CROWS ---
def es_three_black_crows(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    rsi_actual = df['rsi'].iloc[idx]
    if rsi_actual < 25: return False
    
    es_c1_roja = c1['close'] < c1['open']
    es_c2_roja = c2['close'] < c2['open']
    es_c3_roja = c3['close'] < c3['open']
    
    if es_c1_roja and es_c2_roja and es_c3_roja:
        bajada_escalonada = (c2['close'] < c1['close']) and (c3['close'] < c2['close'])
        if bajada_escalonada and cierre_fuerte(c2, "bajista") and cierre_fuerte(c3, "bajista"):
            return True
    return False

# === DETECTOR MAESTRO NISON ===
def detectar_patron_nison(df, soporte, resistencia, idx=-2):
    if len(df) < 15: 
        return False, None, None, {}
        
    atr_actual = df['atr'].iloc[idx]
    
    min_patron = df['low'].iloc[idx-2 : idx+1].min()
    max_patron = df['high'].iloc[idx-2 : idx+1].max()
    
    tendencia_previa = tendencia_previa_micro(df, idx)
    
    # V91.7: Tolerancia ajustada a 1.5x ATR o un mínimo de 80 dólares (ZONA SNIPER)
    tolerancia_zona = max(atr_actual * 1.5, 80)
    
    en_soporte = cerca_de_nivel(min_patron, soporte, tolerancia_zona) 
    en_resistencia = cerca_de_nivel(max_patron, resistencia, tolerancia_zona) 
    
    log_zonas = {
        "min_patron": min_patron, "max_patron": max_patron,
        "dist_soporte": abs(min_patron - soporte),
        "dist_resistencia": abs(max_patron - resistencia),
        "tolerancia": tolerancia_zona,
        "en_soporte": en_soporte, "en_resistencia": en_resistencia,
        "micro_tendencia": tendencia_previa
    }
    
    # --- PATRONES DE COMPRA ---
    if en_soporte:
        if es_hammer_nison(df, idx): return True, "Buy", "Nison Hammer", log_zonas
        if es_bullish_engulfing_nison(df, idx): return True, "Buy", "Nison Bullish Engulfing", log_zonas
        if es_piercing_nison(df, idx): return True, "Buy", "Nison Piercing Pattern", log_zonas
        if es_morning_star_nison(df, idx): return True, "Buy", "Nison Morning Star", log_zonas
        if es_tweezer_bottom_nison(df, idx): return True, "Buy", "Nison Tweezer Bottoms", log_zonas
        if es_three_white_soldiers(df, idx): return True, "Buy", "Three White Soldiers", log_zonas

    # --- PATRONES DE VENTA ---
    if en_resistencia:
        if es_shooting_star_nison(df, idx): return True, "Sell", "Nison Shooting Star", log_zonas
        if es_bearish_engulfing_nison(df, idx): return True, "Sell", "Nison Bearish Engulfing", log_zonas
        if es_dark_cloud_nison(df, idx): return True, "Sell", "Nison Dark Cloud Cover", log_zonas
        if es_evening_star_nison(df, idx): return True, "Sell", "Nison Evening Star", log_zonas
        if es_tweezer_top_nison(df, idx): return True, "Sell", "Nison Tweezer Tops", log_zonas
        if es_three_black_crows(df, idx): return True, "Sell", "Three Black Crows", log_zonas

    return False, None, None, log_zonas


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
            if closes[i] >= opens[i]:
                color_vela = 'green'
            else:
                color_vela = 'red'
                
            ax.vlines(x_valores[i], lows[i], highs[i], color=color_vela, linewidth=1)
            
            cuerpo_y = min(opens[i], closes[i])
            cuerpo_h = abs(closes[i] - opens[i])
            if cuerpo_h == 0:
                cuerpo_h = 0.0001
                
            rectangulo_vela = plt.Rectangle((x_valores[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color_vela, alpha=0.9)
            ax.add_patch(rectangulo_vela)

        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")

        y_tendencia_base = closes
        slope_plot, intercept_plot, r_val, p_val, err = linregress(x_valores, y_tendencia_base)
        linea_tendencia = intercept_plot + slope_plot * x_valores
        
        ax.plot(x_valores, linea_tendencia, color='white', linewidth=2, linestyle='-', label="Tendencia Macro")

        residuos = y_tendencia_base - linea_tendencia
        desviacion = np.std(residuos)
        
        banda_superior = linea_tendencia + (desviacion * 1.5)
        banda_inferior = linea_tendencia - (desviacion * 1.5)
        
        ax.plot(x_valores, banda_superior, linestyle='--', linewidth=2, color='red')
        ax.plot(x_valores, banda_inferior, linestyle='--', linewidth=2, color='green')

        entrada_x_idx = len(df_plot) - 2
        entrada_precio_final = closes[-2]
        
        if decision == 'Buy':
            ax.scatter(entrada_x_idx, entrada_precio_final, s=200, marker='^', color='lime', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='lime', linestyle=':', linewidth=2)
        else:
            ax.scatter(entrada_x_idx, entrada_precio_final, s=200, marker='v', color='red', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='red', linestyle=':', linewidth=2)

        rsi_actual = df['rsi'].iloc[-2]
        texto_razones = "\n".join(razones)
        
        texto_panel = f"OPERACION: {decision.upper()}\nPrecio: {df['close'].iloc[-1]:.2f}\nRSI Contexto: {rsi_actual:.1f}\n\nRazones:\n{texto_razones}"
        
        ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_title(f"BOT V91.7 - BTCUSDT - Entrada {decision} Sniper")
        ax.grid(True, alpha=0.2)
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
        motivo_cierre = trade_data['motivo']
        balance_actual = trade_data['balance']
        
        df_plot = df.copy().tail(120)
        if df_plot.empty:
            return None

        fig, ax = plt.subplots(figsize=(14, 7))

        for i, (idx, row) in enumerate(df_plot.iterrows()):
            if row['close'] >= row['open']:
                color_vela = 'green'
            else:
                color_vela = 'red'
                
            ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
            ax.plot([i, i],[row['open'], row['close']], color=color_vela, linewidth=6)

        ax.axhline(entrada_price, color='blue', linestyle='--', linewidth=1.5, label='Nivel de Entrada Original')
        ax.axhline(salida_price, color='orange', linestyle='--', linewidth=1.5, label='Nivel de Cierre Definitivo')

        indice_salida_x = len(df_plot) - 1
        
        if pnl_obtenido > 0:
            color_marcador = 'lime'
            forma_marcador = '^'
            texto_resultado = "GANADA (+)"  
        else:
            color_marcador = 'red'
            forma_marcador = 'v'
            texto_resultado = "PERDIDA (-)" 
            
        ax.scatter([indice_salida_x],[salida_price], s=200, c=color_marcador, marker=forma_marcador, edgecolors='black', zorder=5)

        texto_panel_salida = f"CIERRE DE OPERACION {decision_original}\nMotivo de Salida: {motivo_cierre}\nResultado PnL: {pnl_obtenido:.4f} USD\nNuevo Balance: {balance_actual:.2f} USD"
        ax.text(0.02, 0.95, texto_panel_salida, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title(f"BOT V91.7 - DETALLE DE CIERRE - {texto_resultado}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"🚨 ERROR GRAFICO SALIDA: {e}")
        return None

def log_colab(df, tendencia, slope, soporte, resistencia, decision, razones, log_zonas, idx=-2):
    ahora = datetime.now(timezone.utc)
    precio = df['close'].iloc[idx]
    atr = df['atr'].iloc[idx]

    print("="*100)
    print(f"🕒 {ahora} | 💰 Precio Analizado: {precio:.2f}")
    
    if log_zonas:
        print(f"🔎 Distancia al Soporte: {log_zonas.get('dist_soporte', 0):.2f} USD | En Zona: {log_zonas.get('en_soporte')}")
        print(f"🔎 Distancia a Resistencia: {log_zonas.get('dist_resistencia', 0):.2f} USD | En Zona: {log_zonas.get('en_resistencia')}")
        print(f"🔎 Micro Tendencia: {log_zonas.get('micro_tendencia').upper()} (Informativo) (Tolerancia Zona: {log_zonas.get('tolerancia', 0):.2f} USD)")

    if decision:
        print(f"🎯 DECISIÓN TOMADA: {decision.upper()}")
    else:
        print(f"🎯 DECISIÓN TOMADA: MANTENER AL MARGEN (NO TRADE)")
        
    for razon in razones:
        print(f"🧠 Lógica: {razon}")
        
    print("="*100)

# ======================================================
# MOTOR FINANCIERO Y GESTIÓN (V91.7 REALISTIC TRACKER)
# ======================================================

def paper_abrir_posicion(decision, precio, atr, soporte, resistencia, razones, tiempo):
    global PAPER_POSICION_ACTIVA
    global PAPER_PRECIO_ENTRADA
    global PAPER_SL
    global PAPER_TP1
    global PAPER_TP2
    global PAPER_SIZE_USD
    global PAPER_SIZE_BTC
    global PAPER_SIZE_BTC_RESTANTE
    global PAPER_DECISION_ACTIVA
    global PAPER_PARTIAL_ACTIVADO
    global PAPER_TP1_EJECUTADO
    global PAPER_PNL_PARCIAL
    
    if PAPER_POSICION_ACTIVA is not None: 
        return False

    # El riesgo que estamos dispuestos a asumir ($100 * 2% = $2.00 USD perdidos si toca SL)
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    
    if decision == "Buy":
        sl = precio - atr 
        tp1 = precio + atr 
    elif decision == "Sell":
        sl = precio + atr 
        tp1 = precio - atr 
    else:
        return False
    
    distancia_riesgo = abs(precio - sl)
    if distancia_riesgo == 0: 
        return False

    # Calculamos cuántos BTC compraríamos para que la caída al SL sea exactamente $2.00 de pérdida
    size_en_cripto_ideal = riesgo_usd / distancia_riesgo
    size_en_dolares_ideal = size_en_cripto_ideal * precio
    
    # ⚠️ VALIDADOR DE MARGEN Y APALANCAMIENTO ⚠️
    poder_de_compra_maximo = PAPER_BALANCE * LEVERAGE
    
    if size_en_dolares_ideal > poder_de_compra_maximo:
        size_en_dolares_real = poder_de_compra_maximo
        size_en_cripto_real = size_en_dolares_real / precio
        margen_usado = PAPER_BALANCE 
    else:
        size_en_dolares_real = size_en_dolares_ideal
        size_en_cripto_real = size_en_cripto_ideal
        margen_usado = size_en_dolares_real / LEVERAGE
    
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

    print(f"💰 FINANZAS: Margen Usado: {margen_usado:.2f} USD | Posición Total: {PAPER_SIZE_USD:.2f} USD | Apalancamiento: {LEVERAGE}x")

    return True

def paper_revisar_sl_tp(df):
    global PAPER_SL
    global PAPER_TP1
    global PAPER_PRECIO_ENTRADA
    global PAPER_DECISION_ACTIVA
    global PAPER_POSICION_ACTIVA
    global PAPER_BALANCE
    global PAPER_PNL_GLOBAL
    global PAPER_TRADES_TOTALES
    global PAPER_ULTIMO_RESULTADO
    global PAPER_ULTIMO_PNL
    global PAPER_SIZE_BTC
    global PAPER_SIZE_BTC_RESTANTE
    global PAPER_TP1_EJECUTADO
    global PAPER_WIN
    global PAPER_LOSS
    global PAPER_PNL_PARCIAL

    if PAPER_POSICION_ACTIVA is None: 
        return None

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    close = df['close'].iloc[-1]
    atr_actual = df['atr'].iloc[-1]
    
    cerrar_total = False
    motivo = None
    
    TRAILING_MULT = 1.2 

    # ==========================
    # REVISIÓN DE COMPRAS (LONG)
    # ==========================
    if PAPER_POSICION_ACTIVA == "Buy":
        if PAPER_TP1_EJECUTADO == False:
            if high >= PAPER_TP1:
                mitad_posicion = PAPER_SIZE_BTC / 2
                PAPER_PNL_PARCIAL = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * mitad_posicion
                
                PAPER_BALANCE += PAPER_PNL_PARCIAL
                PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
                PAPER_SIZE_BTC_RESTANTE = mitad_posicion
                PAPER_TP1_EJECUTADO = True
                
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). 50% cerrado, SL a Break Even. Iniciando Trailing...")

        if PAPER_TP1_EJECUTADO == True:
            nuevo_sl_dinamico = close - (atr_actual * TRAILING_MULT)
            if nuevo_sl_dinamico > PAPER_SL:
                PAPER_SL = nuevo_sl_dinamico 

        if low <= PAPER_SL:
            cerrar_total = True
            if PAPER_TP1_EJECUTADO:
                motivo = "Trailing Stop Dinámico" 
            else:
                motivo = "Stop Loss"

    # ==========================
    # REVISIÓN DE VENTAS (SHORT)
    # ==========================
    elif PAPER_POSICION_ACTIVA == "Sell":
        if PAPER_TP1_EJECUTADO == False:
            if low <= PAPER_TP1:
                mitad_posicion = PAPER_SIZE_BTC / 2
                PAPER_PNL_PARCIAL = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * mitad_posicion
                
                PAPER_BALANCE += PAPER_PNL_PARCIAL
                PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
                PAPER_SIZE_BTC_RESTANTE = mitad_posicion
                PAPER_TP1_EJECUTADO = True
                
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). 50% cerrado, SL a Break Even. Iniciando Trailing...")

        if PAPER_TP1_EJECUTADO == True:
            nuevo_sl_dinamico = close + (atr_actual * TRAILING_MULT)
            if nuevo_sl_dinamico < PAPER_SL:
                PAPER_SL = nuevo_sl_dinamico 

        if high >= PAPER_SL:
            cerrar_total = True
            if PAPER_TP1_EJECUTADO:
                motivo = "Trailing Stop Dinámico" 
            else:
                motivo = "Stop Loss"

    if cerrar_total == True:
        if PAPER_POSICION_ACTIVA == "Buy":
            pnl_final = (PAPER_SL - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE 
        else:
            pnl_final = (PAPER_PRECIO_ENTRADA - PAPER_SL) * PAPER_SIZE_BTC_RESTANTE
            
        decision_almacenada = PAPER_DECISION_ACTIVA
        precio_entrada_almacenado = PAPER_PRECIO_ENTRADA
        precio_salida_almacenado = PAPER_SL
        
        PAPER_BALANCE += pnl_final
        PAPER_PNL_GLOBAL += pnl_final
        PAPER_TRADES_TOTALES += 1
        
        if PAPER_TP1_EJECUTADO:
            pnl_total_trade = PAPER_PNL_PARCIAL + pnl_final
        else:
            pnl_total_trade = pnl_final
            
        if pnl_total_trade > 0:
            PAPER_WIN += 1
        else:
            PAPER_LOSS += 1
            
        winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100
        crecimiento_pct = ((PAPER_BALANCE - PAPER_BALANCE_INICIAL) / PAPER_BALANCE_INICIAL) * 100
        
        PAPER_POSICION_ACTIVA = None
        PAPER_DECISION_ACTIVA = None
        PAPER_PRECIO_ENTRADA = None
        PAPER_SL = None
        PAPER_TP1 = None
        PAPER_SIZE_BTC = 0.0
        PAPER_SIZE_BTC_RESTANTE = 0.0
        PAPER_TP1_EJECUTADO = False
        PAPER_PNL_PARCIAL = 0.0

        texto_cierre = f"📤 TRADE CERRADO: Salida por {motivo}.\n"
        texto_cierre += f"💵 Ganancia/Pérdida Neta del Trade: {pnl_total_trade:.2f} USD\n\n"
        texto_cierre += f"📊 ESTADO DE LA CUENTA (PAPER TRADING)\n"
        texto_cierre += f"Balance Inicial: {PAPER_BALANCE_INICIAL:.2f} USD\n"
        texto_cierre += f"Balance Actual: {PAPER_BALANCE:.2f} USD\n"
        texto_cierre += f"Rendimiento (ROI): {crecimiento_pct:.2f}%\n"
        texto_cierre += f"Winrate Global: {winrate:.1f}% ({PAPER_WIN}W / {PAPER_LOSS}L)"
        
        telegram_mensaje(texto_cierre)
        
        data_de_retorno = {
            "decision": decision_almacenada, 
            "motivo": motivo, 
            "entrada": precio_entrada_almacenado, 
            "salida": precio_salida_almacenado, 
            "pnl": pnl_total_trade, 
            "balance": PAPER_BALANCE
        }
        return data_de_retorno

    return None

def risk_management_check():
    global PAPER_PAUSE_UNTIL
    global PAPER_STOPPED_TODAY
    global PAPER_DAILY_START_BALANCE
    global PAPER_CURRENT_DAY
    global PAPER_BALANCE
    global PAPER_CONSECUTIVE_LOSSES
    
    ahora_utc = datetime.now(timezone.utc)
    hoy_utc = ahora_utc.date()
    
    if PAPER_CURRENT_DAY != hoy_utc:
        PAPER_CURRENT_DAY = hoy_utc
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
        PAPER_CONSECUTIVE_LOSSES = 0
        telegram_mensaje(f"🔄 Nuevo ciclo diario UTC. Balance base establecido en: {PAPER_BALANCE:.2f} USD.")
        
    diferencia_balance = PAPER_BALANCE - PAPER_DAILY_START_BALANCE
    porcentaje_drawdown = diferencia_balance / PAPER_DAILY_START_BALANCE
    
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
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.partial_wins = 0
        self.total_rr = 0.0
        self.equity_curve =[]
        self.trade_log =[]

    def register_trade(self, result_rr, partial=False):
        self.total_trades += 1
        self.total_rr += result_rr
        
        if partial == True: 
            self.partial_wins += 1
        elif result_rr > 0: 
            self.wins += 1
        else: 
            self.losses += 1
            
        self.equity_curve.append(self.total_rr)

    def winrate(self):
        if self.total_trades == 0: 
            return 0.0
        tasa = (self.wins / self.total_trades) * 100
        return tasa

    def avg_rr(self):
        if self.total_trades == 0: 
            return 0.0
        promedio = self.total_rr / self.total_trades
        return promedio

class ExternalBOSDetector:
    def __init__(self, lookback=50):
        self.lookback = lookback
        self.last_swing_high = None
        self.last_swing_low = None

    def detect_swings(self, df):
        altos = df['high'].values
        bajos = df['low'].values
        
        swing_alto_reciente = max(altos[-self.lookback:])
        swing_bajo_reciente = min(bajos[-self.lookback:])
        
        self.last_swing_high = swing_alto_reciente
        self.last_swing_low = swing_bajo_reciente
        
        return swing_alto_reciente, swing_bajo_reciente

    def is_bos_externo(self, df):
        alto_estructural, bajo_estructural = self.detect_swings(df)
        cierre_actual = df['close'].iloc[-1]
        
        ruptura_al_alza = cierre_actual > alto_estructural
        ruptura_a_la_baja = cierre_actual < bajo_estructural
        
        return ruptura_al_alza, ruptura_a_la_baja, alto_estructural, bajo_estructural

class PullbackValidator:
    def __init__(self, tolerance=0.3):
        self.tolerance = tolerance

    def es_pullback_valido(self, df, nivel_estructura, direccion):
        precio_vela_actual = df['close'].iloc[-1]
        
        if direccion == "long":
            tolerancia_valor = self.tolerance / 100
            zona_aceptable = nivel_estructura * (1 - tolerancia_valor)
            if precio_vela_actual <= zona_aceptable:
                return True
            else:
                return False
                
        if direccion == "short":
            tolerancia_valor = self.tolerance / 100
            zona_aceptable = nivel_estructura * (1 + tolerancia_valor)
            if precio_vela_actual >= zona_aceptable:
                return True
            else:
                return False
                
        return False

class PartialTPManager:
    def __init__(self):
        self.tp1_hit = False
        self.tp2_hit = False

    def gestionar_tp_parcial(self, entry, tp1, tp2, price, side):
        resultado = {
            "cerrar_50": False, 
            "cerrar_total": False, 
            "evento": None
        }
        
        if side == "long":
            if self.tp1_hit == False and price >= tp1:
                self.tp1_hit = True
                resultado["cerrar_50"] = True
                resultado["evento"] = "El TP1 ha sido alcanzado. Cerramos el 50%."
            elif price >= tp2:
                self.tp2_hit = True
                resultado["cerrar_total"] = True
                resultado["evento"] = "El TP2 ha sido alcanzado. Cerramos todo."
                
        if side == "short":
            if self.tp1_hit == False and price <= tp1:
                self.tp1_hit = True
                resultado["cerrar_50"] = True
                resultado["evento"] = "El TP1 ha sido alcanzado. Cerramos el 50%."
            elif price <= tp2:
                self.tp2_hit = True
                resultado["cerrar_total"] = True
                resultado["evento"] = "El TP2 ha sido alcanzado. Cerramos todo."
                
        return resultado

class InstitutionalLogger:
    def __init__(self, telegram_send_func):
        self.send_telegram = telegram_send_func

    def log_operacion_completa(self, data):
        texto_log = f"""
📊 REPORTE DE SISTEMA INSTITUCIONAL EXTERNO

🧠 Análisis de Bloque Estructural:
📈 Dirección Proyectada: {data.get('direccion')}
💰 Nivel de Entrada Teórico: {data.get('entry')}
🎯 Nivel TP1 Teórico: {data.get('tp1')}
🎯 Nivel TP2 Teórico: {data.get('tp2')}
🛑 Nivel SL Teórico: {data.get('sl')}

📊 Ratio Riesgo/Beneficio: {data.get('rr')}
🏆 Tasa de Éxito Actual: {data.get('winrate'):.2f}%
📉 Promedio R/R: {data.get('avg_rr'):.2f}
🔢 Cantidad de Operaciones de prueba: {data.get('total_trades')}
"""
        self.send_telegram(texto_log)

class InstitutionalSecondarySystem:
    def __init__(self, telegram_send_func):
        self.bos_detector = ExternalBOSDetector()
        self.pullback_validator = PullbackValidator()
        self.tp_manager = PartialTPManager()
        self.stats = InstitutionalStats()
        self.logger = InstitutionalLogger(telegram_send_func)

    def evaluar_confirmacion_institucional(self, df):
        bos_alcista, bos_bajista, alto, bajo = self.bos_detector.is_bos_externo(df)
        
        datos_confirmacion = {
            "confirmado": False, 
            "direccion": None, 
            "nivel_estructura": None
        }
        
        if bos_alcista == True:
            datos_confirmacion["confirmado"] = True
            datos_confirmacion["direccion"] = "long"
            datos_confirmacion["nivel_estructura"] = alto
            
        elif bos_bajista == True:
            datos_confirmacion["confirmado"] = True
            datos_confirmacion["direccion"] = "short"
            datos_confirmacion["nivel_estructura"] = bajo
            
        return datos_confirmacion

    def validar_pullback(self, df, direccion, nivel):
        valido = self.pullback_validator.es_pullback_valido(df, nivel, direccion)
        return valido

    def gestionar_trade_vivo(self, entry, tp1, tp2, price, side):
        gestion = self.tp_manager.gestionar_tp_parcial(entry, tp1, tp2, price, side)
        return gestion

    def registrar_resultado(self, rr, parcial=False):
        self.stats.register_trade(rr, parcial)

    def enviar_log_completo(self, trade_data):
        trade_data["winrate"] = self.stats.winrate()
        trade_data["avg_rr"] = self.stats.avg_rr()
        trade_data["total_trades"] = self.stats.total_trades
        self.logger.log_operacion_completa(trade_data)

# ======================================================
# LOOP PRINCIPAL
# ======================================================

def run_bot():
    mensaje_inicio = "🤖 BOT V91.7 BYBIT REAL INICIADO.\nConfiguración SNIPER ZONES habilitada.\nTrailing Dinámico Infinito Online."
    telegram_mensaje(mensaje_inicio)

    sistema_institucional = InstitutionalSecondarySystem(telegram_mensaje)

    while True:
        time.sleep(60) 
        
        try:
            # 1. Obtención de datos
            df_velas_crudas = obtener_velas()
            df = calcular_indicadores(df_velas_crudas)

            # 2. Índice de evaluación. Evaluamos en la vela CERRADA (-2)
            idx_eval = -2
            precio_mercado_actual = df['close'].iloc[-1] 

            # 3. Análisis General
            slope, intercept, tendencia_macro = detectar_tendencia_macro(df, idx=idx_eval)
            soporte, resistencia = detectar_soportes_resistencias(df, idx=idx_eval)
            
            razones_para_entrar =[]
            
            # 4. EL DETECTOR MAESTRO NISON 
            patron_detectado, decision_final, nombre_patron, log_zonas = detectar_patron_nison(df, soporte, resistencia, idx=idx_eval)

            # 5. Registro Consola
            if patron_detectado == True:
                lista_log = [nombre_patron]
            else:
                lista_log =["Buscando patrón Nison CERRADO válido..."]
                
            log_colab(df, tendencia_macro, slope, soporte, resistencia, decision_final, lista_log, log_zonas, idx=idx_eval)

            # 6. Toma de Decisión y Ejecución
            if decision_final is not None:
                razones_para_entrar.append(f"Arquitectura Confirmada: {nombre_patron}")
                razones_para_entrar.append(f"Geometria de S/R Respetada y Validada")
                
                riesgo_valido = risk_management_check()
                
                if riesgo_valido == True:
                    atr_entrada = df['atr'].iloc[-1]
                    apertura_exitosa = paper_abrir_posicion(decision_final, precio_mercado_actual, atr_entrada, soporte, resistencia, razones_para_entrar, df.index[-1])
                    
                    if apertura_exitosa == True:
                        texto_entrada = f"📌 SE HA INICIADO UNA OPERACIÓN {decision_final.upper()}\n"
                        texto_entrada += f"💰 Nivel de Entrada: {precio_mercado_actual:.2f}\n"
                        texto_entrada += f"📍 SL Inicial: {PAPER_SL:.2f} | TP1 Objetivo: {PAPER_TP1:.2f}\n"
                        
                        # Info extra de margen simulado
                        margen_inversion = PAPER_SIZE_USD / LEVERAGE
                        texto_entrada += f"💼 Margen Usado: {margen_inversion:.2f} USD ({LEVERAGE}x)\n"
                        
                        razones_unidas = ', '.join(razones_para_entrar)
                        texto_entrada += f"🧠 Justificación Analítica: {razones_unidas}"
                        
                        telegram_mensaje(texto_entrada)
                        
                        figura_generada = generar_grafico_entrada(df, decision_final, soporte, resistencia, slope, intercept, razones_para_entrar)
                        
                        if figura_generada is not None:
                            telegram_grafico(figura_generada)
                            plt.close(figura_generada)

            # 7. Gestión Contínua de Operaciones Abiertas (Usando datos en vivo, idx = -1)
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
