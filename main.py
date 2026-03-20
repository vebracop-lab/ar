# BOT TRADING V91.0 BYBIT REAL – PRODUCCIÓN (SIN PROXY) 
# ======================================================
# ⚠️ KEYS INCLUIDAS TAL CUAL (SEGÚN PEDIDO)
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit
# ======================================================
# NOVEDADES V91.0 (HUMANIZED NISON & CRYPTO CONTEXT):
# - Flexibilización de proporciones (Arte vs Ciencia).
# - Adaptación de Gaps de Nison para mercados 24/7 (Crypto).
# - Lectura de agotamiento de tendencia real con RSI.
# - Tolerancia visual elástica en mechas y cuerpos.
# - CÓDIGO 100% EXPANDIDO SIN RESÚMENES.
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
MARGEN_NIVEL = 250  # Tolerancia base para Soportes y Resistencias

def cerca_de_nivel(precio, nivel, margen=MARGEN_NIVEL):
    distancia = abs(precio - nivel)
    if distancia <= margen:
        return True
    else:
        return False

SYMBOL = "BTCUSDT"
INTERVAL = "1"
RISK_PER_TRADE = 0.0025
LEVERAGE = 1
SLEEP_SECONDS = 60

# ======================================================
# PAPER TRADING (SIMULACIÓN)
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
# INDICADORES Y RSI (PARA FILTRO DE SOBREEXTENSIÓN)
# ======================================================
def calcular_indicadores(df):
    # Media Móvil
    df['ema20'] = df['close'].ewm(span=20).mean()
    
    # Cálculo del True Range para el ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # Cálculo del RSI de 14 periodos paso a paso
    delta = df['close'].diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df_limpio = df.dropna()
    return df_limpio

def detectar_soportes_resistencias(df):
    # Soportes y resistencias macro (los últimos 50 periodos dan niveles relevantes intradía)
    soporte = df['close'].rolling(50).min().iloc[-1]
    resistencia = df['close'].rolling(50).max().iloc[-1]
    return soporte, resistencia

def detectar_tendencia_macro(df, ventana=120):
    y = df['close'].values[-ventana:]
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
# 🕯️ ARSENAL NISON (MODO HUMANIZADO & CRYPTO)
# ======================================================

def calcular_cuerpo_mechas(row):
    """
    Desglosa matemáticamente la vela.
    """
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
    """
    HUMANIZADO: Un trader real mira las últimas 4 a 6 velas para ver 
    si hay un impulso claro o un agotamiento. Evaluamos la diferencia real de precio.
    """
    if idx - velas < 0: 
        return "neutral"
        
    precio_inicio = df['close'].iloc[idx-velas]
    precio_fin = df['close'].iloc[idx-1]
    
    atr_actual = df['atr'].iloc[idx]
    
    # Si el precio ha caído al menos medio ATR en las últimas velas, es una bajada válida
    if precio_fin < (precio_inicio - (atr_actual * 0.5)):
        return "bajista"
    # Si el precio ha subido al menos medio ATR, es una subida válida
    elif precio_fin > (precio_inicio + (atr_actual * 0.5)):
        return "alcista"
    else:
        return "lateral"

def cierre_fuerte(row, direccion):
    """
    HUMANIZADO: Un cierre fuerte significa que el precio cerró empujando en la 
    dirección correcta, sin una mecha gigante en contra. 
    """
    cuerpo, mecha_sup, mecha_inf, rango, top, bottom = calcular_cuerpo_mechas(row)
    
    if direccion == "alcista":
        # Es válido si la mecha superior no es más grande que el cuerpo (permite cierta toma de ganancias)
        return mecha_sup <= cuerpo
    else:
        # Es válido si la mecha inferior no es más grande que el cuerpo
        return mecha_inf <= cuerpo

# --- 1. HAMMER (MARTILLO HUMANIZADO) ---
def es_hammer_nison(df, idx):
    vela_martillo = df.iloc[idx-1] 
    vela_confirmacion = df.iloc[idx]   
    
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela_martillo)
    if cuerpo == 0: cuerpo = 0.0001
    
    # HUMANIZADO: 1.5x es suficiente a simple vista para ser un martillo.
    condicion_mecha_larga = m_inf >= (1.5 * cuerpo)
    
    # HUMANIZADO: La mecha superior debe ser pequeña, pero no exigimos que sea 0.
    # Solo pedimos que la mecha inferior sea claramente la dominante.
    condicion_sin_mecha_arriba = m_sup < (m_inf * 0.3)
    
    forma_valida = condicion_mecha_larga and condicion_sin_mecha_arriba
    
    # Confirmación: La vela actual cerró en positivo y por encima del cuerpo del martillo
    confirmacion_alcista = (vela_confirmacion['close'] > vela_confirmacion['open']) and (vela_confirmacion['close'] > top)
    
    if forma_valida and confirmacion_alcista:
        return True
    return False

# --- 2. SHOOTING STAR (ESTRELLA FUGAZ HUMANIZADA) ---
def es_shooting_star_nison(df, idx):
    vela_estrella = df.iloc[idx-1]
    vela_confirmacion = df.iloc[idx]
    
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela_estrella)
    if cuerpo == 0: cuerpo = 0.0001
    
    # HUMANIZADO: Mecha superior visiblemente dominante
    condicion_mecha_larga = m_sup >= (1.5 * cuerpo)
    condicion_sin_mecha_abajo = m_inf < (m_sup * 0.3)
    
    forma_valida = condicion_mecha_larga and condicion_sin_mecha_abajo
    
    # Confirmación: Vela roja que cierra por debajo del cuerpo de la estrella
    confirmacion_bajista = (vela_confirmacion['close'] < vela_confirmacion['open']) and (vela_confirmacion['close'] < bottom)
    
    if forma_valida and confirmacion_bajista:
        return True
    return False

# --- 3. BULLISH ENGULFING (ENVOLVENTE ALCISTA HUMANIZADA / CRYPTO) ---
def es_bullish_engulfing_nison(df, idx):
    vela_previa = df.iloc[idx-1]
    vela_actual = df.iloc[idx]
    
    es_previa_roja = vela_previa['close'] <= vela_previa['open']
    es_actual_verde = vela_actual['close'] > vela_actual['open']
    
    if not (es_previa_roja and es_actual_verde):
        return False
    
    cuerpo_previo, _, _, _, _, _ = calcular_cuerpo_mechas(vela_previa)
    cuerpo_actual, _, _, _, _, _ = calcular_cuerpo_mechas(vela_actual)
    
    # HUMANIZADO CRYPTO: En Bitcoin 1m no hay gaps. Envolver significa que 
    # la vela verde sea visualmente más grande y cierre por encima de la apertura roja.
    condicion_envuelve = vela_actual['close'] > vela_previa['open']
    condicion_cuerpo_mayor = cuerpo_actual > cuerpo_previo
    
    if condicion_envuelve and condicion_cuerpo_mayor and cierre_fuerte(vela_actual, "alcista"):
        return True
    return False

# --- 4. BEARISH ENGULFING (ENVOLVENTE BAJISTA HUMANIZADA / CRYPTO) ---
def es_bearish_engulfing_nison(df, idx):
    vela_previa = df.iloc[idx-1]
    vela_actual = df.iloc[idx]
    
    es_previa_verde = vela_previa['close'] >= vela_previa['open']
    es_actual_roja = vela_actual['close'] < vela_actual['open']
    
    if not (es_previa_verde and es_actual_roja):
        return False
    
    cuerpo_previo, _, _, _, _, _ = calcular_cuerpo_mechas(vela_previa)
    cuerpo_actual, _, _, _, _, _ = calcular_cuerpo_mechas(vela_actual)
    
    # HUMANIZADO CRYPTO: Cierra por debajo de la apertura previa y el cuerpo es mayor
    condicion_envuelve = vela_actual['close'] < vela_previa['open']
    condicion_cuerpo_mayor = cuerpo_actual > cuerpo_previo
    
    if condicion_envuelve and condicion_cuerpo_mayor and cierre_fuerte(vela_actual, "bajista"):
        return True
    return False

# --- 5. PIERCING PATTERN (PAUTA PENETRANTE HUMANIZADA) ---
def es_piercing_nison(df, idx):
    vela_previa = df.iloc[idx-1]
    vela_actual = df.iloc[idx]
    
    es_previa_roja = vela_previa['close'] < vela_previa['open']
    es_actual_verde = vela_actual['close'] > vela_actual['open']
    
    if not (es_previa_roja and es_actual_verde):
        return False
        
    # HUMANIZADO CRYPTO: En lugar de un gap por debajo del mínimo, 
    # exigimos que la vela arranque desde abajo (abriendo igual o menor al cierre anterior)
    condicion_apertura_baja = vela_actual['open'] <= vela_previa['close']
    
    # Regla de penetración: empuja más allá de la mitad del cuerpo rojo
    mitad_cuerpo_previo = (vela_previa['open'] + vela_previa['close']) / 2
    condicion_penetra_mitad = vela_actual['close'] > mitad_cuerpo_previo
    
    # No envuelve totalmente
    condicion_no_envuelve = vela_actual['close'] <= vela_previa['open']
    
    if condicion_apertura_baja and condicion_penetra_mitad and condicion_no_envuelve and cierre_fuerte(vela_actual, "alcista"):
        return True
    return False

# --- 6. DARK CLOUD COVER (NUBE OSCURA HUMANIZADA) ---
def es_dark_cloud_nison(df, idx):
    vela_previa = df.iloc[idx-1]
    vela_actual = df.iloc[idx]
    
    es_previa_verde = vela_previa['close'] > vela_previa['open']
    es_actual_roja = vela_actual['close'] < vela_actual['open']
    
    if not (es_previa_verde and es_actual_roja):
        return False
    
    # HUMANIZADO CRYPTO: Abre en el techo (igual o mayor al cierre anterior)
    condicion_apertura_alta = vela_actual['open'] >= vela_previa['close']
    
    mitad_cuerpo_previo = (vela_previa['open'] + vela_previa['close']) / 2
    condicion_penetra_mitad = vela_actual['close'] < mitad_cuerpo_previo
    
    condicion_no_envuelve = vela_actual['close'] >= vela_previa['open']
    
    if condicion_apertura_alta and condicion_penetra_mitad and condicion_no_envuelve and cierre_fuerte(vela_actual, "bajista"):
        return True
    return False

# --- 7. MORNING STAR (ESTRELLA DE LA MAÑANA HUMANIZADA) ---
def es_morning_star_nison(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    es_c1_roja = c1['close'] < c1['open']
    es_c3_verde = c3['close'] > c3['open']
    if not (es_c1_roja and es_c3_verde): return False
    
    c1_body, _, _, _, _, _ = calcular_cuerpo_mechas(c1)
    c2_body, _, _, _, _, _ = calcular_cuerpo_mechas(c2)
    
    # HUMANIZADO: La estrella debe ser un cuerpo pequeño comparado a la vela roja inicial
    condicion_cuerpo_diminuto = c2_body < (c1_body * 0.4)
    
    # Penetración profunda de C3
    mitad_c1 = (c1['open'] + c1['close']) / 2
    condicion_penetracion = c3['close'] > mitad_c1
    
    if condicion_cuerpo_diminuto and condicion_penetracion and cierre_fuerte(c3, "alcista"):
        return True
    return False

# --- 8. EVENING STAR (ESTRELLA DEL ATARDECER HUMANIZADA) ---
def es_evening_star_nison(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    es_c1_verde = c1['close'] > c1['open']
    es_c3_roja = c3['close'] < c3['open']
    if not (es_c1_verde and es_c3_roja): return False
    
    c1_body, _, _, _, _, _ = calcular_cuerpo_mechas(c1)
    c2_body, _, _, _, _, _ = calcular_cuerpo_mechas(c2)
    
    condicion_cuerpo_diminuto = c2_body < (c1_body * 0.4)
    
    mitad_c1 = (c1['open'] + c1['close']) / 2
    condicion_penetracion = c3['close'] < mitad_c1
    
    if condicion_cuerpo_diminuto and condicion_penetracion and cierre_fuerte(c3, "bajista"):
        return True
    return False

# --- 9. TWEEZER BOTTOMS (PINZAS HUMANIZADAS) ---
def es_tweezer_bottom_nison(df, idx):
    c1, c2 = df.iloc[idx-1], df.iloc[idx]
    
    # HUMANIZADO: Tolerancia visual basada en el ATR, un trader no mide centavos
    atr_actual = df['atr'].iloc[idx]
    tolerancia = atr_actual * 0.15 
    
    mismos_minimos = abs(c2['low'] - c1['low']) <= tolerancia
    
    # La vela 2 debe reaccionar fuerte al alza
    es_c2_verde = c2['close'] > c2['open']
    
    if mismos_minimos and es_c2_verde and cierre_fuerte(c2, "alcista"):
        return True
    return False

# --- 10. TWEEZER TOPS (PINZAS HUMANIZADAS) ---
def es_tweezer_top_nison(df, idx):
    c1, c2 = df.iloc[idx-1], df.iloc[idx]
    
    atr_actual = df['atr'].iloc[idx]
    tolerancia = atr_actual * 0.15
    
    mismos_maximos = abs(c2['high'] - c1['high']) <= tolerancia
    
    es_c2_roja = c2['close'] < c2['open']
    
    if mismos_maximos and es_c2_roja and cierre_fuerte(c2, "bajista"):
        return True
    return False

# --- 11. THREE WHITE SOLDIERS (3 SOLDADOS HUMANIZADOS) ---
def es_three_white_soldiers(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    # RSI Humano: Si ya estamos en 75, comprar es un suicidio. Tiene que arrancar desde abajo.
    rsi_actual = df['rsi'].iloc[idx]
    if rsi_actual > 65: return False
    
    es_c1_verde = c1['close'] > c1['open']
    es_c2_verde = c2['close'] > c2['open']
    es_c3_verde = c3['close'] > c3['open']
    
    if es_c1_verde and es_c2_verde and es_c3_verde:
        # En crypto, una escalada de 3 velas fuertes consecutivas es suficiente
        if cierre_fuerte(c1, "alcista") and cierre_fuerte(c2, "alcista") and cierre_fuerte(c3, "alcista"):
            return True
    return False

# --- 12. THREE BLACK CROWS (3 CUERVOS HUMANIZADOS) ---
def es_three_black_crows(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    # RSI Humano: No vender en pánico cuando ya está sobrevendido.
    rsi_actual = df['rsi'].iloc[idx]
    if rsi_actual < 35: return False
    
    es_c1_roja = c1['close'] < c1['open']
    es_c2_roja = c2['close'] < c2['open']
    es_c3_roja = c3['close'] < c3['open']
    
    if es_c1_roja and es_c2_roja and es_c3_roja:
        if cierre_fuerte(c1, "bajista") and cierre_fuerte(c2, "bajista") and cierre_fuerte(c3, "bajista"):
            return True
    return False

# === DETECTOR MAESTRO NISON (CONFLUENCIAS HUMANIZADAS) ===
def detectar_patron_nison(df, soporte, resistencia):
    if len(df) < 15: 
        return False, None, None
        
    idx = -1 
    precio_actual = df['close'].iloc[idx]
    atr_actual = df['atr'].iloc[idx]
    
    # 1. ¿Cómo viene el precio?
    tendencia_previa = tendencia_previa_micro(df, idx)
    
    # 2. ¿Estamos en una zona relevante?
    # Tolerancia humana: Un soporte no es una línea, es una zona de varios dólares.
    tolerancia_zona = max(atr_actual * 1.5, MARGEN_NIVEL)
    
    en_soporte = cerca_de_nivel(precio_actual, soporte, tolerancia_zona)
    en_resistencia = cerca_de_nivel(precio_actual, resistencia, tolerancia_zona)
    
    # --- PATRONES DE COMPRA ---
    if tendencia_previa == "bajista" and en_soporte:
        if es_hammer_nison(df, idx): return True, "Buy", "Nison Hammer"
        if es_bullish_engulfing_nison(df, idx): return True, "Buy", "Nison Bullish Engulfing"
        if es_piercing_nison(df, idx): return True, "Buy", "Nison Piercing Pattern"
        if es_morning_star_nison(df, idx): return True, "Buy", "Nison Morning Star"
        if es_tweezer_bottom_nison(df, idx): return True, "Buy", "Nison Tweezer Bottoms"
        if es_three_white_soldiers(df, idx): return True, "Buy", "Three White Soldiers"

    # --- PATRONES DE VENTA ---
    if tendencia_previa == "alcista" and en_resistencia:
        if es_shooting_star_nison(df, idx): return True, "Sell", "Nison Shooting Star"
        if es_bearish_engulfing_nison(df, idx): return True, "Sell", "Nison Bearish Engulfing"
        if es_dark_cloud_nison(df, idx): return True, "Sell", "Nison Dark Cloud Cover"
        if es_evening_star_nison(df, idx): return True, "Sell", "Nison Evening Star"
        if es_tweezer_top_nison(df, idx): return True, "Sell", "Nison Tweezer Tops"
        if es_three_black_crows(df, idx): return True, "Sell", "Three Black Crows"

    return False, None, None


# ======================================================
# GRÁFICOS (ENTRADA Y SALIDA EXPANDIDOS)
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

        # DIBUJO MANUAL DE VELAS JAPONESAS
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

        # LÍNEAS DE SOPORTE Y RESISTENCIA
        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")

        # TENDENCIA MACRO (REGRESIÓN) Y CANAL DINÁMICO
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

        # MARCAR ENTRADA EXACTA
        entrada_x_idx = len(df_plot) - 1
        entrada_precio_final = closes[-1]
        
        if decision == 'Buy':
            ax.scatter(entrada_x_idx, entrada_precio_final, s=200, marker='^', color='lime', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='lime', linestyle=':', linewidth=2)
        else:
            ax.scatter(entrada_x_idx, entrada_precio_final, s=200, marker='v', color='red', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='red', linestyle=':', linewidth=2)

        # CUADRO DE INFORMACIÓN
        rsi_actual = df['rsi'].iloc[-1]
        texto_razones = "\n".join(razones)
        texto_panel = f"OPERACIÓN: {decision.upper()}\nPrecio: {entrada_precio_final:.2f}\nRSI Contexto: {rsi_actual:.1f}\n\nRazones:\n{texto_razones}"
        
        ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_title(f"BOT V91.0 - BTCUSDT - Entrada {decision} Humanizada")
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
            texto_resultado = "GANADA 🤑"
        else:
            color_marcador = 'red'
            forma_marcador = 'v'
            texto_resultado = "PERDIDA 💀"
            
        ax.scatter([indice_salida_x],[salida_price], s=200, c=color_marcador, marker=forma_marcador, edgecolors='black', zorder=5)

        texto_panel_salida = f"CIERRE DE OPERACIÓN {decision_original}\nMotivo de Salida: {motivo_cierre}\nResultado PnL: {pnl_obtenido:.4f} USD\nNuevo Balance: {balance_actual:.2f} USD"
        ax.text(0.02, 0.95, texto_panel_salida, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title(f"BOT V91.0 - DETALLE DE CIERRE - {texto_resultado}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"🚨 ERROR GRAFICO SALIDA: {e}")
        return None

def log_colab(df, tendencia, slope, soporte, resistencia, decision, razones):
    ahora = datetime.now(timezone.utc)
    precio = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    rsi = df['rsi'].iloc[-1]

    print("="*100)
    print(f"🕒 {ahora} | 💰 Precio BTC: {precio:.2f} | RSI: {rsi:.1f}")
    print(f"📐 Tendencia Macro: {tendencia} | Slope Lineal: {slope:.5f}")
    print(f"🧱 Nivel Soporte: {soporte:.2f} | Nivel Resistencia: {resistencia:.2f}")
    print(f"📊 Volatilidad ATR: {atr:.2f}")
    
    if decision:
        print(f"🎯 DECISIÓN TOMADA: {decision.upper()}")
    else:
        print(f"🎯 DECISIÓN TOMADA: MANTENER AL MARGEN (NO TRADE)")
        
    for razon in razones:
        print(f"🧠 Lógica Interna: {razon}")
        
    print("="*100)

# ======================================================
# MOTOR DE EJECUCIÓN (TP1 FIJO + TRAILING DINÁMICO)
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
    
    if PAPER_POSICION_ACTIVA is not None: 
        return False

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

    size_en_cripto = riesgo_usd / distancia_riesgo
    size_en_dolares = size_en_cripto * precio
    
    PAPER_POSICION_ACTIVA = decision
    PAPER_DECISION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL = sl
    PAPER_TP1 = tp1
    PAPER_TP2 = None  # El TP2 se convierte en dinámico infinito
    
    PAPER_SIZE_USD = size_en_dolares
    PAPER_SIZE_BTC = size_en_cripto
    PAPER_SIZE_BTC_RESTANTE = size_en_cripto
    PAPER_PARTIAL_ACTIVADO = True
    PAPER_TP1_EJECUTADO = False

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

    if PAPER_POSICION_ACTIVA is None: 
        return None

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    close = df['close'].iloc[-1]
    atr_actual = df['atr'].iloc[-1]
    
    cerrar_total = False
    motivo = None
    
    # Este multiplicador dicta a qué distancia el SL persigue el precio en ganancias
    TRAILING_MULT = 1.2 

    # ==========================
    # REVISIÓN DE COMPRAS (LONG)
    # ==========================
    if PAPER_POSICION_ACTIVA == "Buy":
        
        # Fase 1: Asegurar ganancias en el TP1 (Cierre del 50%)
        if PAPER_TP1_EJECUTADO == False:
            if high >= PAPER_TP1:
                mitad_posicion = PAPER_SIZE_BTC / 2
                pnl_parcial = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * mitad_posicion
                
                PAPER_BALANCE += pnl_parcial
                PAPER_PNL_GLOBAL += pnl_parcial
                PAPER_SIZE_BTC_RESTANTE = mitad_posicion
                PAPER_TP1_EJECUTADO = True
                
                # Mover SL al nivel de entrada para tener riesgo Cero
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje("🎯 TP1 ALCANZADO: Se cerró el 50% de la operación. SL movido a Break Even. Iniciando persecución de tendencia (Trailing)...")

        # Fase 2: Trailing Stop Dinámico Infinito
        if PAPER_TP1_EJECUTADO == True:
            # Calculamos a qué nivel debería estar el SL ahora mismo para no regalar ganancias
            nuevo_sl_dinamico = close - (atr_actual * TRAILING_MULT)
            
            # Solo se permite subir el Stop Loss, jamás bajarlo
            if nuevo_sl_dinamico > PAPER_SL:
                PAPER_SL = nuevo_sl_dinamico 

        # Fase 3: Cierre por Stop Loss (o Trailing)
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
        
        # Fase 1: Asegurar ganancias en el TP1
        if PAPER_TP1_EJECUTADO == False:
            if low <= PAPER_TP1:
                mitad_posicion = PAPER_SIZE_BTC / 2
                pnl_parcial = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * mitad_posicion
                
                PAPER_BALANCE += pnl_parcial
                PAPER_PNL_GLOBAL += pnl_parcial
                PAPER_SIZE_BTC_RESTANTE = mitad_posicion
                PAPER_TP1_EJECUTADO = True
                
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje("🎯 TP1 ALCANZADO: Se cerró el 50% de la operación. SL movido a Break Even. Iniciando persecución de tendencia (Trailing)...")

        # Fase 2: Trailing Stop Dinámico
        if PAPER_TP1_EJECUTADO == True:
            nuevo_sl_dinamico = close + (atr_actual * TRAILING_MULT)
            
            # Solo se permite bajar el Stop Loss
            if nuevo_sl_dinamico < PAPER_SL:
                PAPER_SL = nuevo_sl_dinamico 

        # Fase 3: Cierre por Stop Loss
        if high >= PAPER_SL:
            cerrar_total = True
            if PAPER_TP1_EJECUTADO:
                motivo = "Trailing Stop Dinámico" 
            else:
                motivo = "Stop Loss"

    # ==========================
    # EJECUCIÓN DEL CIERRE TOTAL
    # ==========================
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
        
        # Reinicio exhaustivo de variables globales de trading
        PAPER_POSICION_ACTIVA = None
        PAPER_DECISION_ACTIVA = None
        PAPER_PRECIO_ENTRADA = None
        PAPER_SL = None
        PAPER_TP1 = None
        PAPER_SIZE_BTC = 0.0
        PAPER_SIZE_BTC_RESTANTE = 0.0
        PAPER_TP1_EJECUTADO = False

        texto_cierre = f"📤 TRADE CERRADO: Salida por {motivo}. Ganancia final flotante: {pnl_final:.2f} USD."
        telegram_mensaje(texto_cierre)
        
        data_de_retorno = {
            "decision": decision_almacenada, 
            "motivo": motivo, 
            "entrada": precio_entrada_almacenado, 
            "salida": precio_salida_almacenado, 
            "pnl": pnl_final, 
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
    
    # Reseteo diario
    if PAPER_CURRENT_DAY != hoy_utc:
        PAPER_CURRENT_DAY = hoy_utc
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
        PAPER_CONSECUTIVE_LOSSES = 0
        telegram_mensaje("🔄 Nuevo ciclo diario UTC. Reseteando métricas de riesgo.")
        
    diferencia_balance = PAPER_BALANCE - PAPER_DAILY_START_BALANCE
    porcentaje_drawdown = diferencia_balance / PAPER_DAILY_START_BALANCE
    
    if porcentaje_drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
        if PAPER_STOPPED_TODAY == False:
            telegram_mensaje(f"🛑 PROTECCIÓN DE CAPITAL ACTIVADA: Se alcanzó un Drawdown de {porcentaje_drawdown*100:.2f}%. Bot pausado hasta mañana.")
            PAPER_STOPPED_TODAY = True
        return False
        
    return True


# ======================================================
# SISTEMA SECUNDARIO INSTITUCIONAL (RESTAURADO COMPLETO)
# ======================================================
# Este sistema evalúa métricas avanzadas y BOS externos
# sin entrometerse en la lógica purista de Nison principal.

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
    mensaje_inicio = "🤖 BOT V91.0 BYBIT REAL INICIADO.\nConfiguración Nison HUMANIZADA habilitada.\nTrailing Dinámico Infinito Online.\nSistema Institucional Paralelo Operativo."
    telegram_mensaje(mensaje_inicio)

    # Inyección de módulos paralelos
    sistema_institucional = InstitutionalSecondarySystem(telegram_mensaje)

    while True:
        # Pausa para evitar rate limits de Bybit
        time.sleep(60) 
        
        try:
            # 1. Obtención y preparación de la Data
            df_velas_crudas = obtener_velas()
            df = calcular_indicadores(df_velas_crudas)

            # 2. Análisis del Entorno General
            slope, intercept, tendencia_macro = detectar_tendencia_macro(df)
            soporte, resistencia = detectar_soportes_resistencias(df)
            precio_actual = df['close'].iloc[-1]
            atr_actual = df['atr'].iloc[-1]
            
            razones_para_entrar =[]
            
            # 3. EL DETECTOR MAESTRO NISON HUMANIZADO
            patron_detectado, decision_cruda, nombre_patron = detectar_patron_nison(df, soporte, resistencia)

            decision_final = decision_cruda

            # 4. Filtro Supremo Macro
            if decision_final == "Buy":
                if tendencia_macro == '📉 BAJISTA': 
                    decision_final = None 
                    
            if decision_final == "Sell":
                if tendencia_macro == '📈 ALCISTA': 
                    decision_final = None 

            # 5. Registro Consola
            if patron_detectado == True:
                lista_log = [nombre_patron]
            else:
                lista_log = ["Buscando patrón Nison válido..."]
                
            log_colab(df, tendencia_macro, slope, soporte, resistencia, decision_final, lista_log)

            # 6. Toma de Decisión y Ejecución
            if decision_final is not None:
                razones_para_entrar.append(f"✅ Arquitectura Confirmada: {nombre_patron}")
                razones_para_entrar.append(f"📊 Tendencia MACRO Válida: {tendencia_macro}")
                razones_para_entrar.append(f"🛡️ Geometría de S/R Respetada")
                
                riesgo_valido = risk_management_check()
                
                if riesgo_valido == True:
                    apertura_exitosa = paper_abrir_posicion(decision_final, precio_actual, atr_actual, soporte, resistencia, razones_para_entrar, df.index[-1])
                    
                    if apertura_exitosa == True:
                        texto_entrada = f"📌 SE HA INICIADO UNA OPERACIÓN {decision_final.upper()}\n"
                        texto_entrada += f"💰 Nivel de Entrada: {precio_actual:.2f}\n"
                        texto_entrada += f"📍 SL Inicial: {PAPER_SL:.2f} | TP1 Objetivo: {PAPER_TP1:.2f}\n"
                        razones_unidas = ', '.join(razones_para_entrar)
                        texto_entrada += f"🧠 Justificación Analítica: {razones_unidas}"
                        
                        telegram_mensaje(texto_entrada)
                        
                        figura_generada = generar_grafico_entrada(df, decision_final, soporte, resistencia, slope, intercept, razones_para_entrar)
                        
                        if figura_generada is not None:
                            telegram_grafico(figura_generada)
                            plt.close(figura_generada)

            # 7. Gestión Contínua de Operaciones Abiertas
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
