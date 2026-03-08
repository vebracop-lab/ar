# BOT TRADING V90.5 BYBIT REAL – PRODUCCIÓN (SIN PROXY) + ARSENAL NISON COMPLETO
# ======================================================
# ⚠️ KEYS INCLUIDAS TAL CUAL (SEGÚN PEDIDO)
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit
# ======================================================
# NOVEDADES V90.5:
# - Agregados patrones de 3 velas: Morning Star y Evening Star.
# - Agregados patrones de precisión: Tweezer Tops y Bottoms.
# - Mantenimiento de contexto estricto para los 10 patrones.
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
# CONFIGURACIÓN GRÁFICOS
# ======================================================

GRAFICO_VELAS_LIMIT = 120  # cantidad de velas para graficar
MOSTRAR_EMA20 = True
MOSTRAR_ATR = False


# ======================================================
# ======================================================
# CONFIGURACIÓN GENERAL
# ===============================
# MARGEN PROXIMIDAD NIVELES
# ===============================
MARGEN_NIVEL = 80  # puntos de precio BTC

def cerca_de_nivel(precio, nivel, margen=MARGEN_NIVEL):
    return abs(precio - nivel) <= margen

# ======================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1"  # 1 minuto
RISK_PER_TRADE = 0.0025
LEVERAGE = 1
SLEEP_SECONDS = 60  # 1 minuto

# ======================================================
# PAPER TRADING (SIMULACIÓN)
# ======================================================

PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_PNL_GLOBAL = 0.0
PAPER_TRADES = []
PAPER_POSICION_ACTIVA = None
PAPER_PRECIO_ENTRADA = None
PAPER_DECISION_ACTIVA = None
PAPER_TIME_ENTRADA = None
PAPER_SIZE_USD = 0.0
PAPER_SIZE_BTC = 0.0
PAPER_SL = None
PAPER_TP = None
PAPER_ULTIMO_RESULTADO = None
PAPER_ULTIMO_PNL = 0.0
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
PAPER_MAX_DRAWDOWN = 0.0
PAPER_BALANCE_MAX = PAPER_BALANCE_INICIAL

# ======================================================
# EXTENSIÓN INTRABAR + GESTIÓN PARCIAL 50/50 (INTEGRADA)
# ======================================================

PAPER_TP1 = None
PAPER_TP2 = None
PAPER_PARTIAL_ACTIVADO = False
PAPER_SIZE_BTC_RESTANTE = 0.0
PAPER_TP1_EJECUTADO = False


# ======================================================
# CONTROL DINÁMICO DE RIESGO AVANZADO (SIN LÍMITE)
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
# CREDENCIALES (SIN MODIFICAR)
# ======================================================

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise Exception("❌ BYBIT_API_KEY o BYBIT_API_SECRET no configuradas")

# ======================================================
# BYBIT ENDPOINT
# ======================================================

BASE_URL = "https://api.bybit.com"

# ======================================================
# TELEGRAM (SIN PROXY)
# ======================================================

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": texto},
            timeout=10
        )
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
        requests.post(
            url,
            files={'photo': buf},
            data={'chat_id': TELEGRAM_CHAT_ID},
            timeout=15
        )
        buf.close()
    except Exception:
        pass

# ======================================================
# FIRMA BYBIT
# ======================================================

def sign(params):
    query = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(
        BYBIT_API_SECRET.encode(), query.encode(), hashlib.sha256
    ).hexdigest()

# ======================================================
# OBTENER VELAS BYBIT (SIN PROXY)
# ======================================================

def obtener_velas(limit=300):
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": limit
    }

    r = requests.get(
        url,
        params=params,
        timeout=20
    )

    if not r.text:
        raise Exception("Respuesta vacía de Bybit")

    try:
        data_json = r.json()
    except Exception:
        raise Exception(f"Bybit devolvió respuesta no-JSON: {r.text}")

    # ======================================================
    # VALIDACIONES FUERTES (ANTI ERROR 'list')
    # ======================================================

    if not isinstance(data_json, dict):
        raise Exception(f"Bybit devolvió JSON no dict: {type(data_json)} | {data_json}")

    if "retCode" in data_json and data_json["retCode"] != 0:
        raise Exception(
            f"Bybit Error retCode={data_json.get('retCode')} "
            f"retMsg={data_json.get('retMsg')} "
            f"result={data_json.get('result')}"
        )

    if "result" not in data_json:
        raise Exception(f"Respuesta inválida Bybit (sin result): {data_json}")

    if not isinstance(data_json["result"], dict):
        raise Exception(
            f"Bybit devolvió result como {type(data_json['result'])} en vez de dict: {data_json['result']}"
        )

    if "list" not in data_json["result"]:
        raise Exception(f"Bybit result sin 'list': {data_json['result']}")

    if not isinstance(data_json["result"]["list"], list):
        raise Exception(
            f"Bybit devolvió result['list'] como {type(data_json['result']['list'])} en vez de list: {data_json['result']['list']}"
        )

    data = data_json["result"]["list"][::-1]

    if len(data) == 0:
        raise Exception(f"Bybit devolvió lista vacía de velas: {data_json}")

    df = pd.DataFrame(data, columns=[
        'time','open','high','low','close','volume','turnover'
    ])

    df[['open','high','low','close','volume']] = df[[
        'open','high','low','close','volume'
    ]].astype(float)

    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)

    df.set_index('time', inplace=True)
    return df

# ======================================================
# INDICADORES
# ======================================================

def calcular_indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)

    df['atr'] = tr.rolling(14).mean()
    return df.dropna()

# ======================================================
# SOPORTE / RESISTENCIA
# ======================================================

def detectar_soportes_resistencias(df):
    soporte = df['close'].rolling(200).min().iloc[-1]
    resistencia = df['close'].rolling(200).max().iloc[-1]
    return soporte, resistencia

# ======================================================
# TENDENCIA
# ======================================================

def detectar_tendencia(df, ventana=120):
    y = df['close'].values[-ventana:]
    x = np.arange(len(y))
    slope, intercept, r, _, _ = linregress(x, y)

    if slope > 0.01:
        direccion = '📈 ALCISTA'
    elif slope < -0.01:
        direccion = '📉 BAJISTA'
    else:
        direccion = '➡️ LATERAL'

    return slope, intercept, direccion

# ======================================================
# MOTOR V90
# ======================================================

def motor_v90(df):
    soporte, resistencia = detectar_soportes_resistencias(df)
    slope, intercept, tendencia = detectar_tendencia(df)
    precio = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    razones = []

    if tendencia == '📈 ALCISTA' and cerca_de_nivel(precio, soporte) < atr:
        razones.append('Confluencia: soporte + tendencia alcista')
        return 'Buy', soporte, resistencia, razones

    if tendencia == '📉 BAJISTA' and cerca_de_nivel(precio, resistencia) < atr:
        razones.append('Confluencia: resistencia + tendencia bajista')
        return 'Sell', soporte, resistencia, razones

    razones.append('Sin confluencia válida')
    return None, soporte, resistencia, razones


# ======================================================
# 🕯️ ARSENAL DE PATRONES NISON V90.5
# Se incluyen: Hammer, Shooting Star, Engulfing, Piercing/Cloud,
# Morning/Evening Star, Tweezers, Harami.
# ======================================================

def calcular_cuerpo_mechas(row):
    cuerpo = abs(row['close'] - row['open'])
    alto = row['high']
    bajo = row['low']
    top = max(row['open'], row['close'])
    bottom = min(row['open'], row['close'])
    
    mecha_sup = alto - top
    mecha_inf = bottom - bajo
    
    return cuerpo, mecha_sup, mecha_inf

def tendencia_previa_nison(df, velas=8):
    """
    Nison: 'Un patrón de reversión NO existe sin una tendencia previa'.
    """
    if len(df) < velas + 1:
        return "neutral"
    
    reciente = df.iloc[-velas:-1]
    
    inicio = reciente['close'].iloc[0]
    fin = reciente['close'].iloc[-1]
    
    y = reciente['close'].values
    x = np.arange(len(y))
    slope, _, _, _, _ = linregress(x, y)
    
    atr = df['atr'].iloc[-1]
    
    if fin < inicio - (atr * 0.5) and slope < 0:
        return "bajista"
    elif fin > inicio + (atr * 0.5) and slope > 0:
        return "alcista"
    
    return "lateral"

# --- 1. HAMMER (MARTILLO) ---
def es_hammer_nison(df, idx):
    row = df.iloc[idx]
    cuerpo, m_sup, m_inf = calcular_cuerpo_mechas(row)
    if cuerpo == 0: cuerpo = 0.0001
    
    return (m_inf / cuerpo >= 2.0) and (m_sup / cuerpo <= 0.2)

# --- 2. SHOOTING STAR (ESTRELLA FUGAZ) ---
def es_shooting_star_nison(df, idx):
    row = df.iloc[idx]
    cuerpo, m_sup, m_inf = calcular_cuerpo_mechas(row)
    if cuerpo == 0: cuerpo = 0.0001
    
    return (m_sup / cuerpo >= 2.0) and (m_inf / cuerpo <= 0.2)

# --- 3. BULLISH ENGULFING (ENVOLVENTE ALCISTA) ---
def es_bullish_engulfing_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if not (prev['close'] < prev['open'] and curr['close'] > curr['open']):
        return False
    
    return (curr['close'] > prev['open']) and (curr['open'] < prev['close'])

# --- 4. BEARISH ENGULFING (ENVOLVENTE BAJISTA) ---
def es_bearish_engulfing_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if not (prev['close'] > prev['open'] and curr['close'] < curr['open']):
        return False
    
    return (curr['open'] > prev['close']) and (curr['close'] < prev['open'])

# --- 5. PIERCING PATTERN (PAUTA PENETRANTE) ---
def es_piercing_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if not (prev['close'] < prev['open'] and curr['close'] > curr['open']):
        return False
        
    midpoint_prev = (prev['open'] + prev['close']) / 2
    return (curr['close'] > midpoint_prev) and (curr['open'] <= prev['close'])

# --- 6. DARK CLOUD COVER (CUBIERTA DE NUBE OSCURA) ---
def es_dark_cloud_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if not (prev['close'] > prev['open'] and curr['close'] < curr['open']):
        return False
    
    midpoint_prev = (prev['open'] + prev['close']) / 2
    return (curr['close'] < midpoint_prev) and (curr['open'] >= prev['close'])

# --- 7. MORNING STAR (ESTRELLA DE LA MAÑANA) - 3 VELAS ---
# Contexto: Bajista.
# 1: Roja larga. 2: Pequeña (indecisión). 3: Verde fuerte que penetra en la 1.
def es_morning_star_nison(df, idx):
    if idx < 2: return False
    c1 = df.iloc[idx-2] # Roja
    c2 = df.iloc[idx-1] # Estrella
    c3 = df.iloc[idx]   # Verde
    
    atr = df['atr'].iloc[idx]
    
    # Colores
    if not (c1['close'] < c1['open']): return False # C1 Roja
    if not (c3['close'] > c3['open']): return False # C3 Verde
    
    # Tamaños
    c1_body = abs(c1['close'] - c1['open'])
    c2_body = abs(c2['close'] - c2['open'])
    c3_body = abs(c3['close'] - c3['open'])
    
    # Nison: C1 debe ser larga, C2 pequeña (estrella)
    if c1_body < atr * 0.5: return False
    if c2_body > c1_body * 0.4: return False # C2 debe ser pequeña
    
    # Penetración: C3 cierra por encima del 50% de C1
    midpoint_c1 = (c1['open'] + c1['close']) / 2
    if c3['close'] < midpoint_c1: return False
    
    return True

# --- 8. EVENING STAR (ESTRELLA DEL ATARDECER) - 3 VELAS ---
# Contexto: Alcista.
# 1: Verde larga. 2: Pequeña (estrella). 3: Roja fuerte que penetra en la 1.
def es_evening_star_nison(df, idx):
    if idx < 2: return False
    c1 = df.iloc[idx-2] # Verde
    c2 = df.iloc[idx-1] # Estrella
    c3 = df.iloc[idx]   # Roja
    
    atr = df['atr'].iloc[idx]
    
    if not (c1['close'] > c1['open']): return False # C1 Verde
    if not (c3['close'] < c3['open']): return False # C3 Roja
    
    c1_body = abs(c1['close'] - c1['open'])
    c2_body = abs(c2['close'] - c2['open'])
    
    if c1_body < atr * 0.5: return False
    if c2_body > c1_body * 0.4: return False
    
    midpoint_c1 = (c1['open'] + c1['close']) / 2
    if c3['close'] > midpoint_c1: return False
    
    return True

# --- 9. TWEEZER BOTTOMS (PINZAS DE SUELO) ---
# Contexto: Bajista. Dos velas con el mismo mínimo (o muy cerca).
def es_tweezer_bottom_nison(df, idx):
    curr = df.iloc[idx]
    prev = df.iloc[idx-1]
    
    # Margen de tolerancia para "mismo precio"
    tolerancia = df['atr'].iloc[idx] * 0.05
    
    match_low = abs(curr['low'] - prev['low']) < tolerancia
    
    # Confirmación: la vela actual debe cerrar alcista o rechazar fuerte
    rechazo = (curr['close'] > curr['open']) or (es_hammer_nison(df, idx))
    
    return match_low and rechazo

# --- 10. TWEEZER TOPS (PINZAS DE TECHO) ---
# Contexto: Alcista. Dos velas con el mismo máximo.
def es_tweezer_top_nison(df, idx):
    curr = df.iloc[idx]
    prev = df.iloc[idx-1]
    
    tolerancia = df['atr'].iloc[idx] * 0.05
    
    match_high = abs(curr['high'] - prev['high']) < tolerancia
    
    rechazo = (curr['close'] < curr['open']) or (es_shooting_star_nison(df, idx))
    
    return match_high and rechazo

# === MASTER PATTERN DETECTOR ===
def detectar_patron_nison(df, soporte, resistencia, tendencia_global):
    if len(df) < 10:
        return False, None

    idx = -1 
    precio_actual = df['close'].iloc[idx]
    atr = df['atr'].iloc[idx]
    
    # 1. Definir Tendencia Previa Inmediata
    t_prev = tendencia_previa_nison(df, velas=7)
    
    # 2. Definir Zonas
    en_soporte = cerca_de_nivel(precio_actual, soporte, margen=atr*0.8)
    en_resistencia = cerca_de_nivel(precio_actual, resistencia, margen=atr*0.8)
    
    # --- PATRONES ALCISTAS (REQUIEREN SOPORTE + TENDENCIA BAJISTA PREVIA) ---
    if t_prev == "bajista" and en_soporte:
        if es_hammer_nison(df, idx):
            return True, "Nison Hammer"
        if es_bullish_engulfing_nison(df, idx):
            return True, "Nison Bullish Engulfing"
        if es_piercing_nison(df, idx):
            return True, "Nison Piercing Pattern"
        if es_morning_star_nison(df, idx):
            return True, "Nison Morning Star (3 Velas)"
        if es_tweezer_bottom_nison(df, idx):
            return True, "Nison Tweezer Bottoms"

    # --- PATRONES BAJISTAS (REQUIEREN RESISTENCIA + TENDENCIA ALCISTA PREVIA) ---
    if t_prev == "alcista" and en_resistencia:
        if es_shooting_star_nison(df, idx):
            return True, "Nison Shooting Star"
        if es_bearish_engulfing_nison(df, idx):
            return True, "Nison Bearish Engulfing"
        if es_dark_cloud_nison(df, idx):
            return True, "Nison Dark Cloud Cover"
        if es_evening_star_nison(df, idx):
            return True, "Nison Evening Star (3 Velas)"
        if es_tweezer_top_nison(df, idx):
            return True, "Nison Tweezer Tops"

    return False, None


# ======================================================
# FILTRO MAESTRO NISON - FASE 1 (ARQUITECTURA BASE)
# ======================================================

def filtro_maestro_nison(
    patron_detectado,
    zona_valida,
    tendencia_valida,
    estructura_valida
):
    """
    Entrada permitida SOLO si se cumplen:

    Patrón + Zona + Tendencia + Estructura (BOS)
    """

    if patron_detectado and zona_valida and tendencia_valida and estructura_valida:
        return True

    return False

# ======================================================
# GRÁFICO VELAS JAPONESAS + SOPORTE/RESISTENCIA + TENDENCIA
# ======================================================

def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones):
    """
    Genera gráfico de velas japonesas con:
    - Soporte (línea horizontal)
    - Resistencia (línea horizontal)
    - Línea de tendencia inclinada (según slope)
    - EMA20 (opcional)
    - Marcador exacto en la vela de entrada
    """

    try:
        df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)

        if df_plot.empty:
            return None

        # ======================================================
        # DATOS
        # ======================================================

        times = df_plot.index
        opens = df_plot['open'].values
        highs = df_plot['high'].values
        lows = df_plot['low'].values
        closes = df_plot['close'].values

        x = np.arange(len(df_plot))

        # ======================================================
        # CREAR FIGURA
        # ======================================================

        fig, ax = plt.subplots(figsize=(14, 7))

        # ======================================================
        # VELAS JAPONESAS (MATPLOTLIB PURO)
        # ======================================================

        for i in range(len(df_plot)):
            color = 'green' if closes[i] >= opens[i] else 'red'

            # Mecha
            ax.vlines(x[i], lows[i], highs[i], color=color, linewidth=1)

            # Cuerpo
            cuerpo_y = min(opens[i], closes[i])
            cuerpo_h = abs(closes[i] - opens[i])

            if cuerpo_h == 0:
                cuerpo_h = 0.0001

            rect = plt.Rectangle(
                (x[i] - 0.3, cuerpo_y),
                0.6,
                cuerpo_h,
                color=color,
                alpha=0.9
            )
            ax.add_patch(rect)

        # ======================================================
        # SOPORTE / RESISTENCIA (LÍNEAS HORIZONTALES)
        # ======================================================

        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")

        # ======================================================
        # EMA20
        # ======================================================

        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA20')

        # ======================================================
        # LÍNEA DE TENDENCIA INCLINADA (REGRESIÓN LINEAL)
        # ======================================================

        # Creamos tendencia solo sobre el rango graficado
        tendencia_y = intercept + slope * np.arange(len(df))

        # Ajustar para que coincida con el rango mostrado
        # intercept original es para el dataset completo, así que recalculamos sobre df_plot
        y_plot = df_plot['close'].values
        x_plot = np.arange(len(y_plot))
        slope_plot, intercept_plot, r_plot, _, _ = linregress(x_plot, y_plot)

        tendencia_linea = intercept_plot + slope_plot * x_plot

        ax.plot(x_plot, tendencia_linea, color='white', linewidth=2, linestyle='-', label=f"Tendencia slope {slope_plot:.4f}")

        # ======================================================
        # MARCAR VELA DE ENTRADA (ÚLTIMA VELA CERRADA)
        # ======================================================

        entrada_x = len(df_plot) - 1
        entrada_precio = closes[-1]
        entrada_time = times[-1]

        if decision == 'Buy':
            ax.scatter(entrada_x, entrada_precio, s=200, marker='^', color='lime', edgecolors='black', linewidths=1.5, label='Entrada BUY')
            ax.axvline(entrada_x, color='lime', linestyle=':', linewidth=2)
        elif decision == 'Sell':
            ax.scatter(entrada_x, entrada_precio, s=200, marker='v', color='red', edgecolors='black', linewidths=1.5, label='Entrada SELL')
            ax.axvline(entrada_x, color='red', linestyle=':', linewidth=2)

        # ======================================================
        # TEXTO DE ENTRADA
        # ======================================================

        texto_entrada = (
            f"{decision.upper()}\n"
            f"Precio: {entrada_precio:.2f}\n"
            f"Hora: {entrada_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Razones: {', '.join(razones)}"
        )

        ax.text(
            0.02,
            0.98,
            texto_entrada,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='black', alpha=0.6)
        )

        # ======================================================
        # FORMATO
        # ======================================================

        ax.set_title(f"BTCUSDT - Velas Japonesas ({INTERVAL}m) - Entrada {decision}")
        ax.set_xlabel("Velas")
        ax.set_ylabel("Precio")

        ax.grid(True, alpha=0.2)

        # Etiquetas de tiempo (cada 10 velas)
        step = max(1, int(len(df_plot) / 10))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([t.strftime('%H:%M') for t in times[::step]], rotation=45)

        ax.legend(loc='lower left')

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"🚨 ERROR GRAFICO: {e}")
        return None

# ======================================================
# LOG
# ======================================================

def log_colab(df, tendencia, slope, soporte, resistencia, decision, razones):
    ahora = datetime.now(timezone.utc)
    precio = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    print("="*100)
    print("🧠 Groq Analyst:", "ACTIVO" if client_groq else "DESACTIVADO")
    print(f"🕒 {ahora} | 💰 BTC: {precio:.2f}")
    print(f"📐 Tendencia: {tendencia} | Slope: {slope:.5f}")
    print(f"🧱 Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}")
    print(f"📊 ATR: {atr:.2f}")
    print(f"🎯 Decisión: {decision if decision else 'NO TRADE'}")
    print(f"🧠 Razones: {', '.join(razones)}")
    print("="*100)

# ======================================================
# GROQ
# ======================================================

def analizar_con_groq(resumen):
    if not client_groq:
        return None
    prompt = f"""
Eres un trader cuantitativo profesional.
Analiza este resumen de trading y da recomendaciones claras:
{resumen}
Devuelve:
- Diagnóstico
- Qué mejorar
- Qué evitar
"""
    try:
        r = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"Error Groq: {e}"

# ======================================================
# GRÁFICO DE ENTRADA (VELAS + SOPORTE/RESISTENCIA + TENDENCIA)
# ======================================================

# NOTA:
# En tu código había una segunda función generar_grafico_entrada DUPLICADA.
# No la elimino (según tu pedido), pero la arreglo para que no rompa el código.


def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones):
    try:
        df_plot = df.copy().tail(120)

        fig, ax = plt.subplots(figsize=(14, 7))

        # Velas japonesas manuales
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            o = row['open']
            h = row['high']
            l = row['low']
            c = row['close']

            # mecha
            ax.plot([i, i], [l, h], color='black', linewidth=1)

            # cuerpo
            if c >= o:
                ax.plot([i, i], [o, c], color='green', linewidth=6)
            else:
                ax.plot([i, i], [o, c], color='red', linewidth=6)

        # soporte y resistencia horizontales
        ax.axhline(soporte, color='blue', linestyle='--', linewidth=2, label='Soporte')
        ax.axhline(resistencia, color='purple', linestyle='--', linewidth=2, label='Resistencia')

        # tendencia inclinada basada en slope/intercept
        y_trend = intercept + slope * np.arange(len(df_plot))
        ax.plot(np.arange(len(df_plot)), y_trend, color='orange', linewidth=2, label='Tendencia')

        # ======================================================
        # CANAL DE TENDENCIA DINÁMICO (SOPORTE Y RESISTENCIA INCLINADOS)
        # ======================================================
        x_vals = np.arange(len(df_plot))
        precios = df_plot['close'].values

        slope_local, intercept_local, _, _, _ = linregress(x_vals, precios)
        linea_central = intercept_local + slope_local * x_vals

        residuos = precios - linea_central
        desviacion = np.std(residuos)
        factor_canal = 1.5

        canal_superior = linea_central + (desviacion * factor_canal)
        canal_inferior = linea_central - (desviacion * factor_canal)

        ax.plot(x_vals, canal_superior, linestyle='--', linewidth=2, color='red', label='Resistencia dinámica')
        ax.plot(x_vals, canal_inferior, linestyle='--', linewidth=2, color='green', label='Soporte dinámico')


        # marcar vela de entrada (última)
        entrada_index = len(df_plot) - 1
        precio_entrada = df_plot['close'].iloc[-1]

        ax.axvline(entrada_index, color='gold', linestyle='-', linewidth=2, label='Entrada')
        ax.scatter([entrada_index], [precio_entrada], color='gold', s=150, marker='o')

        # Texto de entrada (FIX COMPLETO DEL ERROR DE STRING)
        texto_entrada = (
            f"{decision.upper()}\n"
            f"Precio: {precio_entrada:.2f}\n"
            f"Balance: {PAPER_BALANCE:.2f} USD\n"
            f"PnL Global: {PAPER_PNL_GLOBAL:.4f} USD\n"
            f"Razones: {', '.join(razones)}"
        )

        ax.text(
            0.02,
            0.98,
            texto_entrada,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
        )

        ax.set_title(f"{SYMBOL} - Entrada PAPER {decision}")
        ax.set_xlabel("Velas")
        ax.set_ylabel("Precio")
        ax.grid(True)
        ax.legend()

        return fig

    except Exception as e:
        print(f"🚨 ERROR GRÁFICO: {e}")
        return None

# ======================================================
# GRÁFICO DE SALIDA (NUEVO)
# ======================================================
def generar_grafico_salida(df, trade_data):
    """
    Genera un gráfico mostrando el cierre del trade,
    la línea de entrada y el punto de salida.
    """
    try:
        decision = trade_data['decision']  # Buy o Sell original
        entrada_price = trade_data['entrada']
        salida_price = trade_data['salida']
        pnl = trade_data['pnl']
        motivo = trade_data['motivo']
        balance = trade_data['balance']
        
        # Tomamos las últimas 120 velas para contexto (igual que entrada)
        df_plot = df.copy().tail(120)
        
        if df_plot.empty:
            return None

        fig, ax = plt.subplots(figsize=(14, 7))

        # 1. DIBUJAR VELAS JAPONESAS
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            color = 'green' if c >= o else 'red'
            
            # Mecha
            ax.plot([i, i], [l, h], color='black', linewidth=1)
            # Cuerpo
            ax.plot([i, i], [o, c], color=color, linewidth=6)

        # 2. LÍNEA DE PRECIO DE ENTRADA (Referencia)
        ax.axhline(entrada_price, color='blue', linestyle='--', linewidth=1.5, label=f'Entrada: {entrada_price:.2f}')
        
        # 3. LÍNEA DE PRECIO DE SALIDA
        ax.axhline(salida_price, color='orange', linestyle='--', linewidth=1.5, label=f'Salida: {salida_price:.2f}')

        # 4. MARCAR EL PUNTO DE SALIDA (Última vela)
        indice_salida = len(df_plot) - 1
        
        # Color del marcador según si ganamos o perdimos
        color_salida = 'lime' if pnl > 0 else 'red'
        marker_salida = '^' if pnl > 0 else 'v' 
        
        ax.scatter([indice_salida], [salida_price], s=200, c=color_salida, marker=marker_salida, edgecolors='black', zorder=5, label=f'Cierre ({motivo})')

        # 5. CUADRO DE INFORMACIÓN
        texto_info = (
            f"CIERRE {decision}\n"
            f"Motivo: {motivo}\n"
            f"PnL: {pnl:.4f} USD\n"
            f"Balance: {balance:.2f} USD"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax.text(0.02, 0.95, texto_info, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        # Decoración
        ax.set_title(f"DETALLE DE CIERRE - {decision} - {'GANADA 🤑' if pnl > 0 else 'PERDIDA 💀'}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')

        return fig

    except Exception as e:
        print(f"🚨 ERROR GRÁFICO SALIDA: {e}")
        return None

# ======================================================
# MOTOR PAPER (EJECUCIÓN SIMULADA)
# ======================================================


def paper_abrir_posicion(decision, precio, atr, soporte, resistencia, razones, tiempo):
    global PAPER_POSICION_ACTIVA
    global PAPER_PRECIO_ENTRADA
    global PAPER_SL
    global PAPER_TP
    global PAPER_TP1
    global PAPER_TP2
    global PAPER_SIZE_USD
    global PAPER_SIZE_BTC
    global PAPER_SIZE_BTC_RESTANTE
    global PAPER_TIME_ENTRADA
    global PAPER_DECISION_ACTIVA
    global PAPER_PARTIAL_ACTIVADO
    global PAPER_TP1_EJECUTADO

    if PAPER_POSICION_ACTIVA is not None:
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE

    if decision == "Buy":
        sl = precio - atr
        tp1 = precio + atr
        tp2 = precio + (atr * 2)
    elif decision == "Sell":
        sl = precio + atr
        tp1 = precio - atr
        tp2 = precio - (atr * 2)
    else:
        return False

    distancia_sl = abs(precio - sl)
    if distancia_sl == 0:
        return False

    size_btc = riesgo_usd / distancia_sl
    size_usd = size_btc * precio

    PAPER_POSICION_ACTIVA = decision
    PAPER_DECISION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL = sl
    PAPER_TP = tp2
    PAPER_TP1 = tp1
    PAPER_TP2 = tp2
    PAPER_SIZE_USD = size_usd
    PAPER_SIZE_BTC = size_btc
    PAPER_SIZE_BTC_RESTANTE = size_btc
    PAPER_TIME_ENTRADA = tiempo
    PAPER_PARTIAL_ACTIVADO = True
    PAPER_TP1_EJECUTADO = False

    return True

def paper_calcular_pnl(precio_actual):
    if PAPER_POSICION_ACTIVA is None:
        return 0.0

    if PAPER_POSICION_ACTIVA == "Buy":
        return (precio_actual - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC
    elif PAPER_POSICION_ACTIVA == "Sell":
        return (PAPER_PRECIO_ENTRADA - precio_actual) * PAPER_SIZE_BTC

    return 0.0



def paper_revisar_sl_tp(df):
    global PAPER_SL, PAPER_TP1, PAPER_TP2
    global PAPER_PRECIO_ENTRADA, PAPER_DECISION_ACTIVA
    global PAPER_POSICION_ACTIVA, PAPER_BALANCE, PAPER_PNL_GLOBAL
    global PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES
    global PAPER_BALANCE_MAX, PAPER_MAX_DRAWDOWN
    global PAPER_ULTIMO_RESULTADO, PAPER_ULTIMO_PNL
    global PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE
    global PAPER_TP1_EJECUTADO, PAPER_CONSECUTIVE_LOSSES
    global PAPER_PAUSE_UNTIL

    if PAPER_POSICION_ACTIVA is None:
        return None

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]

    cerrar_total = False
    motivo = None

    # ===== BUY =====
    if PAPER_POSICION_ACTIVA == "Buy":
        if (not PAPER_TP1_EJECUTADO) and high >= PAPER_TP1:
            pnl_parcial = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC / 2)
            PAPER_BALANCE += pnl_parcial
            PAPER_PNL_GLOBAL += pnl_parcial
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC / 2
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA
            telegram_mensaje("🎯 TP1 alcanzado - 50% cerrado y SL a BE")

        if low <= PAPER_SL:
            cerrar_total = True
            motivo = "SL"
        elif high >= PAPER_TP2:
            cerrar_total = True
            motivo = "TP2"

    # ===== SELL =====
    elif PAPER_POSICION_ACTIVA == "Sell":
        if (not PAPER_TP1_EJECUTADO) and low <= PAPER_TP1:
            pnl_parcial = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC / 2)
            PAPER_BALANCE += pnl_parcial
            PAPER_PNL_GLOBAL += pnl_parcial
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC / 2
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA
            telegram_mensaje("🎯 TP1 alcanzado - 50% cerrado y SL a BE")

        if high >= PAPER_SL:
            cerrar_total = True
            motivo = "SL"
        elif low <= PAPER_TP2:
            cerrar_total = True
            motivo = "TP2"

    # ===== CIERRE TOTAL (DENTRO DE LA FUNCIÓN) =====
    if cerrar_total:
        if PAPER_POSICION_ACTIVA == "Buy":
            precio_salida = PAPER_TP2 if motivo == "TP2" else PAPER_SL
            pnl_final = (precio_salida - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE
        else:
            precio_salida = PAPER_TP2 if motivo == "TP2" else PAPER_SL
            pnl_final = (PAPER_PRECIO_ENTRADA - precio_salida) * PAPER_SIZE_BTC_RESTANTE

        decision_guardada = PAPER_DECISION_ACTIVA
        entrada_guardada = PAPER_PRECIO_ENTRADA
        balance_final = PAPER_BALANCE + pnl_final

        PAPER_BALANCE += pnl_final
        PAPER_PNL_GLOBAL += pnl_final
        PAPER_TRADES_TOTALES += 1
        PAPER_ULTIMO_PNL = pnl_final
        PAPER_ULTIMO_RESULTADO = motivo

        # RESET
        PAPER_POSICION_ACTIVA = None
        PAPER_DECISION_ACTIVA = None
        PAPER_PRECIO_ENTRADA = None
        PAPER_SL = None
        PAPER_TP1 = None
        PAPER_TP2 = None
        PAPER_SIZE_BTC = 0.0
        PAPER_SIZE_BTC_RESTANTE = 0.0
        PAPER_TP1_EJECUTADO = False

        telegram_mensaje(f"📤 Trade cerrado por {motivo} | PnL: {pnl_final:.2f} USDT")

        return {
            "decision": decision_guardada,
            "motivo": motivo,
            "entrada": entrada_guardada,
            "salida": precio_salida,
            "pnl": pnl_final,
            "balance": balance_final
        }

    return None

# ======================================================
# FUNCIÓN CONTROL DINÁMICO DE RIESGO
# ======================================================
def risk_management_check():
    global PAPER_PAUSE_UNTIL
    global PAPER_STOPPED_TODAY
    global PAPER_DAILY_START_BALANCE
    global PAPER_CURRENT_DAY
    global PAPER_BALANCE
    global PAPER_CONSECUTIVE_LOSSES

    ahora = datetime.now(timezone.utc)
    hoy = ahora.date()

    # Reset diario automático UTC
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY = hoy
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
        PAPER_CONSECUTIVE_LOSSES = 0
        telegram_mensaje("🔄 Nuevo día UTC detectado - Sistema reactivado.")

    daily_dd_pct = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE

    if daily_dd_pct <= -MAX_DAILY_DRAWDOWN_PCT:
        if not PAPER_STOPPED_TODAY:
            telegram_mensaje(f"🛑 STOP DIARIO ACTIVADO - Drawdown {daily_dd_pct*100:.2f}%")
        PAPER_STOPPED_TODAY = True
        return False

    if PAPER_PAUSE_UNTIL and ahora < PAPER_PAUSE_UNTIL:
        return False

    return True

# ======================================================
# SISTEMA SECUNDARIO INSTITUCIONAL (NO REEMPLAZA EL SISTEMA PRINCIPAL)
# ======================================================
# Funciones añadidas:
# 1 BOS Externo (estructura mayor)
# Pullback válido estructural
# Gestión parcial real (50% TP1 / 50% TP2)
# Estadísticas avanzadas internas
# Log detallado enviado a Telegram (sin CSV)
# ======================================================

class InstitutionalStats:
    def __init__(self):
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.partial_wins = 0
        self.total_rr = 0.0
        self.equity_curve = []
        self.trade_log = []

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
        if self.total_trades == 0:
            return 0
        return (self.wins / self.total_trades) * 100

    def avg_rr(self):
        if self.total_trades == 0:
            return 0
        return self.total_rr / self.total_trades


class ExternalBOSDetector:
    def __init__(self, lookback=50):
        self.lookback = lookback
        self.last_swing_high = None
        self.last_swing_low = None

    def detect_swings(self, df):
        highs = df['high'].values
        lows = df['low'].values

        swing_high = max(highs[-self.lookback:])
        swing_low = min(lows[-self.lookback:])

        self.last_swing_high = swing_high
        self.last_swing_low = swing_low

        return swing_high, swing_low

    def is_bos_externo(self, df):
        swing_high, swing_low = self.detect_swings(df)
        last_close = df['close'].iloc[-1]

        bos_alcista = last_close > swing_high
        bos_bajista = last_close < swing_low

        return bos_alcista, bos_bajista, swing_high, swing_low


class PullbackValidator:
    def __init__(self, tolerance=0.3):
        self.tolerance = tolerance

    def es_pullback_valido(self, df, nivel_estructura, direccion):
        precio_actual = df['close'].iloc[-1]

        if direccion == "long":
            zona_pullback = nivel_estructura * (1 - self.tolerance / 100)
            return precio_actual <= zona_pullback

        if direccion == "short":
            zona_pullback = nivel_estructura * (1 + self.tolerance / 100)
            return precio_actual >= zona_pullback

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
            if not self.tp1_hit and price >= tp1:
                self.tp1_hit = True
                resultado["cerrar_50"] = True
                resultado["evento"] = "TP1 alcanzado - cierre 50%"
            elif price >= tp2:
                self.tp2_hit = True
                resultado["cerrar_total"] = True
                resultado["evento"] = "TP2 alcanzado - cierre total"

        if side == "short":
            if not self.tp1_hit and price <= tp1:
                self.tp1_hit = True
                resultado["cerrar_50"] = True
                resultado["evento"] = "TP1 alcanzado - cierre 50%"
            elif price <= tp2:
                self.tp2_hit = True
                resultado["cerrar_total"] = True
                resultado["evento"] = "TP2 alcanzado - cierre total"

        return resultado


class InstitutionalLogger:
    def __init__(self, telegram_send_func):
        self.send_telegram = telegram_send_func

    def log_operacion_completa(self, data):
        mensaje = f"""
📊 OPERACIÓN INSTITUCIONAL DETECTADA

🧠 Sistema: Secundario (BOS Externo)
📈 Dirección: {data.get('direccion')}
💰 Entry: {data.get('entry')}
🎯 TP1 (50%): {data.get('tp1')}
🎯 TP2 (50%): {data.get('tp2')}
🛑 SL: {data.get('sl')}

📊 RR Esperado: {data.get('rr')}
🏆 Winrate Global: {data.get('winrate'):.2f}%
📉 RR Promedio: {data.get('avg_rr'):.2f}
🔢 Total Trades: {data.get('total_trades')}
"""
        self.send_telegram(mensaje)


# ======================================================
# INTEGRADOR DEL SISTEMA SECUNDARIO (CAPA NO INTRUSIVA)
# ======================================================

class InstitutionalSecondarySystem:
    def __init__(self, telegram_send_func):
        self.bos_detector = ExternalBOSDetector()
        self.pullback_validator = PullbackValidator()
        self.tp_manager = PartialTPManager()
        self.stats = InstitutionalStats()
        self.logger = InstitutionalLogger(telegram_send_func)

    def evaluar_confirmacion_institucional(self, df):
        bos_alcista, bos_bajista, swing_high, swing_low = self.bos_detector.is_bos_externo(df)

        confirmacion = {
            "confirmado": False,
            "direccion": None,
            "nivel_estructura": None
        }

        if bos_alcista:
            confirmacion["confirmado"] = True
            confirmacion["direccion"] = "long"
            confirmacion["nivel_estructura"] = swing_high

        elif bos_bajista:
            confirmacion["confirmado"] = True
            confirmacion["direccion"] = "short"
            confirmacion["nivel_estructura"] = swing_low

        return confirmacion

    def validar_pullback(self, df, direccion, nivel):
        return self.pullback_validator.es_pullback_valido(df, nivel, direccion)

    def gestionar_trade_vivo(self, entry, tp1, tp2, price, side):
        return self.tp_manager.gestionar_tp_parcial(entry, tp1, tp2, price, side)

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
    telegram_mensaje("🤖 BOT V90.5 BYBIT REAL INICIADO (ARSENAL NISON COMPLETO)")

    # ======================================================
    # INICIALIZAR SISTEMA INSTITUCIONAL SECUNDARIO
    # ======================================================
    sistema_institucional = InstitutionalSecondarySystem(telegram_mensaje)

    while True:
        time.sleep(60)  # 🔒 Rate limit protection (1 ciclo por minuto)
        try:
            df = obtener_velas()
            df = calcular_indicadores(df)

            slope, intercept, tendencia = detectar_tendencia(df)
            decision, soporte, resistencia, razones = motor_v90(df)

            # ======================================================
            # VARIABLES FILTRO MAESTRO (NISON CONTEXTUAL)
            # Patrón + Zona + Tendencia + Estructura
            # ======================================================
            patron_detectado = False
            zona_valida = False
            tendencia_valida = False
            estructura_valida = False

            precio_actual = df['close'].iloc[-1]
            atr_actual = df['atr'].iloc[-1]

            # =========================
            # ZONA (Soporte/Resistencia)
            # =========================
            if decision == "Buy" and cerca_de_nivel(precio_actual, soporte) < atr_actual:
                zona_valida = True

            if decision == "Sell" and cerca_de_nivel(precio_actual, resistencia) < atr_actual:
                zona_valida = True

            # =========================
            # TENDENCIA PREVIA
            # =========================
            if decision == "Buy" and tendencia == '📈 ALCISTA':
                tendencia_valida = True

            if decision == "Sell" and tendencia == '📉 BAJISTA':
                tendencia_valida = True

            # =========================
            # ESTRUCTURA (sin BOS, usando slope)
            # =========================
            if decision == "Buy" and slope > 0:
                estructura_valida = True

            if decision == "Sell" and slope < 0:
                estructura_valida = True

            # =========================
            # DETECCIÓN PATRÓN NISON REAL (CON CONTEXTO)
            # =========================
            # NOTA: detectar_patron_nison ahora verifica internamente
            # la tendencia previa inmediata y la zona exacta.
            patron_detectado, nombre_patron = detectar_patron_nison(
                df, soporte, resistencia, tendencia
            )

            if patron_detectado:
                razones.append(f"✅ Patrón Nison Validado: {nombre_patron}")

            # =========================
            # FILTRO MAESTRO FINAL
            # =========================
            if decision:
                permitir = filtro_maestro_nison(
                    patron_detectado,
                    zona_valida,
                    tendencia_valida,
                    estructura_valida
                )

                if not permitir:
                    razones.append("⛔ Filtro Maestro bloqueó entrada")
                    decision = None

            # LOG DEL SISTEMA
            log_colab(df, tendencia, slope, soporte, resistencia, decision, razones)

            # ======================================================
            # APERTURA DE TRADE (PAPER)
            # ======================================================
            if decision and risk_management_check():
                precio = df['close'].iloc[-1]
                tiempo_actual = df.index[-1]

                apertura = paper_abrir_posicion(
                    decision=decision,
                    precio=precio,
                    atr=atr_actual,
                    soporte=soporte,
                    resistencia=resistencia,
                    razones=razones,
                    tiempo=tiempo_actual
                )

                pnl_flotante = paper_calcular_pnl(precio)

                mensaje = (
                    f"📌 ENTRADA PAPER {decision}\n"
                    f"💰 Precio: {precio:.2f}\n"
                    f"📍 SL: {PAPER_SL:.2f} | TP: {PAPER_TP:.2f}\n"
                    f"💵 Balance: {PAPER_BALANCE:.2f} USD\n"
                    f"📈 PnL flotante: {pnl_flotante:.4f} USD\n"
                    f"🧠 {', '.join(razones)}"
                )

                telegram_mensaje(mensaje)

                fig = generar_grafico_entrada(
                    df=df,
                    decision=decision,
                    soporte=soporte,
                    resistencia=resistencia,
                    slope=slope,
                    intercept=intercept,
                    razones=razones
                )

                if fig:
                    telegram_grafico(fig)
                    plt.close(fig)

            # ======================================================
            # GESTIÓN DE POSICIÓN ABIERTA
            # ======================================================
            if PAPER_POSICION_ACTIVA is not None:
                cierre = paper_revisar_sl_tp(df)

                if cierre:
                    # Mensaje texto
                    mensaje_cierre = (
                        f"📌 CIERRE PAPER {cierre['decision']} ({cierre['motivo']})\n"
                        f"💰 PnL: {cierre['pnl']:.4f} USD\n"
                        f"💵 Balance: {cierre['balance']:.2f} USD"
                    )
                    telegram_mensaje(mensaje_cierre)
                    
                    # Grafico Salida (NUEVO)
                    fig_salida = generar_grafico_salida(df, cierre)
                    if fig_salida:
                        telegram_grafico(fig_salida)
                        plt.close(fig_salida)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"🚨 ERROR: {e}")
            telegram_mensaje(f"🚨 ERROR BOT: {e}")
            time.sleep(60)

# ======================================================
# START
# ======================================================

if __name__ == '__main__':
    run_bot()


# ======================================================
# NUEVA GESTIÓN DINÁMICA DE TRADE (ATR TRAILING SYSTEM)
# ======================================================
# Entrada → igual
# TP1 → 1 ATR (50%)
# Break Even → automático
# Trailing → 1.2 ATR
# Sin TP final

TRAILING_ACTIVO = False
MAX_PRECIO_ALCANZADO = None
MIN_PRECIO_ALCANZADO = None

ATR_MULT_SL = 1.1
ATR_MULT_TP1 = 1.0
ATR_TRAILING = 1.2
BE_BUFFER = 0.15

def activar_trailing(precio_actual, direccion):
    global TRAILING_ACTIVO, MAX_PRECIO_ALCANZADO, MIN_PRECIO_ALCANZADO
    
    TRAILING_ACTIVO = True
    
    if direccion == "LONG":
        MAX_PRECIO_ALCANZADO = precio_actual
    else:
        MIN_PRECIO_ALCANZADO = precio_actual


def actualizar_trailing(precio_actual, atr, direccion):
    global MAX_PRECIO_ALCANZADO, MIN_PRECIO_ALCANZADO

    if not TRAILING_ACTIVO:
        return None

    if direccion == "LONG":
        if MAX_PRECIO_ALCANZADO is None or precio_actual > MAX_PRECIO_ALCANZADO:
            MAX_PRECIO_ALCANZADO = precio_actual

        trailing_sl = MAX_PRECIO_ALCANZADO - (ATR_TRAILING * atr)
        return trailing_sl

    elif direccion == "SHORT":
        if MIN_PRECIO_ALCANZADO is None or precio_actual < MIN_PRECIO_ALCANZADO:
            MIN_PRECIO_ALCANZADO = precio_actual

        trailing_sl = MIN_PRECIO_ALCANZADO + (ATR_TRAILING * atr)
        return trailing_sl


def ejecutar_tp1_y_break_even(precio_entrada, atr, direccion):
    """
    Cierra 50% y mueve SL a Break Even + buffer.
    """
    if direccion == "LONG":
        nuevo_sl = precio_entrada + (BE_BUFFER * atr)
    else:
        nuevo_sl = precio_entrada - (BE_BUFFER * atr)

    return nuevo_sl


def reset_trailing():
    global TRAILING_ACTIVO, MAX_PRECIO_ALCANZADO, MIN_PRECIO_ALCANZADO

    TRAILING_ACTIVO = False
    MAX_PRECIO_ALCANZADO = None
    MIN_PRECIO_ALCANZADO = None

# ======================================================
# FIN GESTIÓN DINÁMICA ATR
# ======================================================

# =====================================================
# TREND FOLLOW INTEGRADO (TP2 DINÁMICO EN FLUJO PRINCIPAL)
# =====================================================
# Este módulo integra trailing dinámico al flujo principal
# sin depender de un TP2 fijo. Tras TP1:
#   • Cierra 50%
#   • SL -> Break Even
#   • Activa trailing por ATR
#   • Cierre final por retroceso al SL dinámico

TRAILING_MULTIPLIER = 1.2

class _TrendFollowState:
    trailing_activo = False
    tp1_ejecutado = False

_TF_STATE = _TrendFollowState()

def tf_activar_trailing():
    _TF_STATE.trailing_activo = True

def tf_reset():
    _TF_STATE.trailing_activo = False
    _TF_STATE.tp1_ejecutado = False

def tf_actualizar_trailing(precio_actual, atr, tipo_operacion, sl_actual):
    if not _TF_STATE.trailing_activo:
        return sl_actual

    if tipo_operacion == "BUY":
        nuevo_sl = precio_actual - (atr * TRAILING_MULTIPLIER)
        if nuevo_sl > sl_actual:
            return nuevo_sl

    elif tipo_operacion == "SELL":
        nuevo_sl = precio_actual + (atr * TRAILING_MULTIPLIER)
        if nuevo_sl < sl_actual:
            return nuevo_sl

    return sl_actual

def tf_on_tp1():
    # Llamar justo después de cerrar el 50% en TP1
    _TF_STATE.tp1_ejecutado = True
    tf_activar_trailing()

# -------- Integración automática por monkey patch --------
# Si el bot tiene funciones comunes de gestión, intentamos integrarnos.

def _tf_try_patch():
    g = globals()

    # Patch de función que maneja TP1 parcial
    for name in ["cerrar_parcial_50", "close_partial_tp1", "ejecutar_tp1", "take_profit_parcial"]:
        if name in g and callable(g[name]):
            orig = g[name]
            def wrapped_tp1(*a, __orig=orig, **kw):
                r = __orig(*a, **kw)
                try:
                    tf_on_tp1()
                except Exception:
                    pass
                return r
            g[name] = wrapped_tp1

    # Patch de función que actualiza SL dinámicamente o loop de gestión
    for name in ["actualizar_sl", "update_stop_loss", "gestionar_trade", "manage_trade", "trade_loop"]:
        if name in g and callable(g[name]):
            orig = g[name]
            def wrapped_manage(*a, __orig=orig, **kw):
                r = __orig(*a, **kw)
                try:
                    # Intentamos leer variables típicas del contexto global
                    precio_actual = g.get("precio_actual") or g.get("current_price")
                    atr = g.get("atr") or g.get("ATR")
                    tipo = g.get("tipo_operacion") or g.get("side")
                    sl = g.get("sl_actual") or g.get("stop_loss")

                    if precio_actual and atr and tipo and sl:
                        nuevo_sl = tf_actualizar_trailing(precio_actual, atr, tipo, sl)
                        if "sl_actual" in g:
                            g["sl_actual"] = nuevo_sl
                        if "stop_loss" in g:
                            g["stop_loss"] = nuevo_sl
                except Exception:
                    pass
                return r
            g[name] = wrapped_manage

_tf_try_patch()

# =====================================================
# FIN TREND FOLLOW INTEGRADO
# =====================================================
