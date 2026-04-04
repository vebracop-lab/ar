# BOT TRADING V99.2 BYBIT REAL – GROQ IA (RAILWAY READY)
# ======================================================
# - Análisis avanzado de velas japonesas (estrella fugaz, martillo, etc.)
# - Detección de techos/soportes por múltiples rechazos
# - Prompt inspirado en Steven Nison (contexto visual enriquecido)
# - Gestión de riesgos dinámica y autoaprendizaje
# ======================================================

import os
import time
import io
import requests
import json
import numpy as np
import pandas as pd
from scipy.stats import linregress
from datetime import datetime, timezone
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from groq import Groq

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)
MODELO_GROQ = "llama-3.3-70b-versatile"

SYMBOL = "BTCUSDT"
INTERVAL = "5"
RISK_PER_TRADE = 0.02
LEVERAGE = 10
SLEEP_SECONDS = 60

DEFAULT_SL_MULT = 1.5
DEFAULT_TP1_MULT = 2.5
DEFAULT_TP2_MULT = 4.0
DEFAULT_TRAILING_MULT = 2.0

PORCENTAJE_CIERRE_TP1 = 0.5
GRAFICO_VELAS_LIMIT = 120

# ======================================================
# PAPER TRADING
# ======================================================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_POSICION_ACTIVA = None
PAPER_PRECIO_ENTRADA = None
PAPER_SL_INICIAL = None
PAPER_TP1 = None
PAPER_TP2 = None
PAPER_TRAILING_MULT = DEFAULT_TRAILING_MULT
PAPER_SIZE_BTC = 0.0
PAPER_SIZE_BTC_RESTANTE = 0.0
PAPER_TP1_EJECUTADO = False
PAPER_PNL_PARCIAL = 0.0
PAPER_SL_ACTUAL = None
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
PAPER_LAST_10_PNL = []

TRADE_HISTORY = []

MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

ADAPTIVE_BIAS = 0.0
ADAPTIVE_SL_MULT = DEFAULT_SL_MULT
ADAPTIVE_TP1_MULT = DEFAULT_TP1_MULT
ADAPTIVE_TP2_MULT = DEFAULT_TP2_MULT
ADAPTIVE_TRAILING_MULT = DEFAULT_TRAILING_MULT
ULTIMO_APRENDIZAJE = None

ULTIMA_RAZONES = []
ULTIMO_PATRON = ""
ULTIMOS_MULTIS = (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TP2_MULT, DEFAULT_TRAILING_MULT)

# ======================================================
# TELEGRAM
# ======================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL = "https://api.bybit.com"

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texto}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Error Telegram: {e}")

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
        print(f"Error enviando foto: {e}")

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
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
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
    slope, intercept, _, _, _ = linregress(x_macro, y_macro)
    if slope > 0.01: tendencia_macro = 'ALCISTA'
    elif slope < -0.01: tendencia_macro = 'BAJISTA'
    else: tendencia_macro = 'LATERAL'
    return soporte_horiz, resistencia_horiz, slope, intercept, tendencia_macro

# ======================================================
# ANÁLISIS AVANZADO DE VELAS (patrones Nison)
# ======================================================
def detectar_patron_vela(open_, high, low, close):
    """Detecta patrones individuales de una vela"""
    cuerpo = abs(close - open_)
    rango = high - low
    if rango == 0:
        return "Vela indeterminada"
    cuerpo_pct = cuerpo / rango * 100
    mecha_sup = high - max(close, open_)
    mecha_inf = min(close, open_) - low
    mecha_sup_pct = mecha_sup / rango * 100
    mecha_inf_pct = mecha_inf / rango * 100
    
    es_verde = close >= open_
    
    # Estrella fugaz / Martillo invertido
    if mecha_sup_pct > 60 and cuerpo_pct < 30 and (mecha_inf_pct < 10):
        return "ESTRELLA FUGAZ (reversión bajista)" if not es_verde else "MARTILLO INVERTIDO (posible rebote)"
    # Martillo / Hombre colgado
    if mecha_inf_pct > 60 and cuerpo_pct < 30 and (mecha_sup_pct < 10):
        return "MARTILLO (reversión alcista)" if es_verde else "HOMBRE COLGADO (reversión bajista)"
    # Vela doji
    if cuerpo_pct < 10:
        return "DOJI (indecisión)"
    # Vela larga sin mechas (fuerte impulso)
    if cuerpo_pct > 70 and mecha_sup_pct < 15 and mecha_inf_pct < 15:
        return "VELA LARGA SIN MECHAS (impulso fuerte)"
    # Rechazo con mecha superior larga
    if mecha_sup_pct > 50 and cuerpo_pct < 50:
        return "RECHAZO EN ZONA ALTA (mecha superior larga)"
    # Rechazo con mecha inferior larga
    if mecha_inf_pct > 50 and cuerpo_pct < 50:
        return "REBOTE EN ZONA BAJA (mecha inferior larga)"
    return f"Vela normal ({cuerpo_pct:.0f}% cuerpo, Msup {mecha_sup_pct:.0f}%, Minf {mecha_inf_pct:.0f}%)"

def analizar_velas_nison(df, num_velas=8):
    """Genera descripción detallada de velas como lo haría Steven Nison"""
    ultimas = df.iloc[-num_velas-1:-1]  # Velas cerradas
    analisis = []
    for i, (idx, vela) in enumerate(ultimas.iterrows()):
        patron = detectar_patron_vela(vela['open'], vela['high'], vela['low'], vela['close'])
        color = "VERDE" if vela['close'] >= vela['open'] else "ROJA"
        cuerpo = abs(vela['close'] - vela['open'])
        rango = vela['high'] - vela['low']
        mecha_sup = vela['high'] - max(vela['close'], vela['open'])
        mecha_inf = min(vela['close'], vela['open']) - vela['low']
        analisis.append(f"Vela{i+1}: {color} | Rango:{rango:.2f} | Cuerpo:{cuerpo:.2f} | MechSup:{mecha_sup:.2f} | MechInf:{mecha_inf:.2f} | Patrón: {patron}")
    return "\n".join(analisis)

def detectar_techos_soportes(df, resistencia, soporte, num_velas=20):
    """Detecta si ha habido múltiples rechazos en una zona (techo o soporte)"""
    df_cercano = df.iloc[-num_velas:]
    toques_resistencia = 0
    toques_soporte = 0
    for idx, vela in df_cercano.iterrows():
        if vela['high'] >= resistencia * 0.998:
            toques_resistencia += 1
        if vela['low'] <= soporte * 1.002:
            toques_soporte += 1
    texto = ""
    if toques_resistencia >= 3:
        texto += f"⚠️ ZONA DE RESISTENCIA ({resistencia:.0f}) RECHAZADA {toques_resistencia} VECES en últimas velas. Se ha formado un TECHO SÓLIDO. "
    if toques_soporte >= 3:
        texto += f"✅ ZONA DE SOPORTE ({soporte:.0f}) PROBADA {toques_soporte} VECES. Posible rebote. "
    return texto, toques_resistencia, toques_soporte

def analizar_tendencia_velas(df, num_velas=6):
    """Analiza la secuencia de velas para detectar fuerza o debilidad"""
    ultimas = df.iloc[-num_velas-1:-1]
    colores = []
    cuerpos = []
    for vela in ultimas.iterrows():
        colores.append("VERDE" if vela[1]['close'] >= vela[1]['open'] else "ROJA")
        cuerpos.append(abs(vela[1]['close'] - vela[1]['open']))
    # Secuencia de colores
    secuencia = " → ".join(colores[-5:])
    # Cuerpos promedio
    cuerpo_prom = np.mean(cuerpos[-3:])
    if colores[-3:].count("VERDE") >= 2 and cuerpos[-1] > cuerpo_prom:
        tend_texto = "Últimas velas muestran FUERZA ALCISTA (cuerpos verdes grandes)."
    elif colores[-3:].count("ROJA") >= 2 and cuerpos[-1] > cuerpo_prom:
        tend_texto = "Últimas velas muestran DEBILIDAD BAJISTA (cuerpos rojos grandes)."
    else:
        tend_texto = "Sin impulso claro en las últimas velas."
    return secuencia, tend_texto

# ======================================================
# AUTOAPRENDIZAJE
# ======================================================
def aprender_de_trades():
    global ADAPTIVE_BIAS, ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TP2_MULT, ADAPTIVE_TRAILING_MULT
    global ULTIMO_APRENDIZAJE
    
    if len(TRADE_HISTORY) < 10:
        return
    if ULTIMO_APRENDIZAJE is not None and len(TRADE_HISTORY) - ULTIMO_APRENDIZAJE < 10:
        return
    
    ultimos = TRADE_HISTORY[-10:]
    wins = [t for t in ultimos if t['resultado_win']]
    winrate = len(wins) / 10.0
    
    razones_loss = []
    for loss in [t for t in ultimos if not t['resultado_win']]:
        razones_loss.extend(loss.get('razones_ia', []))
    counter = Counter(razones_loss)
    errores_comunes = counter.most_common(3)
    
    if winrate < 0.4:
        ADAPTIVE_BIAS = max(-0.2, ADAPTIVE_BIAS - 0.05)
        ADAPTIVE_SL_MULT = min(2.5, ADAPTIVE_SL_MULT * 1.1)
        ADAPTIVE_TP1_MULT = max(1.5, ADAPTIVE_TP1_MULT * 0.9)
    elif winrate > 0.6:
        ADAPTIVE_BIAS = min(0.3, ADAPTIVE_BIAS + 0.02)
        ADAPTIVE_SL_MULT = max(1.0, ADAPTIVE_SL_MULT * 0.95)
        ADAPTIVE_TP1_MULT = min(4.0, ADAPTIVE_TP1_MULT * 1.05)
    
    mensaje = f"📚 AUTOAPRENDIZAJE (últimos 10 trades)\nWinrate: {winrate*100:.1f}%\nErrores: {errores_comunes}\nSesgo: {ADAPTIVE_BIAS:.2f}\nSL mult: {ADAPTIVE_SL_MULT:.2f}"
    telegram_mensaje(mensaje)
    print(mensaje)
    ULTIMO_APRENDIZAJE = len(TRADE_HISTORY)

# ======================================================
# IA GROQ CON PROMPT ESTILO STEVEN NISON
# ======================================================
def analizar_con_groq_texto(df, soporte, resistencia, tendencia, slope, intercept, idx=-2):
    precio = df['close'].iloc[idx]
    atr = df['atr'].iloc[idx]
    ema_val = df['ema20'].iloc[idx]
    rsi, rsi_estado, tend_macd, hist_macd = analizar_macd_rsi(df, idx)
    
    # Análisis de velas al estilo Nison
    analisis_velas_nison = analizar_velas_nison(df, num_velas=8)
    secuencia_velas, tendencia_velas = analizar_tendencia_velas(df)
    texto_techos, toques_res, toques_sop = detectar_techos_soportes(df, resistencia, soporte)
    
    # Posición respecto a EMA
    diff_ema_pct = (precio - ema_val) / ema_val * 100
    if abs(diff_ema_pct) < 0.2:
        pos_ema = "PRECIO JUSTO EN LA EMA20 (tocando exactamente)"
    elif precio > ema_val:
        pos_ema = f"PRECIO ENCIMA DE EMA20 (+{diff_ema_pct:.2f}%)"
    else:
        pos_ema = f"PRECIO DEBAJO DE EMA20 ({diff_ema_pct:.2f}%)"
    
    # Rol de la EMA basado en toques
    toques_ema = sum(1 for i in range(max(0, idx-5), idx+1) if df['low'].iloc[i] <= df['ema20'].iloc[i] <= df['high'].iloc[i])
    if toques_ema >= 2 and precio > ema_val:
        rol_ema = "EMA20 actúa como SOPORTE DINÁMICO (múltiples toques desde arriba)"
    elif toques_ema >= 2 and precio < ema_val:
        rol_ema = "EMA20 actúa como RESISTENCIA DINÁMICA (múltiples toques desde abajo)"
    else:
        rol_ema = "EMA20 sin rol claro de soporte/resistencia"
    
    # Ruptura de tendencia?
    ruptura = ""
    if len(df) > 20:
        pendiente_reciente, _, _, _, _ = linregress(np.arange(20), df['close'].iloc[-20:].values)
        if slope > 0 and pendiente_reciente < 0:
            ruptura = "⚠️ ¡RUPTURA DE TENDENCIA ALCISTA! La pendiente reciente es negativa mientras que la macro era alcista."
        elif slope < 0 and pendiente_recente > 0:
            ruptura = "✅ ¡RUPTURA DE TENDENCIA BAJISTA! La pendiente reciente es positiva."

    prompt = f"""
Eres Steven Nison, el mayor experto mundial en velas japonesas. Analiza el gráfico de BTCUSDT en 5 minutos como si lo estuvieras viendo. Usa tu conocimiento profundo de la acción del precio, patrones de velas y estructura de mercado. Sé directo, humano y contundente. Si ves una señal de reversión clara, ACTÚA.

=== CONTEXTO ACTUAL ===
Precio actual: {precio:.2f}
ATR (volatilidad): {atr:.2f}
Soporte horizontal: {soporte:.2f}
Resistencia horizontal: {resistencia:.2f}
Tendencia macro (120 velas): {tendencia} (pendiente {slope:.6f})
{ruptura}
RSI: {rsi:.1f} - {rsi_estado}
MACD: {tend_macd} (histograma {hist_macd:.2f})
EMA20: {ema_val:.2f} - {pos_ema} - {rol_ema}
{texto_techos}

=== ANÁLISIS DE VELAS (estilo Nison) ===
{analisis_velas_nison}

Secuencia de colores últimas 5 velas: {secuencia_velas}
{ tendencia_velas }

=== INSTRUCCIONES PARA TU DECISIÓN ===
1. **Mira la vela más reciente**: si tiene una mecha superior muy larga y golpeó la resistencia, es una ESTRELLA FUGAZ → SELL. Si tiene mecha inferior larga en soporte → BUY.
2. **Si ha habido 3 o más rechazos en la resistencia** (techo sólido) y el precio está debajo de la EMA20 → SELL.
3. **Si el precio rompió la línea de tendencia alcista** (pendiente reciente negativa) y las velas son rojas → SELL.
4. **Si hay falta de seguimiento alcista** (velas pequeñas después de un intento de subida) → SELL.
5. **Si la EMA20 está actuando como resistencia** (precio debajo, múltiples toques) y MACD bajista → SELL.
6. **Si ves un martillo en soporte** o vela de rebote con mecha inferior larga → BUY.
7. **No tengas miedo de vender** cuando el mercado muestre debilidad. Los patrones de reversión son poderosos.

Recomienda multiplicadores SL/TP según la volatilidad (ATR). Sé conservador: SL 1.2-2.0x ATR, TP1 2.0-3.5x, TP2 3.0-5.0x, trailing 1.5-2.5x.

Responde SOLO en JSON con este formato:
{{
  "decision": "Buy/Sell/Hold",
  "patron": "nombre del patrón detectado (ej. 'Estrella fugaz en resistencia', 'Triple techo + ruptura tendencia')",
  "razones": ["razón1","razón2","razón3"],
  "sl_mult": 1.5,
  "tp1_mult": 2.5,
  "tp2_mult": 4.0,
  "trailing_mult": 2.0
}}
"""
    try:
        completion = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        datos = json.loads(completion.choices[0].message.content)
        
        def sanitize(value, default):
            if value is None or not isinstance(value, (int, float)):
                return default
            return value
        
        sl_mult = sanitize(datos.get("sl_mult"), DEFAULT_SL_MULT)
        tp1_mult = sanitize(datos.get("tp1_mult"), DEFAULT_TP1_MULT)
        tp2_mult = sanitize(datos.get("tp2_mult"), DEFAULT_TP2_MULT)
        trailing_mult = sanitize(datos.get("trailing_mult"), DEFAULT_TRAILING_MULT)
        
        if not isinstance(datos.get("razones"), list):
            datos["razones"] = ["Análisis de velas"]
        if not isinstance(datos.get("patron"), str):
            datos["patron"] = "Patrón técnico"
        
        # Sesgo adaptativo muy leve
        if ADAPTIVE_BIAS > 0.2 and datos.get("decision") == "Sell":
            if np.random.random() < 0.2:
                datos["decision"] = "Hold"
        elif ADAPTIVE_BIAS < -0.2 and datos.get("decision") == "Buy":
            if np.random.random() < 0.2:
                datos["decision"] = "Hold"
        
        datos["sl_mult"] = 0.6 * sl_mult + 0.4 * ADAPTIVE_SL_MULT
        datos["tp1_mult"] = 0.6 * tp1_mult + 0.4 * ADAPTIVE_TP1_MULT
        datos["tp2_mult"] = 0.6 * tp2_mult + 0.4 * ADAPTIVE_TP2_MULT
        datos["trailing_mult"] = 0.6 * trailing_mult + 0.4 * ADAPTIVE_TRAILING_MULT
        
        return datos
    except Exception as e:
        print(f"Error Groq: {e}")
        return {"decision": "Hold", "razones": ["Error API"], "patron": "", "sl_mult": DEFAULT_SL_MULT, "tp1_mult": DEFAULT_TP1_MULT, "tp2_mult": DEFAULT_TP2_MULT, "trailing_mult": DEFAULT_TRAILING_MULT}

# ======================================================
# GESTIÓN DE RIESGO Y PAPER TRADING (sin cambios, igual que antes)
# ======================================================
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY, PAPER_BALANCE
    hoy = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY = hoy
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
    dd = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if dd <= -MAX_DAILY_DRAWDOWN_PCT and not PAPER_STOPPED_TODAY:
        telegram_mensaje("🛑 Drawdown máximo diario alcanzado. Bot pausado.")
        PAPER_STOPPED_TODAY = True
        return False
    return not PAPER_STOPPED_TODAY

def paper_abrir_posicion(decision, precio, atr, sl_mult, tp1_mult, tp2_mult, trailing_mult, razones, patron):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TP2
    global PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO
    global PAPER_SL_ACTUAL, PAPER_BALANCE, ULTIMA_RAZONES, ULTIMO_PATRON, ULTIMOS_MULTIS

    if PAPER_POSICION_ACTIVA is not None:
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    if decision == "Buy":
        sl = precio - (atr * sl_mult)
        tp1 = precio + (atr * tp1_mult)
        tp2 = precio + (atr * tp2_mult)
    else:
        sl = precio + (atr * sl_mult)
        tp1 = precio - (atr * tp1_mult)
        tp2 = precio - (atr * tp2_mult)

    distancia_riesgo = abs(precio - sl)
    if distancia_riesgo == 0:
        return False

    size_usd = min((riesgo_usd / distancia_riesgo) * precio, PAPER_BALANCE * LEVERAGE)
    size_btc = size_usd / precio

    PAPER_POSICION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL_INICIAL = sl
    PAPER_TP1 = tp1
    PAPER_TP2 = tp2
    PAPER_TRAILING_MULT = trailing_mult
    PAPER_SIZE_BTC = size_btc
    PAPER_SIZE_BTC_RESTANTE = size_btc
    PAPER_TP1_EJECUTADO = False
    PAPER_SL_ACTUAL = sl

    ULTIMA_RAZONES = razones
    ULTIMO_PATRON = patron
    ULTIMOS_MULTIS = (sl_mult, tp1_mult, tp2_mult, trailing_mult)

    telegram_mensaje(f"📌 OPERACIÓN {decision.upper()} | Precio: {precio:.2f} | SL: {sl:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f}\nRazones: {' | '.join(razones[:2])}")
    return True

def paper_revisar_sl_tp(df, soporte, resistencia, slope, intercept):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TP2
    global PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO
    global PAPER_SL_ACTUAL, PAPER_BALANCE, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS
    global PAPER_TRADES_TOTALES, PAPER_LAST_10_PNL, TRADE_HISTORY
    global ULTIMA_RAZONES, ULTIMO_PATRON, ULTIMOS_MULTIS

    if PAPER_POSICION_ACTIVA is None:
        return None

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    close = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    cerrar_total = False
    motivo = ""

    if not PAPER_TP1_EJECUTADO:
        if (PAPER_POSICION_ACTIVA == "Buy" and high >= PAPER_TP1) or (PAPER_POSICION_ACTIVA == "Sell" and low <= PAPER_TP1):
            beneficio_parcial = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1) if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1)
            PAPER_BALANCE += beneficio_parcial
            PAPER_PNL_PARCIAL = beneficio_parcial
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE_TP1)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL_ACTUAL = PAPER_PRECIO_ENTRADA
            telegram_mensaje(f"🎯 TP1 alcanzado. Beneficio parcial: +{beneficio_parcial:.2f} USD. SL a break-even.")

    if PAPER_TP1_EJECUTADO:
        if PAPER_POSICION_ACTIVA == "Buy":
            nuevo_sl = close - (atr * PAPER_TRAILING_MULT)
            if nuevo_sl > PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
            if low <= PAPER_SL_ACTUAL:
                cerrar_total = True
                motivo = "Trailing Stop"
        else:
            nuevo_sl = close + (atr * PAPER_TRAILING_MULT)
            if nuevo_sl < PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
            if high >= PAPER_SL_ACTUAL:
                cerrar_total = True
                motivo = "Trailing Stop"
    else:
        if (PAPER_POSICION_ACTIVA == "Buy" and low <= PAPER_SL_INICIAL) or (PAPER_POSICION_ACTIVA == "Sell" and high >= PAPER_SL_INICIAL):
            cerrar_total = True
            motivo = "Stop Loss Inicial"
            PAPER_SL_ACTUAL = PAPER_SL_INICIAL

    if cerrar_total:
        precio_salida = PAPER_SL_ACTUAL
        if PAPER_POSICION_ACTIVA == "Buy":
            pnl_restante = (precio_salida - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE
        else:
            pnl_restante = (PAPER_PRECIO_ENTRADA - precio_salida) * PAPER_SIZE_BTC_RESTANTE
        pnl_total = PAPER_PNL_PARCIAL + pnl_restante
        PAPER_BALANCE += pnl_restante
        PAPER_TRADES_TOTALES += 1
        win_status = pnl_total > 0
        PAPER_WIN += 1 if win_status else 0
        PAPER_LOSS += 1 if not win_status else 0
        PAPER_LAST_10_PNL.append(pnl_total)
        if len(PAPER_LAST_10_PNL) > 10:
            PAPER_LAST_10_PNL.pop(0)
        winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0

        try:
            trade_record = {
                "fecha": datetime.now(timezone.utc).isoformat(),
                "decision": PAPER_POSICION_ACTIVA,
                "precio_entrada": PAPER_PRECIO_ENTRADA,
                "precio_salida": precio_salida,
                "pnl": pnl_total,
                "razones_ia": ULTIMA_RAZONES,
                "patron": ULTIMO_PATRON,
                "sl_mult_usado": ULTIMOS_MULTIS[0],
                "tp1_mult_usado": ULTIMOS_MULTIS[1],
                "tp2_mult_usado": ULTIMOS_MULTIS[2],
                "trailing_mult_usado": ULTIMOS_MULTIS[3],
                "resultado_win": win_status
            }
            TRADE_HISTORY.append(trade_record)
            if len(TRADE_HISTORY) % 10 == 0:
                aprender_de_trades()
        except Exception as e:
            print(f"Error guardando historial: {e}")

        telegram_mensaje(f"📤 TRADE CERRADO ({motivo})\nPnL total: {pnl_total:.2f} USD\nBalance: {PAPER_BALANCE:.2f} USD\nWinrate: {winrate:.1f}%")

        fig = generar_grafico_salida(df, PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, precio_salida, pnl_total, win_status, soporte, resistencia, slope, intercept)
        telegram_grafico(fig)
        plt.close(fig)

        PAPER_POSICION_ACTIVA = None
        return True
    return None

# ======================================================
# GRÁFICOS (igual)
# ======================================================
def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones, patron):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(14,7))
    for i in range(len(df_plot)):
        color = 'green' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else 'red'
        ax.vlines(x[i], df_plot['low'].iloc[i], df_plot['high'].iloc[i], color=color, linewidth=1)
        y_cuerpo = min(df_plot['open'].iloc[i], df_plot['close'].iloc[i])
        altura = abs(df_plot['close'].iloc[i] - df_plot['open'].iloc[i])
        ax.add_patch(plt.Rectangle((x[i]-0.3, y_cuerpo), 0.6, max(altura,0.01), color=color, alpha=0.9))
    ax.axhline(soporte, color='cyan', ls='--', label='Soporte')
    ax.axhline(resistencia, color='magenta', ls='--', label='Resistencia')
    ax.plot(x, intercept + slope*x, 'w-.', label='Tendencia')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'y', label='EMA20')
    entrada_x = len(df_plot)-2
    if decision=='Buy':
        ax.scatter(entrada_x, df_plot['close'].iloc[-2]-50, s=300, marker='^', c='lime', edgecolors='black')
    else:
        ax.scatter(entrada_x, df_plot['close'].iloc[-2]+50, s=300, marker='v', c='red', edgecolors='black')
    texto = f"GROQ V99.2 (Nison): {decision.upper()}\nPatrón: {patron[:60]}\n{chr(10).join(razones[:2])}"
    ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.85))
    ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    plt.legend(loc='lower right', facecolor='black', labelcolor='white')
    plt.tight_layout()
    return fig

def generar_grafico_salida(df, posicion, precio_entrada, precio_salida, pnl, win, soporte, resistencia, slope, intercept):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(14,7))
    for i in range(len(df_plot)):
        color = 'green' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else 'red'
        ax.vlines(x[i], df_plot['low'].iloc[i], df_plot['high'].iloc[i], color=color, linewidth=1)
        y_cuerpo = min(df_plot['open'].iloc[i], df_plot['close'].iloc[i])
        altura = abs(df_plot['close'].iloc[i] - df_plot['open'].iloc[i])
        ax.add_patch(plt.Rectangle((x[i]-0.3, y_cuerpo), 0.6, max(altura,0.01), color=color, alpha=0.9))
    ax.axhline(soporte, color='cyan', ls='--', label='Soporte')
    ax.axhline(resistencia, color='magenta', ls='--', label='Resistencia')
    ax.plot(x, intercept + slope*x, 'w-.', label='Tendencia')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'y', label='EMA20')
    ax.axhline(precio_entrada, color='blue', ls=':', label='Entrada')
    ax.axhline(precio_salida, color='orange', ls=':', label='Salida')
    estado = "WIN" if win else "LOSS"
    texto = f"RESULTADO: {estado}\n{posicion.upper()} | PnL: {pnl:.2f} USD"
    ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.85))
    ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    plt.legend(loc='lower right', facecolor='black', labelcolor='white')
    plt.tight_layout()
    return fig

# ======================================================
# LOOP PRINCIPAL
# ======================================================
def run_bot():
    print("🤖 BOT V99.2 INICIADO - Análisis de velas estilo Steven Nison, detección de techos/soportes")
    telegram_mensaje("🤖 BOT V99.2: Incorpora patrones de velas, rechazos múltiples y ruptura de tendencia.")
    ultima_vela_operada = None

    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            idx_eval = -2
            precio_mercado = df['close'].iloc[-1]
            tiempo_vela_cerrada = df.index[-2]

            soporte, resistencia, slope, intercept, tendencia = detectar_zonas_mercado(df, idx_eval)

            print(f"\n💓 Heartbeat | Precio: {precio_mercado:.2f} | Sop: {soporte:.2f} | Res: {resistencia:.2f} | Trades: {PAPER_TRADES_TOTALES} | Sesgo: {ADAPTIVE_BIAS:.2f}")

            if PAPER_POSICION_ACTIVA is None and ultima_vela_operada != tiempo_vela_cerrada:
                respuesta = analizar_con_groq_texto(df, soporte, resistencia, tendencia, slope, intercept, idx_eval)
                decision = respuesta.get("decision", "Hold")
                razones = respuesta.get("razones", [])
                patron = respuesta.get("patron", "")
                sl_mult = respuesta.get("sl_mult", DEFAULT_SL_MULT)
                tp1_mult = respuesta.get("tp1_mult", DEFAULT_TP1_MULT)
                tp2_mult = respuesta.get("tp2_mult", DEFAULT_TP2_MULT)
                trailing_mult = respuesta.get("trailing_mult", DEFAULT_TRAILING_MULT)

                if decision in ["Buy", "Sell"] and risk_management_check():
                    atr_actual = df['atr'].iloc[-1]
                    if paper_abrir_posicion(decision, precio_mercado, atr_actual, sl_mult, tp1_mult, tp2_mult, trailing_mult, razones, patron):
                        ultima_vela_operada = tiempo_vela_cerrada
                        fig = generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones, patron)
                        telegram_grafico(fig)
                        plt.close(fig)
                else:
                    print(f"Hold: {razones[0] if razones else 'Sin señal clara'}")

            if PAPER_POSICION_ACTIVA is not None:
                paper_revisar_sl_tp(df, soporte, resistencia, slope, intercept)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
