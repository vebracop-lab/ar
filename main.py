# BOT TRADING V99.0 BYBIT REAL – GROQ IA (RAILWAY READY)
# ======================================================
# IA GROQ (LLaMA 3.3 Versatile) con análisis contextual avanzado:
# - Percepción de EMA como soporte/resistencia dinámico
# - MACD + RSI + Velas + Estructura de mercado
# - Autoaprendizaje cada 10 trades (análisis de errores y ajuste de sesgo)
# - Gestión de riesgos dinámica con SL, TP1 fijo, TP2 con trailing
# ======================================================

import os
import time
import io
import requests
import json
import numpy as np
import pandas as pd
import textwrap
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
RISK_PER_TRADE = 0.02        # 2% del balance por trade
LEVERAGE = 10
SLEEP_SECONDS = 60

# Parámetros por defecto (pueden ajustarse por aprendizaje)
DEFAULT_SL_MULT = 1.5
DEFAULT_TP1_MULT = 2.5
DEFAULT_TP2_MULT = 4.0
DEFAULT_TRAILING_MULT = 2.0

PORCENTAJE_CIERRE_TP1 = 0.5   # Cerrar 50% en TP1

GRAFICO_VELAS_LIMIT = 120
MOSTRAR_EMA20 = True

# ======================================================
# PAPER TRADING (SIMULADO)
# ======================================================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_POSICION_ACTIVA = None      # "Buy" o "Sell"
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

# Historial para aprendizaje
TRADE_HISTORY = []  # Cada trade: {fecha, decision, precio_entrada, precio_salida, pnl, razones_ia, sl_mult_usado, tp1_mult_usado, tp2_mult_usado, trailing_mult_usado, resultado_win}

# Control de drawdown diario
MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

# Parámetros adaptativos (aprendizaje)
ADAPTIVE_BIAS = 0.0   # Rango -0.3 a +0.3, sesgo para Buy (+) o Sell (-)
ADAPTIVE_SL_MULT = DEFAULT_SL_MULT
ADAPTIVE_TP1_MULT = DEFAULT_TP1_MULT
ADAPTIVE_TP2_MULT = DEFAULT_TP2_MULT
ADAPTIVE_TRAILING_MULT = DEFAULT_TRAILING_MULT
TRADES_SIN_APRENDER = 0
ULTIMO_APRENDIZAJE = None

# Variables globales para guardar datos del último trade
ULTIMA_RAZONES = []
ULTIMO_PATRON = ""
ULTIMOS_MULTIS = (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TP2_MULT, DEFAULT_TRAILING_MULT)

# ======================================================
# CREDENCIALES Y TELEGRAM
# ======================================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
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
# DATOS Y TÉCNICO (mejorado con MACD)
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
    # EMA20
    df['ema20'] = df['close'].ewm(span=20).mean()
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    # EMA touch
    df['ema_touch'] = (df['low'] <= df['ema20']) & (df['high'] >= df['ema20'])
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
# ANÁLISIS CONTEXTUAL ENRIQUECIDO
# ======================================================
def analizar_posicion_respecto_ema(df, idx=-2):
    """Describe si el precio está encima, debajo, tocando o cruzando la EMA20, y si la EMA actúa como soporte/resistencia."""
    precio = df['close'].iloc[idx]
    ema = df['ema20'].iloc[idx]
    diff_pct = (precio - ema) / ema * 100
    # Velas anteriores
    precio_ant = df['close'].iloc[idx-1] if idx-1 >= 0 else precio
    ema_ant = df['ema20'].iloc[idx-1] if idx-1 >= 0 else ema
    
    if abs(diff_pct) < 0.15:
        relacion = "PRECIO SOBRE LA EMA20 (tocando exactamente)"
    elif precio > ema:
        relacion = f"PRECIO ENCIMA DE EMA20 (+{diff_pct:.2f}%)"
    else:
        relacion = f"PRECIO DEBAJO DE EMA20 ({diff_pct:.2f}%)"
    
    # Detectar si la EMA ha actuado como soporte o resistencia en las últimas 5 velas
    toques = 0
    for i in range(max(0, idx-5), idx+1):
        if df['low'].iloc[i] <= df['ema20'].iloc[i] <= df['high'].iloc[i]:
            toques += 1
    if toques >= 3:
        rol = "EMA20 está actuando como SOPORTE/RESISTENCIA dinámico (múltiples toques)."
    else:
        rol = "EMA20 no muestra rol claro de soporte/resistencia."
    
    # Detectar cruce reciente
    cruce = ""
    if idx-1 >= 0:
        if precio_ant <= ema_ant and precio > ema:
            cruce = "¡CRUCE ALCISTA RECIENTE! Precio superó EMA20."
        elif precio_ant >= ema_ant and precio < ema:
            cruce = "¡CRUCE BAJISTA RECIENTE! Precio cayó debajo de EMA20."
    return relacion, rol, cruce, diff_pct

def analizar_velas_detallado(df, num_velas=7):
    ultimas = df.iloc[-num_velas-1:-1]
    analisis = []
    for i, (idx, vela) in enumerate(ultimas.iterrows()):
        cuerpo = abs(vela['close'] - vela['open'])
        rango = vela['high'] - vela['low']
        if rango > 0:
            cuerpo_pct = (cuerpo / rango) * 100
            mecha_sup_pct = ((vela['high'] - max(vela['close'], vela['open'])) / rango) * 100
            mecha_inf_pct = ((min(vela['close'], vela['open']) - vela['low']) / rango) * 100
        else:
            cuerpo_pct = mecha_sup_pct = mecha_inf_pct = 0
        color = "VERDE" if vela['close'] >= vela['open'] else "ROJA"
        analisis.append(f"Vela{i+1}: {color} | Cuerpo:{cuerpo_pct:.0f}% | MechaSup:{mecha_sup_pct:.0f}% | MechaInf:{mecha_inf_pct:.0f}% | Cierre:{vela['close']:.2f}")
    return "\n".join(analisis)

def analizar_ema_cruce(df):
    prev_c = df['close'].iloc[-3]; prev_e = df['ema20'].iloc[-3]
    curr_c = df['close'].iloc[-2]; curr_e = df['ema20'].iloc[-2]
    if prev_c > prev_e and curr_c < curr_e:
        return "CRUCE BAJISTA (precio debajo EMA20)"
    elif prev_c < prev_e and curr_c > curr_e:
        return "CRUCE ALCISTA (precio encima EMA20)"
    return f"Precio {'encima' if curr_c>curr_e else 'debajo'} EMA20 (dist:{abs(curr_c-curr_e):.1f})"

def analizar_rechazo_resistencia(df, resistencia):
    for i in range(-4, 1):
        high = df['high'].iloc[i]
        if high >= resistencia * 0.998:
            rango = df['high'].iloc[i] - df['low'].iloc[i]
            mecha = high - max(df['close'].iloc[i], df['open'].iloc[i])
            if rango > 0 and (mecha/rango) > 0.3:
                return f"RECHAZO en resistencia ({resistencia:.0f}) con mecha superior del {(mecha/rango)*100:.0f}%"
    return "Sin rechazo claro"

def analizar_macd_rsi(df, idx=-2):
    rsi = df['rsi'].iloc[idx]
    macd = df['macd'].iloc[idx]
    signal = df['signal'].iloc[idx]
    hist = df['macd_hist'].iloc[idx]
    # Tendencia MACD
    if macd > signal and hist > 0:
        tend_macd = "ALCISTA (MACD arriba de señal, histograma positivo)"
    elif macd < signal and hist < 0:
        tend_macd = "BAJISTA (MACD debajo de señal, histograma negativo)"
    elif macd > signal and hist < 0:
        tend_macd = "POSIBLE CRUCE ALCISTA (divergencia positiva)"
    else:
        tend_macd = "NEUTRAL o debilitamiento"
    # RSI
    if rsi > 70:
        rsi_estado = "SOBRECOMPRADO (sobre 70) - posible retroceso"
    elif rsi < 30:
        rsi_estado = "SOBREVENDIDO (bajo 30) - posible rebote"
    else:
        rsi_estado = f"NEUTRAL ({rsi:.1f})"
    return rsi, rsi_estado, tend_macd, hist

# ======================================================
# AUTOAPRENDIZAJE CADA 10 TRADES
# ======================================================
def aprender_de_trades():
    global ADAPTIVE_BIAS, ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TP2_MULT, ADAPTIVE_TRAILING_MULT
    global TRADES_SIN_APRENDER, ULTIMO_APRENDIZAJE
    
    if len(TRADE_HISTORY) < 10:
        return
    
    # Solo aprender si han pasado al menos 10 trades desde el último aprendizaje
    if ULTIMO_APRENDIZAJE is not None and len(TRADE_HISTORY) - ULTIMO_APRENDIZAJE < 10:
        return
    
    # Analizar últimos 10 trades
    ultimos = TRADE_HISTORY[-10:]
    wins = [t for t in ultimos if t['resultado_win']]
    losses = [t for t in ultimos if not t['resultado_win']]
    winrate = len(wins) / 10.0
    
    # Analizar decisiones erróneas comunes
    razones_loss = []
    for loss in losses:
        razones_loss.extend(loss.get('razones_ia', []))
    # Contar frecuencias
    counter = Counter(razones_loss)
    errores_comunes = counter.most_common(3)
    
    # Ajustar sesgo adaptativo
    if winrate < 0.4:
        # Si ganamos poco, reducir sesgo y ser más conservadores
        ADAPTIVE_BIAS = max(-0.2, ADAPTIVE_BIAS - 0.05)
        ADAPTIVE_SL_MULT = min(2.5, ADAPTIVE_SL_MULT * 1.1)  # SL más lejano (menos riesgo de salir temprano)
        ADAPTIVE_TP1_MULT = max(1.5, ADAPTIVE_TP1_MULT * 0.9)  # TP más cerca
    elif winrate > 0.6:
        # Si ganamos bien, mantener o aumentar ligeramente confianza
        ADAPTIVE_BIAS = min(0.3, ADAPTIVE_BIAS + 0.02)
        ADAPTIVE_SL_MULT = max(1.0, ADAPTIVE_SL_MULT * 0.95)
        ADAPTIVE_TP1_MULT = min(4.0, ADAPTIVE_TP1_MULT * 1.05)
    else:
        # Estable
        pass
    
    # Análisis detallado para enviar a Telegram
    mensaje = f"📚 AUTOAPRENDIZAJE (últimos 10 trades)\nWinrate: {winrate*100:.1f}%\nErrores comunes: {errores_comunes}\nNuevo sesgo: {ADAPTIVE_BIAS:.2f}\nSL mult: {ADAPTIVE_SL_MULT:.2f}\nTP1 mult: {ADAPTIVE_TP1_MULT:.2f}"
    telegram_mensaje(mensaje)
    print(mensaje)
    
    ULTIMO_APRENDIZAJE = len(TRADE_HISTORY)

# ======================================================
# IA GROQ CON ANÁLISIS VISUAL ENRIQUECIDO Y SANITIZACIÓN
# ======================================================
def analizar_con_groq_texto(df, soporte, resistencia, tendencia, slope, intercept, idx=-2):
    precio = df['close'].iloc[idx]
    atr = df['atr'].iloc[idx]
    relacion_ema, rol_ema, cruce_ema, diff_ema_pct = analizar_posicion_respecto_ema(df, idx)
    rsi, rsi_estado, tend_macd, hist_macd = analizar_macd_rsi(df, idx)
    analisis_velas = analizar_velas_detallado(df)
    analisis_ema = analizar_ema_cruce(df)
    analisis_rechazo = analizar_rechazo_resistencia(df, resistencia)
    
    sesgo_texto = ""
    if ADAPTIVE_BIAS > 0.05:
        sesgo_texto = f" (sesgo adaptativo actual hacia BUY: +{ADAPTIVE_BIAS:.2f})"
    elif ADAPTIVE_BIAS < -0.05:
        sesgo_texto = f" (sesgo adaptativo actual hacia SELL: {ADAPTIVE_BIAS:.2f})"
    
    prompt = f"""
Eres un trader institucional con amplia experiencia en BTCUSDT en gráfico de 5 minutos. Analiza TODO el contexto como si estuvieras viendo la imagen en tiempo real. Tu análisis debe ser humano, considerando la estructura de mercado, la EMA20 como posible soporte/resistencia, el MACD, el RSI y el patrón de velas.

DATOS ACTUALES (última vela cerrada):
- Precio: {precio:.2f}
- ATR (volatilidad): {atr:.2f}
- Soporte horizontal: {soporte:.2f}
- Resistencia horizontal: {resistencia:.2f}
- Tendencia macro (120 velas): {tendencia}, pendiente: {slope:.6f}
- RSI: {rsi:.1f} - {rsi_estado}
- MACD: {tend_macd}, histograma: {hist_macd:.2f}
- EMA20: {df['ema20'].iloc[idx]:.2f}
- Posición respecto a EMA20: {relacion_ema}. {rol_ema} {cruce_ema}
- {analisis_ema}
- {analisis_rechazo}

VELAS RECIENTES (últimas 7):
{analisis_velas}

INSTRUCCIONES:
1. Decide "Buy", "Sell" o "Hold". Sé realista: si el precio está exactamente en la EMA20 y ésta ha actuado como soporte/resistencia, espera confirmación (vela de rechazo o quiebre). No entres sin contexto.
2. Si decides Buy o Sell, justifica con al menos 2 razones claras basadas en los datos (ej: "precio rebotó en soporte + MACD alcista", "rechazo en resistencia + RSI sobrecomprado").
3. Recomienda multiplicadores para SL, TP1, TP2 y trailing basados en el ATR actual y la volatilidad. Sé conservador: SL entre 1.0x y 2.5x ATR, TP1 entre 2.0x y 4.0x, TP2 entre 3.0x y 6.0x, trailing entre 1.5x y 3.0x.
4. TP1 será fijo y cerrará el 50% de la posición. TP2 será dinámico con trailing stop después de alcanzar TP1.
5. IMPORTANTE: Si el mercado está lateral o sin dirección clara, prefiere "Hold".

Responde SOLO en JSON con este formato:
{{
  "decision": "Buy/Sell/Hold",
  "patron": "nombre del patrón detectado (ej: 'Rebote en soporte', 'Rechazo en resistencia', 'Quiebre de EMA20', 'Doji en zona clave')",
  "razones": ["razón1","razón2","razón3"],
  "sl_mult": 1.5,
  "tp1_mult": 2.5,
  "tp2_mult": 4.0,
  "trailing_mult": 2.0
}}
Si es Hold, los multiplicadores pueden ser nulos o por defecto.
"""
    try:
        completion = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.15
        )
        datos = json.loads(completion.choices[0].message.content)
        
        # ========== SANITIZACIÓN: reemplazar None por valores por defecto ==========
        def sanitize(value, default):
            if value is None or not isinstance(value, (int, float)):
                return default
            return value
        
        sl_mult_raw = datos.get("sl_mult", DEFAULT_SL_MULT)
        tp1_mult_raw = datos.get("tp1_mult", DEFAULT_TP1_MULT)
        tp2_mult_raw = datos.get("tp2_mult", DEFAULT_TP2_MULT)
        trailing_mult_raw = datos.get("trailing_mult", DEFAULT_TRAILING_MULT)
        
        sl_mult = sanitize(sl_mult_raw, DEFAULT_SL_MULT)
        tp1_mult = sanitize(tp1_mult_raw, DEFAULT_TP1_MULT)
        tp2_mult = sanitize(tp2_mult_raw, DEFAULT_TP2_MULT)
        trailing_mult = sanitize(trailing_mult_raw, DEFAULT_TRAILING_MULT)
        
        # Asegurar que razones y patron sean strings válidas
        if not isinstance(datos.get("razones"), list):
            datos["razones"] = ["Análisis no disponible"]
        if not isinstance(datos.get("patron"), str):
            datos["patron"] = "Patrón no especificado"
        
        # Aplicar sesgo adaptativo (si la IA dijo Buy y sesgo es negativo, reconsiderar)
        if ADAPTIVE_BIAS > 0.1 and datos.get("decision") == "Sell":
            if np.random.random() < abs(ADAPTIVE_BIAS):
                datos["decision"] = "Hold"
                datos["razones"] = datos.get("razones", []) + ["Sesgo adaptativo anuló señal contraria"]
        elif ADAPTIVE_BIAS < -0.1 and datos.get("decision") == "Buy":
            if np.random.random() < abs(ADAPTIVE_BIAS):
                datos["decision"] = "Hold"
                datos["razones"] = datos.get("razones", []) + ["Sesgo adaptativo anuló señal contraria"]
        
        # Mezclar con adaptativos (media ponderada) - ahora seguros de que no son None
        datos["sl_mult"] = 0.7 * sl_mult + 0.3 * ADAPTIVE_SL_MULT
        datos["tp1_mult"] = 0.7 * tp1_mult + 0.3 * ADAPTIVE_TP1_MULT
        datos["tp2_mult"] = 0.7 * tp2_mult + 0.3 * ADAPTIVE_TP2_MULT
        datos["trailing_mult"] = 0.7 * trailing_mult + 0.3 * ADAPTIVE_TRAILING_MULT
        
        return datos
    except Exception as e:
        print(f"Error Groq: {e}")
        return {"decision": "Hold", "razones": ["Error API"], "patron": "", "sl_mult": DEFAULT_SL_MULT, "tp1_mult": DEFAULT_TP1_MULT, "tp2_mult": DEFAULT_TP2_MULT, "trailing_mult": DEFAULT_TRAILING_MULT}

# ======================================================
# GESTIÓN DE RIESGO Y PAPER TRADING
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
    global PAPER_SL_ACTUAL, PAPER_BALANCE
    global ULTIMA_RAZONES, ULTIMO_PATRON, ULTIMOS_MULTIS

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

    # Guardar razones para el historial
    ULTIMA_RAZONES = razones
    ULTIMO_PATRON = patron
    ULTIMOS_MULTIS = (sl_mult, tp1_mult, tp2_mult, trailing_mult)

    telegram_mensaje(f"📌 OPERACIÓN {decision.upper()} | Precio: {precio:.2f} | SL: {sl:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f} | Trailing mult: {trailing_mult}\nRazones: {' | '.join(razones[:2])}")
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

    # TP1 parcial
    if not PAPER_TP1_EJECUTADO:
        if (PAPER_POSICION_ACTIVA == "Buy" and high >= PAPER_TP1) or (PAPER_POSICION_ACTIVA == "Sell" and low <= PAPER_TP1):
            beneficio_parcial = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1) if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1)
            PAPER_BALANCE += beneficio_parcial
            PAPER_PNL_PARCIAL = beneficio_parcial
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE_TP1)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL_ACTUAL = PAPER_PRECIO_ENTRADA
            telegram_mensaje(f"🎯 TP1 alcanzado. Beneficio parcial: +{beneficio_parcial:.2f} USD. SL movido a break-even. Resta {PAPER_SIZE_BTC_RESTANTE:.4f} BTC")

    # Trailing después de TP1
    if PAPER_TP1_EJECUTADO:
        if PAPER_POSICION_ACTIVA == "Buy":
            nuevo_sl = close - (atr * PAPER_TRAILING_MULT)
            if nuevo_sl > PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔼 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f}")
            if low <= PAPER_SL_ACTUAL:
                cerrar_total = True
                motivo = "Trailing Stop"
        else:
            nuevo_sl = close + (atr * PAPER_TRAILING_MULT)
            if nuevo_sl < PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔽 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f}")
            if high >= PAPER_SL_ACTUAL:
                cerrar_total = True
                motivo = "Trailing Stop"
    else:
        # SL inicial
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

        # Guardar en historial para aprendizaje
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

        # Gráfico
        fig = generar_grafico_salida(df, PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, precio_salida, pnl_total, win_status, soporte, resistencia, slope, intercept)
        telegram_grafico(fig)
        plt.close(fig)

        PAPER_POSICION_ACTIVA = None
        return True
    return None

# ======================================================
# GRÁFICOS
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
    texto = f"GROQ V99.0: {decision.upper()}\nPatrón: {patron[:60]}\n{chr(10).join(razones[:2])}"
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
    print("🤖 BOT V99.0 INICIADO - IA con análisis contextual enriquecido y autoaprendizaje")
    telegram_mensaje("🤖 BOT V99.0 INICIADO: Análisis de EMA como soporte/resistencia, MACD + RSI + Autoaprendizaje cada 10 trades.")
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
                    if decision != "Hold":
                        print(f"Hold (riesgo o drawdown): {razones[0] if razones else 'Sin señal'}")
                    else:
                        print(f"Hold: {razones[0] if razones else 'Sin señal'}")

            if PAPER_POSICION_ACTIVA is not None:
                paper_revisar_sl_tp(df, soporte, resistencia, slope, intercept)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
