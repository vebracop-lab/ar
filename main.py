# BOT TRADING V99.12 – GROQ (llama-3.3-70b-versatile) con DESCRIPCIÓN TEXTUAL ENRIQUECIDA
# ===================================================================================
# - El modelo NO ve imágenes, pero recibe una descripción detallada del gráfico (velas,
#   patrones, EMA, RSI, MACD, soportes/resistencias, tendencia, etc.).
# - Prompt inspirado en Steve Nison: análisis holístico, patrones de reversión y continuación.
# - Autoaprendizaje cada 10 trades, logs completos, gráficos en Telegram.
# ====================================================================================

import os, time, requests, json, re, numpy as np, pandas as pd
from scipy.stats import linregress
from datetime import datetime, timezone
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from groq import Groq

# =================== CONFIGURACIÓN ===================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Falta GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODELO_TEXTO = "llama-3.3-70b-versatile"      # Modelo puramente textual (rápido y potente)

SYMBOL = "BTCUSDT"
INTERVAL = "5"
RISK_PER_TRADE = 0.02
LEVERAGE = 10
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120

DEFAULT_SL_MULT = 1.5
DEFAULT_TP1_MULT = 1.6
DEFAULT_TRAILING_MULT = 1.8
PORCENTAJE_CIERRE_TP1 = 0.5

# =================== PAPER TRADING (SIMULADO) ===================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_POSICION_ACTIVA = None
PAPER_PRECIO_ENTRADA = None
PAPER_SL_INICIAL = None
PAPER_TP1 = None
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
ADAPTIVE_TRAILING_MULT = DEFAULT_TRAILING_MULT
ULTIMO_APRENDIZAJE = None

ULTIMA_DECISION = "Hold"
ULTIMO_MOTIVO = "Esperando señal"
ULTIMA_RAZONES = []
ULTIMO_PATRON = ""

# =================== TELEGRAM ===================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e:
        print(f"Error Telegram: {e}")

def telegram_enviar_imagen(ruta_imagen, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        with open(ruta_imagen, 'rb') as foto:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                          data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                          files={"photo": foto}, timeout=15)
    except Exception as e:
        print(f"Error imagen: {e}")

# =================== DATOS Y TÉCNICO ===================
def obtener_velas(limit=150):
    r = requests.get(f"{BASE_URL}/v5/market/kline", params={"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}, timeout=20)
    data = r.json()["result"]["list"][::-1]
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
    return df.dropna()

def detectar_zonas_mercado(df, idx=-2, ventana_macro=120):
    df_eval = df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    y = df_eval['close'].values[-ventana_macro:] if len(df_eval) >= ventana_macro else df_eval['close'].values
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    tendencia = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tendencia

# =================== DESCRIPCIÓN TEXTUAL DEL GRÁFICO (el "análisis visual" en palabras) ===================
def analizar_posicion_respecto_ema(df, idx=-2):
    precio = df['close'].iloc[idx]
    ema = df['ema20'].iloc[idx]
    diff_pct = (precio - ema) / ema * 100
    # Contar toques en las últimas 6 velas
    toques = 0
    for i in range(max(0, idx-5), idx+1):
        if df['low'].iloc[i] <= df['ema20'].iloc[i] <= df['high'].iloc[i]:
            toques += 1
    if abs(diff_pct) < 0.15:
        pos_texto = "PRECIO JUSTO EN LA EMA20 (tocando exactamente)"
    elif precio > ema:
        pos_texto = f"PRECIO ENCIMA DE EMA20 (+{diff_pct:.2f}%)"
    else:
        pos_texto = f"PRECIO DEBAJO DE EMA20 ({diff_pct:.2f}%)"
    return pos_texto, toques

def detectar_patron_vela(open_, high, low, close):
    cuerpo = abs(close - open_)
    rango = high - low
    if rango == 0: return "Vela indeterminada"
    cuerpo_pct = cuerpo / rango * 100
    mecha_sup = high - max(close, open_)
    mecha_inf = min(close, open_) - low
    mecha_sup_pct = mecha_sup / rango * 100
    mecha_inf_pct = mecha_inf / rango * 100
    es_verde = close >= open_
    if mecha_sup_pct > 60 and cuerpo_pct < 30 and mecha_inf_pct < 10:
        return "ESTRELLA FUGAZ (reversión bajista)" if not es_verde else "MARTILLO INVERTIDO"
    if mecha_inf_pct > 60 and cuerpo_pct < 30 and mecha_sup_pct < 10:
        return "MARTILLO (reversión alcista)" if es_verde else "HOMBRE COLGADO"
    if cuerpo_pct < 10:
        return "DOJI (indecisión)"
    if cuerpo_pct > 70 and mecha_sup_pct < 15 and mecha_inf_pct < 15:
        return "VELA LARGA SIN MECHAS (impulso fuerte)"
    if mecha_sup_pct > 50 and cuerpo_pct < 50:
        return "RECHAZO EN ZONA ALTA (mecha superior larga)"
    if mecha_inf_pct > 50 and cuerpo_pct < 50:
        return "REBOTE EN ZONA BAJA (mecha inferior larga)"
    return f"Vela normal ({cuerpo_pct:.0f}% cuerpo, Msup {mecha_sup_pct:.0f}%, Minf {mecha_inf_pct:.0f}%)"

def generar_descripcion_grafica(df, idx=-2):
    """Devuelve un texto detallado con toda la información del gráfico."""
    precio = df['close'].iloc[idx]
    atr = df['atr'].iloc[idx]
    ema_val = df['ema20'].iloc[idx]
    rsi = df['rsi'].iloc[idx]
    macd = df['macd'].iloc[idx]
    signal = df['signal'].iloc[idx]
    hist = df['macd_hist'].iloc[idx]
    soporte, resistencia, slope, intercept, tendencia = detectar_zonas_mercado(df, idx)
    pos_ema, toques_ema = analizar_posicion_respecto_ema(df, idx)

    # Interpretación RSI
    if rsi > 70: rsi_texto = f"SOBRECOMPRADO (sobre 70) - posible retroceso"
    elif rsi < 30: rsi_texto = f"SOBREVENDIDO (bajo 30) - posible rebote"
    else: rsi_texto = f"NEUTRAL ({rsi:.1f})"

    # Interpretación MACD
    if macd > signal and hist > 0:
        macd_texto = "ALCISTA (MACD arriba de señal, histograma positivo)"
    elif macd < signal and hist < 0:
        macd_texto = "BAJISTA (MACD debajo de señal, histograma negativo)"
    elif macd > signal and hist < 0:
        macd_texto = "POSIBLE CRUCE ALCISTA (divergencia positiva)"
    else:
        macd_texto = "NEUTRAL o debilitamiento"

    # Análisis de velas (últimas 8)
    velas_desc = []
    for i in range(max(0, len(df)-9), len(df)-1):
        v = df.iloc[i]
        patron = detectar_patron_vela(v['open'], v['high'], v['low'], v['close'])
        velas_desc.append(f"Vela {i+1}: {patron} | Cierre: {v['close']:.2f} | Rango: {v['high']-v['low']:.2f}")

    # Detección de techos/soportes por rechazos
    df_cercano = df.iloc[-20:]
    toques_res = sum(1 for _, v in df_cercano.iterrows() if v['high'] >= resistencia * 0.998)
    toques_sop = sum(1 for _, v in df_cercano.iterrows() if v['low'] <= soporte * 1.002)
    estructura = ""
    if toques_res >= 3:
        estructura += f"⚠️ ZONA DE RESISTENCIA ({resistencia:.0f}) RECHAZADA {toques_res} VECES. Se ha formado un TECHO SÓLIDO. "
    if toques_sop >= 3:
        estructura += f"✅ ZONA DE SOPORTE ({soporte:.0f}) PROBADA {toques_sop} VECES. Posible rebote. "

    # Detectar ruptura de tendencia
    if len(df) > 20:
        pendiente_reciente, _, _, _, _ = linregress(np.arange(20), df['close'].iloc[-20:].values)
        ruptura = ""
        if slope > 0 and pendiente_reciente < 0:
            ruptura = "⚠️ ¡RUPTURA DE TENDENCIA ALCISTA! La pendiente reciente es negativa mientras que la macro era alcista."
        elif slope < 0 and pendiente_reciente > 0:
            ruptura = "✅ ¡RUPTURA DE TENDENCIA BAJISTA! La pendiente reciente es positiva."
    else:
        ruptura = ""

    descripcion = f"""
=== CONTEXTO GENERAL ===
- Precio actual: {precio:.2f}
- ATR (volatilidad): {atr:.2f}
- Soporte horizontal: {soporte:.2f}
- Resistencia horizontal: {resistencia:.2f}
- Tendencia macro (120 velas): {tendencia} (pendiente {slope:.6f})
{ruptura}
- RSI (14): {rsi:.1f} - {rsi_texto}
- MACD: {macd_texto} (histograma {hist:.2f})
- EMA20: {ema_val:.2f} - {pos_ema} (toques en últimas 6 velas: {toques_ema})

=== ESTRUCTURA DE MERCADO ===
{estructura}

=== ANÁLISIS DE VELAS (últimas 8, cerradas) ===
{chr(10).join(velas_desc)}

=== PARÁMETROS ADAPTATIVOS ===
- Sesgo adaptativo: {ADAPTIVE_BIAS:+.2f}
- Multiplicador SL: {ADAPTIVE_SL_MULT:.2f}
- Multiplicador TP1: {ADAPTIVE_TP1_MULT:.2f}
- Multiplicador trailing: {ADAPTIVE_TRAILING_MULT:.2f}
"""
    return descripcion

# =================== IA GROQ (solo texto) ===================
def analizar_con_groq_texto(descripcion_grafica):
    try:
        # Mensaje de sistema para fijar la personalidad de Steve Nison
        system_msg = """
        Eres Steve Nison, el mayor experto mundial en velas japonesas y análisis técnico.
        Actúas como un trader institucional experimentado. Tu análisis es multifacético,
        considerando patrones de velas (reversión y continuación), estructura de mercado,
        soportes/resistencias dinámicas, tendencia y confluencia de señales.
        Eres decidido: si hay al menos dos señales a favor, tomas posición (Buy/Sell).
        No esperas confirmación perfecta. El mercado rara vez es ideal.
        """

        user_msg = f"""
        Basándote en la siguiente descripción detallada del gráfico de BTCUSDT en 5 minutos,
        decide si debes comprar (Buy), vender (Sell) o esperar (Hold).

        {descripcion_grafica}

        Responde ÚNICAMENTE con un JSON en este formato:
        {{
          "decision": "Buy/Sell/Hold",
          "patron": "nombre del patrón o situación clave",
          "razones": ["razón1", "razón2", "razón3"]
        }}
        No incluyas texto fuera del JSON.
        """

        respuesta = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=300
        )
        raw = respuesta.choices[0].message.content
        print(f"🔍 Respuesta Groq (texto):\n{raw}")
        # Extraer JSON
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        datos = json.loads(match.group(0) if match else raw)
        return datos.get("decision", "Hold"), datos.get("razones", []), datos.get("patron", ""), raw
    except Exception as e:
        print(f"❌ Error en Groq: {e}")
        return "Hold", ["Error en análisis"], "Error", ""

# =================== GRÁFICOS (para enviar a Telegram, sin análisis de IA) ===================
def generar_grafico_entrada(df, decision, razones, patron, soporte, resistencia, slope, intercept):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16,8))
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = 'green' if c >= o else 'red'
        ax.vlines(x[i], l, h, color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((x[i]-0.3, min(o,c)), 0.6, abs(c-o), color=color, alpha=0.9))
    ax.axhline(soporte, color='cyan', ls='--', label=f'Soporte: {soporte:.2f}')
    ax.axhline(resistencia, color='magenta', ls='--', label=f'Resistencia: {resistencia:.2f}')
    ax.plot(x, intercept + slope * x, 'w-.', label=f'Tendencia ({slope:.2e})')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'y', label='EMA20')
    texto = f"GROQ V99.12 (texto) | Decisión: {decision.upper()}\nPatrón: {patron[:70]}\nRazones:\n" + "\n".join(razones[:3])
    ax.text(0.01, 0.99, texto, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.85), color='white')
    entrada_x = len(df_plot)-2
    precio_entrada = df_plot['close'].iloc[-2]
    if decision == 'Buy':
        ax.scatter(entrada_x, precio_entrada-50, s=300, marker='^', c='lime', edgecolors='black')
    elif decision == 'Sell':
        ax.scatter(entrada_x, precio_entrada+50, s=300, marker='v', c='red', edgecolors='black')
    ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.tick_params(colors='white'); ax.grid(True, alpha=0.2); ax.legend(loc='upper left', facecolor='black', labelcolor='white')
    plt.tight_layout()
    ruta = "/tmp/chart_entry.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta

def generar_grafico_salida(df, posicion, precio_entrada, precio_salida, pnl, win, soporte, resistencia, slope, intercept):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16,8))
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = 'green' if c >= o else 'red'
        ax.vlines(x[i], l, h, color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((x[i]-0.3, min(o,c)), 0.6, abs(c-o), color=color, alpha=0.9))
    ax.axhline(soporte, color='cyan', ls='--', label=f'Soporte: {soporte:.2f}')
    ax.axhline(resistencia, color='magenta', ls='--', label=f'Resistencia: {resistencia:.2f}')
    ax.plot(x, intercept + slope * x, 'w-.', label=f'Tendencia ({slope:.2e})')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'y', label='EMA20')
    ax.axhline(precio_entrada, color='blue', ls=':', label=f'Entrada: {precio_entrada:.2f}')
    ax.axhline(precio_salida, color='orange', ls=':', label=f'Salida: {precio_salida:.2f}')
    estado = "WIN" if win else "LOSS"
    texto = f"RESULTADO: {estado}\n{posicion.upper()} | PnL: {pnl:.2f} USD"
    ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.85), color='white')
    ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.tick_params(colors='white'); ax.grid(True, alpha=0.2); ax.legend(loc='upper left', facecolor='black', labelcolor='white')
    plt.tight_layout()
    ruta = "/tmp/chart_exit.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta

# =================== AUTOAPRENDIZAJE ===================
def aprender_de_trades():
    global ADAPTIVE_BIAS, ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TRAILING_MULT, ULTIMO_APRENDIZAJE
    if len(TRADE_HISTORY) < 10 or (ULTIMO_APRENDIZAJE and len(TRADE_HISTORY) - ULTIMO_APRENDIZAJE < 10):
        return
    ultimos = TRADE_HISTORY[-10:]
    winrate = sum(1 for t in ultimos if t['resultado_win']) / 10.0
    if winrate < 0.4:
        ADAPTIVE_BIAS = max(-0.2, ADAPTIVE_BIAS - 0.05)
        ADAPTIVE_SL_MULT = min(2.5, ADAPTIVE_SL_MULT * 1.1)
        ADAPTIVE_TP1_MULT = max(1.2, ADAPTIVE_TP1_MULT * 0.95)
        ADAPTIVE_TRAILING_MULT = max(1.5, ADAPTIVE_TRAILING_MULT * 0.95)
    elif winrate > 0.6:
        ADAPTIVE_BIAS = min(0.3, ADAPTIVE_BIAS + 0.02)
        ADAPTIVE_SL_MULT = max(1.0, ADAPTIVE_SL_MULT * 0.95)
        ADAPTIVE_TP1_MULT = min(2.2, ADAPTIVE_TP1_MULT * 1.03)
        ADAPTIVE_TRAILING_MULT = min(2.5, ADAPTIVE_TRAILING_MULT * 1.02)
    msg = f"📚 AUTOAPRENDIZAJE\nWinrate: {winrate*100:.1f}%\nSesgo: {ADAPTIVE_BIAS:.2f}\nSL: {ADAPTIVE_SL_MULT:.2f}\nTP1: {ADAPTIVE_TP1_MULT:.2f}\nTrail: {ADAPTIVE_TRAILING_MULT:.2f}"
    telegram_mensaje(msg)
    print(msg)
    ULTIMO_APRENDIZAJE = len(TRADE_HISTORY)

# =================== GESTIÓN DE RIESGO Y PAPER TRADING ===================
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY, PAPER_BALANCE
    hoy = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY = hoy
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
    dd = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if dd <= -MAX_DAILY_DRAWDOWN_PCT and not PAPER_STOPPED_TODAY:
        telegram_mensaje("🛑 Drawdown diario máximo. Bot pausado.")
        PAPER_STOPPED_TODAY = True
        return False
    return not PAPER_STOPPED_TODAY

def paper_abrir_posicion(decision, precio, atr, razones, patron):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL, PAPER_BALANCE, ULTIMA_DECISION, ULTIMO_MOTIVO
    if PAPER_POSICION_ACTIVA: return False
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    sl = precio - (atr * ADAPTIVE_SL_MULT) if decision == "Buy" else precio + (atr * ADAPTIVE_SL_MULT)
    tp1 = precio + (atr * ADAPTIVE_TP1_MULT) if decision == "Buy" else precio - (atr * ADAPTIVE_TP1_MULT)
    distancia = abs(precio - sl)
    if distancia == 0: return False
    size_usd = min((riesgo_usd / distancia) * precio, PAPER_BALANCE * LEVERAGE)
    size_btc = size_usd / precio
    PAPER_POSICION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL_INICIAL = sl
    PAPER_TP1 = tp1
    PAPER_TRAILING_MULT = ADAPTIVE_TRAILING_MULT
    PAPER_SIZE_BTC = size_btc
    PAPER_SIZE_BTC_RESTANTE = size_btc
    PAPER_TP1_EJECUTADO = False
    PAPER_SL_ACTUAL = sl
    msg = f"📌 {decision.upper()} | Entrada: {precio:.2f} | SL: {sl:.2f} | TP1: {tp1:.2f} | RR: {ADAPTIVE_TP1_MULT/ADAPTIVE_SL_MULT:.2f}\nRazones: {' | '.join(razones[:2])}"
    telegram_mensaje(msg)
    ULTIMA_DECISION = decision
    ULTIMO_MOTIVO = razones[0][:50]
    print(f"🚀 {decision} a {precio:.2f}")
    return True

def paper_revisar_sl_tp(df, soporte, resistencia, slope, intercept):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL, PAPER_BALANCE, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES, PAPER_LAST_10_PNL, TRADE_HISTORY, ULTIMA_RAZONES, ULTIMO_PATRON
    if PAPER_POSICION_ACTIVA is None: return None
    high, low, close, atr = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1], df['atr'].iloc[-1]
    cerrar, motivo = False, ""
    if not PAPER_TP1_EJECUTADO:
        if (PAPER_POSICION_ACTIVA == "Buy" and high >= PAPER_TP1) or (PAPER_POSICION_ACTIVA == "Sell" and low <= PAPER_TP1):
            beneficio = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1) if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1)
            PAPER_BALANCE += beneficio
            PAPER_PNL_PARCIAL = beneficio
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE_TP1)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL_ACTUAL = PAPER_PRECIO_ENTRADA
            telegram_mensaje(f"🎯 TP1 alcanzado | Beneficio parcial: +{beneficio:.2f} USD | SL a break-even | Restan {PAPER_SIZE_BTC_RESTANTE:.4f} BTC")
    if PAPER_TP1_EJECUTADO:
        if PAPER_POSICION_ACTIVA == "Buy":
            nuevo_sl = close - (atr * PAPER_TRAILING_MULT)
            if nuevo_sl > PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔼 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f}")
            if low <= PAPER_SL_ACTUAL:
                cerrar, motivo = True, "Trailing Stop"
        else:
            nuevo_sl = close + (atr * PAPER_TRAILING_MULT)
            if nuevo_sl < PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔽 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f}")
            if high >= PAPER_SL_ACTUAL:
                cerrar, motivo = True, "Trailing Stop"
    else:
        if (PAPER_POSICION_ACTIVA == "Buy" and low <= PAPER_SL_INICIAL) or (PAPER_POSICION_ACTIVA == "Sell" and high >= PAPER_SL_INICIAL):
            cerrar, motivo = True, "Stop Loss Inicial"
            PAPER_SL_ACTUAL = PAPER_SL_INICIAL
    if cerrar:
        salida = PAPER_SL_ACTUAL
        pnl_rest = (salida - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - salida) * PAPER_SIZE_BTC_RESTANTE
        pnl_total = PAPER_PNL_PARCIAL + pnl_rest
        PAPER_BALANCE += pnl_rest
        PAPER_TRADES_TOTALES += 1
        win = pnl_total > 0
        PAPER_WIN += 1 if win else 0
        PAPER_LOSS += 1 if not win else 0
        winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0
        TRADE_HISTORY.append({"fecha": datetime.now(timezone.utc).isoformat(), "decision": PAPER_POSICION_ACTIVA, "precio_entrada": PAPER_PRECIO_ENTRADA, "precio_salida": salida, "pnl": pnl_total, "razones_ia": ULTIMA_RAZONES, "patron": ULTIMO_PATRON, "resultado_win": win})
        if len(TRADE_HISTORY) % 10 == 0:
            aprender_de_trades()
        msg = f"📤 CERRADO ({motivo})\n{PAPER_POSICION_ACTIVA.upper()} | Entrada: {PAPER_PRECIO_ENTRADA:.2f} | Salida: {salida:.2f} | PnL: {pnl_total:.2f} USD | Balance: {PAPER_BALANCE:.2f} USD | Winrate: {winrate:.1f}%"
        telegram_mensaje(msg)
        print(msg)
        ruta_grafico = generar_grafico_salida(df, PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, salida, pnl_total, win, soporte, resistencia, slope, intercept)
        telegram_enviar_imagen(ruta_grafico, f"Cierre: {'WIN' if win else 'LOSS'}")
        PAPER_POSICION_ACTIVA = None
        return True
    return None

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON
    print("🤖 BOT V99.12 INICIADO - Groq con descripción textual enriquecida (llama-3.3-70b-versatile)")
    telegram_mensaje("🤖 BOT V99.12 INICIADO - Análisis textual del gráfico como Steve Nison")
    ultima_vela = None
    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            precio = df['close'].iloc[-1]
            vela_cerrada = df.index[-2]
            atr = df['atr'].iloc[-1]
            soporte, resistencia, slope, intercept, tendencia = detectar_zonas_mercado(df)
            pnl_global = PAPER_BALANCE - PAPER_BALANCE_INICIAL
            drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE * 100
            winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0
            print(f"\n💓 Heartbeat | Precio: {precio:.2f} | Sop: {soporte:.2f} | Res: {resistencia:.2f} | Trades: {PAPER_TRADES_TOTALES} | PnL: {pnl_global:+.2f} | Winrate: {winrate:.1f}% | Drawdown: {drawdown:.2f}% | Decisión: {ULTIMA_DECISION} - {ULTIMO_MOTIVO[:50]}")
            if PAPER_POSICION_ACTIVA is None and ultima_vela != vela_cerrada:
                # Generar descripción textual del gráfico
                descripcion = generar_descripcion_grafica(df)
                print(f"📝 Descripción enviada a Groq:\n{descripcion[:500]}...")  # Log parcial
                decision, razones, patron, raw = analizar_con_groq_texto(descripcion)
                ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON = decision, razones[0] if razones else "", razones, patron
                if decision in ["Buy","Sell"] and risk_management_check():
                    if paper_abrir_posicion(decision, precio, atr, razones, patron):
                        ultima_vela = vela_cerrada
                        # Gráfico de entrada (solo para registro visual, la IA ya usó la descripción)
                        ruta_entrada = generar_grafico_entrada(df, decision, razones, patron, soporte, resistencia, slope, intercept)
                        caption = f"🚀 Señal {decision} (Groq-texto)\nPatrón: {patron}\n" + "\n".join(razones[:2])
                        telegram_enviar_imagen(ruta_entrada, caption)
                else:
                    print(f"⏸️ Hold: {ULTIMO_MOTIVO}")
            if PAPER_POSICION_ACTIVA:
                paper_revisar_sl_tp(df, soporte, resistencia, slope, intercept)
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
