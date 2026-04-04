# BOT TRADING V99.4 BYBIT REAL – GROQ IA VISION + AUTOAPRENDIZAJE
# ================================================================
# - IA multimodal con análisis directo de la imagen del gráfico (Llama 4 Scout)
# - Autoaprendizaje cada 10 trades (ajuste de sesgo y multiplicadores SL/TP)
# - Logs detallados en consola y Telegram
# ================================================================

import os
import time
import io
import requests
import json
import numpy as np
import pandas as pd
import base64
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
MODELO_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"

SYMBOL = "BTCUSDT"
INTERVAL = "5"
RISK_PER_TRADE = 0.02
LEVERAGE = 10
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120

# Multiplicadores base (se ajustan con aprendizaje)
DEFAULT_SL_MULT = 1.5
DEFAULT_TP1_MULT = 2.5
DEFAULT_TP2_MULT = 4.0
DEFAULT_TRAILING_MULT = 2.0
PORCENTAJE_CIERRE_TP1 = 0.5

# ======================================================
# PAPER TRADING (SIMULADO)
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

# Historial para aprendizaje
TRADE_HISTORY = []  # Guarda cada trade con detalles

# Control de drawdown diario
MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

# Parámetros adaptativos (aprendizaje)
ADAPTIVE_BIAS = 0.0          # Rango -0.3 a +0.3, sesgo para Buy (+) o Sell (-)
ADAPTIVE_SL_MULT = DEFAULT_SL_MULT
ADAPTIVE_TP1_MULT = DEFAULT_TP1_MULT
ADAPTIVE_TP2_MULT = DEFAULT_TP2_MULT
ADAPTIVE_TRAILING_MULT = DEFAULT_TRAILING_MULT
ULTIMO_APRENDIZAJE = None

# Última decisión para mostrar en logs
ULTIMA_DECISION = "Hold"
ULTIMO_MOTIVO = "Esperando señal"

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

def telegram_grafico(fig, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        archivos = {'photo': buf}
        datos = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
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
# AUTOAPRENDIZAJE CADA 10 TRADES
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

    # Analizar errores comunes
    razones_loss = []
    for loss in [t for t in ultimos if not t['resultado_win']]:
        razones_loss.extend(loss.get('razones_ia', []))
    counter = Counter(razones_loss)
    errores_comunes = counter.most_common(3)

    # Ajustar sesgo y multiplicadores según rendimiento
    if winrate < 0.4:
        ADAPTIVE_BIAS = max(-0.2, ADAPTIVE_BIAS - 0.05)
        ADAPTIVE_SL_MULT = min(2.5, ADAPTIVE_SL_MULT * 1.1)
        ADAPTIVE_TP1_MULT = max(1.5, ADAPTIVE_TP1_MULT * 0.9)
    elif winrate > 0.6:
        ADAPTIVE_BIAS = min(0.3, ADAPTIVE_BIAS + 0.02)
        ADAPTIVE_SL_MULT = max(1.0, ADAPTIVE_SL_MULT * 0.95)
        ADAPTIVE_TP1_MULT = min(4.0, ADAPTIVE_TP1_MULT * 1.05)

    mensaje = (f"📚 AUTOAPRENDIZAJE (últimos 10 trades)\n"
               f"Winrate: {winrate*100:.1f}%\n"
               f"Errores comunes: {errores_comunes}\n"
               f"Nuevo sesgo: {ADAPTIVE_BIAS:.2f}\n"
               f"SL mult: {ADAPTIVE_SL_MULT:.2f}\n"
               f"TP1 mult: {ADAPTIVE_TP1_MULT:.2f}")
    telegram_mensaje(mensaje)
    print(mensaje)
    ULTIMO_APRENDIZAJE = len(TRADE_HISTORY)

# ======================================================
# IA GROQ CON VISIÓN (ANÁLISIS DIRECTO DE LA IMAGEN)
# ======================================================
def analizar_con_groq_vision(ruta_imagen):
    try:
        with open(ruta_imagen, "rb") as img_file:
            imagen_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        prompt = """
        Eres Steven Nison, el mayor experto mundial en velas japonesas y análisis técnico. Analiza la imagen del gráfico de BTCUSDT en tiempo real como si lo estuvieras viendo.
        Tu análisis debe ser directo, humano y contundente. Evalúa TODOS los siguientes puntos críticos:

        1.  **Patrones de Velas y Acción del Precio**: ¿Ves alguna vela clave (estrella fugaz, martillo, doji, envolvente)? ¿Hay mechas largas que indiquen rechazo en algún nivel?
        2.  **Estructura de Mercado**: ¿Identificas un techo sólido o un soporte fuerte por múltiples rechazos?
        3.  **Tendencia y Rupturas**: ¿El mercado está en tendencia alcista o bajista? ¿Se ha roto alguna línea de tendencia clave?
        4.  **Relación con la Media Móvil (EMA20)**: ¿El precio está por encima o por debajo? ¿La EMA actúa como soporte o resistencia?

        Basándote en tu análisis, decide tu acción: **Buy**, **Sell** o **Hold**.
        Si la decisión es Buy o Sell, explica los motivos claramente como lo haría un trader profesional.
        """

        respuesta = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagen_base64}"}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=500
        )

        analisis_completo = respuesta.choices[0].message.content
        print(f"🤖 Análisis de la IA:\n{analisis_completo}")

        # Determinar decisión
        decision = "Hold"
        if "BUY" in analisis_completo.upper():
            decision = "Buy"
        elif "SELL" in analisis_completo.upper():
            decision = "Sell"

        # Extraer razones (primeras líneas no vacías)
        lineas = [l.strip() for l in analisis_completo.split('\n') if l.strip()]
        razones = lineas[:3] if lineas else ["Análisis técnico"]
        patron = razones[0] if razones else "Patrón técnico"

        return decision, razones, patron, analisis_completo

    except Exception as e:
        print(f"❌ Error crítico en la IA de visión: {e}")
        return "Hold", ["Error en el análisis de la imagen"], "Error API", ""

# ======================================================
# GESTIÓN DE RIESGO Y PAPER TRADING CON MULTIPLICADORES ADAPTATIVOS
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

def paper_abrir_posicion(decision, precio, atr, razones, patron):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TP2
    global PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO
    global PAPER_SL_ACTUAL, PAPER_BALANCE, ULTIMA_DECISION, ULTIMO_MOTIVO

    if PAPER_POSICION_ACTIVA is not None:
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE

    # Usar multiplicadores adaptativos
    sl_mult = ADAPTIVE_SL_MULT
    tp1_mult = ADAPTIVE_TP1_MULT
    tp2_mult = ADAPTIVE_TP2_MULT
    trailing_mult = ADAPTIVE_TRAILING_MULT

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

    # Mensaje detallado a Telegram
    msg = (f"📌 OPERACIÓN {decision.upper()} iniciada\n"
           f"💰 Precio entrada: {precio:.2f}\n"
           f"🛑 Stop Loss: {sl:.2f}\n"
           f"🎯 TP1 (50%): {tp1:.2f}\n"
           f"🎯 TP2 (resto): {tp2:.2f}\n"
           f"📊 Trailing mult: {trailing_mult}\n"
           f"📝 Razones: {' | '.join(razones[:2])}")
    telegram_mensaje(msg)

    # Actualizar último motivo para heartbeat
    ULTIMA_DECISION = decision
    ULTIMO_MOTIVO = f"Entrada por: {razones[0][:50]}"

    print(f"🚀 {decision.upper()} ejecutado a {precio:.2f} | SL {sl:.2f} | TP1 {tp1:.2f} | TP2 {tp2:.2f}")
    return True

def paper_revisar_sl_tp(df):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TP2
    global PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO
    global PAPER_SL_ACTUAL, PAPER_BALANCE, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS
    global PAPER_TRADES_TOTALES, PAPER_LAST_10_PNL, TRADE_HISTORY

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
            telegram_mensaje(f"🎯 TP1 alcanzado en {PAPER_TP1:.2f}. Beneficio parcial: +{beneficio_parcial:.2f} USD. SL movido a break-even ({PAPER_PRECIO_ENTRADA:.2f}). Restan {PAPER_SIZE_BTC_RESTANTE:.4f} BTC.")

    # Trailing después de TP1
    if PAPER_TP1_EJECUTADO:
        if PAPER_POSICION_ACTIVA == "Buy":
            nuevo_sl = close - (atr * PAPER_TRAILING_MULT)
            if nuevo_sl > PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔼 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f} (distancia {atr * PAPER_TRAILING_MULT:.2f} pts)")
            if low <= PAPER_SL_ACTUAL:
                cerrar_total = True
                motivo = "Trailing Stop"
        else:
            nuevo_sl = close + (atr * PAPER_TRAILING_MULT)
            if nuevo_sl < PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔽 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f} (distancia {atr * PAPER_TRAILING_MULT:.2f} pts)")
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
                "razones_ia": [],  # Se llenará desde variable global si se guarda
                "patron": "",
                "sl_mult_usado": ADAPTIVE_SL_MULT,
                "tp1_mult_usado": ADAPTIVE_TP1_MULT,
                "tp2_mult_usado": ADAPTIVE_TP2_MULT,
                "trailing_mult_usado": ADAPTIVE_TRAILING_MULT,
                "resultado_win": win_status
            }
            # Intentar añadir razones si están disponibles (se guardan al abrir)
            if 'ULTIMA_RAZONES' in globals():
                trade_record["razones_ia"] = ULTIMA_RAZONES
                trade_record["patron"] = ULTIMO_PATRON
            TRADE_HISTORY.append(trade_record)
            if len(TRADE_HISTORY) % 10 == 0:
                aprender_de_trades()
        except Exception as e:
            print(f"Error guardando historial: {e}")

        msg_cierre = (f"📤 TRADE CERRADO ({motivo})\n"
                      f"Dirección: {PAPER_POSICION_ACTIVA.upper()}\n"
                      f"Entrada: {PAPER_PRECIO_ENTRADA:.2f}\n"
                      f"Salida: {precio_salida:.2f}\n"
                      f"PnL total: {pnl_total:.2f} USD\n"
                      f"Balance: {PAPER_BALANCE:.2f} USD\n"
                      f"Winrate global: {winrate:.1f}%")
        telegram_mensaje(msg_cierre)
        print(msg_cierre)

        # Resetear posición
        PAPER_POSICION_ACTIVA = None
        return True
    return None

# ======================================================
# GRÁFICOS (VERSIÓN MEJORADA PARA ANÁLISIS)
# ======================================================
def generar_y_guardar_grafico(df, decision=None, razones=None, patron=None, es_entrada=True):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16, 8))

    # Velas
    for i in range(len(df_plot)):
        color = 'green' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else 'red'
        ax.vlines(x[i], df_plot['low'].iloc[i], df_plot['high'].iloc[i], color=color, linewidth=1)
        y_cuerpo = min(df_plot['open'].iloc[i], df_plot['close'].iloc[i])
        altura = abs(df_plot['close'].iloc[i] - df_plot['open'].iloc[i])
        ax.add_patch(plt.Rectangle((x[i]-0.3, y_cuerpo), 0.6, max(altura, 0.01), color=color, alpha=0.9))

    # Niveles
    soporte, resistencia, slope, intercept, tendencia = detectar_zonas_mercado(df)
    ax.axhline(soporte, color='cyan', ls='--', lw=1.5, label=f'Soporte: {soporte:.2f}')
    ax.axhline(resistencia, color='magenta', ls='--', lw=1.5, label=f'Resistencia: {resistencia:.2f}')
    x_tend = np.arange(len(df_plot))
    y_tend = intercept + slope * x_tend
    ax.plot(x_tend, y_tend, 'w-.', lw=1.5, label=f'Tendencia (pendiente {slope:.2e})')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'y', lw=2, label='EMA20')

    # Anotaciones de IA si es entrada
    if es_entrada and decision and razones and patron:
        texto_ia = f"GROQ VISION V99.4 | Decisión: {decision.upper()}\nPatrón: {patron[:70]}\n"
        texto_ia += "Razones:\n" + "\n".join(razones[:3])
        ax.text(0.01, 0.99, texto_ia, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.85),
                color='white', family='monospace')

    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', facecolor='black', labelcolor='white')
    plt.tight_layout(pad=2.0)

    ruta = "/tmp/chart_analysis.png"
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ruta

# ======================================================
# LOOP PRINCIPAL CON LOGS MEJORADOS
# ======================================================
def run_bot():
    global ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON
    global ADAPTIVE_BIAS, ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TRAILING_MULT

    print("🤖 BOT V99.4 INICIADO - IA Visión + Autoaprendizaje")
    telegram_mensaje("🤖 BOT V99.4 INICIADO: Análisis visual con Llama 4 Scout + aprendizaje cada 10 trades.")
    ultima_vela_operada = None

    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            precio_mercado = df['close'].iloc[-1]
            tiempo_vela_cerrada = df.index[-2]
            atr_actual = df['atr'].iloc[-1]

            # Detectar zonas para heartbeat
            soporte, resistencia, slope, intercept, tendencia = detectar_zonas_mercado(df, idx=-2)

            # Calcular métricas
            pnl_global = PAPER_BALANCE - PAPER_BALANCE_INICIAL
            drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE * 100
            winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0

            # Heartbeat en consola
            print(f"\n{'='*60}")
            print(f"💓 HEARTBEAT - {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
            print(f"💰 Precio: {precio_mercado:.2f} | ATR: {atr_actual:.2f}")
            print(f"📊 Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}")
            print(f"📈 Tendencia macro: {tendencia} (pendiente {slope:.6f})")
            print(f"📉 EMA20: {df['ema20'].iloc[-1]:.2f} | Diferencia: {(precio_mercado - df['ema20'].iloc[-1])/df['ema20'].iloc[-1]*100:.2f}%")
            print(f"🎯 Trades totales: {PAPER_TRADES_TOTALES} | Wins: {PAPER_WIN} | Losses: {PAPER_LOSS} | Winrate: {winrate:.1f}%")
            print(f"💰 PnL global: {pnl_global:+.2f} USD | Balance: {PAPER_BALANCE:.2f} USD | Drawdown diario: {drawdown:.2f}%")
            print(f"🧠 Sesgo adaptativo: {ADAPTIVE_BIAS:+.2f} | SL mult: {ADAPTIVE_SL_MULT:.2f} | TP1 mult: {ADAPTIVE_TP1_MULT:.2f}")
            print(f"🤖 Última decisión IA: {ULTIMA_DECISION} - Motivo: {ULTIMO_MOTIVO[:80]}")
            print(f"{'='*60}")

            # Si no hay posición y nueva vela, analizar
            if PAPER_POSICION_ACTIVA is None and ultima_vela_operada != tiempo_vela_cerrada:
                # Generar gráfico temporal
                ruta_img = generar_y_guardar_grafico(df, es_entrada=False)
                decision, razones, patron, analisis_completo = analizar_con_groq_vision(ruta_img)

                # Guardar para logs
                ULTIMA_DECISION = decision
                ULTIMO_MOTIVO = razones[0] if razones else "Sin motivo"
                ULTIMA_RAZONES = razones
                ULTIMO_PATRON = patron

                if decision in ["Buy", "Sell"] and risk_management_check():
                    if paper_abrir_posicion(decision, precio_mercado, atr_actual, razones, patron):
                        ultima_vela_operada = tiempo_vela_cerrada
                        # Generar gráfico final con anotaciones y enviar a Telegram
                        ruta_final = generar_y_guardar_grafico(df, decision, razones, patron, es_entrada=True)
                        with open(ruta_final, "rb") as f:
                            # Enviar imagen con el análisis completo como caption
                            caption = f"🤖 Análisis IA ({decision.upper()})\n{analisis_completo[:900]}"
                            telegram_grafico(f, caption=caption)
                else:
                    print(f"⏸️ Hold: {ULTIMO_MOTIVO}")

            # Revisar posición activa
            if PAPER_POSICION_ACTIVA is not None:
                paper_revisar_sl_tp(df)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"❌ ERROR CRÍTICO: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
