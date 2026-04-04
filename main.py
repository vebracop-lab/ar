# BOT TRADING V99.3 BYBIT REAL – GROQ IA (VISION - LLAMA 4 SCOUT)
# ================================================================
# IA multimodal con análisis directo de la imagen del gráfico.
# El bot actúa como Steven Nison, analizando patrones de velas,
# tendencias y estructuras de mercado para tomar decisiones.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from groq import Groq

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# Usando el modelo multimodal de visión
MODELO_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"

SYMBOL = "BTCUSDT"
INTERVAL = "5"
RISK_PER_TRADE = 0.02
LEVERAGE = 10
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120

# Parámetros fijos de SL/TP (ahora sin IA)
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

MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

# Historial para análisis
TRADE_HISTORY = []

# ======================================================
# TELEGRAM
# ======================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL = "https://api.bybit.com"

def telegram_mensaje(texto):
    """Envía un mensaje de texto a Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texto}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Error en Telegram (mensaje): {e}")

def telegram_grafico(fig, caption=""):
    """Envía una imagen de matplotlib a Telegram."""
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
        print(f"Error en Telegram (foto): {e}")

# ======================================================
# DATOS Y TÉCNICO
# ======================================================
def obtener_velas(limit=150):
    """Obtiene los datos históricos de velas de Bybit."""
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
    """Calcula la EMA20 y el ATR."""
    df['ema20'] = df['close'].ewm(span=20).mean()
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df.dropna()

def detectar_zonas_mercado(df, idx=-2, ventana_macro=120):
    """Detecta soporte, resistencia y tendencia."""
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
# IA GROQ CON VISIÓN (ANÁLISIS DIRECTO DE LA IMAGEN)
# ======================================================
def analizar_con_groq_vision(ruta_imagen):
    """
    Envía la imagen del gráfico a Groq para que la IA (como Steven Nison) la analice.
    Retorna: decision (Buy/Sell/Hold), razones, patron
    """
    try:
        # Codificar la imagen a base64
        with open(ruta_imagen, "rb") as img_file:
            imagen_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Prompt para la IA (actuando como Steven Nison)
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

        # Llamada a la API de Groq con la imagen y el prompt
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

        # Procesar la respuesta
        analisis_completo = respuesta.choices[0].message.content
        print(f"🤖 Análisis de la IA:\n{analisis_completo}")

        # Determinar la decisión (Buy/Sell/Hold) basado en el texto
        decision = "Hold"
        if "BUY" in analisis_completo.upper():
            decision = "Buy"
        elif "SELL" in analisis_completo.upper():
            decision = "Sell"
        
        # Extraer razones y patrón (simplificado, toma las primeras líneas)
        lineas = analisis_completo.split('\n')
        razones = [linea for linea in lineas if linea.strip()][:3]
        patron = razones[0] if razones else "Patrón técnico"

        return decision, razones, patron, analisis_completo

    except Exception as e:
        print(f"❌ Error crítico en la IA de visión: {e}")
        return "Hold", ["Error en el análisis de la imagen"], "Error API", ""

# ======================================================
# GESTIÓN DE RIESGO Y PAPER TRADING (SIN CAMBIOS)
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
    global PAPER_SL_ACTUAL, PAPER_BALANCE

    if PAPER_POSICION_ACTIVA is not None:
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE

    if decision == "Buy":
        sl = precio - (atr * DEFAULT_SL_MULT)
        tp1 = precio + (atr * DEFAULT_TP1_MULT)
        tp2 = precio + (atr * DEFAULT_TP2_MULT)
    else:
        sl = precio + (atr * DEFAULT_SL_MULT)
        tp1 = precio - (atr * DEFAULT_TP1_MULT)
        tp2 = precio - (atr * DEFAULT_TP2_MULT)

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
    PAPER_TRAILING_MULT = DEFAULT_TRAILING_MULT
    PAPER_SIZE_BTC = size_btc
    PAPER_SIZE_BTC_RESTANTE = size_btc
    PAPER_TP1_EJECUTADO = False
    PAPER_SL_ACTUAL = sl

    telegram_mensaje(f"📌 OPERACIÓN {decision.upper()} | Precio: {precio:.2f} | SL: {sl:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f}\nRazones: {' | '.join(razones[:2])}")
    return True

def paper_revisar_sl_tp(df):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TP2
    global PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO
    global PAPER_SL_ACTUAL, PAPER_BALANCE, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS
    global PAPER_TRADES_TOTALES

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
            telegram_mensaje(f"🎯 TP1 alcanzado. Beneficio parcial: +{beneficio_parcial:.2f} USD. SL a break-even. Resta {PAPER_SIZE_BTC_RESTANTE:.4f} BTC")

    # Trailing después de TP1
    if PAPER_TP1_EJECUTADO:
        if PAPER_POSICION_ACTIVA == "Buy":
            nuevo_sl = close - (atr * PAPER_TRAILING_MULT)
            if nuevo_sl > PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                # telegram_mensaje(f"🔼 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f}")
            if low <= PAPER_SL_ACTUAL:
                cerrar_total = True
                motivo = "Trailing Stop"
        else:
            nuevo_sl = close + (atr * PAPER_TRAILING_MULT)
            if nuevo_sl < PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                # telegram_mensaje(f"🔽 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f}")
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
        winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0

        telegram_mensaje(f"📤 TRADE CERRADO ({motivo})\nPnL total: {pnl_total:.2f} USD\nBalance: {PAPER_BALANCE:.2f} USD\nWinrate: {winrate:.1f}%")

        # Guardar en historial
        TRADE_HISTORY.append({"pnl": pnl_total, "win": win_status})

        PAPER_POSICION_ACTIVA = None
        return True
    return None

# ======================================================
# GRÁFICOS (VERSIÓN MEJORADA PARA ANÁLISIS)
# ======================================================
def generar_y_guardar_grafico(df, decision=None, razones=None, patron=None, es_entrada=True):
    """Genera un gráfico con toda la información relevante y lo guarda en un archivo."""
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16, 8))  # Aumentado el tamaño para mejor análisis

    # --- Dibujar Velas (Estilo TradingView) ---
    for i in range(len(df_plot)):
        color = 'green' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else 'red'
        # Mecha
        ax.vlines(x[i], df_plot['low'].iloc[i], df_plot['high'].iloc[i], color=color, linewidth=1)
        # Cuerpo
        y_cuerpo = min(df_plot['open'].iloc[i], df_plot['close'].iloc[i])
        altura = abs(df_plot['close'].iloc[i] - df_plot['open'].iloc[i])
        ax.add_patch(plt.Rectangle((x[i]-0.3, y_cuerpo), 0.6, max(altura, 0.01), color=color, alpha=0.9))

    # --- Indicadores y Niveles ---
    # Niveles horizontales (Soporte y Resistencia)
    soporte, resistencia, slope, intercept, tendencia = detectar_zonas_mercado(df)
    ax.axhline(soporte, color='cyan', ls='--', lw=1.5, label=f'Soporte: {soporte:.2f}')
    ax.axhline(resistencia, color='magenta', ls='--', lw=1.5, label=f'Resistencia: {resistencia:.2f}')
    
    # Línea de Tendencia
    x_tendencia = np.arange(len(df_plot))
    y_tendencia = intercept + slope * x_tendencia
    ax.plot(x_tendencia, y_tendencia, 'w-.', lw=1.5, label='Tendencia (Regresión)')
    
    # EMA 20
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'y', lw=2, label='EMA 20')

    # --- Anotaciones de la IA (si es una entrada) ---
    if es_entrada and decision and razones and patron:
        texto_ia = f"GROQ VISION V99.3 | Decisión: {decision.upper()}\nPatrón: {patron[:70]}\n"
        texto_ia += f"Razones:\n" + "\n".join(razones[:3])
        ax.text(0.01, 0.99, texto_ia, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.85),
                color='white', family='monospace')

    # --- Estilo y Formato ---
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', facecolor='black', labelcolor='white')
    plt.tight_layout(pad=2.0)
    
    # Guardar la imagen
    ruta_imagen = "/tmp/chart_analysis.png"
    plt.savefig(ruta_imagen, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ruta_imagen

# ======================================================
# LOOP PRINCIPAL
# ======================================================
def run_bot():
    print("🤖 BOT V99.3 INICIADO - IA con VISIÓN DIRECTA (Llama 4 Scout)")
    telegram_mensaje("🤖 BOT V99.3 INICIADO: Analizando el gráfico con IA de visión (Steven Nison).")
    ultima_vela_operada = None

    while True:
        try:
            # 1. Obtener datos y calcular indicadores
            df = calcular_indicadores(obtener_velas())
            precio_mercado = df['close'].iloc[-1]
            tiempo_vela_cerrada = df.index[-2]
            atr_actual = df['atr'].iloc[-1]

            print(f"\n💓 Heartbeat | Precio: {precio_mercado:.2f} | Trades: {PAPER_TRADES_TOTALES} | Drawdown: { (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE * 100:.2f}%")

            # 2. Si no hay posición activa y es una nueva vela, analizar con IA
            if PAPER_POSICION_ACTIVA is None and ultima_vela_operada != tiempo_vela_cerrada:
                # Generar el gráfico para enviar a la IA
                ruta_imagen = generar_y_guardar_grafico(df, es_entrada=False)

                # Analizar la imagen con Groq Vision
                decision, razones, patron, analisis_completo = analizar_con_groq_vision(ruta_imagen)

                if decision in ["Buy", "Sell"] and risk_management_check():
                    if paper_abrir_posicion(decision, precio_mercado, atr_actual, razones, patron):
                        ultima_vela_operada = tiempo_vela_cerrada
                        # Generar y enviar el gráfico final con anotaciones a Telegram
                        ruta_imagen_final = generar_y_guardar_grafico(df, decision, razones, patron, es_entrada=True)
                        with open(ruta_imagen_final, "rb") as f:
                            telegram_grafico(f, caption=analisis_completo[:1000])  # Enviar la imagen con el análisis
                        print(f"🚀 OPERACIÓN: {decision.upper()} | Razones: {razones}")
                else:
                    print(f"⏸️ Hold: {razones[0] if razones else 'Sin señal clara'}")

            # 3. Revisar SL/TP de la posición activa
            if PAPER_POSICION_ACTIVA is not None:
                paper_revisar_sl_tp(df)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"❌ ERROR CRÍTICO: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
