# BOT TRADING V98.3 BYBIT REAL – GROQ IA (RAILWAY READY)
# ======================================================
# IA GROQ (LLaMA 3.3 Versatile) con análisis contextual avanzado
# y gestión de riesgos dinámica (SL, TP1 fijo, TP2 dinámico con trailing)
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

# Parámetros por defecto (se sobreescriben si IA los modifica)
DEFAULT_SL_MULT = 1.5
DEFAULT_TP1_MULT = 2.5
DEFAULT_TP2_MULT = 4.0        # TP2 más lejano
DEFAULT_TRAILING_MULT = 2.0   # Múltiplo de ATR para trailing después de TP1

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
PAPER_SL_ACTUAL = None            # SL dinámico después de TP1
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
PAPER_LAST_10_PNL = []

# Control de drawdown diario
MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

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
# ANÁLISIS DE VELAS (igual que antes)
# ======================================================
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

# ======================================================
# IA GROQ CON SUGERENCIA DE SL/TP DINÁMICOS
# ======================================================
def analizar_con_groq_texto(df, soporte, resistencia, tendencia, slope, intercept, idx=-2):
    precio = df['close'].iloc[idx]
    atr = df['atr'].iloc[idx]
    rsi = df['rsi'].iloc[idx]
    ema = df['ema20'].iloc[idx]
    analisis_velas = analizar_velas_detallado(df)
    analisis_ema = analizar_ema_cruce(df)
    analisis_rechazo = analizar_rechazo_resistencia(df, resistencia)

    prompt = f"""
Eres un trader institucional. Analiza BTCUSDT en gráfico de 5 minutos y decide dirección (Buy/Sell/Hold). Además, recomienda niveles de riesgo basados en ATR actual ({atr:.2f}).

DATOS:
- Precio actual (última vela cerrada): {precio:.2f}
- Soporte: {soporte:.2f}, Resistencia: {resistencia:.2f}
- Tendencia macro: {tendencia}, Slope: {slope:.6f}
- RSI: {rsi:.1f}, EMA20: {ema:.2f}
- {analisis_ema}
- {analisis_rechazo}

VELAS RECIENTES:
{analisis_velas}

INSTRUCCIONES:
1. Decide "Buy", "Sell" o "Hold". Si hay rechazo en resistencia + precio debajo EMA20 -> Sell. Si soporte + precio encima EMA20 -> Buy.
2. Si la decisión es Buy o Sell, debes sugerir multiplicadores para SL y TP basados en el ATR (siendo conservadores). Usa valores entre 1.0 y 3.0 para SL, entre 2.0 y 5.0 para TP1, entre 3.0 y 8.0 para TP2. También sugiere un multiplicador para el trailing stop después de TP1 (entre 1.5 y 3.5).
3. TP1 será fijo y cerrará el 50% de la posición. TP2 será dinámico con trailing stop que se ajusta con el precio.

Responde SOLO en JSON con este formato:
{{
  "decision": "Buy/Sell/Hold",
  "patron": "descripción del patrón",
  "fuera_de_zona": false,
  "razones": ["razón1","razón2"],
  "sl_mult": 1.5,
  "tp1_mult": 2.5,
  "tp2_mult": 4.0,
  "trailing_mult": 2.0
}}
Si es Hold, los multiplicadores pueden ser nulos.
"""
    try:
        completion = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        datos = json.loads(completion.choices[0].message.content)
        # Asegurar valores por defecto si faltan
        datos.setdefault("sl_mult", DEFAULT_SL_MULT)
        datos.setdefault("tp1_mult", DEFAULT_TP1_MULT)
        datos.setdefault("tp2_mult", DEFAULT_TP2_MULT)
        datos.setdefault("trailing_mult", DEFAULT_TRAILING_MULT)
        return datos
    except Exception as e:
        print(f"Error Groq: {e}")
        return {"decision": "Hold", "razones": ["Error API"], "sl_mult": DEFAULT_SL_MULT, "tp1_mult": DEFAULT_TP1_MULT, "tp2_mult": DEFAULT_TP2_MULT, "trailing_mult": DEFAULT_TRAILING_MULT}

# ======================================================
# GESTIÓN DE RIESGO DINÁMICA (con SL/TP sugeridos por IA)
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

def paper_abrir_posicion(decision, precio, atr, sl_mult, tp1_mult, tp2_mult, trailing_mult):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TP2
    global PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO
    global PAPER_SL_ACTUAL, PAPER_BALANCE

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

    # Tamaño en dólares con apalancamiento
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
    PAPER_SL_ACTUAL = sl   # SL inicial

    telegram_mensaje(f"📌 OPERACIÓN {decision.upper()} | Precio: {precio:.2f} | SL: {sl:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f} | Trailing mult: {trailing_mult}")
    return True

def paper_revisar_sl_tp(df, soporte, resistencia, slope, intercept):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TP2
    global PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO
    global PAPER_SL_ACTUAL, PAPER_BALANCE, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS
    global PAPER_TRADES_TOTALES, PAPER_LAST_10_PNL

    if PAPER_POSICION_ACTIVA is None:
        return None

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    close = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    cerrar_total = False
    motivo = ""

    # Comprobar TP1 (cierre parcial)
    if not PAPER_TP1_EJECUTADO:
        if (PAPER_POSICION_ACTIVA == "Buy" and high >= PAPER_TP1) or (PAPER_POSICION_ACTIVA == "Sell" and low <= PAPER_TP1):
            # Cerrar 50%
            beneficio_parcial = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1) if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1)
            PAPER_BALANCE += beneficio_parcial
            PAPER_PNL_PARCIAL = beneficio_parcial
            PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE_TP1)
            PAPER_TP1_EJECUTADO = True
            # Mover SL a break-even (precio de entrada)
            PAPER_SL_ACTUAL = PAPER_PRECIO_ENTRADA
            telegram_mensaje(f"🎯 TP1 alcanzado. Beneficio parcial: +{beneficio_parcial:.2f} USD. SL movido a break-even. Resta {PAPER_SIZE_BTC_RESTANTE:.4f} BTC")

    # Después de TP1, gestionar trailing stop dinámico hacia TP2
    if PAPER_TP1_EJECUTADO:
        if PAPER_POSICION_ACTIVA == "Buy":
            nuevo_sl = close - (atr * PAPER_TRAILING_MULT)
            if nuevo_sl > PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔼 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f}")
            if low <= PAPER_SL_ACTUAL:
                cerrar_total = True
                motivo = "Trailing Stop"
        else:  # Sell
            nuevo_sl = close + (atr * PAPER_TRAILING_MULT)
            if nuevo_sl < PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔽 Trailing SL actualizado a {PAPER_SL_ACTUAL:.2f}")
            if high >= PAPER_SL_ACTUAL:
                cerrar_total = True
                motivo = "Trailing Stop"
    else:
        # Antes de TP1, verificar SL inicial
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
        telegram_mensaje(f"📤 TRADE CERRADO ({motivo})\nPnL total: {pnl_total:.2f} USD\nBalance: {PAPER_BALANCE:.2f} USD\nWinrate: {winrate:.1f}%")

        # Enviar gráfico de salida
        fig = generar_grafico_salida(df, PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, precio_salida, pnl_total, win_status, soporte, resistencia, slope, intercept)
        telegram_grafico(fig)
        plt.close(fig)

        PAPER_POSICION_ACTIVA = None
        return True
    return None

# ======================================================
# GRÁFICOS (versión simplificada)
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
    texto = f"GROQ V98.3: {decision.upper()}\nPatrón: {patron[:60]}\n{chr(10).join(razones[:2])}"
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
    print("🤖 BOT V98.3 INICIADO - IA con gestión de riesgos dinámica (SL/TP sugeridos)")
    telegram_mensaje("🤖 BOT V98.3 INICIADO: IA decide dirección y niveles de SL/TP dinámicos.")
    ultima_vela_operada = None

    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            idx_eval = -2
            precio_mercado = df['close'].iloc[-1]
            tiempo_vela_cerrada = df.index[-2]

            soporte, resistencia, slope, intercept, tendencia = detectar_zonas_mercado(df, idx_eval)

            print(f"\n💓 Heartbeat | Precio: {precio_mercado:.2f} | Sop: {soporte:.2f} | Res: {resistencia:.2f} | Trades: {PAPER_TRADES_TOTALES}")

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
                    if paper_abrir_posicion(decision, precio_mercado, atr_actual, sl_mult, tp1_mult, tp2_mult, trailing_mult):
                        ultima_vela_operada = tiempo_vela_cerrada
                        fig = generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones, patron)
                        telegram_grafico(fig)
                        plt.close(fig)
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
