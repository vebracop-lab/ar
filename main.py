# BOT TRADING V98.2 BYBIT REAL – GROQ IA INSTITUCIONAL (EDICIÓN EXTENSA)
# =====================================================================
# ⚠️ IA GROQ (LLaMA 3.3 70b): Análisis de Geometría de Mercado Pro
# Diseñado para detectar absorción, presión de ruptura y trampas.
# PROHIBIDA LA SIMPLIFICACIÓN. CÓDIGO ÍNTEGRO Y DETALLADO.
# =====================================================================

import os
import time
import io
import hmac
import hashlib
import requests
import json
import base64
import numpy as np
import pandas as pd
import textwrap
from scipy.stats import linregress
from datetime import datetime, timezone
from PIL import Image

# Configuración crucial para Railway (Servidor sin pantalla)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURACIÓN GROQ API
# ======================================================
from groq import Groq

# Railway inyectará la API KEY desde sus Variables de Entorno
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# Uso de Llama 3.3 70b para razonamiento profundo institucional
MODELO_GROQ = "llama-3.3-70b-versatile"

plt.rcParams['figure.figsize'] = (12, 6)

# ======================================================
# CONFIGURACIÓN PARÁMETROS DE TRADING
# ======================================================
GRAFICO_VELAS_LIMIT = 120
MOSTRAR_EMA20 = True

SYMBOL = "BTCUSDT"
INTERVAL = "5"  
RISK_PER_TRADE = 0.02  # 2% de riesgo por operación
LEVERAGE = 10          # 10x de apalancamiento
SLEEP_SECONDS = 60     

MULT_SL = 1.5          
MULT_TP1 = 2.5         
MULT_TRAILING = 2.0    
PORCENTAJE_CIERRE = 0.5 

# ======================================================
# PAPER TRADING (SISTEMA DE GESTIÓN DE ESTADO)
# ======================================================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_PNL_GLOBAL = 0.0
PAPER_POSICION_ACTIVA = None
PAPER_PRECIO_ENTRADA = None
PAPER_DECISION_ACTIVA = None
PAPER_SIZE_BTC = 0.0
PAPER_SL = None
PAPER_TP1 = None
PAPER_TP1_EJECUTADO = False
PAPER_PNL_PARCIAL = 0.0  
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
PAPER_LAST_10_PNL = [] 

# CONTROL DINÁMICO DE RIESGO
MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

# ======================================================
# CREDENCIALES BYBIT Y TELEGRAM
# ======================================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL = "https://api.bybit.com"

# ======================================================
# FUNCIONES DE COMUNICACIÓN (TELEGRAM)
# ======================================================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texto}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Error enviando mensaje a Telegram: {e}")

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
        print(f"Error enviando foto a Telegram: {e}")

# ======================================================
# MOTOR DE DATOS E INDICADORES (ANÁLISIS ESPACIAL)
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

def calcular_indicadores_institucionales(df):
    # Medias Móviles y Volatilidad
    df['ema20'] = df['close'].ewm(span=20).mean()
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # RSI de 14 períodos
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean(); avg_loss = loss.rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (avg_gain/avg_loss)))

    # --- ANATOMÍA DE VELA (Métricas para la IA) ---
    df['cuerpo'] = (df['close'] - df['open']).abs()
    df['rango_total'] = df['high'] - df['low']
    
    # % de Mecha Superior (Presión de Venta/Absorción)
    df['pc_mecha_sup'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / df['rango_total'] * 100).fillna(0)
    # % de Mecha Inferior (Presión de Compra/Absorción)
    df['pc_mecha_inf'] = ((df[['open', 'close']].min(axis=1) - df['low']) / df['rango_total'] * 100).fillna(0)
    
    # Distancia a la EMA 20 (Para detectar sobreextensión o rebotes precisos)
    df['dist_ema'] = ((df['close'] - df['ema20']) / df['ema20']) * 100
    df['ema_touch'] = (df['low'] <= df['ema20']) & (df['high'] >= df['ema20'])
    
    return df.dropna()

def analizar_geometria_avanzada(df, idx=-2):
    df_eval = df.iloc[:idx+1]
    
    # SOPORTES Y RESISTENCIAS MULTI-CAPA
    # Macro (50 velas) para estructura principal
    sup_macro = df_eval['low'].rolling(50).min().iloc[-1]
    res_macro = df_eval['high'].rolling(50).max().iloc[-1]
    
    # Micro (15 velas) para reacción inmediata del precio
    sup_micro = df_eval['low'].rolling(15).min().iloc[-1]
    res_micro = df_eval['high'].rolling(15).max().iloc[-1]
    
    # CÁLCULO DE PENDIENTE Y ÁNGULO (GEOMETRÍA)
    y_macro = df_eval['close'].values[-60:]
    x_macro = np.arange(len(y_macro))
    slope, intercept, _, _, _ = linregress(x_macro, y_macro)
    
    # El ángulo ayuda a la IA a entender la fuerza de la tendencia
    angulo = np.degrees(np.arctan(slope))
    
    if angulo > 5: tendencia_str = 'ALCISTA (BULLISH)'
    elif angulo < -5: tendencia_str = 'BAJISTA (BEARISH)'
    else: tendencia_str = 'LATERAL (NEUTRAL)'
    
    # ANÁLISIS DE "PRESIÓN": Toques a la zona sin retroceso
    umbral = df_eval['close'].iloc[-1] * 0.0008
    toques_res = (df_eval['high'] > (res_macro - umbral)).tail(20).sum()
    toques_sup = (df_eval['low'] < (sup_macro + umbral)).tail(20).sum()

    return {
        "sup_macro": sup_macro, "res_macro": res_macro,
        "sup_micro": sup_micro, "res_micro": res_micro,
        "angulo": angulo, "tendencia": tendencia_str,
        "slope": slope, "intercept": intercept,
        "toques_res": toques_res, "toques_sup": toques_sup
    }

# ======================================================
# MOTOR DE INTELIGENCIA GROQ (RAZONAMIENTO PROFUNDO)
# ======================================================
def analizar_con_groq_institucional(df, geo, idx=-2):
    precio_actual = df['close'].iloc[idx]
    rsi_actual = df['rsi'].iloc[idx]
    ema_actual = df['ema20'].iloc[idx]
    dist_ema = df['dist_ema'].iloc[idx]
    
    # Estructura de datos de velas para el prompt
    contexto_velas = df.iloc[idx-9:idx+1][['open', 'high', 'low', 'close', 'pc_mecha_sup', 'pc_mecha_inf', 'dist_ema']]
    texto_velas = contexto_velas.to_string()

    prompt = f"""
Eres un Master Trader Institucional de Futuros Perpetuos. 
Tu objetivo es realizar un análisis de GEOMETRÍA Y PRESIÓN DE MERCADO.

ESTADO ESPACIAL DEL GRÁFICO:
- Precio: {precio_actual:.2f} | RSI: {rsi_actual:.2f}
- NIVEL MACRO: Soporte {geo['sup_macro']:.2f} | Resistencia {geo['res_macro']:.2f}
- NIVEL MICRO (Zona de Reacción): Soporte {geo['sup_micro']:.2f} | Resistencia {geo['res_micro']:.2f}
- ÁNGULO DE TENDENCIA: {geo['angulo']:.2f}° | Carácter: {geo['tendencia']}
- PRESIÓN DE RUPTURA: Toques en Resistencia ({geo['toques_res']}) | Toques en Soporte ({geo['toques_sup']}) en el último bloque.
- DISTANCIA A EMA 20: {dist_ema:.4f}% (Valores cercanos a 0 indican testeo de EMA).

ANATOMÍA DE LAS ÚLTIMAS 10 VELAS (Incluye OHLC y % de Mechas):
{texto_velas}

DIRECTRICES DE ANÁLISIS PROFESIONAL:
1. Evalúa la "Presión de Ruptura": Si el precio golpea una zona repetidamente ({geo['toques_res']} o {geo['toques_sup']}) sin rebotar con fuerza, es probable una ruptura.
2. Analiza las "Mechas": Un alto % de mecha superior cerca de resistencia indica ABSORCIÓN DE COMPRA.
3. EMA 20: Decide si la EMA está actuando como un imán (lateral) o como un trampolín (tendencia).
4. Define el contexto del patrón de los 5 grupos principales. No simplifiques el nombre.

Devuelve tu decisión en este formato JSON exacto:
{{
  "decision": "Buy" | "Sell" | "Hold",
  "patron_detectado": "Nombre institucional completo y descripción del contexto",
  "fuera_de_zona": true | false,
  "razones": [
    "Análisis de estructura y niveles micro/macro",
    "Interpretación de la presión detectada por toques y ángulo",
    "Evaluación de la anatomía de las mechas actuales"
  ]
}}
"""
    try:
        completion = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"🚨 Error en API Groq: {e}")
        return {"decision": "Hold", "patron_detectado": "Error", "fuera_de_zona": False, "razones": ["Falla de conexión"]}

# ======================================================
# SISTEMA DE GRÁFICOS Y LOGS DETALLADOS
# ======================================================
def generar_grafico_v98(df, decision, geo, razones, patron):
    df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
    x_eje = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16, 9))

    # Dibujo de velas manual (Alta definición)
    for i in range(len(df_plot)):
        color = 'lime' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else 'red'
        ax.vlines(x_eje[i], df_plot['low'].iloc[i], df_plot['high'].iloc[i], color=color, linewidth=1.2)
        base = min(df_plot['open'].iloc[i], df_plot['close'].iloc[i])
        altura = max(abs(df_plot['close'].iloc[i] - df_plot['open'].iloc[i]), 0.1)
        ax.add_patch(plt.Rectangle((x_eje[i] - 0.3, base), 0.6, altura, color=color, alpha=0.8))

    # Visualización de Zonas Geométricas
    ax.axhline(geo['res_macro'], color='magenta', linestyle='--', linewidth=2, label="RES Macro")
    ax.axhline(geo['sup_macro'], color='cyan', linestyle='--', linewidth=2, label="SUP Macro")
    ax.axhline(geo['res_micro'], color='magenta', linestyle=':', linewidth=1, alpha=0.5, label="RES Micro")
    ax.axhline(geo['sup_micro'], color='cyan', linestyle=':', linewidth=1, alpha=0.5, label="SUP Micro")
    
    if MOSTRAR_EMA20:
        ax.plot(x_eje, df_plot['ema20'].values, color='yellow', linewidth=2.5, label='EMA 20')

    # Línea de Tendencia calculada
    linea_tendencia = geo['intercept'] + geo['slope'] * x_eje
    ax.plot(x_eje, linea_tendencia, color='white', linestyle='-.', alpha=0.3)

    # Panel de Texto (Uso de textwrap para evitar desbordamiento)
    razones_unidas = " - ".join(razones)
    txt_ia = f"DECISIÓN: {decision.upper()}\nPATRÓN: {textwrap.fill(patron, 65)}\nRAZONES: {textwrap.fill(razones_unidas, 85)}"
    
    ax.text(0.01, 0.98, txt_ia, transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='black', edgecolor='white', alpha=0.9), color='white')

    ax.set_title(f"TRADING MAESTRO V98.2 - {SYMBOL} ({INTERVAL}m) - GEOMETRÍA IA", color='white', fontsize=15)
    ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
    ax.tick_params(colors='white'); ax.grid(True, alpha=0.1)
    plt.legend(loc="lower right", facecolor='black', labelcolor='white')
    
    try: plt.tight_layout()
    except: pass
    return fig

# ======================================================
# GESTIÓN DE RIESGO Y ÓRDENES (SISTEMA COMPLETO)
# ======================================================
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY, PAPER_BALANCE
    ahora_utc = datetime.now(timezone.utc).date()
    
    if PAPER_CURRENT_DAY != ahora_utc:
        PAPER_CURRENT_DAY = ahora_utc
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
    
    drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
        if not PAPER_STOPPED_TODAY:
            telegram_mensaje("🛑 CRÍTICO: Drawdown diario del 20% alcanzado. Bot pausado por seguridad.")
            PAPER_STOPPED_TODAY = True
        return False
    return True

def paper_abrir_posicion(decision, precio, atr, patron):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_SIZE_BTC
    global PAPER_TP1_EJECUTADO, PAPER_SIZE_BTC_RESTANTE

    if PAPER_POSICION_ACTIVA is not None: return False
    
    # Cálculo de Niveles Institucionales
    distancia_sl = atr * MULT_SL
    if decision == "Buy":
        PAPER_SL = precio - distancia_sl
        PAPER_TP1 = precio + (atr * MULT_TP1)
    else:
        PAPER_SL = precio + distancia_sl
        PAPER_TP1 = precio - (atr * MULT_TP1)
    
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    PAPER_SIZE_BTC = riesgo_usd / distancia_sl
    PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC
    PAPER_POSICION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_TP1_EJECUTADO = False
    
    return True

def gestionar_salidas_y_log(df, geo):
    global PAPER_BALANCE, PAPER_POSICION_ACTIVA, PAPER_TP1_EJECUTADO, PAPER_SL, PAPER_TRADES_TOTALES
    global PAPER_WIN, PAPER_LOSS, PAPER_LAST_10_PNL
    
    if PAPER_POSICION_ACTIVA is None: return

    vela = df.iloc[-1]
    pnl_local = 0
    debe_cerrar = False
    motivo = ""

    if PAPER_POSICION_ACTIVA == "Buy":
        # Revisión TP1 Parcial
        if not PAPER_TP1_EJECUTADO and vela['high'] >= PAPER_TP1:
            PAPER_BALANCE += (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA # Movemos a Breakeven
            telegram_mensaje("🎯 TP1 ALCANZADO (50% cerrado). SL movido a Breakeven.")
        
        # Revisión Stop o Trailing
        if vela['low'] <= PAPER_SL:
            pnl_local = (PAPER_SL - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_EJECUTADO else 1.0))
            debe_cerrar = True
            motivo = "Trailing/Stop Loss"
            
    elif PAPER_POSICION_ACTIVA == "Sell":
        if not PAPER_TP1_EJECUTADO and vela['low'] <= PAPER_TP1:
            PAPER_BALANCE += (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA
            telegram_mensaje("🎯 TP1 ALCANZADO (50% cerrado). SL movido a Breakeven.")
            
        if vela['high'] >= PAPER_SL:
            pnl_local = (PAPER_PRECIO_ENTRADA - PAPER_SL) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_EJECUTADO else 1.0))
            debe_cerrar = True
            motivo = "Trailing/Stop Loss"

    if debe_cerrar:
        PAPER_BALANCE += pnl_local
        PAPER_TRADES_TOTALES += 1
        win = pnl_local > 0
        if win: PAPER_WIN += 1
        else: PAPER_LOSS += 1
        
        PAPER_LAST_10_PNL.append(pnl_local)
        if len(PAPER_LAST_10_PNL) > 10: PAPER_LAST_10_PNL.pop(0)
        
        # Envío de gráfico de salida
        fig_out = generar_grafico_v98(df, "SALIDA", geo, [f"Motivo: {motivo}"], f"Trade finalizado. PnL: {pnl_local:.2f} USD")
        telegram_grafico(fig_out)
        plt.close(fig_out)
        
        PAPER_POSICION_ACTIVA = None
        telegram_mensaje(f"📤 POSICIÓN CERRADA: {motivo}\n💰 PnL Trade: {pnl_local:.2f} USD\n📊 Balance: {PAPER_BALANCE:.2f} USD")

# ======================================================
# BUCLE DE EJECUCIÓN MAESTRO
# ======================================================
def iniciar_maestro():
    print("🤖 TRADING MAESTRO V98.2 ACTIVADO - LLaMA 3.3 70B")
    telegram_mensaje("🚀 BOT V98.2 INICIADO: Geometría Institucional y Análisis de Presión activados.")
    ultima_vela_procesada = None

    while True:
        try:
            # 1. Obtención y cálculo
            df = calcular_indicadores_institucionales(obtener_velas())
            geo = analizar_geometria_avanzada(df)
            
            # 2. LOG HEARTBEAT (Requerimiento del usuario)
            pnl_total_10 = sum(PAPER_LAST_10_PNL)
            log_msg = f"\n💓 [HEARTBEAT] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            log_msg += f"\n   Tendencia: {geo['tendencia']} | Ángulo: {geo['angulo']:.2f}°"
            log_msg += f"\n   Global PnL (10 trades): {pnl_total_10:.2f} USD | Contador: {PAPER_TRADES_TOTALES}"
            log_msg += f"\n   Zonas: RES {geo['res_macro']:.2f} | SUP {geo['sup_macro']:.2f}"
            print(log_msg)

            # 3. Lógica de decisión
            if PAPER_POSICION_ACTIVA is None and ultima_vela_procesada != df.index[-2]:
                print("🔍 Analizando oportunidad con IA Groq...")
                resultado_ia = analizar_con_groq_institucional(df, geo)
                
                decision = resultado_ia.get("decision", "Hold")
                patron = resultado_ia.get("patron_detectado", "Ninguno")
                razones = resultado_ia.get("razones", ["Sin datos"])
                fuera_de_zona = resultado_ia.get("fuera_de_zona", False)

                if decision in ["Buy", "Sell"]:
                    if fuera_de_zona:
                        print(f"⚠️ Patrón detectado ({patron}) pero fuera de zona operativa.")
                    else:
                        print(f"✅ TRADE APROBADO: {decision} | Patrón: {patron}")
                        if risk_management_check():
                            if paper_abrir_posicion(decision, df['close'].iloc[-1], df['atr'].iloc[-1], patron):
                                # Notificación de entrada
                                telegram_mensaje(f"🔔 ENTRADA INSTITUCIONAL {decision.upper()}\n👁️ {patron}\n💵 Precio: {df['close'].iloc[-1]:.2f}")
                                fig_ent = generar_grafico_v98(df, decision, geo, razones, patron)
                                telegram_grafico(fig_ent)
                                plt.close(fig_ent)
                                ultima_vela_procesada = df.index[-2]
                else:
                    print(f"❌ TRADE RECHAZADO: Razón: {razones[0]}")

            # 4. Gestión de posición abierta
            gestionar_salidas_y_log(df, geo)
            
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"🚨 ERROR CRÍTICO EN EL BUCLE: {e}")
            time.sleep(60)

if __name__ == '__main__':
    iniciar_maestro()
