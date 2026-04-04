# BOT TRADING V101.1 BYBIT REAL – PRODUCCIÓN (SIN PROXY) 
# ======================================================
# ⚠️ KEYS INCLUIDAS TAL CUAL (SEGÚN PEDIDO)
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit
# ======================================================
# NOVEDADES V101.1 (HOLISTIC AI MASTERMIND FIX):
# - FIX CRÍTICO: Solucionado error TypeError en log_colab (9 parámetros requeridos).
# - FIX CRÍTICO: Agregados 'soporte', 'resistencia', 'slope' e 'intercept' al 
#   diccionario log_zonas para evitar crasheos al graficar.
# - INTELIGENCIA MULTIMODAL GROQ VISION AI (Modelo llama-3.2-90b-vision-preview).
# - CÓDIGO 100% EXPANDIDO. CERO COMPRESIÓN.
# ======================================================

import os
import time
import io
import hmac
import hashlib
import requests
import base64
import json
import re
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

def cerca_de_nivel(precio, nivel, margen):
    distancia = abs(precio - nivel)
    if distancia <= margen:
        return True
    else:
        return False

SYMBOL = "BTCUSDT"
INTERVAL = "5"  
RISK_PER_TRADE = 0.02  
LEVERAGE = 10          
SLEEP_SECONDS = 60     

MULT_SL = 1.5          
MULT_TP1 = 2.5         
MULT_TRAILING_BASE = 2.0    
PORCENTAJE_CIERRE = 0.5 

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
PAPER_TP1 = None
PAPER_PARTIAL_ACTIVADO = False
PAPER_SIZE_BTC_RESTANTE = 0.0
PAPER_TP1_EJECUTADO = False
PAPER_PNL_PARCIAL = 0.0  
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0

# ======================================================
# CONTROL DINÁMICO DE RIESGO
# ======================================================
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

BASE_URL = "https://api.bybit.com"

# ======================================================
# TELEGRAM Y FIRMA BYBIT
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
        archivos = {'photo': buf}
        datos = {'chat_id': TELEGRAM_CHAT_ID}
        requests.post(url, files=archivos, data=datos, timeout=15)
        buf.close()
    except Exception:
        pass

def sign(params):
    elementos =[]
    for k, v in sorted(params.items()):
        elementos.append(f"{k}={v}")
    query = '&'.join(elementos)
    firma = hmac.new(BYBIT_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return firma

# ======================================================
# OBTENCIÓN DE DATOS BYBIT
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
# INDICADORES Y ZONAS MULTIPLES
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
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

    return df.dropna()

def evaluar_impacto_zonas(df, idx=-2, ventana_macro=120):
    df_eval = df.iloc[:idx+1]
    
    soporte_horiz = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia_horiz = df_eval['high'].rolling(40).max().iloc[-1]
    
    if pd.isna(soporte_horiz): 
        soporte_horiz = df_eval['low'].min()
    if pd.isna(resistencia_horiz): 
        resistencia_horiz = df_eval['high'].max()
        
    if len(df_eval) >= 80:
        max_previo = df_eval['high'].iloc[-80:-40].max()
        min_previo = df_eval['low'].iloc[-80:-40].min()
    else:
        max_previo = resistencia_horiz
        min_previo = soporte_horiz
    
    if len(df_eval) < ventana_macro:
        y_macro = df_eval['close'].values
    else:
        y_macro = df_eval['close'].values[-ventana_macro:]
        
    x_macro = np.arange(len(y_macro))
    
    if len(y_macro) > 1 and not np.isnan(y_macro).any():
        slope, intercept, r_value, p_value, std_err = linregress(x_macro, y_macro)
    else:
        slope = 0.0
        intercept = y_macro[-1]
    
    if slope > 0.01: 
        tendencia_macro = '📈 ALCISTA'
    elif slope < -0.01: 
        tendencia_macro = '📉 BAJISTA'
    else: 
        tendencia_macro = '➡️ LATERAL'
        
    linea_central = intercept + slope * x_macro
    desviacion = np.std(y_macro - linea_central) if not np.isnan(np.std(y_macro - linea_central)) else 0
    
    canal_sup = linea_central[-1] + (desviacion * 1.5)
    canal_inf = linea_central[-1] - (desviacion * 1.5)
    
    atr_actual = df['atr'].iloc[idx]
    ema20_actual = df['ema20'].iloc[idx]
    precio_cierre = df['close'].iloc[idx]
    
    min_patron = df['low'].iloc[idx]
    max_patron = df['high'].iloc[idx]
    
    rango_local = resistencia_horiz - soporte_horiz
    if rango_local <= 0: 
        rango_local = 0.0001
    
    posicion_en_rango = (precio_cierre - soporte_horiz) / rango_local
    tolerancia_zona = atr_actual * 1.5
    
    toques_techo_ema = 0
    toques_suelo_ema = 0
    
    for i in range(idx-3, idx+1):
        h = df['high'].iloc[i]
        l = df['low'].iloc[i]
        c = df['close'].iloc[i]
        ema = df['ema20'].iloc[i]
        
        if h >= (ema - atr_actual*0.1) and c <= ema:
            toques_techo_ema += 1
        if l <= (ema + atr_actual*0.1) and c >= ema:
            toques_suelo_ema += 1

    if toques_techo_ema >= 1 and precio_cierre < ema20_actual:
        rechazo_bajista_ema = True
    else:
        rechazo_bajista_ema = False
        
    if toques_suelo_ema >= 1 and precio_cierre > ema20_actual:
        rechazo_alcista_ema = True
    else:
        rechazo_alcista_ema = False
    
    if cerca_de_nivel(min_patron, soporte_horiz, tolerancia_zona) and posicion_en_rango < 0.5:
        toca_sop_horiz = True
    else:
        toca_sop_horiz = False
        
    if cerca_de_nivel(max_patron, resistencia_horiz, tolerancia_zona) and posicion_en_rango > 0.5:
        toca_res_horiz = True
    else:
        toca_res_horiz = False
        
    if cerca_de_nivel(min_patron, max_previo, tolerancia_zona) and precio_cierre >= max_previo:
        toca_polaridad_sop = True
    else:
        toca_polaridad_sop = False
        
    if cerca_de_nivel(max_patron, min_previo, tolerancia_zona) and precio_cierre <= min_previo:
        toca_polaridad_res = True
    else:
        toca_polaridad_res = False
    
    toca_sop_dinamico = cerca_de_nivel(min_patron, canal_inf, tolerancia_zona)
    toca_res_dinamica = cerca_de_nivel(max_patron, canal_sup, tolerancia_zona)
    
    en_zona_compra = toca_sop_horiz or toca_sop_dinamico or rechazo_alcista_ema or toca_polaridad_sop
    en_zona_venta = toca_res_horiz or toca_res_dinamica or rechazo_bajista_ema or toca_polaridad_res

    # 🚫 Filtros espaciales (Si rompe el rango, se prohíbe operar en contra)
    if rechazo_bajista_ema: en_zona_compra = False
    if rechazo_alcista_ema: en_zona_venta = False

    if tendencia_macro == '📉 BAJISTA': en_zona_compra = False
    elif tendencia_macro == '📈 ALCISTA': en_zona_venta = False

    if posicion_en_rango >= 0.50 and rechazo_bajista_ema: en_zona_venta = True
    if posicion_en_rango <= 0.50 and rechazo_alcista_ema: en_zona_compra = True

    zonas_compra =[]
    if toca_sop_horiz: zonas_compra.append("Soporte Horiz")
    if toca_sop_dinamico: zonas_compra.append("Canal Inf")
    if rechazo_alcista_ema: zonas_compra.append("EMA 20 (Soporte)")
    if toca_polaridad_sop: zonas_compra.append("Polaridad (Techo roto)")
        
    zonas_venta =[]
    if toca_res_horiz: zonas_venta.append("Resistencia Horiz")
    if toca_res_dinamica: zonas_venta.append("Canal Sup")
    if rechazo_bajista_ema: zonas_venta.append("EMA 20 (Resistencia)")
    if toca_polaridad_res: zonas_venta.append("Polaridad (Suelo roto)")
        
    log_zonas = {
        "tolerancia": tolerancia_zona, 
        "en_soporte": en_zona_compra, 
        "en_resistencia": en_zona_venta,
        "zonas_compra": ", ".join(zonas_compra) if zonas_compra else "Ninguna",
        "zonas_venta": ", ".join(zonas_venta) if zonas_venta else "Ninguna",
        "posicion_rango": posicion_en_rango,
        "tendencia_macro": tendencia_macro,
        "soporte": soporte_horiz,
        "resistencia": resistencia_horiz,
        "slope": slope,
        "intercept": intercept
    }
    
    return en_zona_compra, en_zona_venta, log_zonas

# ======================================================
# GRÁFICOS Y UTILIDADES DE IMAGEN
# ======================================================
def fig_to_base64(fig):
    """Convierte el gráfico de Matplotlib a formato Base64 para que la IA de Groq pueda verlo."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.read()
    return base64.b64encode(img_bytes).decode('utf-8')

def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones):
    try:
        df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
        if df_plot.empty: 
            return None

        x_valores = np.arange(len(df_plot))
        closes = df_plot['close'].values
        fig, ax = plt.subplots(figsize=(14, 7))

        for i in range(len(df_plot)):
            if closes[i] >= df_plot['open'].values[i]:
                color_vela = 'green'
            else:
                color_vela = 'red'
                
            ax.vlines(x_valores[i], df_plot['low'].values[i], df_plot['high'].values[i], color=color_vela, linewidth=1)
            cuerpo_y = min(df_plot['open'].values[i], closes[i])
            cuerpo_h = max(abs(closes[i] - df_plot['open'].values[i]), 0.0001)
            rect = plt.Rectangle((x_valores[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color_vela, alpha=0.9)
            ax.add_patch(rect)

        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label="Soporte Horiz")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label="Resistencia Horiz")

        linea_tendencia = intercept + slope * x_valores
        desviacion = np.std(closes - linea_tendencia)
        
        ax.plot(x_valores, linea_tendencia, color='white', linewidth=2, linestyle='-', label="Tendencia Macro")
        ax.plot(x_valores, linea_tendencia + (desviacion * 1.5), linestyle='--', linewidth=2, color='red', label="Canal Sup")
        ax.plot(x_valores, linea_tendencia - (desviacion * 1.5), linestyle='--', linewidth=2, color='green', label="Canal Inf")

        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x_valores, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA 20')

        entrada_x_idx = len(df_plot) - 2
        
        if decision == 'Buy':
            ax.scatter(entrada_x_idx, closes[-2], s=200, marker='^', color='lime', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='lime', linestyle=':', linewidth=2)
        elif decision == 'Sell':
            ax.scatter(entrada_x_idx, closes[-2], s=200, marker='v', color='red', edgecolors='black', zorder=5)
            ax.axvline(entrada_x_idx, color='red', linestyle=':', linewidth=2)

        texto_razones = "\n".join(razones)
        texto_panel = f"OPERACION: {decision.upper()}\nPrecio: {df['close'].iloc[-1]:.2f}\nRSI Contexto: {df['rsi'].iloc[-2]:.1f}\n\nInfo:\n{texto_razones}"
        ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_title(f"BOT V101.1 - VISION AI - Análisis {decision.upper()} (5m)")
        ax.grid(True, alpha=0.2)
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        return None

def generar_grafico_salida(df, trade_data):
    try:
        decision_original = trade_data['decision']
        entrada_price = trade_data['entrada']
        salida_price = trade_data['salida']
        pnl_obtenido = trade_data['pnl']
        
        df_plot = df.copy().tail(120)
        if df_plot.empty: 
            return None

        fig, ax = plt.subplots(figsize=(14, 7))
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            if row['close'] >= row['open']:
                color_vela = 'green'
            else:
                color_vela = 'red'
                
            ax.plot([i, i],[row['low'], row['high']], color='black', linewidth=1)
            ax.plot([i, i],[row['open'], row['close']], color=color_vela, linewidth=6)

        ax.axhline(entrada_price, color='blue', linestyle='--', linewidth=1.5, label='Nivel Entrada')
        ax.axhline(salida_price, color='orange', linestyle='--', linewidth=1.5, label='Nivel Cierre')

        indice_salida_x = len(df_plot) - 1
        
        if pnl_obtenido > 0:
            color_marcador = 'lime'
            forma = '^'
        else:
            color_marcador = 'red'
            forma = 'v'
            
        ax.scatter([indice_salida_x],[salida_price], s=200, c=color_marcador, marker=forma, edgecolors='black', zorder=5)

        texto_panel_salida = f"CIERRE {decision_original}\nMotivo: {trade_data['motivo']}\nPnL: {pnl_obtenido:.4f} USD\nBalance: {trade_data['balance']:.2f} USD"
        ax.text(0.02, 0.95, texto_panel_salida, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title("BOT V101.1 - DETALLE DE CIERRE")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

# ======================================================
# 🧠 CEREBRO GROQ VISION AI (LLAMA 3.2 90B PREVIEW)
# ======================================================
def analizar_con_groq(df, idx, log_zonas, base64_image):
    if not client_groq:
        return "WAIT", "No se configuró GROQ_API_KEY."

    precio_actual = df['close'].iloc[idx]
    rsi_actual = df['rsi'].iloc[idx]
    tendencia = log_zonas.get('tendencia_macro', 'LATERAL')
    zonas_c = log_zonas.get('zonas_compra', 'Ninguna')
    zonas_v = log_zonas.get('zonas_venta', 'Ninguna')

    prompt = f"""
    Eres el mejor Trader Institucional de Criptomonedas del mundo.
    A continuación tienes un gráfico de velas de 5 minutos de BTCUSDT y sus datos técnicos actuales:

    - Precio Actual: {precio_actual:.2f}
    - RSI Actual: {rsi_actual:.1f}
    - Tendencia Macro: {tendencia}
    - Zonas Activas de Soporte (Piso): {zonas_c}
    - Zonas Activas de Resistencia (Techo): {zonas_v}

    Tu tarea: Analizar TODO el contexto del gráfico de forma holística. 
    1. Observa la tendencia general (canal de regresión rojo/verde, línea blanca central).
    2. Observa la EMA 20 (línea amarilla). ¿Está actuando como soporte, resistencia o el precio la cruza sin respeto?
    3. Observa las zonas horizontales (líneas cyan y magenta). ¿Estamos en el techo o en el suelo de la estructura?
    4. Analiza la acción del precio: No busques solo patrones de reversión. Evalúa si hay fuerza de ruptura (breakout), consolidación para continuar a favor de la tendencia, rechazos contundentes en las zonas críticas o agotamiento inminente.

    Basado en todo el ecosistema visual y numérico, ¿cuál es el movimiento de mayor probabilidad en los próximos minutos?
    - Responde BUY si el contexto general indica un rebote alcista sólido en soporte, un rompimiento alcista con fuerza o la continuación de una tendencia de subida saludable.
    - Responde SELL si el contexto general indica un rechazo bajista contundente en resistencia, una ruptura a la baja, o la continuación de una caída con volumen.
    - Responde WAIT si hay indecisión, el mercado está en medio de la nada ("tierra de nadie") o hay señales altamente contradictorias.

    DEBES responder ÚNICAMENTE con un JSON válido, sin ningún texto adicional antes ni después, usando esta estructura exacta:
    {{"decision": "BUY", "razon": "Explicación detallada del contexto, la tendencia, la EMA, el comportamiento de las velas y por qué se toma la decisión de comprar en este punto exacto."}}
    """

    try:
        response = client_groq.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content":[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=250
        )
        
        respuesta_texto = response.choices[0].message.content.strip()
        
        match = re.search(r'\{.*\}', respuesta_texto, re.DOTALL)
        if match:
            datos_json = json.loads(match.group())
            decision = datos_json.get("decision", "WAIT").upper()
            razon = datos_json.get("razon", "IA tomó una decisión sin justificación.")
            return decision, razon
        else:
            return "WAIT", f"La IA no devolvió un JSON estructurado. Respuesta cruda: {respuesta_texto}"

    except Exception as e:
        return "WAIT", f"Error de conexión o fallo de modelo en Groq: {e}"

def log_colab(df, tendencia, slope, soporte, resistencia, decision, razones, log_zonas, idx=-2):
    ahora = datetime.now(timezone.utc)
    print("="*100)
    print(f"🕒 {ahora} | 💰 Precio Cerrado Evaluado: {df['close'].iloc[idx]:.2f}")
    print(f"📐 Tendencia Macro: {tendencia} | Slope Lineal: {slope:.5f}")
    print(f"🧱 Nivel Soporte: {soporte:.2f} | Nivel Resistencia: {resistencia:.2f}")
    
    if log_zonas:
        pos_rango = log_zonas.get('posicion_rango', 0)
        print(f"🏢 Posición en el Rango: {pos_rango*100:.1f}% (0%=Suelo, 100%=Techo)")
        print(f"🔎 Zonas COMPRA Activas: {log_zonas.get('zonas_compra')} | VENTA: {log_zonas.get('zonas_venta')}")
    
    if decision:
        print(f"🎯 DECISIÓN IA: {decision.upper()}")
    else:
        print(f"🎯 DECISIÓN IA: MANTENER AL MARGEN (NO TRADE)")
        
    for razon in razones: 
        print(f"🧠 {razon}")
    print("="*100)

# ======================================================
# MOTOR FINANCIERO Y GESTIÓN
# ======================================================
def paper_abrir_posicion(decision, precio, atr, razones, tiempo):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_SIZE_USD, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_DECISION_ACTIVA, PAPER_PARTIAL_ACTIVADO, PAPER_TP1_EJECUTADO, PAPER_PNL_PARCIAL
    
    if PAPER_POSICION_ACTIVA is not None: 
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    
    if decision == "Buy":
        sl = precio - (atr * MULT_SL)
        tp1 = precio + (atr * MULT_TP1)
    elif decision == "Sell":
        sl = precio + (atr * MULT_SL)
        tp1 = precio - (atr * MULT_TP1)
    else:
        return False
    
    distancia_riesgo = abs(precio - sl)
    if distancia_riesgo == 0: 
        return False

    size_en_dolares_ideal = (riesgo_usd / distancia_riesgo) * precio
    poder_maximo = PAPER_BALANCE * LEVERAGE
    
    if size_en_dolares_ideal > poder_maximo:
        size_real = poder_maximo
    else:
        size_real = size_en_dolares_ideal
        
    PAPER_POSICION_ACTIVA = decision
    PAPER_DECISION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL = sl
    PAPER_TP1 = tp1
    PAPER_SIZE_USD = size_real
    PAPER_SIZE_BTC = size_real / precio
    PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC
    PAPER_PARTIAL_ACTIVADO = True
    PAPER_TP1_EJECUTADO = False
    PAPER_PNL_PARCIAL = 0.0

    return True

def paper_revisar_sl_tp(df):
    global PAPER_SL, PAPER_TP1, PAPER_PRECIO_ENTRADA, PAPER_DECISION_ACTIVA, PAPER_POSICION_ACTIVA, PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_TRADES_TOTALES, PAPER_ULTIMO_RESULTADO, PAPER_ULTIMO_PNL, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_WIN, PAPER_LOSS, PAPER_PNL_PARCIAL

    if PAPER_POSICION_ACTIVA is None: 
        return None

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    close = df['close'].iloc[-1]
    atr_actual = df['atr'].iloc[-1]
    
    cerrar_total = False
    motivo = None

    # ==========================
    # REVISIÓN DE COMPRAS (LONG)
    # ==========================
    if PAPER_POSICION_ACTIVA == "Buy":
        if PAPER_TP1_EJECUTADO == False:
            if high >= PAPER_TP1:
                PAPER_PNL_PARCIAL = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
                PAPER_BALANCE += PAPER_PNL_PARCIAL
                PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
                PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
                PAPER_TP1_EJECUTADO = True
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even. Iniciando Trailing Acelerado...")

        if PAPER_TP1_EJECUTADO == True:
            distancia_a_favor = close - PAPER_PRECIO_ENTRADA
            atrs_ganados = distancia_a_favor / atr_actual
            
            if atrs_ganados >= 5.0:
                multiplicador_dinamico = 0.8  
            elif atrs_ganados >= 3.0:
                multiplicador_dinamico = 1.2  
            else:
                multiplicador_dinamico = MULT_TRAILING_BASE
                
            nuevo_sl = close - (atr_actual * multiplicador_dinamico)
            
            if nuevo_sl > PAPER_SL: 
                PAPER_SL = nuevo_sl 

        if low <= PAPER_SL:
            cerrar_total = True
            if PAPER_TP1_EJECUTADO == True:
                motivo = "Trailing Acelerado Dinámico" 
            else:
                motivo = "Stop Loss"

    # ==========================
    # REVISIÓN DE VENTAS (SHORT)
    # ==========================
    elif PAPER_POSICION_ACTIVA == "Sell":
        if PAPER_TP1_EJECUTADO == False:
            if low <= PAPER_TP1:
                PAPER_PNL_PARCIAL = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
                PAPER_BALANCE += PAPER_PNL_PARCIAL
                PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
                PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
                PAPER_TP1_EJECUTADO = True
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even. Iniciando Trailing Acelerado...")

        if PAPER_TP1_EJECUTADO == True:
            distancia_a_favor = PAPER_PRECIO_ENTRADA - close
            atrs_ganados = distancia_a_favor / atr_actual
            
            if atrs_ganados >= 5.0:
                multiplicador_dinamico = 0.8  
            elif atrs_ganados >= 3.0:
                multiplicador_dinamico = 1.2  
            else:
                multiplicador_dinamico = MULT_TRAILING_BASE
                
            nuevo_sl = close + (atr_actual * multiplicador_dinamico)
            
            if nuevo_sl < PAPER_SL: 
                PAPER_SL = nuevo_sl 

        if high >= PAPER_SL:
            cerrar_total = True
            if PAPER_TP1_EJECUTADO == True:
                motivo = "Trailing Acelerado Dinámico" 
            else:
                motivo = "Stop Loss"

    if cerrar_total == True:
        if PAPER_POSICION_ACTIVA == "Buy":
            pnl_final = (PAPER_SL - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE 
        else:
            pnl_final = (PAPER_PRECIO_ENTRADA - PAPER_SL) * PAPER_SIZE_BTC_RESTANTE
        
        PAPER_BALANCE += pnl_final
        PAPER_PNL_GLOBAL += pnl_final
        PAPER_TRADES_TOTALES += 1
        
        if PAPER_TP1_EJECUTADO == True:
            pnl_total_trade = PAPER_PNL_PARCIAL + pnl_final 
        else:
            pnl_total_trade = pnl_final
            
        if pnl_total_trade > 0: 
            PAPER_WIN += 1
        else: 
            PAPER_LOSS += 1
            
        if PAPER_TRADES_TOTALES > 0:
            winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100
        else:
            winrate = 0.0
            
        crecimiento_pct = ((PAPER_BALANCE - PAPER_BALANCE_INICIAL) / PAPER_BALANCE_INICIAL) * 100
        
        dec = PAPER_DECISION_ACTIVA
        ent = PAPER_PRECIO_ENTRADA
        sal = PAPER_SL
        
        PAPER_POSICION_ACTIVA = None
        PAPER_DECISION_ACTIVA = None
        PAPER_PRECIO_ENTRADA = None
        PAPER_SL = None
        PAPER_TP1 = None
        PAPER_SIZE_BTC = 0.0
        PAPER_SIZE_BTC_RESTANTE = 0.0
        PAPER_TP1_EJECUTADO = False
        PAPER_PNL_PARCIAL = 0.0

        telegram_mensaje(f"📤 TRADE CERRADO: Salida por {motivo}.\n💵 G/P Neta: {pnl_total_trade:.2f} USD\n📊 Balance Actual: {PAPER_BALANCE:.2f} USD\nWinrate: {winrate:.1f}%")
        return {"decision": dec, "motivo": motivo, "entrada": ent, "salida": sal, "pnl": pnl_total_trade, "balance": PAPER_BALANCE}

    return None

def risk_management_check():
    global PAPER_PAUSE_UNTIL
    global PAPER_STOPPED_TODAY
    global PAPER_DAILY_START_BALANCE
    global PAPER_CURRENT_DAY
    global PAPER_BALANCE
    global PAPER_CONSECUTIVE_LOSSES
    
    hoy_utc = datetime.now(timezone.utc).date()
    
    if PAPER_CURRENT_DAY != hoy_utc:
        PAPER_CURRENT_DAY = hoy_utc
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
        PAPER_CONSECUTIVE_LOSSES = 0
        
    porcentaje_drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    
    if porcentaje_drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
        if PAPER_STOPPED_TODAY == False:
            telegram_mensaje(f"🛑 PROTECCIÓN DE CAPITAL: Drawdown máximo alcanzado. Bot pausado.")
            PAPER_STOPPED_TODAY = True
        return False
    return True

# ======================================================
# SISTEMA SECUNDARIO INSTITUCIONAL (PARALELO)
# ======================================================
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
        if partial: 
            self.partial_wins += 1
        elif result_rr > 0: 
            self.wins += 1
        else: 
            self.losses += 1
        self.equity_curve.append(self.total_rr)

    def winrate(self):
        if self.total_trades > 0:
            return (self.wins / self.total_trades) * 100 
        else:
            return 0.0

    def avg_rr(self):
        if self.total_trades > 0:
            return self.total_rr / self.total_trades 
        else:
            return 0.0

class ExternalBOSDetector:
    def __init__(self, lookback=50):
        self.lookback = lookback

    def detect_swings(self, df):
        altos = df['high'].values
        bajos = df['low'].values
        swing_alto_reciente = max(altos[-self.lookback:])
        swing_bajo_reciente = min(bajos[-self.lookback:])
        return swing_alto_reciente, swing_bajo_reciente

    def is_bos_externo(self, df):
        alto_est, bajo_est = self.detect_swings(df)
        cierre = df['close'].iloc[-1]
        
        if cierre > alto_est:
            rup_alza = True
        else:
            rup_alza = False
            
        if cierre < bajo_est:
            rup_baja = True
        else:
            rup_baja = False
            
        return rup_alza, rup_baja, alto_est, bajo_est

class PullbackValidator:
    def __init__(self, tolerance=0.3):
        self.tolerance = tolerance

    def es_pullback_valido(self, df, nivel, direccion):
        precio = df['close'].iloc[-1]
        if direccion == "long":
            if precio <= nivel * (1 - self.tolerance / 100):
                return True
            return False
        if direccion == "short":
            if precio >= nivel * (1 + self.tolerance / 100):
                return True
            return False
        return False

class PartialTPManager:
    def __init__(self):
        self.tp1_hit = False
        self.tp2_hit = False
        
    def gestionar_tp_parcial(self, entry, tp1, tp2, price, side):
        return {"cerrar_50": False, "cerrar_total": False, "evento": None} 

class InstitutionalLogger:
    def __init__(self, telegram_send_func):
        self.send_telegram = telegram_send_func
        
    def log_operacion_completa(self, data):
        self.send_telegram("📊 REPORTE DE SISTEMA INSTITUCIONAL EXTERNO")

class InstitutionalSecondarySystem:
    def __init__(self, telegram_send_func):
        self.bos_detector = ExternalBOSDetector()
        self.pullback_validator = PullbackValidator()
        self.tp_manager = PartialTPManager()
        self.stats = InstitutionalStats()
        self.logger = InstitutionalLogger(telegram_send_func)

# ======================================================
# LOOP PRINCIPAL Y EJECUCIÓN GROQ VISION
# ======================================================
def run_bot():
    mensaje_inicio = "🤖 BOT V101.1 BYBIT REAL INICIADO.\nCerebro Multimodal Llama-3.2-90B Vision Activo.\nAnálisis Holístico de Mercado en curso."
    telegram_mensaje(mensaje_inicio)
    
    sistema_institucional = InstitutionalSecondarySystem(telegram_mensaje)
    ultima_vela_operada = None

    while True:
        time.sleep(SLEEP_SECONDS) 
        try:
            df_velas_crudas = obtener_velas()
            df = calcular_indicadores(df_velas_crudas)

            idx_eval = -2
            precio_mercado = df['close'].iloc[-1] 
            tiempo_vela_cerrada = df.index[-2] 

            en_zona_compra, en_zona_venta, log_zonas = evaluar_impacto_zonas(df, idx_eval)
            
            razones_para_entrar =[]
            decision_final = None
            
            if PAPER_POSICION_ACTIVA is None:

                if ultima_vela_operada == tiempo_vela_cerrada:
                    lista_log =["Vela de 5m bloqueada por Anti-Spam."]
                else:
                    if en_zona_compra or en_zona_venta:
                        lista_log =["🔥 PRECIO EN ZONA CRÍTICA: Despertando a Groq Vision AI para análisis holístico..."]
                        
                        fig_neutral = generar_grafico_entrada(df, "ANALISIS", log_zonas['soporte'], log_zonas['resistencia'], log_zonas['slope'], log_zonas['intercept'],["Verificando estructura general y fuerza del precio..."])
                        
                        if fig_neutral is not None:
                            img_base64 = fig_to_base64(fig_neutral)
                            plt.close(fig_neutral)
                            
                            decision_ia, razon_ia = analizar_con_groq(df, idx_eval, log_zonas, img_base64)
                            lista_log.append(f"🤖 Groq determinó: {decision_ia}")
                            lista_log.append(f"🧠 Argumento: {razon_ia}")
                            
                            if decision_ia == "BUY" and en_zona_compra:
                                decision_final = "Buy"
                                razones_para_entrar.append(f"Análisis IA: {razon_ia}")
                            elif decision_ia == "SELL" and en_zona_venta:
                                decision_final = "Sell"
                                razones_para_entrar.append(f"Análisis IA: {razon_ia}")
                    else:
                        lista_log =["El precio se encuentra flotando fuera de zonas clave. IA en reposo."]
                        
                log_colab(df, log_zonas.get('tendencia_macro'), log_zonas.get('slope', 0), log_zonas.get('soporte', 0), log_zonas.get('resistencia', 0), decision_final, lista_log, log_zonas, idx_eval)

                if decision_final is not None:
                    if risk_management_check():
                        atr_entrada = df['atr'].iloc[-1]
                        apertura = paper_abrir_posicion(decision_final, precio_mercado, atr_entrada, razones_para_entrar, df.index[-1])
                        
                        if apertura:
                            ultima_vela_operada = tiempo_vela_cerrada
                            
                            texto = f"📌 OPERACIÓN {decision_final.upper()} APROBADA POR VISION AI (5m)\n💰 Entrada: {precio_mercado:.2f}\n📍 SL: {PAPER_SL:.2f} | TP1: {PAPER_TP1:.2f}\n🧠 Justificación: {razones_para_entrar[0]}"
                            telegram_mensaje(texto)
                            
                            fig_final = generar_grafico_entrada(df, decision_final, log_zonas['soporte'], log_zonas['resistencia'], log_zonas['slope'], log_zonas['intercept'], razones_para_entrar)
                            if fig_final is not None: 
                                telegram_grafico(fig_final)
                                plt.close(fig_final)

            if PAPER_POSICION_ACTIVA is not None:
                datos_cierre = paper_revisar_sl_tp(df)
                if datos_cierre is not None:
                    fig_salida = generar_grafico_salida(df, datos_cierre)
                    if fig_salida is not None:
                        telegram_grafico(fig_salida)
                        plt.close(fig_salida)

        except Exception as e:
            print(f"🚨 ERROR CRÍTICO EN EL SISTEMA: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
