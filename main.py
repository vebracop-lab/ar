# BOT TRADING V96.0 BYBIT REAL – PRODUCCIÓN (SIN PROXY) 
# ======================================================
# ⚠️ KEYS INCLUIDAS TAL CUAL (SEGÚN PEDIDO)
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit
# ======================================================
# NOVEDADES V96.0 (TOTAL UNBLOCKED SCALPER):
# - FIX DEFINITIVO: Eliminada por completo la restricción de "Micro-Tendencia" 
#   (Pullback) de las funciones de Nison. Ahora el bot dispara inmediatamente
#   al detectar la forma de la vela en la zona de Soporte/Resistencia.
# - Patrones relajados al extremo para imitar el "ojo humano" en Crypto.
# - CÓDIGO 100% EXPANDIDO. CERO COMPRESIÓN.
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

def cerca_de_nivel(precio, nivel, margen):
    distancia = abs(precio - nivel)
    if distancia <= margen:
        return True
    else:
        return False

SYMBOL = "BTCUSDT"
INTERVAL = "5"  
RISK_PER_TRADE = 0.02  # 2% de riesgo ($2.00 USD por trade en cuenta de $100)
LEVERAGE = 10          # 10x de apalancamiento
SLEEP_SECONDS = 60     

# MULTIPLICADORES DE RIESGO/BENEFICIO (EQUILIBRADO 5m)
MULT_SL = 1.5          
MULT_TP1 = 2.5         
MULT_TRAILING = 2.0    
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
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df.dropna()

def detectar_zonas_mercado(df, idx=-2, ventana_macro=120):
    df_eval = df.iloc[:idx+1]
    
    # Soportes Horizontales (40 velas = 3.3 horas en 5m)
    soporte_horiz = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia_horiz = df_eval['high'].rolling(40).max().iloc[-1]
    
    # Tendencia Macro y Canal Dinámico
    if len(df_eval) < ventana_macro:
        y_macro = df_eval['close'].values
    else:
        y_macro = df_eval['close'].values[-ventana_macro:]
        
    x_macro = np.arange(len(y_macro))
    slope, intercept, r_value, p_value, std_err = linregress(x_macro, y_macro)
    
    if slope > 0.01: 
        tendencia_macro = '📈 ALCISTA'
    elif slope < -0.01: 
        tendencia_macro = '📉 BAJISTA'
    else: 
        tendencia_macro = '➡️ LATERAL'
        
    linea_central = intercept + slope * x_macro
    residuos = y_macro - linea_central
    desviacion = np.std(residuos)
    
    canal_sup = linea_central[-1] + (desviacion * 1.5)
    canal_inf = linea_central[-1] - (desviacion * 1.5)
    
    return soporte_horiz, resistencia_horiz, canal_sup, canal_inf, slope, intercept, tendencia_macro

def tendencia_previa_micro(df, idx, velas=6):
    """ Función puramente informativa para los logs. Ya no bloquea trades. """
    if idx - velas < 0: 
        return "neutral"
    
    rango_eval = df.iloc[idx-velas : idx+1]
    max_rango = rango_eval['high'].max()
    min_rango = rango_eval['low'].min()
    rango_total = max_rango - min_rango
    
    if rango_total == 0: 
        return "neutral"
        
    precio_patron = df['close'].iloc[idx]
    posicion_pct = (precio_patron - min_rango) / rango_total
    
    if posicion_pct <= 0.35: 
        return "bajista" 
    elif posicion_pct >= 0.65: 
        return "alcista" 
    else:
        return "lateral"

# ======================================================
# 🕯️ ARSENAL NISON (TOTALMENTE DESBLOQUEADO Y RELAJADO)
# ======================================================
def calcular_cuerpo_mechas(row):
    cuerpo = abs(row['close'] - row['open'])
    if cuerpo == 0:
        cuerpo = 0.0001
        
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

# --- 1. HAMMER (MARTILLO) ---
def es_hammer_nison(df, idx):
    vela = df.iloc[idx]
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela)
    
    # 1.2x el cuerpo es un martillo visualmente válido. Mecha superior no estorba.
    if m_inf >= (1.2 * cuerpo) and m_sup <= (m_inf * 0.8):
        return True
    return False

# --- 2. SHOOTING STAR (ESTRELLA FUGAZ) ---
def es_shooting_star_nison(df, idx):
    vela = df.iloc[idx]
    cuerpo, m_sup, m_inf, rango, top, bottom = calcular_cuerpo_mechas(vela)
    
    if m_sup >= (1.2 * cuerpo) and m_inf <= (m_sup * 0.8):
        return True
    return False

# --- 3. BULLISH ENGULFING (ENVOLVENTE ALCISTA) ---
def es_bullish_engulfing_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    # La vela anterior debe ser roja (o doji) y la actual verde
    if prev['close'] <= prev['open'] and curr['close'] > curr['open']:
        # Solo exige que el cuerpo verde supere la apertura de la roja
        if curr['close'] >= prev['open'] and curr['open'] <= prev['close']:
            return True
    return False

# --- 4. BEARISH ENGULFING (ENVOLVENTE BAJISTA) ---
def es_bearish_engulfing_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if prev['close'] >= prev['open'] and curr['close'] < curr['open']:
        if curr['close'] <= prev['open'] and curr['open'] >= prev['close']:
            return True
    return False

# --- 5. PIERCING PATTERN (PAUTA PENETRANTE) ---
def es_piercing_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if prev['close'] < prev['open'] and curr['close'] > curr['open']:
        # Recuperar aunque sea un poco del cuerpo rojo es válido en Crypto (No pide Gaps)
        if curr['close'] > prev['close']:
            return True
    return False

# --- 6. DARK CLOUD COVER (NUBE OSCURA) ---
def es_dark_cloud_nison(df, idx):
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if prev['close'] > prev['open'] and curr['close'] < curr['open']:
        if curr['close'] < prev['close']:
            return True
    return False

# --- 7. MORNING STAR (ESTRELLA DE LA MAÑANA) ---
def es_morning_star_nison(df, idx):
    if idx < 2: 
        return False
        
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    if c1['close'] < c1['open'] and c3['close'] > c3['open']:
        # La vela 3 debe recuperar por encima de la estrella
        if c3['close'] > c2['high']:
            return True
    return False

# --- 8. EVENING STAR (ESTRELLA DEL ATARDECER) ---
def es_evening_star_nison(df, idx):
    if idx < 2: 
        return False
        
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    if c1['close'] > c1['open'] and c3['close'] < c3['open']:
        if c3['close'] < c2['low']:
            return True
    return False

# --- 9. TWEEZER BOTTOMS ---
def es_tweezer_bottom_nison(df, idx):
    c1 = df.iloc[idx-1]
    c2 = df.iloc[idx]
    
    tolerancia = df['atr'].iloc[idx] * 0.50 
    
    if abs(c2['low'] - c1['low']) <= tolerancia:
        if c2['close'] > c2['open']:  # La segunda rechaza al alza
            return True
    return False

# --- 10. TWEEZER TOPS ---
def es_tweezer_top_nison(df, idx):
    c1 = df.iloc[idx-1]
    c2 = df.iloc[idx]
    
    tolerancia = df['atr'].iloc[idx] * 0.50
    
    if abs(c2['high'] - c1['high']) <= tolerancia:
        if c2['close'] < c2['open']:
            return True
    return False

# --- 11. THREE WHITE SOLDIERS ---
def es_three_white_soldiers(df, idx):
    if idx < 2: 
        return False
        
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    if c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open']:
        if c2['close'] > c1['close'] and c3['close'] > c2['close']:
            return True
    return False

# --- 12. THREE BLACK CROWS ---
def es_three_black_crows(df, idx):
    if idx < 2: 
        return False
        
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    if c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open']:
        if c2['close'] < c1['close'] and c3['close'] < c2['close']:
            return True
    return False

# === DETECTOR MAESTRO NISON MULTI-ZONA ===
def detectar_patron_nison(df, sop_horiz, res_horiz, canal_sup, canal_inf, idx=-2):
    if len(df) < 15: 
        return False, None, None, {}
        
    atr_actual = df['atr'].iloc[idx]
    ema20_actual = df['ema20'].iloc[idx]
    precio_cierre = df['close'].iloc[idx]
    
    # Evalúa el mínimo de todo el patrón posible (últimas 3 velas)
    min_patron = df['low'].iloc[idx-2 : idx+1].min()
    max_patron = df['high'].iloc[idx-2 : idx+1].max()
    
    micro_tendencia = tendencia_previa_micro(df, idx)
    
    # Tolerancia balanceada (2.0x ATR o mínimo 150 USD) -> ¡Abre el embudo!
    tolerancia_zona = max(atr_actual * 2.0, 150)
    
    toca_sop_horiz = cerca_de_nivel(min_patron, sop_horiz, tolerancia_zona) 
    toca_res_horiz = cerca_de_nivel(max_patron, res_horiz, tolerancia_zona)
    
    toca_sop_dinamico = cerca_de_nivel(min_patron, canal_inf, tolerancia_zona)
    toca_res_dinamica = cerca_de_nivel(max_patron, canal_sup, tolerancia_zona)
    
    toca_ema20_sop = cerca_de_nivel(min_patron, ema20_actual, tolerancia_zona) and (precio_cierre >= ema20_actual)
    toca_ema20_res = cerca_de_nivel(max_patron, ema20_actual, tolerancia_zona) and (precio_cierre <= ema20_actual)
    
    en_zona_compra = toca_sop_horiz or toca_sop_dinamico or toca_ema20_sop
    en_zona_venta = toca_res_horiz or toca_res_dinamica or toca_ema20_res
    
    zonas_compra =[]
    if toca_sop_horiz: 
        zonas_compra.append("Soporte Horizontal")
    if toca_sop_dinamico: 
        zonas_compra.append("Canal Inferior")
    if toca_ema20_sop: 
        zonas_compra.append("EMA 20")
        
    zonas_venta =[]
    if toca_res_horiz: 
        zonas_venta.append("Resistencia Horizontal")
    if toca_res_dinamica: 
        zonas_venta.append("Canal Superior")
    if toca_ema20_res: 
        zonas_venta.append("EMA 20")
    
    if len(zonas_compra) > 0:
        texto_zonas_compra = ", ".join(zonas_compra)
    else:
        texto_zonas_compra = "Ninguna"
        
    if len(zonas_venta) > 0:
        texto_zonas_venta = ", ".join(zonas_venta)
    else:
        texto_zonas_venta = "Ninguna"
    
    log_zonas = {
        "tolerancia": tolerancia_zona, 
        "en_soporte": en_zona_compra, 
        "en_resistencia": en_zona_venta,
        "zonas_compra": texto_zonas_compra,
        "zonas_venta": texto_zonas_venta,
        "micro_tendencia": micro_tendencia
    }
    
    # --- PATRONES DE COMPRA (SIN RESTRICCIÓN DE MICRO-TENDENCIA) ---
    if en_zona_compra:
        if es_hammer_nison(df, idx): return True, "Buy", "Nison Hammer", log_zonas
        if es_bullish_engulfing_nison(df, idx): return True, "Buy", "Nison Bullish Engulfing", log_zonas
        if es_piercing_nison(df, idx): return True, "Buy", "Nison Piercing Pattern", log_zonas
        if es_morning_star_nison(df, idx): return True, "Buy", "Nison Morning Star", log_zonas
        if es_tweezer_bottom_nison(df, idx): return True, "Buy", "Nison Tweezer Bottoms", log_zonas
        if es_three_white_soldiers(df, idx): return True, "Buy", "Three White Soldiers", log_zonas

    # --- PATRONES DE VENTA (SIN RESTRICCIÓN DE MICRO-TENDENCIA) ---
    if en_zona_venta:
        if es_shooting_star_nison(df, idx): return True, "Sell", "Nison Shooting Star", log_zonas
        if es_bearish_engulfing_nison(df, idx): return True, "Sell", "Nison Bearish Engulfing", log_zonas
        if es_dark_cloud_nison(df, idx): return True, "Sell", "Nison Dark Cloud Cover", log_zonas
        if es_evening_star_nison(df, idx): return True, "Sell", "Nison Evening Star", log_zonas
        if es_tweezer_top_nison(df, idx): return True, "Sell", "Nison Tweezer Tops", log_zonas
        if es_three_black_crows(df, idx): return True, "Sell", "Three Black Crows", log_zonas

    return False, None, None, log_zonas

# ======================================================
# GRÁFICOS MATPLOTLIB
# ======================================================
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

        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label="Soporte")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label="Resistencia")

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
        else:
            ax.scatter(entrada_x_idx, closes[-2], s=200, marker='v', color='red', edgecolors='black', zorder=5)

        texto_razones = "\n".join(razones)
        texto_panel = f"OPERACION: {decision.upper()}\nPrecio: {df['close'].iloc[-1]:.2f}\nRSI Contexto: {df['rsi'].iloc[-2]:.1f}\n\nRazones:\n{texto_razones}"
        
        ax.text(0.02, 0.98, texto_panel, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_title(f"BOT V96.0 - Entrada {decision} Multi-Zona (5m)")
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

        texto_panel = f"CIERRE {decision_original}\nMotivo: {motivo_cierre}\nPnL: {pnl_obtenido:.4f} USD\nBalance: {balance_actual:.2f} USD"
        ax.text(0.02, 0.95, texto_panel, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title("BOT V96.0 - DETALLE DE CIERRE")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        return None

def log_colab(df, tendencia, slope, soporte, resistencia, decision, razones, log_zonas, idx=-2):
    ahora = datetime.now(timezone.utc)
    precio = df['close'].iloc[idx]
    
    print("="*100)
    print(f"🕒 {ahora} | 💰 Precio Cerrado: {precio:.2f}")
    
    if log_zonas:
        print(f"🔎 Zonas COMPRA Activas: {log_zonas.get('zonas_compra')} | VENTA: {log_zonas.get('zonas_venta')}")
        print(f"🔎 Micro Tendencia Real (Rango Swing): {log_zonas.get('micro_tendencia').upper()} (Tolerancia Zona: {log_zonas.get('tolerancia', 0):.2f} USD)")
        
    if decision:
        print(f"🎯 DECISIÓN: {decision.upper()}")
    else:
        print(f"🎯 DECISIÓN: NO TRADE")
        
    for razon in razones: 
        print(f"🧠 Lógica: {razon}")
    print("="*100)

# ======================================================
# MOTOR FINANCIERO Y GESTIÓN
# ======================================================
def paper_abrir_posicion(decision, precio, atr, razones, tiempo):
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
    global PAPER_PNL_PARCIAL
    
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
        
    size_en_cripto_real = size_real / precio
    
    PAPER_POSICION_ACTIVA = decision
    PAPER_DECISION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL = sl
    PAPER_TP1 = tp1
    PAPER_TP2 = None  
    PAPER_SIZE_USD = size_real
    PAPER_SIZE_BTC = size_en_cripto_real
    PAPER_SIZE_BTC_RESTANTE = size_en_cripto_real
    PAPER_PARTIAL_ACTIVADO = True
    PAPER_TP1_EJECUTADO = False
    PAPER_PNL_PARCIAL = 0.0

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
    global PAPER_WIN
    global PAPER_LOSS
    global PAPER_PNL_PARCIAL

    if PAPER_POSICION_ACTIVA is None: 
        return None

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    close = df['close'].iloc[-1]
    atr_actual = df['atr'].iloc[-1]
    
    cerrar_total = False
    motivo = None

    if PAPER_POSICION_ACTIVA == "Buy":
        if PAPER_TP1_EJECUTADO == False:
            if high >= PAPER_TP1:
                PAPER_PNL_PARCIAL = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
                PAPER_BALANCE += PAPER_PNL_PARCIAL
                PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
                PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
                PAPER_TP1_EJECUTADO = True
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even. Iniciando Trailing...")

        if PAPER_TP1_EJECUTADO == True:
            if close - (atr_actual * MULT_TRAILING) > PAPER_SL: 
                PAPER_SL = close - (atr_actual * MULT_TRAILING)

        if low <= PAPER_SL:
            cerrar_total = True
            if PAPER_TP1_EJECUTADO == True:
                motivo = "Trailing Dinámico" 
            else:
                motivo = "Stop Loss"

    elif PAPER_POSICION_ACTIVA == "Sell":
        if PAPER_TP1_EJECUTADO == False:
            if low <= PAPER_TP1:
                PAPER_PNL_PARCIAL = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE)
                PAPER_BALANCE += PAPER_PNL_PARCIAL
                PAPER_PNL_GLOBAL += PAPER_PNL_PARCIAL
                PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC * (1 - PORCENTAJE_CIERRE)
                PAPER_TP1_EJECUTADO = True
                PAPER_SL = PAPER_PRECIO_ENTRADA 
                telegram_mensaje(f"🎯 TP1 ALCANZADO (+{PAPER_PNL_PARCIAL:.2f} USD). SL a Break Even. Iniciando Trailing...")

        if PAPER_TP1_EJECUTADO == True:
            if close + (atr_actual * MULT_TRAILING) < PAPER_SL: 
                PAPER_SL = close + (atr_actual * MULT_TRAILING)

        if high >= PAPER_SL:
            cerrar_total = True
            if PAPER_TP1_EJECUTADO == True:
                motivo = "Trailing Dinámico" 
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
        
        datos_retorno = {
            "decision": dec, 
            "motivo": motivo, 
            "entrada": ent, 
            "salida": sal, 
            "pnl": pnl_total_trade, 
            "balance": PAPER_BALANCE
        }
        return datos_retorno

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
        
        if cierre_actual > alto_estructural:
            ruptura_al_alza = True
        else:
            ruptura_al_alza = False
            
        if cierre_actual < bajo_estructural:
            ruptura_a_la_baja = True
        else:
            ruptura_a_la_baja = False
            
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
🔢 Cantidad de Operaciones: {data.get('total_trades')}
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
# LOOP PRINCIPAL Y ANTI-SPAM
# ======================================================
def run_bot():
    mensaje_inicio = "🤖 BOT V96.0 BYBIT REAL INICIADO.\nConfiguración FLUID SNIPER (5m MULTI-ZONA).\nRestricción de Micro-Tendencia Eliminada de los Patrones."
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

            # Detección integral unificada
            soporte_horiz, resistencia_horiz, canal_sup, canal_inf, slope, intercept, tendencia_macro = detectar_zonas_mercado(df, idx_eval)
            
            razones_para_entrar =[]
            
            if PAPER_POSICION_ACTIVA is None:
                patron_detectado, decision_final, nombre_patron, log_zonas = detectar_patron_nison(df, soporte_horiz, resistencia_horiz, canal_sup, canal_inf, idx=idx_eval)

                if ultima_vela_operada == tiempo_vela_cerrada:
                    decision_final = None
                    if patron_detectado:
                        lista_log =[f"Patrón {nombre_patron} bloqueado (Anti-Spam de vela)"]
                        patron_detectado = False
                    else:
                        lista_log =["Buscando patrón Nison CERRADO válido..."]
                else:
                    if patron_detectado == True:
                        lista_log = [nombre_patron]
                    else:
                        lista_log =["Buscando patrón Nison CERRADO válido..."]
                    
                log_colab(df, tendencia_macro, slope, soporte_horiz, resistencia_horiz, decision_final, lista_log, log_zonas, idx_eval)

                if decision_final is not None:
                    if decision_final == "Buy":
                        zonas_activadas = log_zonas.get('zonas_compra')
                    else:
                        zonas_activadas = log_zonas.get('zonas_venta')
                        
                    razones_para_entrar.append(f"Patrón Validado: {nombre_patron}")
                    razones_para_entrar.append(f"Rebote Confirmado en: {zonas_activadas}")
                    
                    riesgo_valido = risk_management_check()
                    
                    if riesgo_valido == True:
                        atr_entrada = df['atr'].iloc[-1]
                        apertura = paper_abrir_posicion(decision_final, precio_mercado, atr_entrada, razones_para_entrar, df.index[-1])
                        
                        if apertura == True:
                            ultima_vela_operada = tiempo_vela_cerrada
                            
                            texto = f"📌 OPERACIÓN {decision_final.upper()} (5m)\n💰 Entrada: {precio_mercado:.2f}\n📍 SL: {PAPER_SL:.2f} | TP1: {PAPER_TP1:.2f}\n🧠 {', '.join(razones_para_entrar)}"
                            telegram_mensaje(texto)
                            
                            fig = generar_grafico_entrada(df, decision_final, soporte_horiz, resistencia_horiz, slope, intercept, razones_para_entrar)
                            if fig is not None: 
                                telegram_grafico(fig)
                                plt.close(fig)

            if PAPER_POSICION_ACTIVA is not None:
                datos_cierre = paper_revisar_sl_tp(df)
                if datos_cierre is not None:
                    fig_salida = generar_grafico_salida(df, datos_cierre)
                    if fig_salida is not None:
                        telegram_grafico(fig_salida)
                        plt.close(fig_salida)

        except Exception as e:
            texto_error = f"🚨 ERROR CRÍTICO EN EL SISTEMA: {e}"
            print(texto_error)
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
