# BOT TRADING V99.18 – GROQ (STEVE NISON COMPLETO: PATRONES, ESTRUCTURA, CONTINUACIÓN)
# ====================================================================================
# - Detecta patrones de velas: Doji, Martillo, Estrella fugaz, Envolvente, Tres soldados, etc.
# - Interpreta confluencia con soportes/resistencias, EMA, tendencia, rupturas.
# - Sugiere SL/TP adaptados al contexto y timeframe 5m.
# - Autoaprendizaje y logs completos.

import os, time, requests, json, re, numpy as np, pandas as pd
from scipy.stats import linregress
from datetime import datetime, timezone
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Falta GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODELO_TEXTO = "llama-3.3-70b-versatile"

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

# =================== PAPER TRADING ===================
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
ULTIMOS_MULTIS = (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

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
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    y = df_eval['close'].values[-ventana_macro:] if len(df_eval) >= ventana_macro else df_eval['close'].values
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    tendencia = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tendencia

# =================== PATRONES DE VELAS (NISON) ===================
def detectar_patron_vela(open_, high, low, close):
    """Retorna (nombre_patron, tipo: 'alcista', 'bajista', 'indecision', 'continuacion') y descripción"""
    cuerpo = abs(close - open_)
    rango = high - low
    if rango == 0:
        return "Vela indeterminada", "neutral", "Rango cero"
    cuerpo_pct = (cuerpo / rango) * 100
    sombra_sup = high - max(close, open_)
    sombra_inf = min(close, open_) - low
    sombra_sup_pct = (sombra_sup / rango) * 100
    sombra_inf_pct = (sombra_inf / rango) * 100
    es_verde = close >= open_

    # Patrones de indecisión
    if cuerpo_pct < 10:
        return "DOJI", "indecision", f"Cuerpo {cuerpo_pct:.1f}%, sombras iguales"

    # Patrones de reversión alcista
    if es_verde and sombra_inf_pct > 60 and cuerpo_pct < 30 and sombra_sup_pct < 10:
        return "MARTILLO", "alcista", "Mecha inferior larga, cuerpo pequeño arriba, rebote en soporte"
    if not es_verde and sombra_sup_pct > 60 and cuerpo_pct < 30 and sombra_inf_pct < 10:
        return "ESTRELLA FUGAZ", "bajista", "Mecha superior larga, cuerpo pequeño abajo, rechazo en resistencia"
    if not es_verde and sombra_inf_pct > 60 and cuerpo_pct < 30 and sombra_sup_pct < 10:
        return "HOMBRE COLGADO", "bajista", "Mecha inferior larga en vela roja, señal de agotamiento alcista"
    
    # Patrones de continuación (vela larga sin mechas)
    if cuerpo_pct > 70 and sombra_sup_pct < 15 and sombra_inf_pct < 15:
        if es_verde:
            return "VELA LARGA ALCISTA", "continuacion_alcista", "Fuerte presión compradora"
        else:
            return "VELA LARGA BAJISTA", "continuacion_bajista", "Fuerte presión vendedora"
    
    # Rechazo / rebote (no necesariamente patrón clásico pero útil)
    if sombra_sup_pct > 50 and cuerpo_pct < 50:
        return "RECHAZO EN RESISTENCIA", "bajista", "Mecha superior larga, probable venta"
    if sombra_inf_pct > 50 and cuerpo_pct < 50:
        return "REBOTE EN SOPORTE", "alcista", "Mecha inferior larga, probable compra"
    
    return "VELA NORMAL", "neutral", f"Cuerpo {cuerpo_pct:.0f}%"

def detectar_patron_envolvente(df, idx):
    """Detecta patrón envolvente alcista o bajista en la última vela cerrada y la anterior"""
    if idx < 1: return None
    vela1 = df.iloc[idx-1]  # anterior
    vela2 = df.iloc[idx]    # actual (última cerrada)
    cuerpo1 = vela1['close'] - vela1['open']
    cuerpo2 = vela2['close'] - vela2['open']
    # Envolvente alcista: vela actual verde que envuelve completamente a la vela anterior roja
    if cuerpo2 > 0 and cuerpo1 < 0:
        if vela2['open'] < vela1['close'] and vela2['close'] > vela1['open']:
            return "ENVOLVENTE ALCISTA", "alcista", "Vela verde envuelve vela roja anterior, reversión alcista"
    # Envolvente bajista: vela actual roja que envuelve a la anterior verde
    if cuerpo2 < 0 and cuerpo1 > 0:
        if vela2['open'] > vela1['close'] and vela2['close'] < vela1['open']:
            return "ENVOLVENTE BAJISTA", "bajista", "Vela roja envuelve vela verde anterior, reversión bajista"
    return None

def detectar_patrones_multiples(df, idx):
    """Tres soldados blancos / tres cuervos negros (continuación)"""
    if idx < 2: return None
    ultimas = df.iloc[idx-2:idx+1]
    colores = ['VERDE' if c >= o else 'ROJA' for o, c in zip(ultimas['open'], ultimas['close'])]
    cuerpos = [abs(c - o) for o, c in zip(ultimas['open'], ultimas['close'])]
    # Tres soldados blancos: 3 velas verdes consecutivas con cuerpos crecientes
    if all(c == 'VERDE' for c in colores) and cuerpos[-1] > cuerpos[-2] > cuerpos[-3]:
        return "TRES SOLDADOS BLANCOS", "continuacion_alcista", "Fuerte continuación alcista"
    # Tres cuervos negros: 3 velas rojas consecutivas con cuerpos crecientes
    if all(c == 'ROJA' for c in colores) and cuerpos[-1] > cuerpos[-2] > cuerpos[-3]:
        return "TRES CUERVOS NEGROS", "continuacion_bajista", "Fuerte continuación bajista"
    return None

# =================== DESCRIPCIÓN TEXTUAL ENRIQUECIDA ===================
def generar_descripcion_nison(df, idx=-2):
    precio = df['close'].iloc[idx]
    atr = df['atr'].iloc[idx]
    ema_val = df['ema20'].iloc[idx]
    rsi = df['rsi'].iloc[idx]
    macd = df['macd'].iloc[idx]
    signal = df['signal'].iloc[idx]
    hist = df['macd_hist'].iloc[idx]
    soporte, resistencia, slope, intercept, tendencia = detectar_zonas_mercado(df, idx)
    
    diff_ema_pct = (precio - ema_val) / ema_val * 100
    toques_ema = sum(1 for i in range(max(0, idx-5), idx+1) if df['low'].iloc[i] <= df['ema20'].iloc[i] <= df['high'].iloc[i])
    if abs(diff_ema_pct) < 0.15:
        pos_ema = f"PRECIO JUSTO EN LA EMA20 (tocando exactamente, {toques_ema} toques)"
    elif precio > ema_val:
        pos_ema = f"PRECIO ENCIMA DE EMA20 (+{diff_ema_pct:.2f}%, {toques_ema} toques)"
    else:
        pos_ema = f"PRECIO DEBAJO DE EMA20 ({diff_ema_pct:.2f}%, {toques_ema} toques)"
    
    if toques_ema >= 2 and precio > ema_val:
        rol_ema = "✅ EMA20 actúa como SOPORTE DINÁMICO"
    elif toques_ema >= 2 and precio < ema_val:
        rol_ema = "❌ EMA20 actúa como RESISTENCIA DINÁMICA"
    else:
        rol_ema = "⚠️ EMA20 sin rol claro"
    
    # Análisis de velas individuales (últimas 8)
    velas_info = []
    for i, v in df.iloc[max(0, len(df)-9):-1].iterrows():
        patron, tipo, desc = detectar_patron_vela(v['open'], v['high'], v['low'], v['close'])
        velas_info.append(f"{i.strftime('%H:%M')}: {patron} ({tipo}) - {desc} | Cierre {v['close']:.2f}")
    
    # Patrón envolvente (última vela cerrada con la anterior)
    env = detectar_patron_envolvente(df, idx)
    if env:
        velas_info.append(f"✨ PATRÓN ENVOLVENTE: {env[0]} - {env[2]}")
    
    # Patrones múltiples (tres soldados, tres cuervos)
    multi = detectar_patrones_multiples(df, idx)
    if multi:
        velas_info.append(f"✨ PATRÓN MÚLTIPLE: {multi[0]} - {multi[2]}")
    
    # Detección de mechas largas en últimas 5 velas (corregido)
    mechas_inf = 0
    mechas_sup = 0
    for _, row in df.iloc[-6:-1].iterrows():
        rango = row['high'] - row['low']
        if rango > 0:
            som_inf = (min(row['close'], row['open']) - row['low']) / rango
            som_sup = (row['high'] - max(row['close'], row['open'])) / rango
            if som_inf > 0.5:
                mechas_inf += 1
            if som_sup > 0.5:
                mechas_sup += 1
    alerta_mechas = ""
    if mechas_inf >= 2:
        alerta_mechas += f"⚠️ {mechas_inf} velas con MECHA INFERIOR LARGA (soporte, presión compradora → ALCISTA). "
    if mechas_sup >= 2:
        alerta_mechas += f"⚠️ {mechas_sup} velas con MECHA SUPERIOR LARGA (resistencia, presión vendedora → BAJISTA). "
    
    # Estructura de mercado (rechazos en S/R)
    df_cercano = df.iloc[-20:]
    toques_res = sum(1 for _, v in df_cercano.iterrows() if v['high'] >= resistencia * 0.998)
    toques_sop = sum(1 for _, v in df_cercano.iterrows() if v['low'] <= soporte * 1.002)
    estructura = ""
    if toques_res >= 3:
        estructura += f"⚠️ RESISTENCIA ({resistencia:.0f}) RECHAZADA {toques_res} VECES (TECHO SÓLIDO) → BAJISTA. "
    if toques_sop >= 3:
        estructura += f"✅ SOPORTE ({soporte:.0f}) PROBADO {toques_sop} VECES (SUELO SÓLIDO) → ALCISTA. "
    
    # Ruptura de tendencia
    ruptura = ""
    if len(df) > 20:
        pend_reciente, _, _, _, _ = linregress(np.arange(20), df['close'].iloc[-20:].values)
        if slope > 0 and pend_reciente < 0:
            ruptura = "⚠️ RUPTURA DE TENDENCIA ALCISTA (pendiente reciente negativa) → BAJISTA"
        elif slope < 0 and pend_reciente > 0:
            ruptura = "✅ RUPTURA DE TENDENCIA BAJISTA (pendiente reciente positiva) → ALCISTA"
    
    # RSI y MACD
    if rsi > 70: rsi_texto = "SOBRECOMPRADO (>70) → posible BAJISTA"
    elif rsi < 30: rsi_texto = "SOBREVENDIDO (<30) → posible ALCISTA"
    else: rsi_texto = f"NEUTRAL ({rsi:.1f})"
    
    if macd > signal and hist > 0:
        macd_texto = "ALCISTA (MACD > señal, histograma +)"
    elif macd < signal and hist < 0:
        macd_texto = "BAJISTA (MACD < señal, histograma -)"
    else:
        macd_texto = "NEUTRAL"
    
    descripcion = f"""
=== ANÁLISIS STEVE NISON – BTCUSDT 5m ===
Precio: {precio:.2f} | ATR: {atr:.2f}
Tendencia macro: {tendencia} (pendiente {slope:.6f})
{ruptura}
RSI: {rsi:.1f} - {rsi_texto}
MACD: {macd_texto} (hist {hist:.2f})
{pos_ema}
{rol_ema}
Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}
{estructura}
{alerta_mechas}
PATRONES DE VELAS (últimas 8):
{chr(10).join(velas_info)}
"""
    return descripcion, atr

# =================== IA GROQ (PROMPT MEJORADO) ===================
def analizar_con_groq_texto(descripcion, atr):
    try:
        system_msg = """
Eres Steve Nison, el mayor experto mundial en velas japonesas y análisis técnico.
Tu conocimiento incluye patrones de reversión (martillo, estrella fugaz, hombre colgado, envolvente), patrones de continuación (tres soldados blancos, tres cuervos negros, velas largas sin mechas), patrones de indecisión (doji), y estructura de mercado (soportes/resistencias, tendencia, rupturas).
Interpreta TODO el contexto que se te proporciona: velas individuales, patrones múltiples, mechas largas, toques en EMA, rechazos en niveles, RSI, MACD, ruptura de tendencia.

REGLAS PARA DECIDIR (Buy/Sell/Hold):
- Si ves un PATRÓN DE REVERSIÓN ALCISTA (martillo, envolvente alcista, rebote en soporte) Y el precio está en una zona de soporte (horizontal, EMA o línea de tendencia) → BUY.
- Si ves un PATRÓN DE REVERSIÓN BAJISTA (estrella fugaz, hombre colgado, envolvente bajista, rechazo en resistencia) Y el precio está en resistencia → SELL.
- Si ves PATRONES DE CONTINUACIÓN (tres soldados blancos, velas largas alcistas) en una tendencia alcista → BUY. Si son bajistas en tendencia bajista → SELL.
- Si hay CONFLUENCIA de señales (ej. martillo + soporte + RSI sobrevendido) → mayor convicción.
- Si el mercado está lateral y no hay patrones claros, o las señales son contradictorias → HOLD.
- Además, sugiere multiplicadores SL, TP1 y Trailing basados en el ATR y la volatilidad actual, teniendo en cuenta que es gráfico de 5 minutos (rangos conservadores: SL 1.0-2.0, TP1 1.2-2.5, Trailing 1.2-2.0).

Responde ÚNICAMENTE con un JSON válido en este formato:
{
  "decision": "Buy/Sell/Hold",
  "patron": "nombre del patrón o situación clave",
  "razones": ["razón1","razón2","razón3"],
  "sl_mult": 1.5,
  "tp1_mult": 1.8,
  "trailing_mult": 1.6
}
No incluyas texto adicional fuera del JSON.
"""
        user_msg = f"{descripcion}\nATR actual: {atr:.2f}\n\nRecomienda multiplicadores realistas para 5m."
        respuesta = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=400
        )
        raw = respuesta.choices[0].message.content
        print(f"🔍 Respuesta Groq:\n{raw}")
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = raw
        try:
            datos = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decodificando JSON: {e}. Intentando limpiar...")
            json_str_clean = re.sub(r'[\x00-\x1f\x7f]', '', json_str)
            datos = json.loads(json_str_clean)
        if not isinstance(datos, dict):
            print(f"Error: datos no es dict, es {type(datos)}. Valor: {datos}")
            datos = {}
        decision = datos.get("decision", "Hold")
        patron = datos.get("patron", "")
        razones = datos.get("razones", [])
        if not isinstance(razones, list):
            razones = [str(razones)]
        sl_mult = float(datos.get("sl_mult", DEFAULT_SL_MULT))
        tp1_mult = float(datos.get("tp1_mult", DEFAULT_TP1_MULT))
        trailing_mult = float(datos.get("trailing_mult", DEFAULT_TRAILING_MULT))
        sl_mult = max(1.0, min(2.5, sl_mult))
        tp1_mult = max(1.2, min(3.0, tp1_mult))
        trailing_mult = max(1.2, min(2.5, trailing_mult))
        return decision, razones, patron, (sl_mult, tp1_mult, trailing_mult), raw
    except Exception as e:
        print(f"Error Groq: {e}")
        import traceback
        traceback.print_exc()
        return "Hold", ["Error"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT), ""

# =================== GRÁFICOS (sin cambios) ===================
def generar_grafico_entrada(df, decision, razones, patron, soporte, resistencia, slope, intercept, multiplicadores):
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
    sl_m, tp1_m, trail_m = multiplicadores
    texto = f"GROQ V99.18 | Decisión: {decision.upper()}\nPatrón: {patron[:70]}\nSL mult: {sl_m:.2f} | TP1 mult: {tp1_m:.2f} | Trail mult: {trail_m:.2f}\nRazones:\n" + "\n".join(razones[:3])
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
        ADAPTIVE_TRAILING_MULT = max(1.2, ADAPTIVE_TRAILING_MULT * 0.95)
    elif winrate > 0.6:
        ADAPTIVE_BIAS = min(0.3, ADAPTIVE_BIAS + 0.02)
        ADAPTIVE_SL_MULT = max(1.0, ADAPTIVE_SL_MULT * 0.95)
        ADAPTIVE_TP1_MULT = min(3.0, ADAPTIVE_TP1_MULT * 1.03)
        ADAPTIVE_TRAILING_MULT = min(2.5, ADAPTIVE_TRAILING_MULT * 1.02)
    msg = f"📚 AUTOAPRENDIZAJE (10 trades)\nWinrate: {winrate*100:.1f}%\nSesgo: {ADAPTIVE_BIAS:.2f}\nSL: {ADAPTIVE_SL_MULT:.2f}\nTP1: {ADAPTIVE_TP1_MULT:.2f}\nTrail: {ADAPTIVE_TRAILING_MULT:.2f}"
    telegram_mensaje(msg)
    print(msg)
    ULTIMO_APRENDIZAJE = len(TRADE_HISTORY)

# =================== GESTIÓN DE RIESGO Y PAPER TRADING (sin cambios) ===================
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

def paper_abrir_posicion(decision, precio, atr, razones, patron, multiplicadores_ia):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1
    global PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO
    global PAPER_SL_ACTUAL, PAPER_BALANCE, ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMOS_MULTIS
    if PAPER_POSICION_ACTIVA: return False
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    sl_mult = 0.6 * multiplicadores_ia[0] + 0.4 * ADAPTIVE_SL_MULT
    tp1_mult = 0.6 * multiplicadores_ia[1] + 0.4 * ADAPTIVE_TP1_MULT
    trailing_mult = 0.6 * multiplicadores_ia[2] + 0.4 * ADAPTIVE_TRAILING_MULT
    sl_mult = max(1.0, min(2.5, sl_mult))
    tp1_mult = max(1.2, min(3.0, tp1_mult))
    trailing_mult = max(1.2, min(2.5, trailing_mult))
    if decision == "Buy":
        sl = precio - (atr * sl_mult)
        tp1 = precio + (atr * tp1_mult)
    else:
        sl = precio + (atr * sl_mult)
        tp1 = precio - (atr * tp1_mult)
    distancia = abs(precio - sl)
    if distancia == 0: return False
    size_usd = min((riesgo_usd / distancia) * precio, PAPER_BALANCE * LEVERAGE)
    size_btc = size_usd / precio
    PAPER_POSICION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_SL_INICIAL = sl
    PAPER_TP1 = tp1
    PAPER_TRAILING_MULT = trailing_mult
    PAPER_SIZE_BTC = size_btc
    PAPER_SIZE_BTC_RESTANTE = size_btc
    PAPER_TP1_EJECUTADO = False
    PAPER_SL_ACTUAL = sl
    ULTIMOS_MULTIS = (sl_mult, tp1_mult, trailing_mult)
    msg = f"📌 {decision.upper()} | Entrada: {precio:.2f} | SL: {sl:.2f} (dist {atr*sl_mult:.1f}) | TP1: {tp1:.2f} (RR {tp1_mult/sl_mult:.2f}) | Trail: {trailing_mult}x ATR\nRazones: {' | '.join(razones[:2])}"
    telegram_mensaje(msg)
    ULTIMA_DECISION = decision
    ULTIMO_MOTIVO = razones[0][:50] if razones else ""
    print(f"🚀 {decision} a {precio:.2f} | SL={sl:.2f} TP1={tp1:.2f} Trail={trailing_mult}")
    return True

def paper_revisar_sl_tp(df, soporte, resistencia, slope, intercept):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1
    global PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO
    global PAPER_SL_ACTUAL, PAPER_BALANCE, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS
    global PAPER_TRADES_TOTALES, PAPER_LAST_10_PNL, TRADE_HISTORY, ULTIMA_RAZONES, ULTIMO_PATRON, ULTIMOS_MULTIS
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
            telegram_mensaje(f"🎯 TP1 alcanzado en {PAPER_TP1:.2f} | Beneficio parcial: +{beneficio:.2f} USD | SL a break-even ({PAPER_PRECIO_ENTRADA:.2f}) | Restan {PAPER_SIZE_BTC_RESTANTE:.4f} BTC | Trailing activo")
    if PAPER_TP1_EJECUTADO:
        if PAPER_POSICION_ACTIVA == "Buy":
            nuevo_sl = close - (atr * PAPER_TRAILING_MULT)
            if nuevo_sl > PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔼 Trailing SL sube a {PAPER_SL_ACTUAL:.2f} (dist {atr*PAPER_TRAILING_MULT:.1f})")
            if low <= PAPER_SL_ACTUAL:
                cerrar, motivo = True, "Trailing Stop"
        else:
            nuevo_sl = close + (atr * PAPER_TRAILING_MULT)
            if nuevo_sl < PAPER_SL_ACTUAL:
                PAPER_SL_ACTUAL = nuevo_sl
                telegram_mensaje(f"🔽 Trailing SL baja a {PAPER_SL_ACTUAL:.2f} (dist {atr*PAPER_TRAILING_MULT:.1f})")
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
        TRADE_HISTORY.append({
            "fecha": datetime.now(timezone.utc).isoformat(),
            "decision": PAPER_POSICION_ACTIVA,
            "precio_entrada": PAPER_PRECIO_ENTRADA,
            "precio_salida": salida,
            "pnl": pnl_total,
            "razones_ia": ULTIMA_RAZONES,
            "patron": ULTIMO_PATRON,
            "resultado_win": win,
            "sl_mult": ULTIMOS_MULTIS[0],
            "tp1_mult": ULTIMOS_MULTIS[1],
            "trailing_mult": ULTIMOS_MULTIS[2]
        })
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
    global ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON, ULTIMOS_MULTIS
    print("🤖 BOT V99.18 INICIADO - Steve Nison completo: patrones, estructura, continuación")
    telegram_mensaje("🤖 BOT V99.18 INICIADO - Análisis técnico integral (velas japonesas, tendencia, S/R)")
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
            print(f"\n💓 Heartbeat | Precio: {precio:.2f} | ATR: {atr:.2f} | Sop: {soporte:.2f} | Res: {resistencia:.2f} | Trades: {PAPER_TRADES_TOTALES} | PnL: {pnl_global:+.2f} | Winrate: {winrate:.1f}% | Drawdown: {drawdown:.2f}% | Decisión: {ULTIMA_DECISION} - {ULTIMO_MOTIVO[:50]}")
            if PAPER_POSICION_ACTIVA is None and ultima_vela != vela_cerrada:
                descripcion, atr_val = generar_descripcion_nison(df)
                print(f"📝 Descripción enviada a Groq (primeros 600 chars):\n{descripcion[:600]}...")
                decision, razones, patron, multiplicadores_ia, raw = analizar_con_groq_texto(descripcion, atr_val)
                ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON = decision, razones[0] if razones else "", razones, patron
                if decision in ["Buy","Sell"] and risk_management_check():
                    if paper_abrir_posicion(decision, precio, atr, razones, patron, multiplicadores_ia):
                        ultima_vela = vela_cerrada
                        ruta_entrada = generar_grafico_entrada(df, decision, razones, patron, soporte, resistencia, slope, intercept, multiplicadores_ia)
                        caption = f"🚀 Señal {decision} (Nison)\nPatrón: {patron}\nSL mult: {multiplicadores_ia[0]:.2f} | TP1 mult: {multiplicadores_ia[1]:.2f} | Trail: {multiplicadores_ia[2]:.2f}\n" + "\n".join(razones[:2])
                        telegram_enviar_imagen(ruta_entrada, caption)
                else:
                    print(f"⏸️ Hold: {ULTIMO_MOTIVO}")
            if PAPER_POSICION_ACTIVA:
                paper_revisar_sl_tp(df, soporte, resistencia, slope, intercept)
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
