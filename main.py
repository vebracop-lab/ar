# BOT TRADING V99.28 – GROQ (NISON HOLÍSTICO + CABALGAR EMA Y ESTRUCTURA GLOBAL)
# ==============================================================================
import os, time, requests, json, re, numpy as np, pandas as pd
from scipy.stats import linregress
from datetime import datetime, timezone
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

DEFAULT_SL_MULT = 1.2
DEFAULT_TP1_MULT = 1.5
DEFAULT_TRAILING_MULT = 1.8
PORCENTAJE_CIERRE_TP1 = 0.5

# =================== PAPER TRADING Y GLOBALES ===================
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
PAPER_MAX_PRECIO_ALCANZADO = None

PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
TRADE_HISTORY = []

MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

# Variables de Autoaprendizaje Evolutivo
ADAPTIVE_BIAS = 0.0
ADAPTIVE_SL_MULT = DEFAULT_SL_MULT
ADAPTIVE_TP1_MULT = DEFAULT_TP1_MULT
ADAPTIVE_TRAILING_MULT = DEFAULT_TRAILING_MULT
ULTIMO_APRENDIZAJE = 0
REGLAS_APRENDIDAS = "Aún no hay trades suficientes. Analiza el gráfico completo sin sesgos."

ULTIMA_DECISION = "Hold"
ULTIMO_MOTIVO = "Esperando señal"
ULTIMA_RAZONES = []
ULTIMO_PATRON = ""
ULTIMOS_MULTIS = (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

# =================== COMUNICACIÓN ===================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e:
        print(f"Error Telegram: {e}")

def telegram_enviar_imagen(ruta_imagen, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        with open(ruta_imagen, 'rb') as foto:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
    except Exception as e:
        print(f"Error imagen: {e}")

# =================== DATOS E INDICADORES ===================
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
    df['ema50'] = df['close'].ewm(span=50).mean()
    tr = pd.concat([(df['high'] - df['low']), (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))
    return df.dropna()

def detectar_zonas_mercado(df, idx=-2, ventana_macro=120):
    df_eval = df if idx == -1 else df.iloc[:idx+1]
    
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    
    y = df_eval['close'].values[-ventana_macro:] if len(df_eval) >= ventana_macro else df_eval['close'].values
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    
    micro_slope, _, _, _, _ = linregress(np.arange(8), df_eval['close'].values[-8:])
    micro_tendencia = 'CAYENDO' if micro_slope < -0.2 else 'SUBIENDO' if micro_slope > 0.2 else 'CONSOLIDANDO'
    tendencia = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    
    return soporte, resistencia, slope, intercept, tendencia, micro_tendencia

# =================== MOTOR HOLÍSTICO NISON (ACTUALIZADO) ===================
def analizar_patron_nison_completo(df, idx):
    if idx < 3: return "Datos insuficientes"
    
    v3 = df.iloc[idx]
    v2 = df.iloc[idx-1]
    v1 = df.iloc[idx-2]
    
    def stats(v):
        rango = v['high'] - v['low']
        cuerpo = abs(v['close'] - v['open'])
        c_pct = (cuerpo / rango) * 100 if rango > 0 else 0
        s_sup = ((v['high'] - max(v['close'], v['open'])) / rango) * 100 if rango > 0 else 0
        s_inf = ((min(v['close'], v['open']) - v['low']) / rango) * 100 if rango > 0 else 0
        es_verde = v['close'] > v['open']
        return rango, cuerpo, c_pct, s_sup, s_inf, es_verde

    r3, c3, cp3, sup3, inf3, verde3 = stats(v3)
    r2, c2, cp2, sup2, inf2, verde2 = stats(v2)
    r1, c1, cp1, sup1, inf1, verde1 = stats(v1)

    patrones = []

    # Reversiones Mayores
    if not verde1 and c1 > r1*0.5 and cp2 < 30 and verde3 and c3 > r3*0.5 and v3['close'] > (v1['open']+v1['close'])/2:
        patrones.append("🌟 ESTRELLA DE LA MAÑANA")
    if verde1 and c1 > r1*0.5 and cp2 < 30 and not verde3 and c3 > r3*0.5 and v3['close'] < (v1['open']+v1['close'])/2:
        patrones.append("🌟 ESTRELLA DEL ATARDECER")
    
    # Continuaciones Fuertes
    if verde1 and verde2 and verde3 and c3 > c2 and c2 > c1 and inf3 < 20 and inf2 < 20:
        patrones.append("🚀 TRES SOLDADOS BLANCOS")
    if not verde1 and not verde2 and not verde3 and c3 > c2 and c2 > c1 and sup3 < 20 and sup2 < 20:
        patrones.append("🩸 TRES CUERVOS NEGROS")

    # Envolventes
    if not verde2 and verde3 and v3['close'] > v2['open'] and v3['open'] < v2['close']:
        patrones.append("🐂 ENVOLVENTE ALCISTA")
    if verde2 and not verde3 and v3['close'] < v2['open'] and v3['open'] > v2['close']:
        patrones.append("🐻 ENVOLVENTE BAJISTA")

    # Velas Individuales y Control de Mercado (Añadido para captar Cierres Fuertes)
    if verde3 and cp3 > 65 and sup3 < 10:
        patrones.append("📈 VELA VERDE SÓLIDA CERRANDO EN MÁXIMOS (Control Toro Total)")
    elif not verde3 and cp3 > 65 and inf3 < 10:
        patrones.append("📉 VELA ROJA SÓLIDA CERRANDO EN MÍNIMOS (Control Oso Total)")
    elif cp3 < 10:
        patrones.append("⚖️ DOJI")
    elif cp3 < 25 and sup3 > 30 and inf3 > 30:
        patrones.append("🌀 PEONZA")
    else:
        if inf3 > 65 and cp3 < 25 and sup3 < 10:
            patrones.append("🔨 MARTILLO / PINBAR ALCISTA")
        elif sup3 > 65 and cp3 < 25 and inf3 < 10:
            patrones.append("🌠 ESTRELLA FUGAZ / PINBAR BAJISTA")

    if not patrones:
        patrones.append(f"Vela {'Verde' if verde3 else 'Roja'} Estándar")
        
    return " | ".join(patrones)

def generar_descripcion_nison(df, idx=-2):
    vela_actual = df.iloc[idx]
    precio = vela_actual['close']
    atr = df['atr'].iloc[idx]
    ema20 = df['ema20'].iloc[idx]
    
    soporte, resistencia, slope, intercept, tendencia, micro = detectar_zonas_mercado(df, idx)
    patron = analizar_patron_nison_completo(df, idx)

    # Evaluación de Cabalgar EMA y Soportes Dinámicos
    if precio > ema20:
        dist_ema = (precio - ema20) / ema20 * 100
        if vela_actual['low'] <= ema20:
            interaccion_ema = "✅ Testeando EMA20 desde arriba y rebotando (Soporte Dinámico Confirmado, presión compradora)."
        else:
            interaccion_ema = f"✅ Cabalgando sobre la EMA20 (+{dist_ema:.2f}% de distancia). Los compradores tienen el control del flujo."
    else:
        dist_ema = (ema20 - precio) / ema20 * 100
        if vela_actual['high'] >= ema20:
            interaccion_ema = "❌ Testeando EMA20 desde abajo y siendo rechazado (Resistencia Dinámica Confirmada, presión vendedora)."
        else:
            interaccion_ema = f"❌ Atrapado bajo la EMA20 (-{dist_ema:.2f}% de distancia). Los vendedores tienen el control del flujo."

    # Evaluación de Rupturas (Throwbacks y Pullbacks)
    df_reciente = df.iloc[idx-20:] if idx == -1 else df.iloc[idx-20:idx+1]
    
    zona_actual = f"El precio actual es {precio:.2f}."
    if precio > resistencia and vela_actual['low'] <= resistencia * 1.002:
        zona_actual += " 🔥 RUPTURA CONFIRMADA: Resistencia rota ahora actuando como Soporte (Throwback) con defensa compradora."
    elif precio < soporte and vela_actual['high'] >= soporte * 0.998:
        zona_actual += " 🚨 RUPTURA CONFIRMADA: Soporte roto ahora actuando como Resistencia (Pullback) con defensa vendedora."
    elif precio <= soporte + (atr * 1.5):
        zona_actual += f" Apoyándose en el suelo clave de {soporte:.2f}."
    elif precio >= resistencia - (atr * 1.5):
        zona_actual += f" Chocando contra el techo clave de {resistencia:.2f}."
    else:
        zona_actual += " Flotando en el flujo de la tendencia media."

    descripcion = f"""
=== INFORME PRICE ACTION GLOBAL (NISON & ESTRUCTURA) ===

1. EL FLUJO Y LA TENDENCIA
- Tendencia Macro: {tendencia} | Micro-Tendencia: {micro}
- Relación con Media Móvil: {interaccion_ema}

2. ZONAS ESTRUCTURALES
- Soportes/Resistencias: {zona_actual}
*(Nota: Observa si el precio ha roto una estructura reciente y está continuando, o si está rebotando).*

3. ACCIÓN DEL PRECIO Y VELAS ACTUALES
- Comportamiento Reciente: {patron}
"""
    return descripcion, atr

# =================== IA GROQ: DECISIONES DES-SESGADAS ===================
def analizar_con_groq_texto(descripcion, atr, reglas_aprendidas):
    try:
        system_msg = f"""
        Eres un experto Chartista Institucional que sigue fielmente la filosofía de Steve Nison y la Acción del Precio pura.
        Tu trabajo NO es buscar obsesivamente pequeños rechazos de mechas o reversiones forzadas, sino leer el FLUJO DEL MERCADO de forma holística.

        🔥 ATENCIÓN - LECCIÓN APRENDIDA DE TUS ÚLTIMOS TRADES 🔥
        "{reglas_aprendidas}"

        LÓGICA DE DECISIÓN HOLÍSTICA:
        1. TENDENCIA Y EMA: Si el gráfico muestra una tendencia alcista clara, el precio cabalga sobre la EMA20, o acaba de romper una resistencia (con throwback), tu deber es seguir la tendencia (BUY).
        2. CONTROL DE LA VELA: Si la última vela es verde, sólida, y cierra cerca de su máximo, los toros controlan. NO busques shorts falsos si no hay un patrón de reversión evidente (Estrella Fugaz/Evening Star).
        3. REVERSIONES: Solo opera en contra del movimiento si ves un choque evidente en resistencia con un rechazo bajista fuerte, o en soporte con un rechazo alcista.
        4. No tengas miedo a comprar si todo el contexto indica continuación.

        Responde ÚNICAMENTE con un JSON válido en este formato:
        {{
          "decision": "Buy/Sell/Hold",
          "patron": "Explica la estructura global. Ej: Tendencia alcista + Cabalgando EMA20 + Cierre Toro Sólido",
          "razones": ["Razón de estructura general", "Razón de velas/EMA"],
          "sl_mult": 1.2,
          "tp1_mult": 1.5,
          "trailing_mult": 1.8
        }}
        """
        user_msg = f"{descripcion}\n\nATR: {atr:.2f}. Lee el panorama completo, no te estanques en detalles aislados. Determina el movimiento de mayor probabilidad."
        
        respuesta = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.15, 
            max_tokens=400
        )
        raw = respuesta.choices[0].message.content
        print(f"\n🔍 Respuesta Groq IA:\n{raw}\n")
        
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        datos = json.loads(re.sub(r'[\x00-\x1f\x7f]', '', match.group(0) if match else raw))
        
        decision = datos.get("decision", "Hold")
        sl_m = max(0.5, min(2.5, float(datos.get("sl_mult", 1.2))))
        tp_m = max(0.8, min(3.0, float(datos.get("tp1_mult", 1.5))))
        tr_m = max(1.0, min(3.0, float(datos.get("trailing_mult", 1.8))))
        
        return decision, datos.get("razones", []), datos.get("patron", ""), (sl_m, tp_m, tr_m)
    except Exception as e:
        print(f"Error IA: {e}")
        return "Hold", ["Error IA"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

# =================== AUTOAPRENDIZAJE IA (INTELIGENCIA EVOLUTIVA) ===================
def aprender_de_trades():
    global ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TRAILING_MULT, ULTIMO_APRENDIZAJE, REGLAS_APRENDIDAS
    
    if len(TRADE_HISTORY) < 10 or (len(TRADE_HISTORY) - ULTIMO_APRENDIZAJE < 10):
        return
    
    ultimos = TRADE_HISTORY[-10:]
    wins = sum(1 for t in ultimos if t['resultado_win'])
    losses = 10 - wins
    winrate = wins / 10.0
    
    resumen_trades = ""
    for i, t in enumerate(ultimos):
        estado = "WIN ✅" if t['resultado_win'] else "LOSS ❌"
        resumen_trades += f"Trade {i+1} ({estado}): {t['decision'].upper()} | Gatillo: {t.get('patron', 'Desconocido')} | PnL: {t['pnl']:.2f}\n"

    system_msg = """
    Eres el Mentor de Trading de una Inteligencia Artificial orientada al Price Action.
    Tu objetivo es revisar los últimos 10 trades de tu alumno y decirle en qué se equivocó y qué hizo bien.
    Encuentra el patrón común de las pérdidas (ej. ignorar la tendencia principal, SL muy corto) y formula una nueva "Regla de Oro" holística.
    
    Responde ÚNICAMENTE con un JSON válido:
    {
      "analisis": "Breve explicación de por qué ganamos o perdimos estos trades",
      "nueva_regla": "Instrucción clara de 1 o 2 oraciones para no repetir errores y seguir el flujo del mercado",
      "sl_mult_sugerido": 1.5,
      "tp1_mult_sugerido": 1.5,
      "trailing_mult_sugerido": 1.8
    }
    """
    
    user_msg = f"Winrate reciente: {winrate*100:.0f}%. ({wins} Ganados, {losses} Perdidos).\n\nHistorial detallado:\n{resumen_trades}\n\nEscribe tu análisis y dicta la nueva regla."
    
    try:
        respuesta = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.3, max_tokens=500
        )
        raw = respuesta.choices[0].message.content
        print(f"\n🧠 [REFLEXIÓN DE APRENDIZAJE CRUDA]:\n{raw}\n")
        
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        datos = json.loads(re.sub(r'[\x00-\x1f\x7f]', '', match.group(0) if match else raw))
        
        analisis = datos.get("analisis", "Análisis completado.")
        REGLAS_APRENDIDAS = datos.get("nueva_regla", REGLAS_APRENDIDAS)
        
        ADAPTIVE_SL_MULT = max(0.5, min(2.5, float(datos.get("sl_mult_sugerido", ADAPTIVE_SL_MULT))))
        ADAPTIVE_TP1_MULT = max(0.8, min(3.0, float(datos.get("tp1_mult_sugerido", ADAPTIVE_TP1_MULT))))
        ADAPTIVE_TRAILING_MULT = max(1.0, min(3.0, float(datos.get("trailing_mult_sugerido", ADAPTIVE_TRAILING_MULT))))
        
        msg_telegram = f"""🧠 IA AUTOAPRENDIZAJE (10 Trades Evaluados)
📊 Winrate: {winrate*100:.1f}% ({wins}W / {losses}L)
🧐 Análisis: {analisis}
📜 NUEVA REGLA: "{REGLAS_APRENDIDAS}"
⚙️ Riesgo Base Ajustado -> SL: {ADAPTIVE_SL_MULT:.2f} | TP1: {ADAPTIVE_TP1_MULT:.2f} | Trail: {ADAPTIVE_TRAILING_MULT:.2f}"""
        
        telegram_mensaje(msg_telegram)
        print(f"\n{msg_telegram}\n")
        ULTIMO_APRENDIZAJE = len(TRADE_HISTORY)
        
    except Exception as e:
        print(f"Error en IA de Aprendizaje: {e}")

# =================== GRÁFICOS ===================
def generar_grafico(df, decision, razones, patron, soporte, resistencia, slope, intercept, multi, tipo="Entrada", salida_data=None):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16,8))
    
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
        
    ax.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte Horizontal')
    ax.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia Horizontal')
    if 'ema20' in df_plot.columns: ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA 20')
    
    sl_m, tp1_m, trail_m = multi
    
    if tipo == "Entrada":
        p_act = df_plot['close'].iloc[-2]
        ax.scatter(len(df_plot)-2, p_act + (-30 if decision=='Buy' else 30), s=400, marker='^' if decision=='Buy' else 'v', c='lime' if decision=='Buy' else 'red', zorder=5)
        txt = f"DECISIÓN: {decision.upper()}\nPanorama Global: {patron}\nSL mult: {sl_m:.2f} | TP1 mult: {tp1_m:.2f} | Trail mult: {trail_m:.2f}\nRazones:\n" + "\n".join(razones[:2])
    else:
        p_ent, p_sal, win, pnl = salida_data
        ax.axhline(p_ent, color='blue', ls=':', lw=2, label='Precio de Entrada')
        ax.axhline(p_sal, color='white', ls=':', lw=2, label='Precio de Salida')
        estado = "WIN" if win else "LOSS"
        txt = f"RESULTADO DEL TRADE: {estado} | PnL: {pnl:.2f} USD\nSL mult: {sl_m:.2f} | TP1 mult: {tp1_m:.2f} | Trail mult: {trail_m:.2f}"

    ax.text(0.01, 0.99, txt, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_facecolor('#121212'); fig.patch.set_facecolor('#121212'); ax.tick_params(colors='white'); ax.grid(True, alpha=0.1)
    ax.legend(loc='lower right', facecolor='black', labelcolor='white')
    plt.tight_layout()
    ruta = f"/tmp/chart_{tipo.lower()}.png"
    plt.savefig(ruta, dpi=120)
    plt.close()
    return ruta

# =================== GESTIÓN DE RIESGO Y TRAILING ===================
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY
    hoy = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY = hoy
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
    
    drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if drawdown <= -MAX_DAILY_DRAWDOWN_PCT and not PAPER_STOPPED_TODAY:
        telegram_mensaje("🛑 Drawdown diario máximo alcanzado. Bot pausado por hoy.")
        PAPER_STOPPED_TODAY = True
    return not PAPER_STOPPED_TODAY

def paper_abrir_posicion(decision, precio, atr, razones, patron, multis_ia):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TRAILING_MULT
    global PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL, ULTIMOS_MULTIS, PAPER_MAX_PRECIO_ALCANZADO
    if PAPER_POSICION_ACTIVA: return False
    
    sl_m = (multis_ia[0] * 0.6) + (ADAPTIVE_SL_MULT * 0.4)
    tp_m = (multis_ia[1] * 0.6) + (ADAPTIVE_TP1_MULT * 0.4)
    tr_m = (multis_ia[2] * 0.6) + (ADAPTIVE_TRAILING_MULT * 0.4)
    
    PAPER_SL_INICIAL = precio - (atr * sl_m) if decision == "Buy" else precio + (atr * sl_m)
    PAPER_TP1 = precio + (atr * tp_m) if decision == "Buy" else precio - (atr * tp_m)
    
    distancia = abs(precio - PAPER_SL_INICIAL)
    if distancia == 0: return False
    
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    PAPER_SIZE_BTC = min((riesgo_usd / distancia) * precio, PAPER_BALANCE * LEVERAGE) / precio
    
    PAPER_POSICION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = precio
    PAPER_TRAILING_MULT = tr_m
    PAPER_SIZE_BTC_RESTANTE = PAPER_SIZE_BTC
    PAPER_TP1_EJECUTADO = False
    PAPER_SL_ACTUAL = PAPER_SL_INICIAL
    PAPER_MAX_PRECIO_ALCANZADO = precio
    ULTIMOS_MULTIS = (sl_m, tp_m, tr_m)
    
    msg = f"""📌 NUEVA POSICIÓN ABIERTA: {decision.upper()}
Entrada: {precio:.2f}
🎯 TP1 (50%): {PAPER_TP1:.2f} (Distancia: {tp_m:.2f}x ATR)
🛑 Stop Loss: {PAPER_SL_INICIAL:.2f} (Distancia: {sl_m:.2f}x ATR)
🔄 Trailing Dinámico: {tr_m:.2f}x ATR
📊 Tamaño: {PAPER_SIZE_BTC:.4f} BTC
🔍 Panorama: {patron}"""
    telegram_mensaje(msg)
    return True

def paper_revisar_sl_tp(df, sop, res, slo, inter):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TRAILING_MULT
    global PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL, PAPER_MAX_PRECIO_ALCANZADO
    global PAPER_BALANCE, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES, TRADE_HISTORY, ULTIMO_PATRON
    
    if not PAPER_POSICION_ACTIVA: return None
    
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    c = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    cerrar = False
    motivo = ""
    
    if PAPER_POSICION_ACTIVA == "Buy": 
        PAPER_MAX_PRECIO_ALCANZADO = max(PAPER_MAX_PRECIO_ALCANZADO, h)
    else: 
        PAPER_MAX_PRECIO_ALCANZADO = min(PAPER_MAX_PRECIO_ALCANZADO, l)
    
    if not PAPER_TP1_EJECUTADO:
        if (PAPER_POSICION_ACTIVA == "Buy" and h >= PAPER_TP1) or (PAPER_POSICION_ACTIVA == "Sell" and l <= PAPER_TP1):
            PAPER_PNL_PARCIAL = abs(PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1)
            PAPER_BALANCE += PAPER_PNL_PARCIAL
            PAPER_SIZE_BTC_RESTANTE *= (1 - PORCENTAJE_CIERRE_TP1)
            PAPER_TP1_EJECUTADO = True
            PAPER_SL_ACTUAL = PAPER_PRECIO_ENTRADA 
            telegram_mensaje(f"🎯 ¡TP1 ALCANZADO!\nSe cerró el 50% de la posición asegurando +{PAPER_PNL_PARCIAL:.2f} USD.\n🛡️ El Stop Loss se ha movido a Break-Even ({PAPER_PRECIO_ENTRADA:.2f}). Dejamos correr el resto con Trailing.")
            
    if PAPER_TP1_EJECUTADO:
        n_sl = PAPER_MAX_PRECIO_ALCANZADO - (atr * PAPER_TRAILING_MULT) if PAPER_POSICION_ACTIVA == "Buy" else PAPER_MAX_PRECIO_ALCANZADO + (atr * PAPER_TRAILING_MULT)
        if (PAPER_POSICION_ACTIVA == "Buy" and n_sl > PAPER_SL_ACTUAL) or (PAPER_POSICION_ACTIVA == "Sell" and n_sl < PAPER_SL_ACTUAL):
            PAPER_SL_ACTUAL = n_sl
            print(f"🔄 Trailing Stop ajustado dinámicamente a: {PAPER_SL_ACTUAL:.2f}")
            
        if (PAPER_POSICION_ACTIVA == "Buy" and l <= PAPER_SL_ACTUAL) or (PAPER_POSICION_ACTIVA == "Sell" and h >= PAPER_SL_ACTUAL):
            cerrar = True
            motivo = "Trailing Stop Ejecutado"
    else:
        if (PAPER_POSICION_ACTIVA == "Buy" and l <= PAPER_SL_INICIAL) or (PAPER_POSICION_ACTIVA == "Sell" and h >= PAPER_SL_INICIAL):
            cerrar = True
            motivo = "Stop Loss Inicial Alcanzado"
            PAPER_SL_ACTUAL = PAPER_SL_INICIAL

    if cerrar:
        pnl_rest = (PAPER_SL_ACTUAL - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - PAPER_SL_ACTUAL) * PAPER_SIZE_BTC_RESTANTE
        pnl_total = PAPER_PNL_PARCIAL + pnl_rest
        PAPER_BALANCE += pnl_rest
        PAPER_TRADES_TOTALES += 1
        win = pnl_total > 0
        if win: PAPER_WIN += 1 
        else: PAPER_LOSS += 1
        
        TRADE_HISTORY.append({
            "fecha": datetime.now(timezone.utc).isoformat(),
            "decision": PAPER_POSICION_ACTIVA,
            "patron": ULTIMO_PATRON,
            "pnl": pnl_total,
            "resultado_win": win
        })
        
        if PAPER_TRADES_TOTALES > 0 and PAPER_TRADES_TOTALES % 10 == 0:
            aprender_de_trades()
            
        msg = f"""📤 POSICIÓN CERRADA ({motivo})
Dirección: {PAPER_POSICION_ACTIVA.upper()}
Precio de Entrada: {PAPER_PRECIO_ENTRADA:.2f}
Precio de Salida: {PAPER_SL_ACTUAL:.2f}
💰 PnL Total del Trade: {pnl_total:.2f} USD
🏦 Balance Actualizado: {PAPER_BALANCE:.2f} USD"""
        telegram_mensaje(msg)
        
        ruta_img = generar_grafico(df, PAPER_POSICION_ACTIVA, [], "", sop, res, slo, inter, ULTIMOS_MULTIS, "Salida", (PAPER_PRECIO_ENTRADA, PAPER_SL_ACTUAL, win, pnl_total))
        telegram_enviar_imagen(ruta_img, msg)
        
        PAPER_POSICION_ACTIVA = None
        return True
        
    return None

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON
    print("🤖 BOT V99.28 INICIADO - Enfoque Holístico Des-Sesgado + IA Evolutiva")
    telegram_mensaje("🤖 BOT V99.28 INICIADO - Monitoreando Estructura, Tendencia y Velas Sólidas.")
    
    ultima_vela = None
    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            vela_cerrada = df.index[-2]
            precio = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            sop, res, slo, inter, tend, micro = detectar_zonas_mercado(df)
            
            pnl_global = PAPER_BALANCE - PAPER_BALANCE_INICIAL
            drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE * 100
            winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0
            
            print(f"\n💓 Heartbeat | Precio: {precio:.2f} | ATR: {atr:.2f} | Sop: {sop:.2f} | Res: {res:.2f} | Trades: {PAPER_TRADES_TOTALES} | PnL: {pnl_global:+.2f} | Winrate: {winrate:.1f}% | Drawdown: {drawdown:.2f}%")
            print(f"📊 Última Evaluación: {ULTIMA_DECISION} - {ULTIMO_MOTIVO[:60]}")
            
            if PAPER_POSICION_ACTIVA is None and ultima_vela != vela_cerrada:
                desc, atr_val = generar_descripcion_nison(df)
                print(f"--- Escaneando Vela Cerrada: {vela_cerrada.strftime('%H:%M')} ---")
                print(f"📝 Reporte generado:\n{desc}\n")
                
                decision, razones, patron, multis = analizar_con_groq_texto(desc, atr_val, REGLAS_APRENDIDAS)
                ULTIMA_DECISION = decision
                ULTIMO_MOTIVO = razones[0] if razones else "Sin razones"
                ULTIMA_RAZONES = razones
                ULTIMO_PATRON = patron
                
                if decision in ["Buy","Sell"] and risk_management_check():
                    if paper_abrir_posicion(decision, precio, atr_val, razones, patron, multis):
                        ultima_vela = vela_cerrada
                        ruta_img = generar_grafico(df, decision, razones, patron, sop, res, slo, inter, multis, "Entrada")
                        telegram_enviar_imagen(ruta_img, f"🚀 SEÑAL CONFIRMADA: {decision.upper()}\nPanorama Global: {patron}\n{razones[0]}")
                else:
                    print(f"⏸️ Se mantiene Hold. Motivo: {ULTIMO_MOTIVO[:80]}...")
                    ultima_vela = vela_cerrada
            
            if PAPER_POSICION_ACTIVA:
                sop, res, slo, inter, _, _ = detectar_zonas_mercado(df, -1)
                paper_revisar_sl_tp(df, sop, res, slo, inter)
            
            time.sleep(SLEEP_SECONDS)
            
        except Exception as e:
            print(f"❌ ERROR EN EL LOOP PRINCIPAL: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
