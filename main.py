# BOT TRADING V99.32 – GROQ (MULTI-TRADES + BARRIDOS DE LIQUIDEZ/FAKEOUTS)
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
MODELO_TEXTO = "openai/gpt-oss-120b"   # ← se mantiene tu modelo

# ====== MEJORAS IA (MEMORIA + JSON ROBUSTO) ======
MEMORY_FILE = "memoria_bot.json"

def guardar_memoria():
    data = {
        "TRADE_HISTORY": TRADE_HISTORY,
        "REGLAS_APRENDIDAS": REGLAS_APRENDIDAS,
        "ADAPTIVE_SL_MULT": ADAPTIVE_SL_MULT,
        "ADAPTIVE_TP1_MULT": ADAPTIVE_TP1_MULT,
        "ADAPTIVE_TRAILING_MULT": ADAPTIVE_TRAILING_MULT,
        "PAPER_BALANCE": PAPER_BALANCE,
        "PAPER_WIN": PAPER_WIN,
        "PAPER_LOSS": PAPER_LOSS,
        "PAPER_TRADES_TOTALES": PAPER_TRADES_TOTALES
    }
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print("Error guardando memoria:", e)

def cargar_memoria():
    global TRADE_HISTORY, REGLAS_APRENDIDAS
    global ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TRAILING_MULT
    global PAPER_BALANCE, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES

    if not os.path.exists(MEMORY_FILE):
        return
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)

        TRADE_HISTORY = data.get("TRADE_HISTORY", [])
        REGLAS_APRENDIDAS = data.get("REGLAS_APRENDIDAS", REGLAS_APRENDIDAS)
        ADAPTIVE_SL_MULT = data.get("ADAPTIVE_SL_MULT", ADAPTIVE_SL_MULT)
        ADAPTIVE_TP1_MULT = data.get("ADAPTIVE_TP1_MULT", ADAPTIVE_TP1_MULT)
        ADAPTIVE_TRAILING_MULT = data.get("ADAPTIVE_TRAILING_MULT", ADAPTIVE_TRAILING_MULT)
        PAPER_BALANCE = data.get("PAPER_BALANCE", PAPER_BALANCE)
        PAPER_WIN = data.get("PAPER_WIN", 0)
        PAPER_LOSS = data.get("PAPER_LOSS", 0)
        PAPER_TRADES_TOTALES = data.get("PAPER_TRADES_TOTALES", 0)

        print("🧠 Memoria cargada")
    except Exception as e:
        print("Error cargando memoria:", e)

def parse_json_seguro(raw):
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        json_str = match.group(0) if match else raw
        json_str = re.sub(r'[\x00-\x1f\x7f]', '', json_str)
        json_str = json_str.replace("'", '"')
        return json.loads(json_str)
    except Exception as e:
        print("⚠️ Error JSON:", e)
        print("RAW:", raw)
        return None
# =================================================

SYMBOL = "BTCUSDT"
INTERVAL = "5"
RISK_PER_TRADE = 0.02
LEVERAGE = 10
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 3

DEFAULT_SL_MULT = 1.2
DEFAULT_TP1_MULT = 1.5
DEFAULT_TRAILING_MULT = 1.8
PORCENTAJE_CIERRE_TP1 = 0.5

# =================== PAPER TRADING ===================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_ACTIVE_TRADES = {}
TRADE_COUNTER = 0

PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
TRADE_HISTORY = []

MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

ADAPTIVE_SL_MULT = DEFAULT_SL_MULT
ADAPTIVE_TP1_MULT = DEFAULT_TP1_MULT
ADAPTIVE_TRAILING_MULT = DEFAULT_TRAILING_MULT
ULTIMO_APRENDIZAJE = 0
REGLAS_APRENDIDAS = "Aún no hay trades suficientes. Busca confluencia y trampas de liquidez."

ULTIMA_DECISION = "Hold"
ULTIMA_MOTIVO = "Esperando señal"

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
    micro_tendencia = 'CAYENDO' if micro_slope < -0.2 else 'SUBIENDO' if micro_slope > 0.2 else 'LATERAL'
    tendencia = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tendencia, micro_tendencia

# =================== MOTOR HOLÍSTICO Y BARRIDOS ===================
def analizar_anatomia_vela(v):
    rango = v['high'] - v['low']
    if rango == 0: return "Doji Plano (0%)"
    c_pct = (abs(v['close'] - v['open']) / rango) * 100
    s_sup = ((v['high'] - max(v['close'], v['open'])) / rango) * 100
    s_inf = ((min(v['close'], v['open']) - v['low']) / rango) * 100
    color = "VERDE" if v['close'] > v['open'] else "ROJA"
    return f"{color} (Cuerpo:{c_pct:.0f}% | M.Sup:{s_sup:.0f}% | M.Inf:{s_inf:.0f}%)"

def analizar_patrones_conjuntos(df, idx):
    if idx < 3: return "Datos insuficientes"
    v3, v2, v1 = df.iloc[idx], df.iloc[idx-1], df.iloc[idx-2]

    r3 = v3['high'] - v3['low']
    c3_pct = (abs(v3['close'] - v3['open']) / r3) * 100 if r3 > 0 else 0
    sup3 = ((v3['high'] - max(v3['close'], v3['open'])) / r3) * 100 if r3 > 0 else 0
    inf3 = ((min(v3['close'], v3['open']) - v3['low']) / r3) * 100 if r3 > 0 else 0
    verde3 = v3['close'] > v3['open']
    verde2 = v2['close'] > v2['open']
    verde1 = v1['close'] > v1['open']

    patrones = []
    if not verde1 and verde3 and v3['close'] > (v1['open']+v1['close'])/2: patrones.append("🌟 ESTRELLA DE LA MAÑANA")
    if verde1 and not verde3 and v3['close'] < (v1['open']+v1['close'])/2: patrones.append("🌟 ESTRELLA DEL ATARDECER")
    if verde1 and verde2 and verde3 and v3['close'] > v2['close'] and v2['close'] > v1['close']: patrones.append("🚀 TRES SOLDADOS BLANCOS")
    if not verde1 and not verde2 and not verde3 and v3['close'] < v2['close'] and v2['close'] < v1['close']: patrones.append("🩸 TRES CUERVOS NEGROS")
    if not verde2 and verde3 and v3['close'] > v2['open'] and v3['open'] < v2['close']: patrones.append("🐂 ENVOLVENTE ALCISTA")
    if verde2 and not verde3 and v3['close'] < v2['open'] and v3['open'] > v2['close']: patrones.append("🐻 ENVOLVENTE BAJISTA")

    if verde3 and c3_pct > 70 and sup3 < 10: patrones.append("📈 VELA SÓLIDA EN MÁXIMOS (Toros controlan)")
    elif not verde3 and c3_pct > 70 and inf3 < 10: patrones.append("📉 VELA SÓLIDA EN MÍNIMOS (Osos controlan)")
    elif c3_pct < 15 and sup3 > 25 and inf3 > 25: patrones.append("⚖️ DOJI / INDECISIÓN")
    elif inf3 > 60 and c3_pct < 25 and sup3 < 15: patrones.append("🔨 MARTILLO / PINBAR ALCISTA")
    elif sup3 > 60 and c3_pct < 25 and inf3 < 15: patrones.append("🌠 ESTRELLA FUGAZ / SHOOTING STAR")

    return " | ".join(patrones) if patrones else "Formación de consolidación normal"

def generar_descripcion_nison(df, idx=-2):
    vela_actual = df.iloc[idx]
    precio = vela_actual['close']
    atr = df['atr'].iloc[idx]
    ema20 = df['ema20'].iloc[idx]

    soporte, resistencia, slope, intercept, tendencia, micro = detectar_zonas_mercado(df, idx)
    patrones_generales = analizar_patrones_conjuntos(df, idx)

    anat_v1 = analizar_anatomia_vela(df.iloc[idx-2])
    anat_v2 = analizar_anatomia_vela(df.iloc[idx-1])
    anat_v3 = analizar_anatomia_vela(df.iloc[idx])

    margen_fakeout = atr * 0.4
    if precio > ema20:
        if vela_actual['low'] < (ema20 - margen_fakeout):
            rol_ema = "🔥 BARRIDO DE LIQUIDEZ EN EMA20: La mecha perforó profundo hacia abajo atrapando vendedores (cazando stops), pero cerró por encima. Fuerte rechazo alcista."
        elif vela_actual['low'] <= ema20:
            rol_ema = "✅ EMA20 actuando como SOPORTE DINÁMICO normal."
        else:
            rol_ema = "Cabalgando SOBRE la EMA20 (Fuerza Compradora Dominante)."
    else:
        if vela_actual['high'] > (ema20 + margen_fakeout):
            rol_ema = "🔥 BARRIDO DE LIQUIDEZ EN EMA20: La mecha perforó profundo hacia arriba atrapando compradores (cazando stops), pero cerró por debajo. Fuerte rechazo bajista."
        elif vela_actual['high'] >= ema20:
            rol_ema = "❌ EMA20 actuando como RESISTENCIA DINÁMICA normal."
        else:
            rol_ema = "Presionado BAJO la EMA20 (Fuerza Vendedora Dominante)."

    df_reciente = df.iloc[idx-20:] if idx == -1 else df.iloc[idx-20:idx+1]
    toques_res = (df_reciente['high'] >= resistencia * 0.998).sum()
    toques_sop = (df_reciente['low'] <= soporte * 1.002).sum()

    polaridad = f"Precio actual: {precio:.2f}. "
    if vela_actual['low'] < soporte and precio > soporte:
        polaridad += f"🔥 BARRIDO EN SOPORTE (SPRING): Perforó el soporte de {soporte:.2f} quitando liquidez, pero regresó arriba. Alta probabilidad alcista."
    elif vela_actual['high'] > resistencia and precio < resistencia:
        polaridad += f"🚨 BARRIDO EN RESISTENCIA (UPTHRUST): Perforó la resistencia de {resistencia:.2f} quitando liquidez, pero regresó abajo. Alta probabilidad bajista."
    elif precio > resistencia and vela_actual['low'] <= resistencia * 1.005:
        polaridad += f"POLARIDAD ALCISTA: Acaba de romper resistencia y la testea como SOPORTE (Throwback)."
    elif precio < soporte and vela_actual['high'] >= soporte * 0.995:
        polaridad += f"POLARIDAD BAJISTA: Acaba de romper soporte y lo testea como RESISTENCIA (Pullback)."
    else:
        polaridad += "Dentro del flujo de mercado."

    df_mechas = df.iloc[idx-8:] if idx == -1 else df.iloc[idx-8:idx+1]
    rangos = df_mechas['high'] - df_mechas['low']
    mechas_sup = (df_mechas['high'] - df_mechas[['close', 'open']].max(axis=1)) / rangos.replace(0, 0.001)
    mechas_inf = (df_mechas[['close', 'open']].min(axis=1) - df_mechas['low']) / rangos.replace(0, 0.001)
    cluster_txt = f"{sum(mechas_sup > 0.55)} mechas superiores (venta) | {sum(mechas_inf > 0.55)} mechas inferiores (compra)."

    descripcion = f"""
=== MATRIZ DE CONFLUENCIA & TRAMPAS DE LIQUIDEZ (5M) ===

1. TENDENCIA Y ESTRUCTURA GLOBAL
- Tendencia Macro: {tendencia} | Impulso Micro: {micro}
- Soportes/Resistencias: {polaridad}

2. EMA 20 Y BARRIDOS
- Acción sobre EMA: {rol_ema}
(NOTA: Un nivel perforado que devuelve el precio rápidamente es una barrida de liquidez, señal de muy alta confirmación en dirección contraria a la mecha).

3. CLÚSTERES Y PRESIÓN ACUMULADA
- Presión oculta (8 velas): {cluster_txt}

4. ANATOMÍA EXACTA DE VELAS (Cuerpos y Mechas)
- Vela Antepenúltima: {anat_v1}
- Vela Penúltima: {anat_v2}
- Vela Actual (Gatillo): {anat_v3}

5. PATRONES IDENTIFICADOS (Conjunto)
- Lectura: {patrones_generales}
"""
    return descripcion, atr

# =================== IA GROQ: DECISIÓN MATRIZ TOTAL ===================
def analizar_con_groq_texto(descripcion, atr, reglas_aprendidas):
    try:
        system_msg = f"""
        Eres un Maestro del Price Action. Lees la "MATRIZ DE CONFLUENCIA" de forma holística.
        Entiendes que los Soportes, Resistencias y EMAs NO SON LÍNEAS EXACTAS.
        Si el mercado "perfora" una zona pero el cuerpo de la vela cierra devolviéndose (Barrido de Liquidez / Fakeout), sabes que es una confirmación enorme, porque han cazado los Stop Loss de los novatos.

        🔥 REGLA EVOLUTIVA DE TU MENTOR:
        "{reglas_aprendidas}"

        LÓGICA OPERATIVA:
        - Puedes operar Reversiones, Continuaciones, Rompimientos o Trampas de Liquidez en CUALQUIER zona del gráfico si la suma del contexto lo apoya.
        - NO ignores la vela actual. Si el contexto es alcista pero la vela actual es una gran Estrella Fugaz (Oso), se anulan y es "Hold".
        
        Responde ÚNICAMENTE con un JSON válido en este formato:
        {{
          "decision": "Buy/Sell/Hold",
          "patron": "Ej: Falso Rompimiento en EMA20 (Barrido) + Martillo en Tendencia Alcista",
          "razones": ["Razón de Estructura/Liquidez", "Razón de Velas Exactas"],
          "sl_mult": 1.2,
          "tp1_mult": 1.5,
          "trailing_mult": 1.8
        }}
        """
        user_msg = f"{descripcion}\n\nATR: {atr:.2f}. Analiza trampas de liquidez y flujos. Toma tu decisión:"

        respuesta = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.1, max_tokens=400
        )
        raw = respuesta.choices[0].message.content
        datos = parse_json_seguro(raw)
        if not datos:
            return "Hold", ["JSON inválido"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

        decision = datos.get("decision", "Hold")
        sl_m = max(0.5, min(2.5, float(datos.get("sl_mult", 1.2))))
        tp_m = max(0.8, min(3.0, float(datos.get("tp1_mult", 1.5))))
        tr_m = max(1.0, min(3.0, float(datos.get("trailing_mult", 1.8))))

        return decision, datos.get("razones", []), datos.get("patron", ""), (sl_m, tp_m, tr_m)
    except Exception as e:
        print(f"Error IA: {e}")
        return "Hold", ["Error IA"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

# =================== AUTOAPRENDIZAJE IA (CORREGIDO) ===================
def aprender_de_trades():
    global ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TRAILING_MULT, ULTIMO_APRENDIZAJE, REGLAS_APRENDIDAS
    if len(TRADE_HISTORY) < 10 or (len(TRADE_HISTORY) - ULTIMO_APRENDIZAJE < 10): return

    ultimos = TRADE_HISTORY[-10:]
    wins = sum(1 for t in ultimos if t['resultado_win'])
    losses = 10 - wins
    winrate = wins / 10.0

    resumen_trades = ""
    for i, t in enumerate(ultimos):
        estado = "WIN ✅" if t['resultado_win'] else "LOSS ❌"
        resumen_trades += f"Trade {i+1} ({estado}): {t['decision'].upper()} | {t.get('patron', 'Desconocido')} | PnL: {t['pnl']:.2f}\n"

    system_msg = """
    Eres el Mentor de Trading de una IA. Analiza los últimos 10 trades.
    Responde ÚNICAMENTE con un JSON:
    {
      "analisis": "Explicación de aciertos y errores en la lectura de fluidez y barridos",
      "nueva_regla": "Instrucción de mejora continua para la matriz",
      "sl_mult_sugerido": 1.5,
      "tp1_mult_sugerido": 1.5,
      "trailing_mult_sugerido": 1.8
    }
    """
    user_msg = f"Winrate: {winrate*100:.0f}%. ({wins}W, {losses}L).\n\nHistorial:\n{resumen_trades}\n\nDicta la nueva regla."

    try:
        respuesta = client.chat.completions.create(model=MODELO_TEXTO, messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}], temperature=0.3, max_tokens=500)
        raw = respuesta.choices[0].message.content
        datos = parse_json_seguro(raw)
        if not datos:
            return

        analisis = datos.get("analisis", "Análisis completado.")
        REGLAS_APRENDIDAS = datos.get("nueva_regla", REGLAS_APRENDIDAS)
        ADAPTIVE_SL_MULT = max(0.5, min(2.5, float(datos.get("sl_mult_sugerido", ADAPTIVE_SL_MULT))))
        ADAPTIVE_TP1_MULT = max(0.8, min(3.0, float(datos.get("tp1_mult_sugerido", ADAPTIVE_TP1_MULT))))
        ADAPTIVE_TRAILING_MULT = max(1.0, min(3.0, float(datos.get("trailing_mult_sugerido", ADAPTIVE_TRAILING_MULT))))

        msg_telegram = f"""🧠 IA AUTOAPRENDIZAJE (10 Trades)
📊 Winrate: {winrate*100:.1f}% ({wins}W / {losses}L)
🧐 Análisis: {analisis}
📜 NUEVA REGLA: "{REGLAS_APRENDIDAS}"
⚙️ Riesgo Ajustado -> SL:{ADAPTIVE_SL_MULT:.2f} TP1:{ADAPTIVE_TP1_MULT:.2f} Trail:{ADAPTIVE_TRAILING_MULT:.2f}"""
        telegram_mensaje(msg_telegram)
        print(f"\n{msg_telegram}\n")
        ULTIMO_APRENDIZAJE = len(TRADE_HISTORY)
        guardar_memoria()
    except Exception as e:
        print(f"Error Aprendizaje: {e}")

# =================== GRÁFICOS MULTI-TRADE ===================
def generar_grafico(df, trade_info, soporte, resistencia, slope, intercept, tipo="Entrada"):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16,8))

    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))

    ax.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte')
    ax.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia')
    ax.plot(x, intercept + slope * x, color='white', linestyle='-.', linewidth=1.5, alpha=0.6, label='Tendencia Inclinada')
    if 'ema20' in df_plot.columns: ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA 20')

    if tipo == "Entrada":
        decision = trade_info['decision']
        p_act = df_plot['close'].iloc[-2]
        ax.scatter(len(df_plot)-2, p_act + (-30 if decision=='Buy' else 30), s=400, marker='^' if decision=='Buy' else 'v', c='lime' if decision=='Buy' else 'red', zorder=5)
        txt = f"[TRADE #{trade_info['id']}] {decision.upper()}\nMatriz Global: {trade_info['patron']}\nSL: {trade_info['sl_inicial']:.2f} | TP1: {trade_info['tp1']:.2f} | Trail: {trade_info['trailing_mult']}x"
    else:
        win = trade_info['resultado_win']
        ax.axhline(trade_info['entrada'], color='blue', ls=':', lw=2, label='Precio de Entrada')
        ax.axhline(trade_info['sl_actual'], color='white', ls=':', lw=2, label='Precio de Salida')
        estado = "WIN" if win else "LOSS"
        txt = f"[TRADE #{trade_info['id']}] {estado} | PnL: {trade_info['pnl']:.2f} USD\nDirección: {trade_info['decision']}"

    ax.text(0.01, 0.99, txt, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.1)
    ax.legend(loc='lower right', facecolor='black', labelcolor='white')
    plt.tight_layout()
    ruta = f"/tmp/chart_{tipo.lower()}_{trade_info['id']}.png"
    plt.savefig(ruta, dpi=120)
    plt.close()
    return ruta

# =================== GESTIÓN MULTI-TRADE ===================
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY
    hoy = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY = hoy
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False

    drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if drawdown <= -MAX_DAILY_DRAWDOWN_PCT and not PAPER_STOPPED_TODAY:
        telegram_mensaje("🛑 Drawdown diario máximo. Bot pausado por hoy.")
        PAPER_STOPPED_TODAY = True
    return not PAPER_STOPPED_TODAY

def paper_abrir_posicion(decision, precio, atr, razones, patron, multis_ia, df, sop, res, slo, inter):
    global PAPER_BALANCE, TRADE_COUNTER
    if len(PAPER_ACTIVE_TRADES) >= MAX_CONCURRENT_TRADES:
        return False

    for t in PAPER_ACTIVE_TRADES.values():
        if t['decision'] == decision and abs(t['entrada'] - precio) < atr * 0.2:
            return False

    sl_m = (multis_ia[0] * 0.6) + (ADAPTIVE_SL_MULT * 0.4)
    tp_m = (multis_ia[1] * 0.6) + (ADAPTIVE_TP1_MULT * 0.4)
    tr_m = (multis_ia[2] * 0.6) + (ADAPTIVE_TRAILING_MULT * 0.4)

    sl_inicial = precio - (atr * sl_m) if decision == "Buy" else precio + (atr * sl_m)
    tp1 = precio + (atr * tp_m) if decision == "Buy" else precio - (atr * tp_m)
    distancia = abs(precio - sl_inicial)
    if distancia == 0: return False

    TRADE_COUNTER += 1
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    size_btc = min((riesgo_usd / distancia) * precio, PAPER_BALANCE * LEVERAGE) / precio

    trade = {
        "id": TRADE_COUNTER,
        "decision": decision,
        "entrada": precio,
        "sl_inicial": sl_inicial,
        "tp1": tp1,
        "trailing_mult": tr_m,
        "size_btc": size_btc,
        "size_restante": size_btc,
        "tp1_ejecutado": False,
        "pnl_parcial": 0.0,
        "sl_actual": sl_inicial,
        "max_precio": precio,
        "patron": patron,
        "atr_entrada": atr
    }

    PAPER_ACTIVE_TRADES[TRADE_COUNTER] = trade

    msg = f"""📌 [TRADE #{TRADE_COUNTER}] ABIERTO: {decision.upper()}
Entrada: {precio:.2f}
🎯 TP1 (50%): {tp1:.2f}
🛑 Stop Loss: {sl_inicial:.2f}
🔄 Trailing: {tr_m:.2f}x ATR
🔍 Matriz: {patron}"""
    print(f"🚀 [TRADE #{TRADE_COUNTER}] {decision} a {precio:.2f}")
    telegram_mensaje(msg)

    ruta_img = generar_grafico(df, trade, sop, res, slo, inter, "Entrada")
    telegram_enviar_imagen(ruta_img, f"🚀 [TRADE #{TRADE_COUNTER}] {decision.upper()}\n{razones[0]}")
    return True

def paper_revisar_sl_tp(df, sop, res, slo, inter):
    global PAPER_BALANCE, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES, TRADE_HISTORY

    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    c = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    trades_a_cerrar = []

    for t_id, t in PAPER_ACTIVE_TRADES.items():
        cerrar = False
        motivo = ""

        if t['decision'] == "Buy":
            t['max_precio'] = max(t['max_precio'], h)
        else:
            t['max_precio'] = min(t['max_precio'], l)

        if not t['tp1_ejecutado']:
            if (t['decision'] == "Buy" and h >= t['tp1']) or (t['decision'] == "Sell" and l <= t['tp1']):
                beneficio = abs(t['tp1'] - t['entrada']) * (t['size_btc'] * PORCENTAJE_CIERRE_TP1)
                t['pnl_parcial'] = beneficio
                PAPER_BALANCE += beneficio
                t['size_restante'] *= (1 - PORCENTAJE_CIERRE_TP1)
                t['tp1_ejecutado'] = True
                t['sl_actual'] = t['entrada']

                msg_tp1 = f"🎯 [TRADE #{t_id}] ¡TP1 ALCANZADO!\nSe cerró el 50% asegurando +{beneficio:.2f} USD.\n🛡️ SL movido a Break-Even ({t['entrada']:.2f})"
                telegram_mensaje(msg_tp1)
                print(msg_tp1)

        if t['tp1_ejecutado']:
            n_sl = t['max_precio'] - (t['atr_entrada'] * t['trailing_mult']) if t['decision'] == "Buy" else t['max_precio'] + (t['atr_entrada'] * t['trailing_mult'])

            if (t['decision'] == "Buy" and n_sl > t['sl_actual']) or (t['decision'] == "Sell" and n_sl < t['sl_actual']):
                t['sl_actual'] = n_sl
                msg_trail = f"🔄 [TRADE #{t_id}] Trailing Stop ajustado a: {n_sl:.2f}"
                telegram_mensaje(msg_trail)
                print(msg_trail)

            if (t['decision'] == "Buy" and l <= t['sl_actual']) or (t['decision'] == "Sell" and h >= t['sl_actual']):
                cerrar = True
                motivo = "Trailing Stop"
        else:
            if (t['decision'] == "Buy" and l <= t['sl_inicial']) or (t['decision'] == "Sell" and h >= t['sl_inicial']):
                cerrar = True
                motivo = "Stop Loss Inicial"
                t['sl_actual'] = t['sl_inicial']

        if cerrar:
            pnl_rest = (t['sl_actual'] - t['entrada']) * t['size_restante'] if t['decision'] == "Buy" else (t['entrada'] - t['sl_actual']) * t['size_restante']
            pnl_total = t['pnl_parcial'] + pnl_rest
            PAPER_BALANCE += pnl_rest
            PAPER_TRADES_TOTALES += 1
            win = pnl_total > 0
            if win:
                PAPER_WIN += 1
            else:
                PAPER_LOSS += 1

            t['pnl'] = pnl_total
            t['resultado_win'] = win
            trades_a_cerrar.append(t_id)

            TRADE_HISTORY.append({
                "fecha": datetime.now(timezone.utc).isoformat(),
                "decision": t['decision'],
                "patron": t['patron'],
                "pnl": pnl_total,
                "resultado_win": win
            })
            guardar_memoria()

            msg_cierre = f"📤 [TRADE #{t_id}] CERRADO ({motivo})\nDirección: {t['decision'].upper()}\nSalida: {t['sl_actual']:.2f}\n💰 PnL Total: {pnl_total:.2f} USD\n🏦 Balance: {PAPER_BALANCE:.2f} USD"
            telegram_mensaje(msg_cierre)
            print(msg_cierre)

            ruta_img = generar_grafico(df, t, sop, res, slo, inter, "Salida")
            telegram_enviar_imagen(ruta_img, msg_cierre)

    for t_id in trades_a_cerrar:
        del PAPER_ACTIVE_TRADES[t_id]

    if trades_a_cerrar and PAPER_TRADES_TOTALES > 0 and PAPER_TRADES_TOTALES % 10 == 0:
        aprender_de_trades()

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global ULTIMA_DECISION, ULTIMO_MOTIVO
    cargar_memoria()
    print("🤖 BOT V99.32 INICIADO - Multi-Trades y Detección de Barridos (Liquidity Sweeps)")
    telegram_mensaje("🤖 BOT V99.32 INICIADO - Sistema Multi-Trades Activado.")

    ultima_vela = None
    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            vela_cerrada = df.index[-2]
            precio = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            sop, res, slo, inter, tend, micro = detectar_zonas_mercado(df)

            pnl_global = PAPER_BALANCE - PAPER_BALANCE_INICIAL
            winrate = (PAPER_WIN / PAPER_TRADES_TOTALES) * 100 if PAPER_TRADES_TOTALES > 0 else 0
            activos_count = len(PAPER_ACTIVE_TRADES)

            print(f"\n💓 Heartbeat | P: {precio:.2f} | ATR: {atr:.2f} | Trades Activos: {activos_count}/{MAX_CONCURRENT_TRADES} | Cerrados: {PAPER_TRADES_TOTALES} | PnL: {pnl_global:+.2f} | Winrate: {winrate:.1f}%")

            if activos_count < MAX_CONCURRENT_TRADES and ultima_vela != vela_cerrada:
                desc, atr_val = generar_descripcion_nison(df)
                print(f"--- Evaluando Matriz de Vela: {vela_cerrada.strftime('%H:%M')} ---")

                decision, razones, patron, multis = analizar_con_groq_texto(desc, atr_val, REGLAS_APRENDIDAS)
                ULTIMA_DECISION, ULTIMO_MOTIVO = decision, razones[0] if razones else "Sin razones"

                if decision in ["Buy","Sell"] and risk_management_check():
                    paper_abrir_posicion(decision, precio, atr_val, razones, patron, multis, df, sop, res, slo, inter)
                else:
                    print(f"⏸️ Hold: {ULTIMO_MOTIVO[:80]}...")

                ultima_vela = vela_cerrada

            if PAPER_ACTIVE_TRADES:
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
