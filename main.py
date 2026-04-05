# BOT TRADING V99.20 – GROQ (NISON HOLÍSTICO: REVERSIÓN, CONTINUACIÓN Y CONTEXTO)
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

DEFAULT_SL_MULT = 1.5
DEFAULT_TP1_MULT = 1.6
DEFAULT_TRAILING_MULT = 1.8
PORCENTAJE_CIERRE_TP1 = 0.5

# =================== VARIABLES GLOBALES ===================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_POSICION_ACTIVA = None
PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1 = None, None, None
PAPER_TRAILING_MULT = DEFAULT_TRAILING_MULT
PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE = 0.0, 0.0
PAPER_TP1_EJECUTADO = False
PAPER_PNL_PARCIAL = 0.0
PAPER_SL_ACTUAL = None
PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES = 0, 0, 0

MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON = "Hold", "Esperando señal", [], ""
ULTIMOS_MULTIS = (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

# =================== COMUNICACIÓN ===================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e: print(f"Error Telegram: {e}")

def telegram_enviar_imagen(ruta_imagen, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        with open(ruta_imagen, 'rb') as foto:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
    except Exception as e: print(f"Error imagen: {e}")

# =================== DATOS E INDICADORES ===================
def obtener_velas(limit=150):
    r = requests.get(f"{BASE_URL}/v5/market/kline", params={"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}, timeout=20)
    data = r.json()["result"]["list"][::-1]
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','turnover'])
    for col in ['open','high','low','close','volume']: df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
    df.set_index('time', inplace=True)
    return df

def calcular_indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    tr = pd.concat([(df['high'] - df['low']), (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    delta = df['close'].diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / -delta.where(delta < 0, 0).rolling(14).mean())))
    return df.dropna()

def detectar_zonas_mercado(df, idx=-2, ventana_macro=120):
    df_eval = df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    y = df_eval['close'].values[-ventana_macro:] if len(df_eval) >= ventana_macro else df_eval['close'].values
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    
    micro_slope, _, _, _, _ = linregress(np.arange(8), df_eval['close'].values[-8:])
    micro_tendencia = 'CAYENDO' if micro_slope < -0.2 else 'SUBIENDO' if micro_slope > 0.2 else 'CONSOLIDANDO'
    tendencia = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    
    return soporte, resistencia, slope, intercept, tendencia, micro_tendencia

# =================== MOTOR HOLÍSTICO NISON ===================
def analizar_patron_nison_completo(df, idx):
    if idx < 3: return "Datos insuficientes"
    
    # Extraer las últimas 3 velas
    v3 = df.iloc[idx]   # Actual
    v2 = df.iloc[idx-1] # Anterior
    v1 = df.iloc[idx-2] # Pre-anterior
    
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

    # 1. PATRONES DE 3 VELAS (Los más fuertes)
    if not verde1 and c1 > r1*0.5 and cp2 < 30 and verde3 and c3 > r3*0.5 and v3['close'] > (v1['open']+v1['close'])/2:
        patrones.append("🌟 ESTRELLA DE LA MAÑANA (Morning Star - Reversión Alcista Mayor)")
    if verde1 and c1 > r1*0.5 and cp2 < 30 and not verde3 and c3 > r3*0.5 and v3['close'] < (v1['open']+v1['close'])/2:
        patrones.append("🌟 ESTRELLA DEL ATARDECER (Evening Star - Reversión Bajista Mayor)")
    if verde1 and verde2 and verde3 and c3 > c2 and c2 > c1 and inf3 < 20 and inf2 < 20:
        patrones.append("🚀 TRES SOLDADOS BLANCOS (Continuación Alcista Fuerte)")
    if not verde1 and not verde2 and not verde3 and c3 > c2 and c2 > c1 and sup3 < 20 and sup2 < 20:
        patrones.append("🩸 TRES CUERVOS NEGROS (Continuación Bajista Fuerte)")

    # 2. PATRONES DE 2 VELAS
    # Envolventes
    if not verde2 and verde3 and v3['close'] > v2['open'] and v3['open'] < v2['close']:
        patrones.append("🐂 ENVOLVENTE ALCISTA (Bullish Engulfing)")
    if verde2 and not verde3 and v3['close'] < v2['open'] and v3['open'] > v2['close']:
        patrones.append("🐻 ENVOLVENTE BAJISTA (Bearish Engulfing)")
    
    # Harami (Mujer Embarazada) - Pérdida de momentum de la tendencia anterior
    if not verde2 and verde3 and c2 > r2*0.6 and v3['high'] < v2['open'] and v3['low'] > v2['close']:
        patrones.append("⚠️ HARAMI ALCISTA (Agotamiento Bajista - Posible Reversión)")
    if verde2 and not verde3 and c2 > r2*0.6 and v3['high'] < v2['close'] and v3['low'] > v2['open']:
        patrones.append("⚠️ HARAMI BAJISTA (Agotamiento Alcista - Posible Reversión)")

    # Perforantes y Nube Oscura
    if not verde2 and verde3 and v3['open'] < v2['close'] and v3['close'] > (v2['open'] + v2['close'])/2:
        patrones.append("🔪 PAUTA PENETRANTE (Piercing Line - Alcista)")
    if verde2 and not verde3 and v3['open'] > v2['close'] and v3['close'] < (v2['open'] + v2['close'])/2:
        patrones.append("⛈️ NUBE OSCURA (Dark Cloud Cover - Bajista)")

    # Pinzas (Tweezers)
    if abs(v3['low'] - v2['low']) < r3 * 0.1 and inf3 > 40 and inf2 > 40:
        patrones.append("✂️ PINZAS DE SUELO (Tweezer Bottom - Soporte Fuerte)")
    if abs(v3['high'] - v2['high']) < r3 * 0.1 and sup3 > 40 and sup2 > 40:
        patrones.append("✂️ PINZAS DE TECHO (Tweezer Top - Resistencia Fuerte)")

    # 3. PATRONES DE 1 VELA (Contextuales)
    if cp3 < 10:
        patrones.append("⚖️ DOJI (Indecisión / Equilibrio total)")
    elif cp3 < 25 and sup3 > 30 and inf3 > 30:
        patrones.append("🌀 PEONZA (Spinning Top - Pérdida de Momentum)")
    else:
        if inf3 > 65 and cp3 < 25 and sup3 < 10:
            patrones.append("🔨 MARTILLO / PINBAR ALCISTA (Rechazo a la baja)")
        elif sup3 > 65 and cp3 < 25 and inf3 < 10:
            patrones.append("🌠 ESTRELLA FUGAZ / PINBAR BAJISTA (Rechazo al alza)")
        elif cp3 > 85:
            patrones.append(f"🧱 MARUBOZU {'ALCISTA' if verde3 else 'BAJISTA'} (Fuerza Dominante Unidireccional)")

    if not patrones:
        patrones.append(f"Vela {'Verde' if verde3 else 'Roja'} Estándar (Cuerpo: {cp3:.0f}%)")

    return " | ".join(patrones)

# =================== GENERACIÓN DEL INFORME (CONTEXTO) ===================
def generar_descripcion_nison(df, idx=-2):
    vela_actual = df.iloc[idx]
    precio = vela_actual['close']
    atr = df['atr'].iloc[idx]
    ema20 = df['ema20'].iloc[idx]
    ema50 = df['ema50'].iloc[idx]
    soporte, resistencia, slope, intercept, tendencia, micro_tendencia = detectar_zonas_mercado(df, idx)
    
    # Detectar el patrón actual complejo
    patron_detectado = analizar_patron_nison_completo(df, idx)

    # Contexto del precio vs Soporte/Resistencia
    zona_actual = "En rango medio (Tierra de nadie)"
    if precio <= soporte + (atr * 1.5): zona_actual = f"🔥 EN ZONA DE SOPORTE CLAVE ({soporte:.2f})"
    elif precio >= resistencia - (atr * 1.5): zona_actual = f"🚨 EN ZONA DE RESISTENCIA CLAVE ({resistencia:.2f})"

    # Contexto EMA (Rechazos dinámicos)
    interaccion_ema = "Fluctuando lejos de la EMA20"
    if vela_actual['low'] <= ema20 <= vela_actual['high']:
        if vela_actual['close'] > ema20: interaccion_ema = "✅ RECHAZO DESDE ABAJO EN EMA20 (Validada como Soporte Dinámico)"
        elif vela_actual['close'] < ema20: interaccion_ema = "❌ RECHAZO DESDE ARRIBA EN EMA20 (Validada como Resistencia Dinámica)"
    elif precio > ema20 and precio < ema20 + (atr * 0.5): interaccion_ema = "Apoyándose firmemente sobre EMA20 (Soporte)"
    elif precio < ema20 and precio > ema20 - (atr * 0.5): interaccion_ema = "Presionado firmemente bajo EMA20 (Resistencia)"

    descripcion = f"""
=== INFORME HOLÍSTICO STEVE NISON (BTCUSDT 5m) ===

1. EL PANORAMA (MACRO Y MICRO)
- Tendencia Principal (Fondo): {tendencia}
- Micro-Tendencia (Últimas 8 velas): {micro_tendencia}
- Fase del Precio: {zona_actual}

2. NIVELES CLAVE Y MEDIAS
- Resistencia Techo: {resistencia:.2f}
- Soporte Suelo: {soporte:.2f}
- EMA 20 (Corto plazo): {interaccion_ema}
- EMA 50 (Medio plazo): {ema50:.2f}

3. LECTURA DE VELAS (PATRÓN ACTUAL COMBINADO)
- Acción del Precio Reciente (Velas 1 a 3): {patron_detectado}

4. LÓGICA DE DECISIÓN NISON:
- Los patrones de REVERSIÓN (Estrellas, Envolventes, Martillos) SOLO son válidos si ocurren en la "Zona de Soporte/Resistencia" adecuada y van EN CONTRA de la micro-tendencia agotada.
- Los patrones de CONTINUACIÓN (Soldados, Cuervos, Marubozus) son señales para ENTRAR A FAVOR del rompimiento o tendencia.
- Si el precio está en "Tierra de nadie" y forma un Doji o Peonza, indica consolidación.
"""
    return descripcion, atr

# =================== IA GROQ ===================
def analizar_con_groq_texto(descripcion, atr):
    try:
        system_msg = """
        Eres Steve Nison. Analizas todo el "Panorama Holístico": Tendencia, Zonas Clave y Formaciones de Velas Combinadas.

        REGLAS DE ORO:
        1. NO operar en el medio de un rango ("Tierra de nadie") a menos que haya un patrón de continuación muy agresivo (Tres Soldados / Marubozu).
        2. Para un "Buy" de reversión, EXIGE que la micro-tendencia sea bajista, el precio esté en Zona de Soporte o EMA20, y haya un patrón alcista (Estrella de la mañana, Pinzas de suelo, Envolvente alcista).
        3. Para un "Sell" de reversión, EXIGE que la micro-tendencia sea alcista, el precio esté en Zona de Resistencia o EMA20, y haya un patrón bajista (Estrella del atardecer, Nube Oscura, Envolvente bajista).
        4. Evita comprar contra la tendencia dominante sin una vela de confirmación fuerte.

        Responde ÚNICAMENTE con un JSON válido en este formato estricto:
        {
          "decision": "Buy/Sell/Hold",
          "patron": "Explica la suma de la confluencia (Ej: Envolvente en Soporte + EMA20 a favor)",
          "razones": ["Razón 1", "Razón 2"],
          "sl_mult": 1.5,
          "tp1_mult": 2.0,
          "trailing_mult": 1.5
        }
        """
        user_msg = f"{descripcion}\n\nATR Actual: {atr:.2f}. Piensa como un Chartista Institucional y dicta tu veredicto."
        
        respuesta = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.1, 
            max_tokens=400
        )
        raw = respuesta.choices[0].message.content
        print(f"\n🔍 Análisis Groq:\n{raw}\n")
        
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        json_str = match.group(0) if match else raw
        datos = json.loads(re.sub(r'[\x00-\x1f\x7f]', '', json_str))
        
        decision = datos.get("decision", "Hold")
        patron = datos.get("patron", "")
        razones = datos.get("razones", ["Sin razones"])
        sl_m = max(1.0, min(2.5, float(datos.get("sl_mult", DEFAULT_SL_MULT))))
        tp_m = max(1.2, min(3.0, float(datos.get("tp1_mult", DEFAULT_TP1_MULT))))
        tr_m = max(1.2, min(2.5, float(datos.get("trailing_mult", DEFAULT_TRAILING_MULT))))
        
        return decision, razones, patron, (sl_m, tp_m, tr_m)
    except Exception as e:
        print(f"Error IA: {e}")
        return "Hold", ["Error Parseo IA"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

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
        
    ax.axhline(soporte, color='cyan', ls='--', lw=2, label=f'Soporte ({soporte:.1f})')
    ax.axhline(resistencia, color='magenta', ls='--', lw=2, label=f'Resistencia ({resistencia:.1f})')
    if 'ema20' in df_plot.columns: ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    if 'ema50' in df_plot.columns: ax.plot(x, df_plot['ema50'], 'orange', lw=1.5, ls=':', label='EMA50')
    
    if tipo == "Entrada":
        precio = df_plot['close'].iloc[-2]
        marcador = '^' if decision == 'Buy' else 'v'
        c_marcador = 'lime' if decision == 'Buy' else 'red'
        ax.scatter(len(df_plot)-2, precio + (-30 if decision=='Buy' else 30), s=400, marker=marcador, c=c_marcador, edgecolors='white', zorder=5)
        txt = f"DECISIÓN: {decision.upper()}\nContexto: {patron}\n" + "\n".join(razones[:2])
    else:
        p_ent, p_sal, win, pnl = salida_data
        ax.axhline(p_ent, color='blue', ls=':', lw=2, label=f'Entrada: {p_ent:.1f}')
        ax.axhline(p_sal, color='white', ls=':', lw=2, label=f'Salida: {p_sal:.1f}')
        txt = f"CIERRE: {'WIN ✅' if win else 'LOSS ❌'} | PnL: {pnl:.2f} USD"

    ax.text(0.01, 0.99, txt, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_facecolor('#121212'); fig.patch.set_facecolor('#121212'); ax.tick_params(colors='white'); ax.grid(True, alpha=0.1)
    ax.legend(loc='upper right', facecolor='black', labelcolor='white')
    plt.tight_layout()
    ruta = f"/tmp/chart_{tipo.lower()}.png"
    plt.savefig(ruta, dpi=120)
    plt.close()
    return ruta

# =================== GESTIÓN ===================
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY
    hoy = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY, PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY = hoy, PAPER_BALANCE, False
    if (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE <= -MAX_DAILY_DRAWDOWN_PCT:
        PAPER_STOPPED_TODAY = True
    return not PAPER_STOPPED_TODAY

def paper_abrir_posicion(decision, precio, atr, razones, patron, multis):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL, ULTIMOS_MULTIS
    if PAPER_POSICION_ACTIVA: return False
    
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    sl_m, tp_m, tr_m = multis
    
    PAPER_SL_INICIAL = precio - (atr * sl_m) if decision == "Buy" else precio + (atr * sl_m)
    PAPER_TP1 = precio + (atr * tp_m) if decision == "Buy" else precio - (atr * tp_m)
    if abs(precio - PAPER_SL_INICIAL) == 0: return False
    
    PAPER_SIZE_BTC = min((riesgo_usd / abs(precio - PAPER_SL_INICIAL)) * precio, PAPER_BALANCE * LEVERAGE) / precio
    PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_TRAILING_MULT = decision, precio, tr_m
    PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL = PAPER_SIZE_BTC, False, PAPER_SL_INICIAL
    ULTIMOS_MULTIS = multis
    
    telegram_mensaje(f"📌 {decision.upper()} Ejecutado | In: {precio:.1f} | SL: {PAPER_SL_INICIAL:.1f} | TP: {PAPER_TP1:.1f}\n🔍 Contexto: {patron}")
    return True

def paper_revisar_sl_tp(df, sop, res, slo, inter):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL, PAPER_BALANCE, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES
    if not PAPER_POSICION_ACTIVA: return None
    
    h, l, c, atr = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1], df['atr'].iloc[-1]
    cerrar, motivo = False, ""
    
    if not PAPER_TP1_EJECUTADO:
        if (PAPER_POSICION_ACTIVA == "Buy" and h >= PAPER_TP1) or (PAPER_POSICION_ACTIVA == "Sell" and l <= PAPER_TP1):
            PAPER_PNL_PARCIAL = abs(PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_TP1)
            PAPER_BALANCE += PAPER_PNL_PARCIAL
            PAPER_SIZE_BTC_RESTANTE *= (1 - PORCENTAJE_CIERRE_TP1)
            PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL = True, PAPER_PRECIO_ENTRADA
            telegram_mensaje(f"🎯 TP1 Alcanzado! SL a Break-Even.")
            
    if PAPER_TP1_EJECUTADO:
        n_sl = c - (atr * PAPER_TRAILING_MULT) if PAPER_POSICION_ACTIVA == "Buy" else c + (atr * PAPER_TRAILING_MULT)
        if (PAPER_POSICION_ACTIVA == "Buy" and n_sl > PAPER_SL_ACTUAL) or (PAPER_POSICION_ACTIVA == "Sell" and n_sl < PAPER_SL_ACTUAL):
            PAPER_SL_ACTUAL = n_sl
        if (PAPER_POSICION_ACTIVA == "Buy" and l <= PAPER_SL_ACTUAL) or (PAPER_POSICION_ACTIVA == "Sell" and h >= PAPER_SL_ACTUAL):
            cerrar, motivo = True, "Trailing Stop"
    else:
        if (PAPER_POSICION_ACTIVA == "Buy" and l <= PAPER_SL_INICIAL) or (PAPER_POSICION_ACTIVA == "Sell" and h >= PAPER_SL_INICIAL):
            cerrar, motivo = True, "Stop Loss"
            PAPER_SL_ACTUAL = PAPER_SL_INICIAL

    if cerrar:
        pnl_rest = (PAPER_SL_ACTUAL - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC_RESTANTE if PAPER_POSICION_ACTIVA == "Buy" else (PAPER_PRECIO_ENTRADA - PAPER_SL_ACTUAL) * PAPER_SIZE_BTC_RESTANTE
        pnl_total = PAPER_PNL_PARCIAL + pnl_rest
        PAPER_BALANCE += pnl_rest
        PAPER_TRADES_TOTALES += 1
        win = pnl_total > 0
        if win: PAPER_WIN += 1 
        else: PAPER_LOSS += 1
        
        msg = f"📤 CIERRE ({motivo}) | {PAPER_POSICION_ACTIVA} | PnL: {pnl_total:.2f} USD | Bal: {PAPER_BALANCE:.2f}"
        telegram_mensaje(msg)
        ruta_img = generar_grafico(df, PAPER_POSICION_ACTIVA, [], "", sop, res, slo, inter, ULTIMOS_MULTIS, "Salida", (PAPER_PRECIO_ENTRADA, PAPER_SL_ACTUAL, win, pnl_total))
        telegram_enviar_imagen(ruta_img, msg)
        PAPER_POSICION_ACTIVA = None
        return True
    return None

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON
    print("🤖 BOT V99.20 INICIADO - Nison Holístico (15+ Patrones, Contexto y Confluencia)")
    ultima_vela = None
    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            vela_cerrada = df.index[-2]
            if PAPER_POSICION_ACTIVA is None and ultima_vela != vela_cerrada:
                sop, res, slo, inter, tend, micro = detectar_zonas_mercado(df)
                desc, atr_val = generar_descripcion_nison(df)
                print(f"\n--- Evaluando Vela {vela_cerrada.strftime('%H:%M')} ---")
                
                decision, razones, patron, multis = analizar_con_groq_texto(desc, atr_val)
                ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON = decision, razones[0] if razones else "", razones, patron
                
                if decision in ["Buy","Sell"] and risk_management_check():
                    if paper_abrir_posicion(decision, df['close'].iloc[-1], atr_val, razones, patron, multis):
                        ultima_vela = vela_cerrada
                        ruta_img = generar_grafico(df, decision, razones, patron, sop, res, slo, inter, multis, "Entrada")
                        telegram_enviar_imagen(ruta_img, f"🚀 {decision} (Nison)\n{patron}\n{razones[0]}")
                else:
                    print(f"⏸️ Hold: {ULTIMO_MOTIVO[:60]}...")
                    ultima_vela = vela_cerrada
            
            if PAPER_POSICION_ACTIVA:
                sop, res, slo, inter, _, _ = detectar_zonas_mercado(df, -1)
                paper_revisar_sl_tp(df, sop, res, slo, inter)
            
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
