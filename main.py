# BOT TRADING V99.30 – GROQ (NISON MATRIZ DE CONFLUENCIA + VETOS DE VELAS)
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
REGLAS_APRENDIDAS = "Aún no hay trades suficientes. Usa la Matriz de Confluencia Nison."

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

# =================== MOTOR HOLÍSTICO NISON ===================
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
        patrones.append("🌟 ESTRELLA DE LA MAÑANA (Reversión Alcista)")
    if verde1 and c1 > r1*0.5 and cp2 < 30 and not verde3 and c3 > r3*0.5 and v3['close'] < (v1['open']+v1['close'])/2:
        patrones.append("🌟 ESTRELLA DEL ATARDECER (Reversión Bajista)")
    
    # Continuaciones Fuertes
    if verde1 and verde2 and verde3 and c3 > c2 and c2 > c1 and inf3 < 20 and inf2 < 20:
        patrones.append("🚀 TRES SOLDADOS BLANCOS (Continuación Alcista)")
    if not verde1 and not verde2 and not verde3 and c3 > c2 and c2 > c1 and sup3 < 20 and sup2 < 20:
        patrones.append("🩸 TRES CUERVOS NEGROS (Continuación Bajista)")

    # Envolventes
    if not verde2 and verde3 and v3['close'] > v2['open'] and v3['open'] < v2['close']:
        patrones.append("🐂 ENVOLVENTE ALCISTA")
    if verde2 and not verde3 and v3['close'] < v2['open'] and v3['open'] > v2['close']:
        patrones.append("🐻 ENVOLVENTE BAJISTA")

    # Rechazos y Fuerzas Individuales
    if verde3 and cp3 > 70 and sup3 < 10:
        patrones.append("📈 VELA SÓLIDA EN MÁXIMOS (Toros en Control)")
    elif not verde3 and cp3 > 70 and inf3 < 10:
        patrones.append("📉 VELA SÓLIDA EN MÍNIMOS (Osos en Control)")
    elif cp3 < 15 and sup3 > 25 and inf3 > 25:
        patrones.append("⚖️ DOJI / INDECISIÓN TOTAL")
    else:
        # Aquí el bot detecta explícitamente el peligro
        if inf3 > 60 and cp3 < 25 and sup3 < 15:
            patrones.append("🔨 MARTILLO (Rechazo a la baja - Presión Compradora)")
        elif sup3 > 60 and cp3 < 25 and inf3 < 15:
            patrones.append("🌠 ESTRELLA FUGAZ / SHOOTING STAR (Rechazo al alza - Fuerte Presión Vendedora)")

    if not patrones:
        patrones.append(f"Vela {'Verde' if verde3 else 'Roja'} Estándar (Cuerpo: {cp3:.0f}%, Mecha Sup: {sup3:.0f}%, Mecha Inf: {inf3:.0f}%)")
        
    return " | ".join(patrones)

def generar_descripcion_nison(df, idx=-2):
    vela_actual = df.iloc[idx]
    precio = vela_actual['close']
    atr = df['atr'].iloc[idx]
    ema20 = df['ema20'].iloc[idx]
    
    soporte, resistencia, slope, intercept, tendencia, micro = detectar_zonas_mercado(df, idx)
    patron = analizar_patron_nison_completo(df, idx)

    # Secuencia visual de colores y cuerpos
    secuencia_velas = []
    rango_velas = range(idx-4, idx+1) if idx == -1 else range(idx-4, idx+1)
    for i in rango_velas:
        try:
            v = df.iloc[i]
            color = "VERDE" if v['close'] > v['open'] else "ROJA"
            secuencia_velas.append(color)
        except: pass
    secuencia_str = " -> ".join(secuencia_velas) + " (Actual)"

    # Evaluación Profunda de EMA y Sobre-extensión
    dist_ema_pct = (precio - ema20) / ema20 * 100
    if precio > ema20:
        if vela_actual['low'] <= ema20:
            interaccion_ema = "✅ Testeando EMA20 desde arriba (Actuando como Soporte Dinámico)."
        elif dist_ema_pct > 0.4:
            interaccion_ema = f"⚠️ Precio muy extendido sobre EMA20 (+{dist_ema_pct:.2f}%). Cuidado con compras en la cima (Riesgo de Reversión)."
        else:
            interaccion_ema = f"✅ Cabalgando sanamente sobre EMA20 (+{dist_ema_pct:.2f}%)."
    else:
        if vela_actual['high'] >= ema20:
            interaccion_ema = "❌ Testeando EMA20 desde abajo (Actuando como Resistencia Dinámica)."
        elif dist_ema_pct < -0.4:
            interaccion_ema = f"⚠️ Precio muy extendido bajo EMA20 ({dist_ema_pct:.2f}%). Cuidado con ventas en el fondo."
        else:
            interaccion_ema = f"❌ Presionado sanamente bajo EMA20 ({dist_ema_pct:.2f}%)."

    # Evaluación de Rupturas (S/R Flips) y Zonas Clave
    zona_actual = f"Precio: {precio:.2f}."
    if precio > resistencia and vela_actual['low'] <= resistencia * 1.002:
        zona_actual += f" 🔥 RUPTURA: La Resistencia ({resistencia:.2f}) se rompió y ahora es Soporte (Throwback)."
    elif precio < soporte and vela_actual['high'] >= soporte * 0.998:
        zona_actual += f" 🚨 RUPTURA: El Soporte ({soporte:.2f}) se rompió y ahora es Resistencia (Pullback)."
    elif precio <= soporte + (atr * 1.2):
        zona_actual += f" 🛡️ Cerca de Zona de Soporte Mayor ({soporte:.2f})."
    elif precio >= resistencia - (atr * 1.2):
        zona_actual += f" 🧱 Cerca de Zona de Resistencia Mayor ({resistencia:.2f})."
    else:
        zona_actual += " 🌊 En el canal intermedio (Lejos de Soportes y Resistencias horizontales)."

    # Clústeres (Presión Acumulada)
    df_mechas = df.iloc[idx-8:] if idx == -1 else df.iloc[idx-8:idx+1]
    rangos = df_mechas['high'] - df_mechas['low']
    mechas_sup = (df_mechas['high'] - df_mechas[['close', 'open']].max(axis=1)) / rangos.replace(0, 0.001)
    mechas_inf = (df_mechas[['close', 'open']].min(axis=1) - df_mechas['low']) / rangos.replace(0, 0.001)
    
    cluster_txt = f"{sum(mechas_sup > 0.5)} velas recientes con Mecha Superior larga (Rechazo Bajista). "
    cluster_txt += f"{sum(mechas_inf > 0.5)} velas recientes con Mecha Inferior larga (Rechazo Alcista)."

    descripcion = f"""
=== INFORME PRICE ACTION 360° (5 MINUTOS) ===

1. ESTRUCTURA Y TENDENCIA
- Tendencia Macro: {tendencia}
- Zona Horizontal: {zona_actual}
- Relación con EMA20: {interaccion_ema}

2. PRESIÓN ACUMULADA (Clústeres)
- Impulso Reciente: {micro}
- Secuencia de Colores: {secuencia_str}
- Rechazos Acumulados: {cluster_txt}

3. ACCIÓN DEL PRECIO INMEDIATA (El Gatillo)
- Vela/Patrón Actual: {patron}
"""
    return descripcion, atr

# =================== IA GROQ: DECISIONES BLINDADAS ===================
def analizar_con_groq_texto(descripcion, atr, reglas_aprendidas):
    try:
        system_msg = f"""
        Eres Steve Nison. Analizas la CONFLUENCIA TOTAL (Soportes/Resistencias, Velas, Mechas, Línea de Tendencia, EMA20, S/R Flips).
        Operamos gráficos de 5 MINUTOS (Scalping).

        🔥 MATRIZ DE CONFLUENCIA Y VETOS ABSOLUTOS 🔥
        1. LA VELA MANDA SOBRE LA EMA: Si el precio está sobre la EMA20 (Alcista) PERO la última vela es una "ESTRELLA FUGAZ", "ENVOLVENTE BAJISTA", o tiene una mecha superior enorme, ¡TIENES PROHIBIDO COMPRAR (BUY)! Es señal de reversión o debilidad extrema. Da "Hold" o "Sell".
        2. COMPRA (BUY) PERFECTA: Tendencia/Micro alcista + Apoyo en EMA20 o Soporte (Throwback) + Vela Gatillo Alcista (Martillo, Envolvente Verde, Cierre Sólido Verde).
        3. VENTA (SELL) PERFECTA: Tendencia/Micro bajista + Choque en EMA20 o Resistencia (Pullback) + Vela Gatillo Bajista (Estrella Fugaz, Envolvente Roja, Cierre Sólido Rojo).
        4. INDECISIÓN = HOLD: Si la tendencia es Alcista pero la vela es Bajista (o viceversa), se anulan mutuamente. El resultado debe ser "Hold".

        REGLA EVOLUTIVA DEL MENTOR: "{reglas_aprendidas}"

        Responde ÚNICAMENTE con un JSON válido en este formato:
        {{
          "decision": "Buy/Sell/Hold",
          "patron": "Resumen de la Confluencia. Ej: Tendencia Alcista anulada por Estrella Fugaz en Resistencia (Veto de Compra)",
          "razones": ["Razón de Estructura/EMA", "Razón de Vela Gatillo"],
          "sl_mult": 1.2,
          "tp1_mult": 1.5,
          "trailing_mult": 1.8
        }}
        """
        user_msg = f"{descripcion}\n\nATR: {atr:.2f}. Evalúa la confluencia. Si la vela actual contradice la EMA, respeta el VETO de Nison. Decide:"
        
        respuesta = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.1, # Muy bajo para que respete las reglas de veto estrictamente
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
        resumen_trades += f"Trade {i+1} ({estado}): {t['decision'].upper()} | Contexto: {t.get('patron', 'Desconocido')} | PnL: {t['pnl']:.2f}\n"

    system_msg = """
    Eres el Mentor de Trading de una IA orientada al Price Action.
    Revisa los últimos 10 trades. Detecta si la IA está ignorando la confluencia (ej. comprando estrellas fugaces solo por estar sobre la EMA, o vendiendo martillos).
    
    Responde ÚNICAMENTE con un JSON válido:
    {
      "analisis": "Breve explicación de los aciertos y errores estructurales",
      "nueva_regla": "Instrucción estricta de 1 oración para obligar a buscar confluencia real y no caer en trampas",
      "sl_mult_sugerido": 1.5,
      "tp1_mult_sugerido": 1.5,
      "trailing_mult_sugerido": 1.8
    }
    """
    
    user_msg = f"Winrate: {winrate*100:.0f}%. ({wins} Ganados, {losses} Perdidos).\n\nHistorial:\n{resumen_trades}\n\nAnaliza y dicta la nueva regla."
    
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
    
    # Línea de Tendencia Inclinada Macro
    ax.plot(x, intercept + slope * x, color='white', linestyle='-.', linewidth=1.5, alpha=0.6, label='Tendencia Inclinada')
    
    if 'ema20' in df_plot.columns: ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA 20')
    
    sl_m, tp1_m, trail_m = multi
    
    if tipo == "Entrada":
        p_act = df_plot['close'].iloc[-2]
        ax.scatter(len(df_plot)-2, p_act + (-30 if decision=='Buy' else 30), s=400, marker='^' if decision=='Buy' else 'v', c='lime' if decision=='Buy' else 'red', zorder=5)
        txt = f"DECISIÓN: {decision.upper()}\nConfluencia: {patron}\nSL mult: {sl_m:.2f} | TP1 mult: {tp1_m:.2f} | Trail mult: {trail_m:.2f}\nRazones:\n" + "\n".join(razones[:2])
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
🔍 Confluencia: {patron}"""
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
    print("🤖 BOT V99.30 INICIADO - Confluencia Perfecta Nison + Vetos de Vela")
    telegram_mensaje("🤖 BOT V99.30 INICIADO - Matriz de Confluencia y Vetos activados.")
    
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
                print(f"📝 Reporte generado para Groq:\n{desc}\n")
                
                decision, razones, patron, multis = analizar_con_groq_texto(desc, atr_val, REGLAS_APRENDIDAS)
                ULTIMA_DECISION = decision
                ULTIMO_MOTIVO = razones[0] if razones else "Sin razones"
                ULTIMA_RAZONES = razones
                ULTIMO_PATRON = patron
                
                if decision in ["Buy","Sell"] and risk_management_check():
                    if paper_abrir_posicion(decision, precio, atr_val, razones, patron, multis):
                        ultima_vela = vela_cerrada
                        ruta_img = generar_grafico(df, decision, razones, patron, sop, res, slo, inter, multis, "Entrada")
                        telegram_enviar_imagen(ruta_img, f"🚀 SEÑAL CONFIRMADA: {decision.upper()}\nConfluencia: {patron}\n{razones[0]}")
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
