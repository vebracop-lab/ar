# BOT TRADING V99.19 – GROQ (NISON ESTRICTO: CONTEXTO, RECHAZOS Y VETOS)
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
PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES = 0, 0, 0
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

ULTIMA_DECISION, ULTIMO_MOTIVO, ULTIMA_RAZONES, ULTIMO_PATRON = "Hold", "Esperando señal", [], ""
ULTIMOS_MULTIS = (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

# =================== TELEGRAM ===================
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
    
    # Micro tendencia (últimas 5 velas)
    micro_slope, _, _, _, _ = linregress(np.arange(5), df_eval['close'].values[-5:])
    micro_tendencia = 'CAYENDO FUERTE' if micro_slope < -0.5 else 'SUBIENDO FUERTE' if micro_slope > 0.5 else 'LATERAL'
    
    tendencia = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tendencia, micro_tendencia

# =================== ANÁLISIS NISON (CONTEXTUAL) ===================
def analizar_patron_nison(v_actual, v_previa, v_pre_previa, micro_tendencia):
    # Calcular anatomía vela actual
    rango = v_actual['high'] - v_actual['low']
    if rango == 0: return "Doji", "Sin rango"
    cuerpo = abs(v_actual['close'] - v_actual['open'])
    cuerpo_pct = (cuerpo / rango) * 100
    sombra_sup_pct = ((v_actual['high'] - max(v_actual['close'], v_actual['open'])) / rango) * 100
    sombra_inf_pct = ((min(v_actual['close'], v_actual['open']) - v_actual['low']) / rango) * 100
    es_verde = v_actual['close'] > v_actual['open']
    previa_verde = v_previa['close'] > v_previa['open']

    anatomia = f"Cuerpo:{cuerpo_pct:.0f}% | Msup:{sombra_sup_pct:.0f}% | Minf:{sombra_inf_pct:.0f}%"
    patron = f"Vela {'Verde' if es_verde else 'Roja'} Normal"

    # 1. DOJI
    if cuerpo_pct < 10: patron = "DOJI (Indecisión)"
    
    # 2. PATRONES DE REVERSIÓN (REQUIEREN CONTEXTO DE TENDENCIA PREVIA)
    if "CAYENDO" in micro_tendencia:
        if sombra_inf_pct > 65 and cuerpo_pct < 25 and sombra_sup_pct < 10:
            patron = "🔥 MARTILLO (Reversión Alcista muy fuerte si está en soporte)"
        elif es_verde and not previa_verde and v_actual['close'] > v_previa['open'] and v_actual['open'] < v_previa['close']:
            patron = "🔥 ENVOLVENTE ALCISTA (Fuerte señal de compra)"
        elif es_verde and not previa_verde and v_actual['close'] > (v_previa['open'] + v_previa['close'])/2 and v_actual['open'] < v_previa['close']:
            patron = "✅ PAUTA PENETRANTE (Posible reversión alcista)"

    elif "SUBIENDO" in micro_tendencia:
        if sombra_sup_pct > 65 and cuerpo_pct < 25 and sombra_inf_pct < 10:
            patron = "🚨 ESTRELLA FUGAZ (Reversión Bajista muy fuerte si está en resistencia)"
        if sombra_inf_pct > 65 and cuerpo_pct < 25 and sombra_sup_pct < 10:
            patron = "🚨 HOMBRE COLGADO (Reversión Bajista, el soporte falló)"
        elif not es_verde and previa_verde and v_actual['close'] < v_previa['open'] and v_actual['open'] > v_previa['close']:
            patron = "🚨 ENVOLVENTE BAJISTA (Fuerte señal de venta)"
        elif not es_verde and previa_verde and v_actual['close'] < (v_previa['open'] + v_previa['close'])/2 and v_actual['open'] > v_previa['close']:
            patron = "⚠️ NUBE OSCURA (Posible reversión bajista)"

    # Patrón de continuación sin importar microtendencia
    if cuerpo_pct > 80:
        patron = "VELA MARUBOZU / IMPULSO " + ("ALCISTA" if es_verde else "BAJISTA")

    return patron, anatomia

# =================== DESCRIPCIÓN ENRIQUECIDA ===================
def generar_descripcion_nison(df, idx=-2):
    vela_actual = df.iloc[idx]
    precio = vela_actual['close']
    atr = df['atr'].iloc[idx]
    ema20 = df['ema20'].iloc[idx]
    ema50 = df['ema50'].iloc[idx]
    rsi = df['rsi'].iloc[idx]
    soporte, resistencia, slope, intercept, tendencia, micro_tendencia = detectar_zonas_mercado(df, idx)
    
    # Análisis de Interacción con EMA (Rechazos Reales)
    distancia_ema20 = (precio - ema20) / ema20 * 100
    interaccion_ema = "Lejos de EMA20"
    if vela_actual['low'] <= ema20 <= vela_actual['high']:
        if vela_actual['close'] > ema20 and (ema20 - vela_actual['low']) > abs(vela_actual['close'] - vela_actual['open']):
            interaccion_ema = "✅ RECHAZO DESDE ABAJO EN EMA20 (Mecha inferior la perforó, pero cerró arriba = SOPORTE DINÁMICO CONFIRMADO)"
        elif vela_actual['close'] < ema20 and (vela_actual['high'] - ema20) > abs(vela_actual['close'] - vela_actual['open']):
            interaccion_ema = "❌ RECHAZO DESDE ARRIBA EN EMA20 (Mecha superior la perforó, pero cerró abajo = RESISTENCIA DINÁMICA CONFIRMADA)"
        else:
            interaccion_ema = "Atravesando EMA20 (Consolidación/Indecisión)"
    elif precio > ema20:
        interaccion_ema = f"Precio firmemente por ENCIMA de EMA20 (+{distancia_ema20:.2f}%)"
    else:
        interaccion_ema = f"Precio firmemente por DEBAJO de EMA20 ({distancia_ema20:.2f}%)"

    # Historial detallado de las últimas 5 velas (para ver secuencias)
    historial_velas = []
    for i in range(idx-4, idx+1):
        if i < 1: continue
        v = df.iloc[i]; v_prev = df.iloc[i-1]; v_prev2 = df.iloc[i-2]
        pat, anat = analizar_patron_nison(v, v_prev, v_prev2, micro_tendencia)
        historial_velas.append(f"Vela {df.index[i].strftime('%H:%M')}: {pat} | {anat}")

    # Estructura horizontal
    dist_soporte = (precio - soporte) / precio * 100
    dist_resistencia = (resistencia - precio) / precio * 100

    descripcion = f"""
=== INFORME STEVE NISON (BTCUSDT 5m) ===

1. CONTEXTO DE TENDENCIA (LA REGLA #1 DE NISON)
- Tendencia Macro (120 velas): {tendencia}
- Micro-Tendencia (Últimas 5 velas): {micro_tendencia} 
  (¡Crucial!: Los patrones de reversión solo valen si van en contra de esta micro-tendencia).

2. SOPORTES Y RESISTENCIAS HORIZONTALES
- Resistencia Techo: {resistencia:.2f} (a {dist_resistencia:.2f}% de distancia)
- Soporte Suelo: {soporte:.2f} (a {dist_soporte:.2f}% de distancia)
*(Si el precio está muy cerca del soporte, buscar patrones alcistas. Si está cerca de resistencia, bajistas).*

3. EMA20 (SOPORTE/RESISTENCIA DINÁMICO)
- Valor EMA20: {ema20:.2f}
- Acción: {interaccion_ema}

4. SECUENCIA DE VELAS EXACTA (LEER DE ARRIBA HACIA ABAJO)
{chr(10).join(historial_velas)}

5. INDICADORES EXTRA
- RSI: {rsi:.1f}
"""
    return descripcion, atr

# =================== IA GROQ ===================
def analizar_con_groq_texto(descripcion, atr):
    try:
        system_msg = """
        Eres Steve Nison. Analizas velas japonesas estrictamente por su CONTEXTO.
        
        REGLAS DE VETO ABSOLUTO (INQUEBRANTABLES):
        1. VETO DE CUCHILLO CAYENDO: NUNCA sugieras "Buy" si la Micro-Tendencia es "CAYENDO FUERTE" y la última vela es ROJA. Debes esperar a una vela verde de confirmación (ej. Envolvente Alcista).
        2. VETO DE MARTILLO: Si en las últimas 3 velas hubo un "MARTILLO" en zona de soporte, NUNCA sugieras "Sell", ya que los grandes capitales acaban de rechazar precios bajos.
        3. VETO DE ESTRELLA FUGAZ: Si hubo una "ESTRELLA FUGAZ" o "ENVOLVENTE BAJISTA", NUNCA sugieras "Buy" hasta que la EMA20 sea recuperada.
        4. VETO DE RANGO MEDIO: Si el precio está en el medio del rango (lejos de soporte, resistencia y EMA20), la decisión debe ser "Hold".

        LÓGICA DE DECISIÓN:
        - BUY: Requiere que el precio rebote en Soporte o EMA20 (rechazo inferior) + Patrón alcista (Martillo, Envolvente verde).
        - SELL: Requiere rechazo en Resistencia o EMA20 por debajo (mecha superior) + Patrón bajista (Estrella fugaz, Envolvente roja).
        - HOLD: Si hay indecisión, velas Doji, o falta de confluencia.

        Responde ÚNICAMENTE con un JSON válido:
        {
          "decision": "Buy/Sell/Hold",
          "patron": "Situación principal identificada (ej. Martillo en soporte + Confirmación)",
          "razones": ["razón1 detallada", "razón2"],
          "sl_mult": 1.5,
          "tp1_mult": 2.0,
          "trailing_mult": 1.5
        }
        """
        user_msg = f"{descripcion}\n\nATR: {atr:.2f}. Analiza la secuencia paso a paso mentalmente, aplica los vetos, y da tu decisión."
        
        respuesta = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.1, # Más bajo para mayor lógica estricta
            max_tokens=400
        )
        raw = respuesta.choices[0].message.content
        print(f"\n🔍 Groq:\n{raw}\n")
        
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
    
    if tipo == "Entrada":
        precio = df_plot['close'].iloc[-2]
        marcador = '^' if decision == 'Buy' else 'v'
        c_marcador = 'lime' if decision == 'Buy' else 'red'
        ax.scatter(len(df_plot)-2, precio + (-30 if decision=='Buy' else 30), s=400, marker=marcador, c=c_marcador, edgecolors='white', zorder=5)
        txt = f"DECISIÓN: {decision.upper()}\nPatrón: {patron}\n" + "\n".join(razones[:2])
    else:
        p_ent, p_sal, win, pnl = salida_data
        ax.axhline(p_ent, color='blue', ls=':', lw=2, label=f'Entrada: {p_ent:.1f}')
        ax.axhline(p_sal, color='orange', ls=':', lw=2, label=f'Salida: {p_sal:.1f}')
        txt = f"CIERRE: {'WIN ✅' if win else 'LOSS ❌'} | PnL: {pnl:.2f} USD"

    ax.text(0.01, 0.99, txt, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_facecolor('#121212'); fig.patch.set_facecolor('#121212'); ax.tick_params(colors='white'); ax.grid(True, alpha=0.1)
    ax.legend(loc='upper right', facecolor='black', labelcolor='white')
    plt.tight_layout()
    ruta = f"/tmp/chart_{tipo.lower()}.png"
    plt.savefig(ruta, dpi=120)
    plt.close()
    return ruta

# =================== RISK MGMT Y GESTIÓN ===================
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
    distancia = abs(precio - PAPER_SL_INICIAL)
    if distancia == 0: return False
    
    PAPER_SIZE_BTC = min((riesgo_usd / distancia) * precio, PAPER_BALANCE * LEVERAGE) / precio
    PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_TRAILING_MULT = decision, precio, tr_m
    PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL = PAPER_SIZE_BTC, False, PAPER_SL_INICIAL
    ULTIMOS_MULTIS = multis
    
    telegram_mensaje(f"📌 {decision.upper()} | In: {precio:.1f} | SL: {PAPER_SL_INICIAL:.1f} | TP: {PAPER_TP1:.1f}\nMotivo: {patron}")
    return True

def paper_revisar_sl_tp(df, sop, res, slo, inter):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL_INICIAL, PAPER_TP1, PAPER_TRAILING_MULT, PAPER_SIZE_BTC, PAPER_SIZE_BTC_RESTANTE, PAPER_TP1_EJECUTADO, PAPER_SL_ACTUAL, PAPER_BALANCE, PAPER_PNL_PARCIAL, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES, TRADE_HISTORY
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
    print("🤖 BOT V99.19 INICIADO - Nison Puro (Contexto y Vetos de IA)")
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
                    print(f"⏸️ Hold: {ULTIMO_MOTIVO}")
                    ultima_vela = vela_cerrada # Marcar evaluada aunque sea hold
            
            if PAPER_POSICION_ACTIVA:
                sop, res, slo, inter, _, _ = detectar_zonas_mercado(df, -1)
                paper_revisar_sl_tp(df, sop, res, slo, inter)
            
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
