# =====================================================================
# BOT TRADING MAESTRO V101.0 - PROTOCOLO DE ANALISIS ESTRUCTURAL TOTAL
# =====================================================================
# ⚠️ ESTRICTAMENTE PROHIBIDA LA SIMPLIFICACIÓN O RESUMEN DE CÓDIGO.
# ⚠️ ENFOQUE: ACCIÓN DE PRECIO (PRICE ACTION) Y RECHAZO DE MECHAS.
# ⚠️ INTEGRACIÓN: GROQ IA (LLAMA 3.3 70B) + BYBIT V5.
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

# Configuración de entorno para ejecución en la nube (Railway/VPS)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ======================================================
# 1. CONFIGURACIÓN DE CONECTIVIDAD (APIs)
# ======================================================
from groq import Groq

# Las llaves se cargan desde las variables de entorno del sistema
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)
MODELO_GROQ = "llama-3.3-70b-versatile"

# Credenciales de Exchange y Mensajería
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL = "https://api.bybit.com"

# ======================================================
# 2. PARÁMETROS OPERATIVOS ESTRATÉGICOS
# ======================================================
SYMBOL = "BTCUSDT"
INTERVAL = "5"        # Temporalidad principal: 5 minutos
LEVERAGE = 10         # Apalancamiento institucional
RISK_PER_TRADE = 0.02 # Arriesgar el 2% del balance por operación
SLEEP_SECONDS = 60    # Frecuencia de actualización del bot

# Configuración de Gestión de Salidas (ATR-Based)
MULT_SL = 1.8         # Stop Loss más holgado para evitar "stop hunts"
MULT_TP1 = 2.5        # Objetivo de beneficio 1
MULT_TRAILING = 2.0   # Distancia para el Trailing Stop
PORCENTAJE_CIERRE_PARCIAL = 0.5 

# ======================================================
# 3. SISTEMA DE GESTIÓN DE CAPITAL (PAPER TRADING)
# ======================================================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_POSICION_ACTIVA = None # 'Buy', 'Sell', None
PAPER_PRECIO_ENTRADA = 0.0
PAPER_SL = 0.0
PAPER_TP1 = 0.0
PAPER_TP1_LISTO = False
PAPER_SIZE_BTC = 0.0
PAPER_TRADES_TOTALES = 0
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_LAST_10_PNL = []

# ======================================================
# 4. FUNCIONES DE TELEMETRÍA (TELEGRAM)
# ======================================================
def telegram_enviar_texto(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texto, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=12)
    except Exception as e:
        print(f"Error enviando texto a Telegram: {e}")

def telegram_enviar_grafico(fig):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': buf}
        data = {'chat_id': TELEGRAM_CHAT_ID}
        requests.post(url, files=files, data=data, timeout=25)
        buf.close()
    except Exception as e:
        print(f"Error enviando gráfico a Telegram: {e}")

# ======================================================
# 5. MOTOR DE DATOS Y CÁLCULOS TÉCNICOS PROFUNDOS
# ======================================================
def fetch_mercado_data(limit=250):
    url = f"{BASE_URL}/v5/market/kline"
    params = {"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=20)
        raw = r.json()["result"]["list"][::-1]
        df = pd.DataFrame(raw, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"Error crítico en fetch de datos: {e}")
        return pd.DataFrame()

def inyectar_analisis_institucional(df):
    # --- INDICADORES DINÁMICOS ---
    # EMA 20: La línea de equilibrio del mercado
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # ATR: Medición de la volatilidad real para SL y TP
    tr = pd.concat([df['high']-df['low'], 
                    (df['high']-df['close'].shift()).abs(), 
                    (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # RSI: Fuerza relativa para detectar clímax
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/loss)))

    # --- ANATOMÍA DE VELAS (ANÁLISIS DE MECHAS DETALLADO) ---
    df['rango_total'] = df['high'] - df['low']
    df['cuerpo_absoluto'] = (df['close'] - df['open']).abs()
    
    # Cálculo de mechas (Sombras)
    df['mecha_superior'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['mecha_inferior'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Porcentajes de absorción (Lo que Gemini vió y antes ignoramos)
    df['pct_mecha_sup'] = (df['mecha_superior'] / df['rango_total'] * 100).fillna(0)
    df['pct_mecha_inf'] = (df['mecha_inferior'] / df['rango_total'] * 100).fillna(0)
    
    # --- ESTADO DE CRUCE Y MOMENTUM ---
    df['bajo_ema'] = df['close'] < df['ema20']
    df['cruce_bajista'] = (df['close'] < df['ema20']) & (df['close'].shift(1) > df['ema20'])
    df['cruce_alcista'] = (df['close'] > df['ema20']) & (df['close'].shift(1) < df['ema20'])
    
    return df.dropna()

def detectar_geometria_espacial(df):
    # Identificación de techos y suelos (Zonas de Reacción)
    resistencia_max = df['high'].rolling(60).max().iloc[-1]
    soporte_min = df['low'].rolling(60).min().iloc[-1]
    
    # Tendencia por regresión lineal
    y_vals = df['close'].tail(45).values
    x_vals = np.arange(len(y_vals))
    slope, intercept, r_val, p_val, std_err = linregress(x_vals, y_vals)
    angulo = np.degrees(np.arctan(slope))
    
    # Conteo de "testeo de zona"
    margen = df['close'].iloc[-1] * 0.0008
    golpes_techo = (df['high'] > (resistencia_max - margen)).tail(25).sum()
    
    return {
        "res_macro": resistencia_max,
        "sup_macro": soporte_min,
        "angulo": angulo,
        "slope": slope,
        "intercept": intercept,
        "golpes_res": golpes_techo
    }

# ======================================================
# 6. IA GROQ: EL CEREBRO DE TOMA DE DECISIONES
# ======================================================
def consultoria_ia_groq_v101(df, geo):
    # Generamos un historial textual DE LAS ÚLTIMAS 12 VELAS
    # No solo de una, para que la IA vea la SECUENCIA de absorción.
    bloque_velas = df.tail(12)
    historial_detallado = ""
    
    for i, (idx, fila) in enumerate(bloque_velas.iterrows()):
        color = "🟢 VERDE" if fila['close'] >= fila['open'] else "🔴 ROJA"
        posicion = "BAJO EMA" if fila['close'] < fila['ema20'] else "SOBRE EMA"
        historial_detallado += (f"T-{11-i}: {color} | "
                                f"Mecha Sup: {fila['pct_mecha_sup']:.1f}% | "
                                f"Mecha Inf: {fila['pct_mecha_inf']:.1f}% | "
                                f"Cierre: {fila['close']} | {posicion}\n")

    prompt = f"""
AUDITORÍA TÉCNICA DE TRADING - MODELO V101 (PROFESIONAL)
Tu misión es analizar la estructura y evitar trampas de mercado.

CONFIGURACIÓN GEOMÉTRICA:
- Precio de Mercado: {df['close'].iloc[-1]}
- Resistencia Crítica (Línea Morada): {geo['res_macro']:.2f}
- Soporte Base: {geo['sup_macro']:.2f}
- Ángulo de Tendencia: {geo['angulo']:.2f}°
- Golpes en Resistencia: {geo['golpes_res']} (Fuerza de testeo).

ESTADO DE LA EMA 20 (INSTITUCIONAL):
- ¿Precio por debajo de EMA 20?: {"SÍ (PÉRDIDA DE MOMENTO)" if df['bajo_ema'].iloc[-1] else "NO"}
- ¿Cruce bajista en la vela actual?: {"ALERTA: CRUCE BAJISTA CONFIRMADO" if df['cruce_bajista'].iloc[-1] else "NO"}

SECUENCIA ANATÓMICA DE LAS ÚLTIMAS 12 VELAS:
{historial_detallado}

REGLAS DE EJECUCIÓN (PROHIBIDO IGNORAR):
1. RECHAZO DE MECHA: Si el precio está cerca de la línea morada ({geo['res_macro']}) y ves una secuencia de mechas superiores largas (>45%), hay absorción de ventas masiva.
2. CONFIRMACIÓN: No abras Long si el precio acaba de cerrar por debajo de la EMA 20, sin importar lo que diga el RSI.
3. VEREDICTO GANADOR: Si el precio falla en romper la resistencia y pierde la EMA 20 con velas rojas y mechas superiores, la orden es SELL (Short).

Responde en este formato JSON exacto:
{{
  "decision": "Buy" | "Sell" | "Hold",
  "patron_detectado": "Nombre completo del patrón y contexto",
  "razonamiento_espacial": "Análisis de por qué el precio rechazó o aceptó la zona",
  "fuera_de_zona": true | false
}}
"""
    try:
        respuesta = client.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(respuesta.choices[0].message.content)
    except Exception as e:
        print(f"Error en comunicación con IA: {e}")
        return {"decision": "Hold", "razonamiento_espacial": "Error API"}

# ======================================================
# 7. GENERACIÓN DE GRÁFICOS (EVIDENCIA VISUAL)
# ======================================================
def generar_visualizacion_v101(df, geo, res_ia):
    df_plot = df.tail(110)
    fig, ax = plt.subplots(figsize=(15, 9))
    x_axis = np.arange(len(df_plot))
    
    # Dibujo de Velas Manual (Alta Fidelidad)
    for i in range(len(df_plot)):
        color = 'lime' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else 'red'
        # Mechas
        ax.vlines(x_axis[i], df_plot['low'].iloc[i], df_plot['high'].iloc[i], color=color, linewidth=1.2)
        # Cuerpo
        base = min(df_plot['open'].iloc[i], df_plot['close'].iloc[i])
        altura = max(abs(df_plot['close'].iloc[i] - df_plot['open'].iloc[i]), 0.1)
        ax.add_patch(plt.Rectangle((x_axis[i]-0.3, base), 0.6, altura, color=color, alpha=0.85))

    # Líneas Técnicas
    ax.axhline(geo['res_macro'], color='purple', linewidth=2.5, linestyle='--', label='Línea Morada (RES)')
    ax.axhline(geo['sup_macro'], color='cyan', linewidth=2, linestyle='--', label='Soporte Macro')
    ax.plot(x_axis, df_plot['ema20'].values, color='yellow', linewidth=2.5, label='EMA 20')
    
    # Anotación de la IA
    txt_ia = (f"DECISIÓN: {res_ia['decision'].upper()}\n"
              f"PATRÓN: {res_ia.get('patron_detectado')}\n"
              f"RAZÓN: {textwrap.fill(res_ia.get('razonamiento_espacial', ''), 65)}")
    
    ax.text(0.02, 0.98, txt_ia, transform=ax.transAxes, color='white', verticalalignment='top', 
            bbox=dict(facecolor='black', edgecolor='purple', alpha=0.9, boxstyle='round,pad=1'))

    ax.set_title(f"BOT MAESTRO V101 - BTC/USDT - PRECISIÓN INSTITUCIONAL", color='white', fontsize=14)
    ax.set_facecolor('#0a0a0a'); fig.patch.set_facecolor('#0a0a0a')
    ax.tick_params(colors='white'); ax.grid(alpha=0.1)
    plt.legend(loc='lower right', facecolor='black', labelcolor='white')
    
    return fig

# ======================================================
# 8. GESTIÓN DE RIESGO Y ÓRDENES (LÓGICA DE EJECUCIÓN)
# ======================================================
def monitorear_posicion_activa(df):
    global PAPER_POSICION_ACTIVA, PAPER_BALANCE, PAPER_TP1_LISTO, PAPER_TRADES_TOTALES, PAPER_WIN, PAPER_LOSS, PAPER_LAST_10_PNL
    
    if PAPER_POSICION_ACTIVA is None: return

    vela_actual = df.iloc[-1]
    cierre_requerido = False
    pnl_realizado = 0

    if PAPER_POSICION_ACTIVA == "Buy":
        # Gestión de TP1 Parcial
        if not PAPER_TP1_LISTO and vela_actual['high'] >= PAPER_TP1:
            PAPER_BALANCE += (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_PARCIAL)
            PAPER_TP1_LISTO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA # Mover a Break Even
            telegram_enviar_texto("🎯 TP1 ALCANZADO (LONG). 50% cerrado. SL a Break Even.")
        
        # Gestión de Cierre Total (SL o Trailing)
        if vela_actual['low'] <= PAPER_SL:
            pnl_realizado = (PAPER_SL - PAPER_PRECIO_ENTRADA) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_LISTO else 1.0))
            cierre_requerido = True
            
    elif PAPER_POSICION_ACTIVA == "Sell":
        # Gestión de TP1 Parcial
        if not PAPER_TP1_LISTO and vela_actual['low'] <= PAPER_TP1:
            PAPER_BALANCE += (PAPER_PRECIO_ENTRADA - PAPER_TP1) * (PAPER_SIZE_BTC * PORCENTAJE_CIERRE_PARCIAL)
            PAPER_TP1_LISTO = True
            PAPER_SL = PAPER_PRECIO_ENTRADA
            telegram_enviar_texto("🎯 TP1 ALCANZADO (SHORT). 50% cerrado. SL a Break Even.")
            
        if vela_actual['high'] >= PAPER_SL:
            pnl_realizado = (PAPER_PRECIO_ENTRADA - PAPER_SL) * (PAPER_SIZE_BTC * (0.5 if PAPER_TP1_LISTO else 1.0))
            cierre_requerido = True

    if cierre_requerido:
        PAPER_BALANCE += pnl_realizado
        PAPER_TRADES_TOTALES += 1
        if pnl_realizado > 0: PAPER_WIN += 1
        else: PAPER_LOSS += 1
        PAPER_LAST_10_PNL.append(pnl_realizado)
        
        telegram_enviar_texto(f"📤 POSICIÓN CERRADA\n💵 PnL: {pnl_realizado:.2f} USD\n💰 Balance Actual: {PAPER_BALANCE:.2f}")
        PAPER_POSICION_ACTIVA = None

# ======================================================
# 9. BUCLE MAESTRO DE EJECUCIÓN (HEARTBEAT)
# ======================================================
def ejecutar_bot_maestro():
    print("🔥 BOT V101 ACTIVADO - PROTOCOLO DE MÁXIMA EXTENSIÓN")
    telegram_enviar_texto("🚀 *BOT V101 INICIADO*\nModo de Análisis Secuencial de Mechas y Acción de Precio Total.")
    
    ultima_vela_procesada = None

    while True:
        try:
            # Captura de datos
            df_raw = fetch_mercado_data()
            if df_raw.empty: continue
            
            df = inyectar_analisis_institucional(df_raw)
            geo = detectar_geometria_espacial(df)
            
            # --- LOG DE AUDITORÍA (HEARTBEAT) ---
            pnl_global_10 = sum(PAPER_LAST_10_PNL[-10:]) if PAPER_LAST_10_PNL else 0.0
            print(f"\n💓 [{datetime.now().strftime('%H:%M:%S')}] HEARTBEAT")
            print(f"   Precio: {df['close'].iloc[-1]} | EMA20: {df['ema20'].iloc[-1]:.2f} | RSI: {df['rsi'].iloc[-1]:.1f}")
            print(f"   Ángulo: {geo['angulo']:.2f}° | PnL 10 trades: {pnl_global_10:.2f} | Contador: {PAPER_TRADES_TOTALES}")
            print(f"   Niveles: RES {geo['res_macro']:.2f} | SUP {geo['sup_macro']:.2f}")

            # Análisis de entrada (Solo al cierre de vela)
            if PAPER_POSICION_ACTIVA is None and ultima_vela_procesada != df.index[-2]:
                print("🔍 Consultando al experto de IA Groq...")
                resultado_ia = consultoria_ia_groq_v101(df, geo)
                
                decision = resultado_ia.get("decision", "Hold")
                if decision in ["Buy", "Sell"] and not resultado_ia.get("fuera_de_zona", False):
                    # Abrir posición
                    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_SIZE_BTC, PAPER_TP1_LISTO
                    precio_in = df['close'].iloc[-1]
                    atr_in = df['atr'].iloc[-1]
                    
                    dist_stop = atr_in * MULT_SL
                    PAPER_POSICION_ACTIVA = decision
                    PAPER_PRECIO_ENTRADA = precio_in
                    PAPER_TP1_LISTO = False
                    
                    if decision == "Buy":
                        PAPER_SL = precio_in - dist_stop
                        PAPER_TP1 = precio_in + (atr_in * MULT_TP1)
                    else:
                        PAPER_SL = precio_in + dist_stop
                        PAPER_TP1 = precio_in - (atr_in * MULT_TP1)
                    
                    # Cálculo de tamaño basado en riesgo
                    PAPER_SIZE_BTC = (PAPER_BALANCE * RISK_PER_TRADE) / dist_stop
                    
                    telegram_enviar_texto(f"🔔 *ORDEN {decision.upper()} EJECUTADA*\n📍 Patrón: {resultado_ia.get('patron_detectado')}")
                    figura = generar_visualizacion_v101(df, geo, resultado_ia)
                    telegram_enviar_grafico(figura)
                    plt.close(figura)
                    
                    ultima_vela_procesada = df.index[-2]
                
                elif resultado_ia.get("fuera_de_zona"):
                    print(f"⚠️ Patrón {resultado_ia.get('patron_detectado')} fuera de zona operativa.")

            # Gestión continua
            monitorear_posicion_activa(df)
            
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"🚨 ERROR CRÍTICO: {e}")
            time.sleep(40)

if __name__ == "__main__":
    ejecutar_bot_maestro()
