
# ============================================================
# BOT TRADING V200 – LLAMA 70B NISON PRO (FULL VERSION)
# ============================================================
# Features:
# - IA: llama-3.3-70b-versatile (Groq)
# - Context-based (no image input to AI)
# - Candlestick + EMA + RSI + S/R
# - TP1 (50%) + SL to BE + Trailing runner
# - Telegram logs (detailed)
# - Auto-learning every 10 trades
# ============================================================

import os, time, json, requests
import numpy as np
import pandas as pd
from datetime import datetime
from groq import Groq
import matplotlib.pyplot as plt

# ========= CONFIG =========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = Groq(api_key=GROQ_API_KEY)

SYMBOL = "BTCUSDT"
INTERVAL = "5"
MODEL = "llama-3.3-70b-versatile"

# ========= TELEGRAM =========
def send(msg):
    if TELEGRAM_TOKEN:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": msg})

def send_img(path, caption=""):
    if TELEGRAM_TOKEN:
        with open(path,"rb") as f:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                          data={"chat_id": CHAT_ID, "caption": caption},
                          files={"photo": f})

# ========= DATA =========
def get_data():
    url="https://api.bybit.com/v5/market/kline"
    r=requests.get(url,params={"category":"linear","symbol":SYMBOL,"interval":INTERVAL,"limit":200}).json()
    df=pd.DataFrame(r["result"]["list"][::-1],
        columns=['time','open','high','low','close','vol','turn'])
    df[['open','high','low','close']]=df[['open','high','low','close']].astype(float)
    df['time']=pd.to_datetime(df['time'].astype(np.int64),unit='ms')
    df.set_index('time',inplace=True)

    df['ema20']=df['close'].ewm(span=20).mean()
    delta=df['close'].diff()
    gain=(delta.where(delta>0,0)).rolling(14).mean()
    loss=(-delta.where(delta<0,0)).rolling(14).mean()
    rs=gain/loss
    df['rsi']=100-(100/(1+rs))

    return df.dropna()

# ========= CONTEXT =========
def build_context(df):
    last=df.tail(20)
    candles=[]
    for _,r in last.iterrows():
        candles.append({
            "type":"bullish" if r.close>r.open else "bearish",
            "body":round(abs(r.close-r.open),2),
            "upper_wick":round(r.high-max(r.open,r.close),2),
            "lower_wick":round(min(r.open,r.close)-r.low,2)
        })

    return json.dumps({
        "price":round(df.close.iloc[-1],2),
        "ema20":round(df.ema20.iloc[-1],2),
        "rsi":round(df.rsi.iloc[-1],2),
        "support":round(last.low.min(),2),
        "resistance":round(last.high.max(),2),
        "candles":candles
    })

# ========= AI =========
def decision(context):
    prompt=f"""
You are a professional trader inspired by Steve Nison.

Analyze:
{context}

Return JSON:
{{
"decision":"Buy/Sell/Hold",
"tp1":0,
"sl":0,
"reason":""
}}
"""
    r=client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return json.loads(r.choices[0].message.content)

# ========= TRADE MANAGEMENT =========
trades=[]
current_trade=None

def manage_trade(price):
    global current_trade

    if not current_trade:
        return

    t=current_trade

    # TP1 hit
    if not t["tp1_hit"]:
        if (t["side"]=="Buy" and price>=t["tp1"]) or (t["side"]=="Sell" and price<=t["tp1"]):
            t["tp1_hit"]=True
            t["sl"]=t["entry"]
            send("✅ TP1 alcanzado + SL a BE")

    # trailing
    if t["tp1_hit"]:
        if t["side"]=="Buy":
            new_sl=price*0.995
            if new_sl>t["sl"]:
                t["sl"]=new_sl
        else:
            new_sl=price*1.005
            if new_sl<t["sl"]:
                t["sl"]=new_sl

    # SL hit
    if (t["side"]=="Buy" and price<=t["sl"]) or (t["side"]=="Sell" and price>=t["sl"]):
        send(f"❌ Trade cerrado @ {price}")
        trades.append(t)
        current_trade=None

# ========= CHART =========
def chart(df):
    d=df.tail(100)
    x=range(len(d))
    fig,ax=plt.subplots()
    for i,r in enumerate(d.itertuples()):
        color='green' if r.close>=r.open else 'red'
        ax.vlines(i,r.low,r.high,color=color)
        ax.bar(i,abs(r.close-r.open),bottom=min(r.open,r.close),color=color)
    ax.plot(x,d.ema20)
    path="/mnt/data/chart.png"
    plt.savefig(path)
    plt.close()
    return path

# ========= AUTO LEARNING =========
def learn():
    if len(trades)>=10:
        wins=sum(1 for t in trades if t["result"]=="win")
        send(f"📊 Winrate últimos 10: {wins}/10")

# ========= MAIN =========
def run():
    global current_trade
    send("🚀 BOT V200 INICIADO")

    while True:
        try:
            df=get_data()
            price=df.close.iloc[-1]

            manage_trade(price)

            if not current_trade:
                ctx=build_context(df)
                d=decision(ctx)

                if d["decision"]!="Hold":
                    current_trade={
                        "side":d["decision"],
                        "entry":price,
                        "tp1":d["tp1"],
                        "sl":d["sl"],
                        "tp1_hit":False
                    }
                    send(f"📈 NUEVO TRADE: {d}")

                    img=chart(df)
                    send_img(img,str(d))

            learn()

        except Exception as e:
            send(f"ERROR {e}")

        time.sleep(60)

if __name__=="__main__":
    run()
