#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================
  FEED DE VIGILANCIA · VERSION NUBE (GitHub Actions + Telegram)
================================================================
Version AUTONOMA de vigilancia.py (no importa nada del proyecto
local): corre en un runner de GitHub y envia el informe a Telegram.

Necesita dos variables de entorno (GitHub Secrets):
  TELEGRAM_BOT_TOKEN   token del bot (de @BotFather)
  TELEGRAM_CHAT_ID     id del chat de Alejandro con el bot

Guardia horaria: los cron de GitHub van en UTC y Madrid cambia de
huso (CET/CEST). El workflow lanza a las 20:35 y 21:35 UTC y este
script solo continua si en Madrid son las 22:xx (asi corre UNA vez
al dia, siempre a las 22:35 locales). FORZAR=1 salta la guardia
(para ejecuciones manuales).

Uso personal · no constituye asesoramiento financiero.
================================================================
"""

import html
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- Parametros del protocolo (= proyecto local) ----------
ATR_MIN, ATR_MAX = 0.8, 3.6
PROXIMIDAD_MAX = 6.5
VIX_MAX = 22.0
PERIODO = "1y"
EARNINGS_MARGEN_DIAS = 7

UNIVERSO = {
    "AAPL": "Tecnologia", "MSFT": "Tecnologia", "GOOGL": "Tecnologia",
    "META": "Tecnologia", "AVGO": "Tecnologia", "ORCL": "Tecnologia",
    "ADBE": "Tecnologia", "CRM": "Tecnologia", "TXN": "Tecnologia",
    "QCOM": "Tecnologia", "AMD": "Tecnologia", "NVDA": "Tecnologia",
    "CSCO": "Tecnologia", "INTU": "Tecnologia", "IBM": "Tecnologia",
    "JNJ": "Salud", "LLY": "Salud", "ABBV": "Salud", "UNH": "Salud",
    "MRK": "Salud", "PFE": "Salud", "TMO": "Salud", "ABT": "Salud",
    "AMGN": "Salud", "ISRG": "Salud",
    "XOM": "Energia", "CVX": "Energia", "COP": "Energia",
    "SLB": "Energia", "EOG": "Energia",
    "PG": "Consumo", "KO": "Consumo", "COST": "Consumo", "AMZN": "Consumo",
    "PEP": "Consumo", "WMT": "Consumo", "MCD": "Consumo", "HD": "Consumo",
    "LOW": "Consumo", "SBUX": "Consumo", "TGT": "Consumo",
    "JPM": "Financiero", "V": "Financiero", "MA": "Financiero",
    "BAC": "Financiero", "GS": "Financiero", "AXP": "Financiero",
    "MS": "Financiero", "BLK": "Financiero", "SCHW": "Financiero",
    "CAT": "Industrial", "CSX": "Industrial", "DAL": "Industrial",
    "UNP": "Industrial", "HON": "Industrial", "DE": "Industrial",
    "LMT": "Industrial", "GE": "Industrial", "UPS": "Industrial",
    "LIN": "Materiales", "SHW": "Materiales", "APD": "Materiales",
    "FCX": "Materiales",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "TMUS": "Comunicaciones", "VZ": "Comunicaciones", "DIS": "Comunicaciones",
    "PLD": "Inmobiliario", "AMT": "Inmobiliario",
}


# ---------------- Indicadores (identicos a backtesting.py) -------------
def dias_desde_max(ventana):
    return (len(ventana) - 1) - int(np.argmax(ventana))


def preparar_indicadores(df):
    d = df.copy()
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA50_prev5"] = d["SMA50"].shift(5)
    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    d["ATR_pct"] = d["ATR"] / d["Close"] * 100
    d["prev_high"] = d["High"].shift(1)
    d["c1"] = (d["Close"] > d["SMA50"]) & (d["SMA50"] > d["SMA50_prev5"])
    d["c3"] = (d["Close"] > d["Open"]) & (d["Close"] > d["prev_high"])
    d["max_close_8"] = d["Close"].rolling(8).max()
    d["dias_desde_max"] = d["Close"].rolling(8).apply(dias_desde_max, raw=True)
    d["caida_desde_max"] = (d["max_close_8"] - d["Close"]) / d["max_close_8"] * 100
    d["umbral_caida"] = np.maximum(5.0, 1.8 * d["ATR_pct"])
    d["sobre_sma_4"] = (d["Close"] > d["SMA50"]).rolling(4).min()
    d["c2"] = ((d["dias_desde_max"] >= 2)
               & (d["caida_desde_max"] <= d["umbral_caida"])
               & (d["sobre_sma_4"] == 1))
    d["atr_ok"] = (d["ATR_pct"] >= ATR_MIN) & (d["ATR_pct"] <= ATR_MAX)
    d["dist_sma"] = (d["Close"] - d["SMA50"]) / d["SMA50"] * 100
    d["prox_ok"] = (d["dist_sma"] >= 0) & (d["dist_sma"] <= PROXIMIDAD_MAX)
    return d


def earnings_proximos(ticker):
    """Fecha de resultados si esta a <= EARNINGS_MARGEN_DIAS (orientativo)."""
    try:
        cal = yf.Ticker(ticker).calendar
        fechas = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not fechas:
            return None
        hoy = datetime.now(ZoneInfo("Europe/Madrid")).date()
        for f in fechas:
            f = f.date() if hasattr(f, "date") else f
            if hoy <= f <= hoy + timedelta(days=EARNINGS_MARGEN_DIAS):
                return f
    except Exception:
        pass
    return None


def enviar_telegram(token, chat_id, texto):
    """Envia texto monoespaciado, troceado bajo el limite de 4096."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    trozos, actual = [], []
    for lin in texto.split("\n"):
        if sum(len(x) + 1 for x in actual) + len(lin) > 3800:
            trozos.append("\n".join(actual)); actual = []
        actual.append(lin)
    if actual:
        trozos.append("\n".join(actual))
    for t in trozos:
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": f"<pre>{html.escape(t)}</pre>",
            "parse_mode": "HTML",
        }, timeout=30)
        r.raise_for_status()


def main():
    # --- Guardia horaria (una sola ejecucion diaria, 22:xx Madrid) ---
    ahora = datetime.now(ZoneInfo("Europe/Madrid"))
    if os.environ.get("FORZAR") != "1" and ahora.hour != 22:
        print(f"Guardia horaria: en Madrid son las {ahora:%H:%M}, no las 22:xx. "
              "Salgo sin hacer nada (la otra pasada del cron se encargara).")
        return

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("Faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (GitHub Secrets).")
        sys.exit(1)

    try:
        tickers = list(UNIVERSO)
        print(f"Descargando {len(tickers)} tickers + indices...")
        data = yf.download(tickers, period=PERIODO, interval="1d",
                           group_by="ticker", auto_adjust=False,
                           threads=True, progress=False)
        reg = yf.download(["^GSPC", "^VIX"], period=PERIODO, interval="1d",
                          group_by="ticker", auto_adjust=False,
                          threads=True, progress=False)

        gspc = reg["^GSPC"][["Close"]].dropna()
        vix = reg["^VIX"][["Close"]].dropna()
        sma50_spx = gspc["Close"].rolling(50).mean().iloc[-1]
        spx_dist = float((gspc["Close"].iloc[-1] - sma50_spx) / sma50_spx * 100)
        vix_hoy = float(vix["Close"].iloc[-1])
        c0_ok = (spx_dist > 0) and (vix_hoy < VIX_MAX)
        fecha_datos = gspc.index[-1].strftime("%Y-%m-%d")

        gatillo, en_zona = [], []
        for tk, sector in UNIVERSO.items():
            try:
                df = data[tk][["Open", "High", "Low", "Close"]].dropna()
            except Exception:
                continue
            if len(df) < 70:
                continue
            u = preparar_indicadores(df).iloc[-1]
            if pd.isna(u["SMA50"]) or pd.isna(u["ATR_pct"]):
                continue
            base = (bool(u["c1"]) and bool(u["c2"]) and bool(u["atr_ok"])
                    and bool(u["prox_ok"]))
            if not base:
                continue
            (gatillo if bool(u["c3"]) else en_zona).append((tk, sector, u))

        earnings = {tk: earnings_proximos(tk)
                    for tk, _, _ in gatillo + en_zona}
        gatillo.sort(key=lambda x: x[2]["dist_sma"])
        en_zona.sort(key=lambda x: x[2]["dist_sma"])

        # --- Informe ---
        L = []
        w = L.append
        w("VIGILANCIA SWING · datos del " + fecha_datos)
        if c0_ok:
            w(f"C0 OK: S&P {spx_dist:+.1f}% sobre SMA-50, VIX {vix_hoy:.1f}")
        else:
            w(f"C0 ROTO (S&P {spx_dist:+.1f}% / VIX {vix_hoy:.1f}): "
              "protocolo dice NO OPERAR. Lista solo informativa.")
        w("")
        w(f"GATILLO ({len(gatillo)}) - senal completa, confirmar en TradingView:")
        for tk, sector, u in gatillo or []:
            e = f"  !!earnings {earnings.get(tk)}" if earnings.get(tk) else ""
            w(f"  {tk:<5} {sector[:10]:<10} SMA50 {u['dist_sma']:+.1f}% "
              f"ATR {u['ATR_pct']:.1f}%{e}")
        if not gatillo:
            w("  (ninguno)")
        w("")
        w(f"EN ZONA ({len(en_zona)}) - vigilar, falta C3:")
        for tk, sector, u in en_zona or []:
            e = f"  !!earnings {earnings.get(tk)}" if earnings.get(tk) else ""
            w(f"  {tk:<5} {sector[:10]:<10} SMA50 {u['dist_sma']:+.1f}% "
              f"ATR {u['ATR_pct']:.1f}%{e}")
        if not en_zona:
            w("  (ninguno)")
        w("")
        w("Max 2-3 posiciones, sectores distintos. No operar tambien vale.")
        informe = "\n".join(L)
        print(informe)
        enviar_telegram(token, chat_id, informe)
        print("\nEnviado a Telegram.")

    except Exception as e:
        # Aviso de fallo por Telegram (sin inventar datos)
        try:
            enviar_telegram(token, chat_id,
                            f"VIGILANCIA: fallo al generar el feed: {type(e).__name__}: {e}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
