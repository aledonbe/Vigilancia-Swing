#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================
  FEED DE VIGILANCIA · VERSION NUBE v2
  GitHub Actions + Telegram + web-app (GitHub Pages)
================================================================
Cada noche:
  1. Corre el screening del protocolo (C0-C4 + ATR + proximidad).
  2. Envia el resumen a Telegram.
  3. Guarda el feed del dia como JSON en docs/feed/ (historial
     permanente que lee la web-app).
  4. AUDITORIA: simula mecanicamente cada senal GATILLO pasada
     (entrada en la apertura siguiente, stop -2%, objetivo +4%,
     limite 3 sesiones, stop-primero en dias ambiguos) y publica
     la esperanza REAL del feed en docs/rendimiento.json.

Variables de entorno:
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  (GitHub Secrets)
  FORZAR=1        salta la guardia horaria (ejecucion manual)
  SIN_TELEGRAM=1  no envia mensaje (pruebas locales)
  BACKFILL_DIAS=N reconstruye ademas los feeds de las ultimas N
                  sesiones (sin look-ahead: cada feed se calcula
                  solo con datos hasta su fecha). Una sola vez.

Uso personal · no constituye asesoramiento financiero.
================================================================
"""

import html
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ======================================================================
#  BLOQUE EDITABLE: parametros del protocolo y universo.
#  Para "mejorar el screening" se toca SOLO esto (tras validarlo
#  antes en el laboratorio local con backtesting.py).
# ======================================================================
ATR_MIN, ATR_MAX = 0.8, 3.6      # % ATR admitido
PROXIMIDAD_MAX = 6.5             # % max sobre la SMA-50
VIX_MAX = 22.0                   # C0
EARNINGS_MARGEN_DIAS = 7         # C4 (aviso)

# Parametros de la AUDITORIA (= protocolo de referencia)
AUD_OBJETIVO_PCT = 0.04          # +4%
AUD_STOP_PCT = 0.02              # -2%
AUD_LIMITE_DIAS = 3              # cerrar al 3er cierre

POSICION_EUR = 300.0             # para el plan de trade en la app

UNIVERSO = {  # ticker: (nombre, sector)
    "AAPL": ("Apple", "Tecnologia"), "MSFT": ("Microsoft", "Tecnologia"),
    "GOOGL": ("Alphabet (Google)", "Tecnologia"), "META": ("Meta", "Tecnologia"),
    "AVGO": ("Broadcom", "Tecnologia"), "ORCL": ("Oracle", "Tecnologia"),
    "ADBE": ("Adobe", "Tecnologia"), "CRM": ("Salesforce", "Tecnologia"),
    "TXN": ("Texas Instruments", "Tecnologia"), "QCOM": ("Qualcomm", "Tecnologia"),
    "AMD": ("AMD", "Tecnologia"), "NVDA": ("Nvidia", "Tecnologia"),
    "CSCO": ("Cisco", "Tecnologia"), "INTU": ("Intuit", "Tecnologia"),
    "IBM": ("IBM", "Tecnologia"),
    "JNJ": ("Johnson & Johnson", "Salud"), "LLY": ("Eli Lilly", "Salud"),
    "ABBV": ("AbbVie", "Salud"), "UNH": ("UnitedHealth", "Salud"),
    "MRK": ("Merck", "Salud"), "PFE": ("Pfizer", "Salud"),
    "TMO": ("Thermo Fisher", "Salud"), "ABT": ("Abbott", "Salud"),
    "AMGN": ("Amgen", "Salud"), "ISRG": ("Intuitive Surgical", "Salud"),
    "XOM": ("Exxon Mobil", "Energia"), "CVX": ("Chevron", "Energia"),
    "COP": ("ConocoPhillips", "Energia"), "SLB": ("SLB (Schlumberger)", "Energia"),
    "EOG": ("EOG Resources", "Energia"),
    "PG": ("Procter & Gamble", "Consumo"), "KO": ("Coca-Cola", "Consumo"),
    "COST": ("Costco", "Consumo"), "AMZN": ("Amazon", "Consumo"),
    "PEP": ("PepsiCo", "Consumo"), "WMT": ("Walmart", "Consumo"),
    "MCD": ("McDonald's", "Consumo"), "HD": ("Home Depot", "Consumo"),
    "LOW": ("Lowe's", "Consumo"), "SBUX": ("Starbucks", "Consumo"),
    "TGT": ("Target", "Consumo"),
    "JPM": ("JPMorgan Chase", "Financiero"), "V": ("Visa", "Financiero"),
    "MA": ("Mastercard", "Financiero"), "BAC": ("Bank of America", "Financiero"),
    "GS": ("Goldman Sachs", "Financiero"), "AXP": ("American Express", "Financiero"),
    "MS": ("Morgan Stanley", "Financiero"), "BLK": ("BlackRock", "Financiero"),
    "SCHW": ("Charles Schwab", "Financiero"),
    "CAT": ("Caterpillar", "Industrial"), "CSX": ("CSX", "Industrial"),
    "DAL": ("Delta Air Lines", "Industrial"), "UNP": ("Union Pacific", "Industrial"),
    "HON": ("Honeywell", "Industrial"), "DE": ("John Deere", "Industrial"),
    "LMT": ("Lockheed Martin", "Industrial"), "GE": ("GE Aerospace", "Industrial"),
    "UPS": ("UPS", "Industrial"),
    "LIN": ("Linde", "Materiales"), "SHW": ("Sherwin-Williams", "Materiales"),
    "APD": ("Air Products", "Materiales"), "FCX": ("Freeport-McMoRan", "Materiales"),
    "NEE": ("NextEra Energy", "Utilities"), "DUK": ("Duke Energy", "Utilities"),
    "SO": ("Southern Company", "Utilities"),
    "TMUS": ("T-Mobile US", "Comunicaciones"), "VZ": ("Verizon", "Comunicaciones"),
    "DIS": ("Disney", "Comunicaciones"),
    "PLD": ("Prologis", "Inmobiliario"), "AMT": ("American Tower", "Inmobiliario"),
}
# ======================================================================

PERIODO = "1y"
DOCS = Path(__file__).resolve().parent / "docs"
FEED_DIR = DOCS / "feed"


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
                return f.isoformat()
    except Exception:
        pass
    return None


# ---------------- Screening de un dia (sin look-ahead) -----------------
def candidato_json(tk, u, df_hasta, earnings=None):
    nombre, sector = UNIVERSO[tk]
    cierre = float(u["Close"])
    serie = [round(float(x), 2) for x in df_hasta["Close"].tail(30)]
    return {
        "ticker": tk, "nombre": nombre, "sector": sector,
        "cierre": round(cierre, 2),
        "dist_sma": round(float(u["dist_sma"]), 1),
        "atr_pct": round(float(u["ATR_pct"]), 1),
        "caida": round(float(u["caida_desde_max"]), 1),
        "dias_max": int(u["dias_desde_max"]),
        "earnings": earnings,
        "stop": round(cierre * (1 - AUD_STOP_PCT), 2),
        "objetivo": round(cierre * (1 + AUD_OBJETIVO_PCT), 2),
        "acciones_300eur": round(POSICION_EUR / cierre, 2),
        "serie": serie,
    }


def feed_de_fecha(data, gspc, vix, fecha, con_earnings):
    """Feed calculado SOLO con datos hasta 'fecha' (incluida)."""
    g = gspc[gspc.index <= fecha]
    v = vix[vix.index <= fecha]
    if len(g) < 60 or g.index[-1] != fecha:
        return None
    sma50 = g["Close"].rolling(50).mean().iloc[-1]
    spx_dist = float((g["Close"].iloc[-1] - sma50) / sma50 * 100)
    vix_hoy = float(v["Close"].iloc[-1])
    c0_ok = (spx_dist > 0) and (vix_hoy < VIX_MAX)

    gatillo, en_zona = [], []
    for tk in UNIVERSO:
        try:
            df = data[tk][["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            continue
        df = df[df.index <= fecha]
        if len(df) < 70 or df.index[-1] != fecha:
            continue
        u = preparar_indicadores(df).iloc[-1]
        if pd.isna(u["SMA50"]) or pd.isna(u["ATR_pct"]):
            continue
        base = (bool(u["c1"]) and bool(u["c2"]) and bool(u["atr_ok"])
                and bool(u["prox_ok"]))
        if not base:
            continue
        e = earnings_proximos(tk) if con_earnings else None
        (gatillo if bool(u["c3"]) else en_zona).append(candidato_json(tk, u, df, e))

    gatillo.sort(key=lambda x: x["dist_sma"])
    en_zona.sort(key=lambda x: x["dist_sma"])
    return {
        "fecha_datos": fecha.strftime("%Y-%m-%d"),
        "generado": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(timespec="minutes"),
        "c0": {"ok": c0_ok, "vix": round(vix_hoy, 1), "spx_dist": round(spx_dist, 1)},
        "gatillo": gatillo,
        "en_zona": en_zona,
    }


# ---------------- Auditoria de senales GATILLO --------------------------
def auditar(data):
    """Simula el protocolo sobre cada GATILLO del historial de feeds."""
    senales = []
    for fjson in sorted(FEED_DIR.glob("*.json")):
        if fjson.name == "index.json":
            continue
        feed = json.loads(fjson.read_text(encoding="utf-8"))
        for c in feed.get("gatillo", []):
            senales.append((feed["fecha_datos"], c["ticker"], c["nombre"]))

    resultado = []
    for fecha_senal, tk, nombre in senales:
        try:
            df = data[tk][["Open", "High", "Low", "Close"]].dropna()
            pos = df.index.get_loc(pd.Timestamp(fecha_senal))
        except Exception:
            continue
        reg = {"fecha_senal": fecha_senal, "ticker": tk, "nombre": nombre}
        if pos + 1 >= len(df):
            reg.update(estado="pendiente")   # aun no abrio el mercado
            resultado.append(reg)
            continue
        entrada = float(df["Open"].iloc[pos + 1])
        stop = entrada * (1 - AUD_STOP_PCT)
        objetivo = entrada * (1 + AUD_OBJETIVO_PCT)
        salida, motivo = None, None
        ult_cierre = entrada
        dias = 0
        for k in range(AUD_LIMITE_DIAS):
            j = pos + 1 + k
            if j >= len(df):
                break
            dias = k + 1
            hi, lo, cl = (float(df["High"].iloc[j]), float(df["Low"].iloc[j]),
                          float(df["Close"].iloc[j]))
            ult_cierre = cl
            if lo <= stop:                       # stop-primero (conservador)
                salida, motivo = stop, "stop"; break
            if hi >= objetivo:
                salida, motivo = objetivo, "objetivo"; break
        if salida is None and dias == AUD_LIMITE_DIAS:
            salida, motivo = ult_cierre, "tiempo"
        reg["entrada"] = round(entrada, 2)
        if salida is None:                        # operacion aun en curso
            ret = (ult_cierre - entrada) / entrada * 100
            reg.update(estado="abierta", dias=dias,
                       ret_pct=round(ret, 2),
                       r=round(ret / (AUD_STOP_PCT * 100), 2))
        else:
            ret = (salida - entrada) / entrada * 100
            reg.update(estado="cerrada", dias=dias, motivo=motivo,
                       salida=round(salida, 2), ret_pct=round(ret, 2),
                       r=round(ret / (AUD_STOP_PCT * 100), 2))
        resultado.append(reg)

    cerradas = [x for x in resultado if x.get("estado") == "cerrada"]
    resumen = None
    if cerradas:
        rs = [x["r"] for x in cerradas]
        gan = [x for x in cerradas if x["ret_pct"] > 0]
        resumen = {
            "n": len(cerradas),
            "ganadoras": len(gan),
            "win_rate": round(100 * len(gan) / len(cerradas), 1),
            "esperanza_R": round(sum(rs) / len(rs), 2),
            "suma_R": round(sum(rs), 2),
        }
    return {
        "actualizado": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(timespec="minutes"),
        "params": {"objetivo_pct": AUD_OBJETIVO_PCT * 100,
                   "stop_pct": AUD_STOP_PCT * 100,
                   "limite_dias": AUD_LIMITE_DIAS,
                   "nota": "Simulacion mecanica conservadora (stop-primero). "
                           "Mide el feed, no las operaciones reales de Alejandro."},
        "resumen": resumen,
        "senales": sorted(resultado, key=lambda x: x["fecha_senal"], reverse=True),
    }


# ---------------- Telegram ---------------------------------------------
def enviar_telegram(token, chat_id, texto):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    trozos, actual = [], []
    for lin in texto.split("\n"):
        if sum(len(x) + 1 for x in actual) + len(lin) > 3800:
            trozos.append("\n".join(actual)); actual = []
        actual.append(lin)
    if actual:
        trozos.append("\n".join(actual))
    for t in trozos:
        r = requests.post(url, json={"chat_id": chat_id,
                                     "text": f"<pre>{html.escape(t)}</pre>",
                                     "parse_mode": "HTML"}, timeout=30)
        r.raise_for_status()


def texto_telegram(feed, rend):
    L = []
    w = L.append
    w("VIGILANCIA SWING · datos del " + feed["fecha_datos"])
    c0 = feed["c0"]
    if c0["ok"]:
        w(f"C0 OK: S&P {c0['spx_dist']:+.1f}% s/SMA-50, VIX {c0['vix']}")
    else:
        w(f"C0 ROTO (S&P {c0['spx_dist']:+.1f}% / VIX {c0['vix']}): NO OPERAR. "
          "Lista solo informativa.")
    w("")
    w(f"GATILLO ({len(feed['gatillo'])}) - confirmar en TradingView:")
    for c in feed["gatillo"] or []:
        e = f"  !!earnings {c['earnings']}" if c["earnings"] else ""
        w(f"  {c['ticker']:<5} {c['nombre'][:18]:<18} SMA {c['dist_sma']:+.1f}%{e}")
    if not feed["gatillo"]:
        w("  (ninguno)")
    w("")
    w(f"EN ZONA ({len(feed['en_zona'])}) - falta C3:")
    for c in feed["en_zona"] or []:
        e = "  !!earn" if c["earnings"] else ""
        w(f"  {c['ticker']:<5} {c['nombre'][:18]:<18} SMA {c['dist_sma']:+.1f}%{e}")
    if not feed["en_zona"]:
        w("  (ninguno)")
    if rend and rend.get("resumen"):
        s = rend["resumen"]
        w("")
        w(f"Auditoria del feed: {s['n']} senales cerradas, "
          f"win {s['win_rate']}%, E[R] {s['esperanza_R']:+.2f}")
    w("")
    w("App con historial y rendimiento: ver GitHub Pages.")
    w("Max 2-3 posiciones, sectores distintos. No operar tambien vale.")
    return "\n".join(L)


# ---------------- Principal --------------------------------------------
def main():
    ahora = datetime.now(ZoneInfo("Europe/Madrid"))
    if os.environ.get("FORZAR") != "1" and ahora.hour != 22:
        print(f"Guardia horaria: en Madrid son las {ahora:%H:%M}, no las 22:xx. Salgo.")
        return

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    sin_tg = os.environ.get("SIN_TELEGRAM") == "1"
    if not sin_tg and (not token or not chat_id):
        print("Faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.")
        sys.exit(1)

    FEED_DIR.mkdir(parents=True, exist_ok=True)

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

        # --- backfill opcional (primera puesta en marcha) ---
        n_back = int(os.environ.get("BACKFILL_DIAS", "0") or 0)
        if n_back > 0:
            for fecha in gspc.index[-n_back:]:
                destino = FEED_DIR / f"{fecha:%Y-%m-%d}.json"
                if destino.exists():
                    continue
                f = feed_de_fecha(data, gspc, vix, fecha, con_earnings=False)
                if f:
                    f["backfill"] = True
                    destino.write_text(json.dumps(f, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
                    print(f"  backfill {fecha:%Y-%m-%d}: "
                          f"{len(f['gatillo'])}G/{len(f['en_zona'])}Z")

        # --- feed de hoy (ultima sesion disponible) ---
        fecha_ult = gspc.index[-1]
        feed = feed_de_fecha(data, gspc, vix, fecha_ult, con_earnings=True)
        if feed is None:
            raise RuntimeError("No pude calcular el feed de la ultima sesion.")
        (FEED_DIR / f"{feed['fecha_datos']}.json").write_text(
            json.dumps(feed, ensure_ascii=False, indent=1), encoding="utf-8")

        # --- indices para la app ---
        fechas = sorted(p.stem for p in FEED_DIR.glob("*.json")
                        if p.name != "index.json")
        (DOCS / "latest.json").write_text(
            json.dumps(feed, ensure_ascii=False, indent=1), encoding="utf-8")
        (FEED_DIR / "index.json").write_text(
            json.dumps(fechas, ensure_ascii=False), encoding="utf-8")

        # --- auditoria ---
        rend = auditar(data)
        (DOCS / "rendimiento.json").write_text(
            json.dumps(rend, ensure_ascii=False, indent=1), encoding="utf-8")

        # --- telegram ---
        texto = texto_telegram(feed, rend)
        print(texto)
        if not sin_tg:
            enviar_telegram(token, chat_id, texto)
            print("\nEnviado a Telegram.")

    except Exception as e:
        if not sin_tg:
            try:
                enviar_telegram(token, chat_id,
                                f"VIGILANCIA: fallo al generar el feed: "
                                f"{type(e).__name__}: {e}")
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
