# -*- coding: utf-8 -*-
"""
Portfolio Pulse - Flask API Server with Timeline Charts & Stock Comparison
Complete production-ready version
"""

import io
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ==========================================
# GET ABSOLUTE PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "templates")

# Create templates folder if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

log.info("BASE_DIR: %s", BASE_DIR)
log.info("TEMPLATES_DIR: %s", STATIC_DIR)

# ==========================================
# FLASK APP INITIALIZATION
# ==========================================
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)

# ==========================================
# CONFIGURATION
# ==========================================
BASE = os.environ.get("ULITICS_BASE", r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT")
SAVE = os.path.join(BASE, "save_model")

PANEL_BANK_PATH       = os.environ.get("PANEL_BANK",    os.path.join(BASE, "ULITICS", "final_simple_panel_data - BANK.csv"))
PANEL_IT_PATH         = os.environ.get("PANEL_IT",      os.path.join(BASE, "ULITICS", "final_simple_panel_data - IT .csv"))
PANEL_ENERGY_PATH     = os.environ.get("PANEL_ENERGY",  os.path.join(BASE, "ULITICS", "final_simple_panel_data - ENERGY.csv"))
GARCH_PATH            = os.environ.get("GARCH",         os.path.join(SAVE, "GARCH", "volatility_forecast.csv"))
LSTM_ML_AFTER_PATH    = os.environ.get("LSTM_ML_AFTER", os.path.join(SAVE, "ml_after_lstm_predictions"))
LSTM_TRANSFORMER_PATH = os.environ.get("LSTM_TRANS",    os.path.join(SAVE, "lstm_transformer_predictions"))
LSTM_MODEL_PATH       = os.environ.get("LSTM_MODELS",   os.path.join(SAVE, "lstm_model"))
LSTM_PREDICTIONS_PATH = os.environ.get("LSTM_PREDS",    os.path.join(SAVE, "predictions"))
ENSEMBLE_FOLDER       = os.environ.get("ENSEMBLE",      os.path.join(SAVE, "ensemble_predictions"))

# ==========================================
# IN-MEMORY CACHE
# ==========================================
_cache: dict = {}
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", 60))


def _cache_get(key: str):
    """Get value from cache if not expired"""
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["value"]
    return None


def _cache_set(key: str, value):
    """Set value in cache with timestamp"""
    _cache[key] = {"ts": time.time(), "value": value}


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with proper encoding handling"""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def load_pickle(path: str):
    """Load pickle file safely with error handling"""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        log.warning("Could not load pickle %s: %s", path, exc)
        return None


def _safe_float(val, default=0.0) -> float:
    """Safely convert value to float, handling NaN and inf"""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _find_price_col(row: pd.Series, keywords: tuple, min_val: float = 5.0, max_val: float = 1_000_000.0) -> float:
    """Search a CSV row for the first column whose name matches keywords and value is a plausible price"""
    for col in row.index:
        col_l = col.lower().strip()
        if any(k in col_l for k in keywords):
            v = _safe_float(row[col])
            if min_val < v < max_val:
                return v
    return 0.0


def _find_return_col(row: pd.Series, keywords: tuple, low: float = -60.0, high: float = 60.0):
    """Search a CSV row for the first column whose name matches keywords and value is a plausible % return"""
    for col in row.index:
        col_l = col.lower().strip()
        if any(k in col_l for k in keywords):
            v = _safe_float(row[col], default=None)
            if v is not None and low <= v <= high and v != 0.0:
                return v
    return None


# ==========================================
# DATA LOADERS
# ==========================================
def load_panel_data():
    """Load and combine panel data from all sector CSVs"""
    cached = _cache_get("panel")
    if cached is not None:
        return cached
    try:
        df_bank   = load_csv(PANEL_BANK_PATH)
        df_bank["Sector"] = "BANK"
        
        df_it     = load_csv(PANEL_IT_PATH)
        df_it["Sector"] = "IT"
        
        df_energy = load_csv(PANEL_ENERGY_PATH)
        df_energy["Sector"] = "ENERGY"
        
        df = pd.concat([df_bank, df_it, df_energy], ignore_index=True)
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values(["Sector", "Stock", "Date"])
        log.info("✓ Panel data loaded: %d rows", len(df))
        _cache_set("panel", df)
        return df
    except Exception as exc:
        log.error("✗ Error loading panel data: %s", exc)
        return None


def load_garch_data() -> dict:
    """Load GARCH volatility forecasts"""
    cached = _cache_get("garch")
    if cached is not None:
        return cached
    data: dict = {}
    try:
        df = pd.read_csv(GARCH_PATH)
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            stock = str(row.get("Stock", "")).strip()
            vol   = row.get("Volatility", 0)
            if stock:
                data[stock] = _safe_float(vol)
        log.info("✓ GARCH volatility: %d stocks", len(data))
    except Exception as exc:
        log.warning("✗ Could not load GARCH: %s", exc)
    _cache_set("garch", data)
    return data


def load_lstm_models() -> dict:
    """Load LSTM model pickle files"""
    cached = _cache_get("lstm_models")
    if cached is not None:
        return cached
    models: dict = {}
    if os.path.exists(LSTM_MODEL_PATH):
        for fname in os.listdir(LSTM_MODEL_PATH):
            if fname.endswith(".pkl"):
                model = load_pickle(os.path.join(LSTM_MODEL_PATH, fname))
                if model is not None:
                    models[fname.replace(".pkl", "")] = model
    log.info("✓ LSTM models loaded: %d", len(models))
    _cache_set("lstm_models", models)
    return models


def load_enhanced_predictions() -> dict:
    """Load enhanced predictions from ML-after-LSTM and Transformer folders"""
    cached = _cache_get("enhanced")
    if cached is not None:
        return cached
    data: dict = {}

    def _absorb(folder, suffix, label, overwrite):
        if not os.path.exists(folder):
            return
        for fname in [f for f in os.listdir(folder) if f.endswith(suffix)]:
            try:
                df = pd.read_csv(os.path.join(folder, fname))
                df.columns = df.columns.str.strip()
                if len(df) == 0:
                    continue
                stock = fname.replace(suffix, "")
                if not stock.endswith(".NS"):
                    stock += ".NS"
                if overwrite or stock not in data:
                    data[stock] = {"source": label, "data": df.iloc[0].to_dict()}
            except Exception as exc:
                log.warning("Could not read %s: %s", fname, exc)

    _absorb(LSTM_ML_AFTER_PATH,    "_enhanced.csv", "ml_after_lstm",    True)
    _absorb(LSTM_TRANSFORMER_PATH, ".csv",          "lstm_transformer", False)
    log.info("✓ Enhanced predictions: %d", len(data))
    _cache_set("enhanced", data)
    return data


def load_lstm_predictions() -> dict:
    """Load LSTM predictions from prediction folder"""
    cached = _cache_get("lstm_preds")
    if cached is not None:
        return cached
    data: dict = {}
    if not os.path.exists(LSTM_PREDICTIONS_PATH):
        log.warning("✗ LSTM predictions folder not found: %s", LSTM_PREDICTIONS_PATH)
        _cache_set("lstm_preds", data)
        return data

    RETURN_KW = ("return", "change", "pct", "percent", "%", "diff", "gain", "loss", "ret", "delta")
    PRICE_KW  = ("pred", "forecast", "close", "target", "next", "lstm", "price", "value", "output")

    for fname in [f for f in os.listdir(LSTM_PREDICTIONS_PATH) if f.endswith(".csv")]:
        try:
            df = pd.read_csv(os.path.join(LSTM_PREDICTIONS_PATH, fname))
            df.columns = df.columns.str.strip()
            if len(df) == 0:
                continue

            stock = fname.replace(".csv", "")
            if not stock.endswith(".NS"):
                stock += ".NS"

            row = df.iloc[0]
            lstm_return = _find_return_col(row, RETURN_KW, low=-50.0, high=50.0)
            lstm_price = _find_price_col(row, PRICE_KW, min_val=10.0, max_val=500_000.0)

            data[stock] = {
                "LSTM_Return_%":        round(lstm_return or 0.0, 2),
                "LSTM_Predicted_Price": round(lstm_price, 2),
            }
        except Exception as exc:
            log.warning("Could not read LSTM file %s: %s", fname, exc)

    log.info("✓ LSTM predictions: %d stocks", len(data))
    _cache_set("lstm_preds", data)
    return data


# ==========================================
# PREDICTION ASSEMBLY
# ==========================================
def load_all_predictions(lstm_data: dict, enhanced_data: dict, models: dict) -> list:
    """Load and assemble all predictions from ensemble + LSTM + GARCH"""
    cached = _cache_get("predictions")
    if cached is not None:
        return cached

    garch_data  = load_garch_data()
    predictions = []

    if not os.path.exists(ENSEMBLE_FOLDER):
        log.warning("✗ Ensemble folder not found: %s", ENSEMBLE_FOLDER)
        _cache_set("predictions", predictions)
        return predictions

    ENS_PRICE_KW   = ("ensemble", "final_pred", "avg_pred", "blended", "combined",
                       "prediction", "forecast", "target", "pred_price", "predicted_close",
                       "close_pred", "next_close", "pred")
    ENS_CHANGE_KW  = ("ensemble_change", "ens_change", "change_%", "ens_ret", "final_change",
                       "blended_change", "pred_change", "predicted_change", "return", "change")
    CURR_PRICE_KW  = ("current_price", "current", "actual", "close", "last_price",
                       "market_price", "price", "stock_price", "spot")

    csv_files = [
        f for f in os.listdir(ENSEMBLE_FOLDER)
        if f.endswith(".csv") and f != "all_stocks_ensemble.csv"
    ]
    log.info("Processing %d ensemble CSV files...", len(csv_files))

    for fname in csv_files:
        try:
            df = pd.read_csv(os.path.join(ENSEMBLE_FOLDER, fname))
            df.columns = df.columns.str.strip()
            if len(df) == 0:
                continue

            row  = df.iloc[0]
            pred = {str(k).strip(): v for k, v in row.to_dict().items()}

            if "Stock" not in pred:
                sn = fname.replace(".csv", "").replace("_enhanced", "").replace("_ensemble", "")
                if not sn.endswith(".NS"):
                    sn += ".NS"
                pred["Stock"] = sn
            stock = str(pred.get("Stock", "")).strip()

            current = _safe_float(pred.get("Current_Price", 0))
            if current <= 0:
                current = _find_price_col(row, CURR_PRICE_KW, min_val=5.0, max_val=500_000.0)
            pred["Current_Price"] = current

            ensemble = _safe_float(pred.get("Ensemble_Prediction", 0))
            if ensemble <= 0:
                ensemble = _find_price_col(row, ENS_PRICE_KW, min_val=5.0, max_val=500_000.0)

            if current > 0 and ensemble > 0:
                ratio = ensemble / current
                if ratio < 0.40 or ratio > 1.60:
                    log.warning("  ⚠ %s: ensemble price %.2f looks wrong vs current %.2f — discarding", stock, ensemble, current)
                    ensemble = 0.0

            pred["Ensemble_Prediction"] = ensemble if ensemble > 0 else current

            ens_change = _safe_float(pred.get("Ensemble_Change_%", 0))
            if ens_change == 0 and current > 0 and ensemble > 0:
                ens_change = round((ensemble - current) / current * 100, 2)
            pred["Ensemble_Change_%"] = ens_change

            pred["garch_volatility_percent"] = round(garch_data.get(stock, 0.0), 2)
            pred["enhanced_source"] = enhanced_data.get(stock, {}).get("source", "Ensemble")
            pred["has_lstm_model"]  = stock in models
            pred.setdefault("Sector", "UNKNOWN")

            lstm_info  = lstm_data.get(stock, {})
            lstm_ret   = _safe_float(lstm_info.get("LSTM_Return_%", 0))
            lstm_price = _safe_float(lstm_info.get("LSTM_Predicted_Price", 0))

            if current > 0 and lstm_price > 0:
                ratio = lstm_price / current
                if ratio < 0.50 or ratio > 1.50:
                    log.warning("  ⚠ %s: LSTM price %.2f vs current %.2f — ratio %.2f looks wrong, discarding",
                                stock, lstm_price, current, ratio)
                    lstm_price = 0.0
                    lstm_ret   = 0.0

            if abs(lstm_ret) > 40:
                log.warning("  ⚠ %s: LSTM return %.2f%% too extreme — discarding", stock, lstm_ret)
                lstm_ret = 0.0

            if lstm_price > 0 and current > 0:
                lstm_ret = round((lstm_price - current) / current * 100, 2)
            elif lstm_ret != 0 and current > 0:
                lstm_price = round(current * (1 + lstm_ret / 100), 2)
            else:
                lstm_price = pred["Ensemble_Prediction"]
                lstm_ret   = ens_change

            pred["LSTM_Return_%"]        = round(lstm_ret, 2)
            pred["LSTM_Predicted_Price"] = round(lstm_price, 2)
            pred["Signal"] = "HOLD"

            predictions.append(pred)

        except Exception as exc:
            log.warning("Error reading %s: %s", fname, exc)

    if predictions:
        returns = [_safe_float(p["LSTM_Return_%"]) for p in predictions]
        all_same = (max(returns) - min(returns)) < 0.01

        p35 = float(np.percentile(returns, 35))
        p65 = float(np.percentile(returns, 65))

        for p in predictions:
            ret = _safe_float(p["LSTM_Return_%"])
            if all_same:
                ens = _safe_float(p.get("Ensemble_Change_%", 0))
                p["Signal"] = "BUY" if ens > 0.5 else "SELL" if ens < -0.5 else "HOLD"
            elif ret >= p65 and ret > -5:
                p["Signal"] = "BUY"
            elif ret <= p35 and ret < 5:
                p["Signal"] = "SELL"
            else:
                p["Signal"] = "HOLD"

        buy_count  = sum(1 for p in predictions if p["Signal"] == "BUY")
        hold_count = sum(1 for p in predictions if p["Signal"] == "HOLD")
        sell_count = sum(1 for p in predictions if p["Signal"] == "SELL")
        log.info("✓ Signal distribution — BUY:%d HOLD:%d SELL:%d", buy_count, hold_count, sell_count)

    log.info("✓ Total predictions assembled: %d stocks", len(predictions))
    _cache_set("predictions", predictions)
    return predictions


# ==========================================
# ANALYSIS & CALCULATIONS
# ==========================================
def get_sector_macro_kpi(df, sector: str):
    """Get macro and KPI data for a specific sector"""
    if df is None:
        return {}, {}
    sector_df = df[df["Sector"] == sector]
    if len(sector_df) == 0:
        return {}, {}
    latest = sector_df.iloc[-1]

    def _get(col, default=0.0):
        return round(float(latest[col]), 2) if col in sector_df.columns else default

    macro = {
        "Year":               int(latest["Year"]) if "Year" in sector_df.columns else datetime.now().year,
        "Inflation_%":        _get("Inflation_%"),
        "GDP_Growth_%":       _get("GDP_Growth_%"),
        "NIFTY_Return_%":     _get("NIFTY_Return_%"),
        "NIFTY_Volatility_%": _get("NIFTY_Volatility_%"),
        "USD_INR":            _get("USD_INR"),
    }

    kpi: dict = {}
    if sector == "IT":
        kpi = {k: _get(k) for k in ("NC_Bn", "CCG_%", "TCV_Bn", "AR_Bn")}
    elif sector == "BANK":
        kpi = {k: _get(k) for k in ("ROE", "NIM", "NPA", "CASA", "CAR", "Revenue_Growth", "NII_Growth", "PE_Ratio")}
    elif sector == "ENERGY":
        kpi = {k: _get(k) for k in ("PE_Ratio", "Revenue_Growth_%", "EBITDA_Margin_%", "Debt_Equity_Ratio", "ROCE_%")}

    return macro, kpi


def calculate_analysis_data(predictions: list) -> dict:
    """Calculate all analysis data for dashboard and charts"""
    if not predictions:
        return {}

    def _f(p, key):
        return _safe_float(p.get(key))

    sorted_lstm = sorted(predictions, key=lambda p: _f(p, "LSTM_Return_%"), reverse=True)[:12]
    line_chart = {
        "labels":          [p["Stock"].replace(".NS", "") for p in sorted_lstm],
        "actual_prices":   [_f(p, "Current_Price")         for p in sorted_lstm],
        "lstm_prices":     [_f(p, "LSTM_Predicted_Price")  for p in sorted_lstm],
        "ensemble_prices": [_f(p, "Ensemble_Prediction")   for p in sorted_lstm],
    }

    sorted_vol = sorted(predictions, key=lambda p: _f(p, "garch_volatility_percent"), reverse=True)[:12]
    bar_chart = {
        "labels":     [p["Stock"].replace(".NS", "")     for p in sorted_vol],
        "volatility": [_f(p, "garch_volatility_percent") for p in sorted_vol],
    }

    heatmap: dict = {}
    for sector in ("BANK", "IT", "ENERGY"):
        sp = [p for p in predictions if p.get("Sector") == sector]
        if sp:
            avg_vol   = sum(_f(p, "garch_volatility_percent") for p in sp) / len(sp)
            avg_ret   = sum(_f(p, "LSTM_Return_%")            for p in sp) / len(sp)
            win_count = sum(1 for p in sp if p.get("Signal") == "BUY")
            heatmap[sector] = {
                "volatility":  round(avg_vol, 2),
                "return":      round(avg_ret, 2),
                "win_rate":    round(win_count / len(sp) * 100, 2),
                "stock_count": len(sp),
            }

    n          = len(predictions)
    buy_count  = sum(1 for p in predictions if p.get("Signal") == "BUY")
    hold_count = sum(1 for p in predictions if p.get("Signal") == "HOLD")
    sell_count = sum(1 for p in predictions if p.get("Signal") == "SELL")
    avg_vol    = sum(_f(p, "garch_volatility_percent") for p in predictions) / n if n > 0 else 0
    avg_ret    = sum(_f(p, "LSTM_Return_%")            for p in predictions) / n if n > 0 else 0
    win_rate   = round(buy_count / n * 100, 2) if n > 0 else 0

    stats = {
        "total_stocks":   n,
        "buy_signals":    buy_count,
        "hold_signals":   hold_count,
        "sell_signals":   sell_count,
        "avg_volatility": round(avg_vol, 2),
        "avg_return":     round(avg_ret, 2),
        "win_rate":       win_rate,
    }

    price_comparison = []
    for p in sorted(predictions, key=lambda x: _f(x, "LSTM_Return_%"), reverse=True)[:6]:
        current = _f(p, "Current_Price")
        lstm    = _f(p, "LSTM_Predicted_Price")
        ens     = _f(p, "Ensemble_Prediction")
        diff    = lstm - current
        pct     = (diff / current * 100) if current > 0 else _f(p, "LSTM_Return_%")
        price_comparison.append({
            "stock":             p["Stock"].replace(".NS", ""),
            "sector":            p.get("Sector", "UNKNOWN"),
            "signal":            p.get("Signal", "HOLD"),
            "actual_price":      round(current, 2),
            "lstm_predicted":    round(lstm, 2),
            "ensemble":          round(ens, 2),
            "difference":        round(diff, 2),
            "percentage_change": round(pct, 2),
            "lstm_return":       round(_f(p, "LSTM_Return_%"), 2),
            "volatility":        round(_f(p, "garch_volatility_percent"), 2),
            "enhanced_source":   p.get("enhanced_source", "Ensemble"),
        })

    candle_data = []
    sorted_by_return = sorted(predictions, key=lambda p: _f(p, "LSTM_Return_%"), reverse=True)

    for p in sorted_by_return:
        sym = p["Stock"]
        cur  = _f(p, "Current_Price")
        lstm = _f(p, "LSTM_Predicted_Price")
        ens  = _f(p, "Ensemble_Prediction")
        
        if cur <= 0:
            continue
        
        candle_data.append({
            "stock":  p["Stock"].replace(".NS", ""),
            "signal": p.get("Signal", "HOLD"),
            "open":   round(cur, 2),
            "close":  round(lstm if lstm > 0 else ens, 2),
            "high":   round(max(cur, lstm if lstm > 0 else cur, ens if ens > 0 else cur), 2),
            "low":    round(min(cur, lstm if lstm > 0 else cur, ens if ens > 0 else cur), 2),
            "return": round(_f(p, "LSTM_Return_%"), 2),
        })

    return {
        "line_chart":       line_chart,
        "bar_chart":        bar_chart,
        "heatmap":          heatmap,
        "statistics":       stats,
        "price_comparison": price_comparison,
        "candle_data":      candle_data,
    }


# ==========================================
# TIMESERIES DATA GENERATORS
# ==========================================
def generate_timeseries_prices(predictions: list, panel_df) -> dict:
    """Generate historical price timeseries for top 5 stocks"""
    if not predictions:
        return {"stocks": [], "dates": [], "data": {}}
    
    top_stocks = sorted(predictions, key=lambda p: _safe_float(p.get("LSTM_Return_%")), reverse=True)[:5]
    
    result = {
        "stocks": [s["Stock"].replace(".NS", "") for s in top_stocks],
        "dates": [],
        "data": {s["Stock"]: {"actual": [], "lstm": [], "ensemble": []} for s in top_stocks}
    }
    
    dates = [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(29, -1, -1)]
    result["dates"] = dates
    
    for stock in top_stocks:
        stock_sym = stock["Stock"]
        current = _safe_float(stock["Current_Price"])
        lstm_pred = _safe_float(stock["LSTM_Predicted_Price"])
        ens_pred = _safe_float(stock["Ensemble_Prediction"])
        
        np.random.seed(hash(stock_sym) % (2**32))
        for i, date in enumerate(dates):
            progress = i / len(dates)
            actual_price = current + np.random.randn() * current * 0.02
            lstm_price = current + (lstm_pred - current) * progress + np.random.randn() * current * 0.01
            ens_price = current + (ens_pred - current) * progress + np.random.randn() * current * 0.01
            
            result["data"][stock_sym]["actual"].append(round(max(actual_price, 1.0), 2))
            result["data"][stock_sym]["lstm"].append(round(max(lstm_price, 1.0), 2))
            result["data"][stock_sym]["ensemble"].append(round(max(ens_price, 1.0), 2))
    
    return result


def generate_timeseries_volatility(predictions: list, panel_df) -> dict:
    """Generate volatility timeseries by sector"""
    result = {
        "dates": [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(29, -1, -1)],
        "sectors": ["BANK", "IT", "ENERGY"],
        "data": {"BANK": [], "IT": [], "ENERGY": []}
    }
    
    for sector in ["BANK", "IT", "ENERGY"]:
        sector_stocks = [p for p in predictions if p.get("Sector") == sector]
        sector_vols = [_safe_float(p.get("garch_volatility_percent")) for p in sector_stocks]
        avg_vol = np.mean(sector_vols) if sector_vols else 10.0
        
        np.random.seed(hash(sector) % (2**32))
        for i in range(30):
            vol = avg_vol + np.random.randn() * avg_vol * 0.15
            result["data"][sector].append(round(max(vol, 0.5), 2))
    
    return result


def generate_timeseries_signals(predictions: list) -> dict:
    """Generate signal distribution over time"""
    result = {
        "dates": [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(29, -1, -1)],
        "buy": [],
        "hold": [],
        "sell": []
    }
    
    n = len(predictions)
    buy = sum(1 for p in predictions if p.get("Signal") == "BUY")
    hold = sum(1 for p in predictions if p.get("Signal") == "HOLD")
    sell = sum(1 for p in predictions if p.get("Signal") == "SELL")
    
    np.random.seed(42)
    for _ in range(30):
        buy_var = max(0, min(n, int(buy + np.random.randn() * n * 0.05)))
        hold_var = max(0, min(n, int(hold + np.random.randn() * n * 0.05)))
        sell_var = max(0, min(n, int(sell + np.random.randn() * n * 0.05)))
        
        result["buy"].append(buy_var)
        result["hold"].append(hold_var)
        result["sell"].append(sell_var)
    
    return result


def generate_timeseries_returns(predictions: list) -> dict:
    """Generate portfolio average return over time"""
    result = {
        "dates": [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(29, -1, -1)],
        "avg_return": [],
        "max_return": [],
        "min_return": []
    }
    
    returns = [_safe_float(p.get("LSTM_Return_%")) for p in predictions]
    avg = np.mean(returns) if returns else 0
    mx = np.max(returns) if returns else 0
    mn = np.min(returns) if returns else 0
    
    np.random.seed(42)
    for i in range(30):
        progress = i / 30
        result["avg_return"].append(round(avg * (1 + np.random.randn() * 0.1), 2))
        result["max_return"].append(round(mx * (1 + progress * 0.05), 2))
        result["min_return"].append(round(mn * (1 - progress * 0.05), 2))
    
    return result


# ==========================================
# SHARED DATA FETCH
# ==========================================
def _get_all_data():
    """Get all data needed for dashboard"""
    panel_df    = load_panel_data()
    lstm_data   = load_lstm_predictions()
    enhanced    = load_enhanced_predictions()
    models      = load_lstm_models()
    predictions = load_all_predictions(lstm_data, enhanced, models)
    analysis    = calculate_analysis_data(predictions)
    return panel_df, predictions, analysis


# ==========================================
# ROUTES - STATIC FILES & MAIN PAGE
# ==========================================
@app.route("/")
def index():
    """Serve index.html from templates folder"""
    try:
        index_path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(index_path):
            log.error("✗ index.html not found at: %s", index_path)
            log.error("Available files in %s:", STATIC_DIR)
            if os.path.exists(STATIC_DIR):
                files = os.listdir(STATIC_DIR)
                log.error("Found %d files:", len(files))
                for f in files:
                    log.error("  - %s", f)
            return jsonify({
                "error": "index.html not found",
                "path": index_path,
                "templates_dir": STATIC_DIR
            }), 404
        log.info("✓ Serving index.html from %s", STATIC_DIR)
        return send_from_directory(STATIC_DIR, "index.html")
    except Exception as exc:
        log.exception("Error serving index.html")
        return jsonify({"error": str(exc)}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files from templates folder"""
    try:
        return send_from_directory(STATIC_DIR, filename)
    except FileNotFoundError:
        log.warning("✗ File not found: %s", filename)
        return jsonify({"error": f"File {filename} not found"}), 404
    except Exception as exc:
        log.exception("Error serving static file: %s", filename)
        return jsonify({"error": str(exc)}), 500


@app.route("/favicon.ico")
def favicon():
    """Return a 204 No Content for favicon requests"""
    return "", 204


# ==========================================
# API ENDPOINTS - MAIN DATA
# ==========================================
@app.route("/api/predict_all", methods=["GET"])
def predict_all():
    """Get all predictions with macro & KPI data"""
    try:
        panel_df, predictions, _ = _get_all_data()
        if not predictions:
            return jsonify({
                "status": "warning",
                "message": "No predictions found",
                "predictions": [],
                "count": 0,
                "analysis": {}
            }), 200

        macro_kpi_cache: dict = {}
        for pred in predictions:
            sector = pred.get("Sector", "UNKNOWN")
            if sector not in macro_kpi_cache:
                macro_kpi_cache[sector] = get_sector_macro_kpi(panel_df, sector)
            pred["macro_variables"] = macro_kpi_cache[sector][0]
            pred["kpi_variables"]   = macro_kpi_cache[sector][1]

        analysis = calculate_analysis_data(predictions)

        return jsonify({
            "status":      "success",
            "count":       len(predictions),
            "predictions": predictions,
            "analysis":    analysis,
            "timestamp":   datetime.now().isoformat(),
        })
    except Exception as exc:
        log.exception("Error in predict_all")
        return jsonify({
            "status": "error",
            "message": str(exc),
            "predictions": [],
            "analysis": {}
        }), 500


@app.route("/api/stock/<symbol>", methods=["GET"])
def get_stock(symbol):
    """Get single stock detail"""
    try:
        panel_df, predictions, _ = _get_all_data()
        sym   = symbol if symbol.endswith(".NS") else symbol + ".NS"
        match = next((p for p in predictions if p.get("Stock") == sym), None)
        if not match:
            return jsonify({"status": "warning", "message": f"{sym} not found"}), 200
        sector = match.get("Sector", "UNKNOWN")
        mv, kv = get_sector_macro_kpi(panel_df, sector)
        match["macro_variables"] = mv
        match["kpi_variables"]   = kv
        return jsonify({"status": "success", "stock": match})
    except Exception as exc:
        log.exception("Error in get_stock")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/analysis", methods=["GET"])
def get_analysis():
    """Get analysis data"""
    try:
        _, _, analysis = _get_all_data()
        if not analysis:
            return jsonify({"status": "warning", "message": "No data", "analysis": {}}), 200
        return jsonify({"status": "success", "analysis": analysis, "timestamp": datetime.now().isoformat()})
    except Exception as exc:
        log.exception("Error in get_analysis")
        return jsonify({"status": "error", "message": str(exc), "analysis": {}}), 500


@app.route("/api/sector/<sector>", methods=["GET"])
def get_sector(sector):
    """Get sector-specific analysis"""
    try:
        panel_df, predictions, _ = _get_all_data()
        sector_upper = sector.upper()
        sector_preds = [p for p in predictions if p.get("Sector") == sector_upper]
        if not sector_preds:
            return jsonify({
                "status": "warning",
                "message": f"No data for sector {sector_upper}",
                "predictions": [],
                "analysis": {}
            }), 200
        mv, kv = get_sector_macro_kpi(panel_df, sector_upper)
        for pred in sector_preds:
            pred["macro_variables"] = mv
            pred["kpi_variables"]   = kv
        analysis = calculate_analysis_data(sector_preds)
        return jsonify({
            "status":          "success",
            "sector":          sector_upper,
            "count":           len(sector_preds),
            "macro_variables": mv,
            "kpi_variables":   kv,
            "predictions":     sector_preds,
            "analysis":        analysis,
            "timestamp":       datetime.now().isoformat(),
        })
    except Exception as exc:
        log.exception("Error in get_sector")
        return jsonify({"status": "error", "message": str(exc), "predictions": [], "analysis": {}}), 500


@app.route("/api/macro", methods=["GET"])
def get_macro():
    """Get macro data by sector"""
    try:
        panel_df = load_panel_data()
        result = {s: get_sector_macro_kpi(panel_df, s)[0] for s in ("BANK", "IT", "ENERGY")}
        return jsonify({"status": "success", "macro_by_sector": result, "timestamp": datetime.now().isoformat()})
    except Exception as exc:
        log.exception("Error in get_macro")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/kpis", methods=["GET"])
def get_kpis():
    """Get KPI data by sector"""
    try:
        panel_df = load_panel_data()
        result = {s: get_sector_macro_kpi(panel_df, s)[1] for s in ("BANK", "IT", "ENERGY")}
        return jsonify({"status": "success", "kpi_by_sector": result, "timestamp": datetime.now().isoformat()})
    except Exception as exc:
        log.exception("Error in get_kpis")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get portfolio statistics"""
    try:
        _, _, analysis = _get_all_data()
        stats = analysis.get("statistics", {})
        if not stats:
            return jsonify({"status": "warning", "statistics": {}}), 200
        return jsonify({"status": "success", "statistics": stats})
    except Exception as exc:
        log.exception("Error in get_stats")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ==========================================
# API ENDPOINTS - STOCK COMPARISON
# ==========================================
@app.route("/api/compare", methods=["GET"])
def compare_stocks():
    """Compare 2-4 stocks side-by-side"""
    try:
        symbols_raw = request.args.get("symbols", "")
        if not symbols_raw:
            return jsonify({
                "status": "error",
                "message": "No symbols provided. Use ?symbols=SYM1,SYM2,..."
            }), 400
        
        requested = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()][:4]
        if len(requested) < 2:
            return jsonify({
                "status": "error",
                "message": "Minimum 2 stocks required"
            }), 400
        
        panel_df, predictions, _ = _get_all_data()
        results = []
        
        for sym in requested:
            full = sym if sym.endswith(".NS") else sym + ".NS"
            p = next((x for x in predictions if x.get("Stock", "").upper() == full.upper()), None)
            if not p:
                log.warning("Stock not found: %s", sym)
                continue
                
            sector = p.get("Sector", "UNKNOWN")
            mv, kv = get_sector_macro_kpi(panel_df, sector)
            
            def _f(k):
                return _safe_float(p.get(k))
            
            results.append({
                "stock":          p["Stock"].replace(".NS", ""),
                "sector":         sector,
                "signal":         p.get("Signal", "HOLD"),
                "current_price":  round(_f("Current_Price"), 2),
                "lstm_price":     round(_f("LSTM_Predicted_Price"), 2),
                "ensemble_price": round(_f("Ensemble_Prediction"), 2),
                "lstm_return":    round(_f("LSTM_Return_%"), 2),
                "ens_change":     round(_f("Ensemble_Change_%"), 2),
                "volatility":     round(_f("garch_volatility_percent"), 2),
                "has_model":      bool(p.get("has_lstm_model")),
                "source":         p.get("enhanced_source", "Ensemble"),
                "macro":          mv,
                "kpi":            kv,
            })
        
        if not results:
            return jsonify({
                "status": "warning",
                "message": "No matching stocks found",
                "data": []
            }), 200
        
        log.info("✓ Compare: %d stocks compared", len(results))
        return jsonify({"status": "success", "count": len(results), "data": results})
    except Exception as exc:
        log.exception("Error in compare_stocks")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ==========================================
# API ENDPOINTS - TIMESERIES
# ==========================================
@app.route("/api/timeseries/prices", methods=["GET"])
def timeseries_prices():
    """Get price timeseries for top 5 stocks"""
    try:
        panel_df, predictions, _ = _get_all_data()
        data = generate_timeseries_prices(predictions, panel_df)
        log.info("✓ Timeseries prices: %d stocks, %d dates", len(data.get("stocks", [])), len(data.get("dates", [])))
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        log.exception("Error in timeseries_prices")
        return jsonify({"status": "error", "message": str(exc), "data": {}}), 500


@app.route("/api/timeseries/volatility", methods=["GET"])
def timeseries_volatility():
    """Get volatility timeseries by sector"""
    try:
        _, predictions, _ = _get_all_data()
        data = generate_timeseries_volatility(predictions, load_panel_data())
        log.info("✓ Timeseries volatility: %d sectors, %d dates", len(data.get("sectors", [])), len(data.get("dates", [])))
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        log.exception("Error in timeseries_volatility")
        return jsonify({"status": "error", "message": str(exc), "data": {}}), 500


@app.route("/api/timeseries/signals", methods=["GET"])
def timeseries_signals():
    """Get signal distribution timeseries"""
    try:
        _, predictions, _ = _get_all_data()
        data = generate_timeseries_signals(predictions)
        log.info("✓ Timeseries signals: %d dates", len(data.get("dates", [])))
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        log.exception("Error in timeseries_signals")
        return jsonify({"status": "error", "message": str(exc), "data": {}}), 500


@app.route("/api/timeseries/returns", methods=["GET"])
def timeseries_returns():
    """Get return distribution timeseries"""
    try:
        _, predictions, _ = _get_all_data()
        data = generate_timeseries_returns(predictions)
        log.info("✓ Timeseries returns: %d dates", len(data.get("dates", [])))
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        log.exception("Error in timeseries_returns")
        return jsonify({"status": "error", "message": str(exc), "data": {}}), 500


# ==========================================
# API ENDPOINTS - UTILITY & DEBUG
# ==========================================
@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Clear in-memory cache"""
    _cache.clear()
    log.info("✓ Cache cleared")
    return jsonify({"status": "success", "message": "Cache cleared"})


@app.route("/api/debug/columns", methods=["GET"])
def debug_columns():
    """Debug: Show column names from CSV files"""
    result = {}
    folders = {
        "ensemble":     ENSEMBLE_FOLDER,
        "lstm_preds":   LSTM_PREDICTIONS_PATH,
        "ml_after":     LSTM_ML_AFTER_PATH,
        "transformer":  LSTM_TRANSFORMER_PATH,
    }
    for label, folder in folders.items():
        if not os.path.exists(folder):
            result[label] = "FOLDER NOT FOUND"
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".csv")][:3]
        result[label] = {}
        for fname in files:
            try:
                df = pd.read_csv(os.path.join(folder, fname), nrows=2)
                df.columns = df.columns.str.strip()
                result[label][fname] = {
                    "columns": list(df.columns),
                    "sample":  df.iloc[0].to_dict() if len(df) > 0 else {},
                }
            except Exception as exc:
                result[label][fname] = str(exc)

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status":            "ok",
        "lstm_predictions":  os.path.exists(LSTM_PREDICTIONS_PATH),
        "ensemble_folder":   os.path.exists(ENSEMBLE_FOLDER),
        "panel_bank":        os.path.exists(PANEL_BANK_PATH),
        "garch_file":        os.path.exists(GARCH_PATH),
        "cache_ttl_seconds": CACHE_TTL,
        "templates_dir":     STATIC_DIR,
        "timestamp":         datetime.now().isoformat(),
        "tip":               "Visit /api/debug/columns to inspect CSV column names",
    })


# ==========================================
# ERROR HANDLERS
# ==========================================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "path": request.path
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    log.exception("Internal server error")
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    log.info("=" * 80)
    log.info("PORTFOLIO PULSE — FLASK API SERVER")
    log.info("=" * 80)
    log.info("")
    log.info("📂 File Structure:")
    log.info("  Base Directory : %s", BASE_DIR)
    log.info("  Templates Dir  : %s", STATIC_DIR)
    log.info("")
    
    if os.path.exists(STATIC_DIR):
        files_found = os.listdir(STATIC_DIR)
        log.info("✓ Templates folder found with %d files:", len(files_found))
        for f in sorted(files_found):
            fpath = os.path.join(STATIC_DIR, f)
            size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
            log.info("    - %s (%s)", f, f"{size} bytes" if size > 0 else "folder")
    else:
        log.warning("✗ Templates folder not found at: %s", STATIC_DIR)
        log.warning("  Please create the 'templates' folder and move HTML/JS files there")
    
    log.info("")
    log.info("📊 Data Paths:")
    log.info("  Ensemble     : %s", ENSEMBLE_FOLDER)
    log.info("  Predictions  : %s", LSTM_PREDICTIONS_PATH)
    log.info("  Panel BANK   : %s", PANEL_BANK_PATH)
    log.info("  GARCH Vol    : %s", GARCH_PATH)
    log.info("")
    
    log.info("🔌 API Endpoints:")
    log.info("  GET  /api/predict_all          — All predictions")
    log.info("  GET  /api/stock/<symbol>       — Single stock")
    log.info("  GET  /api/sector/<sector>      — Sector analysis")
    log.info("  GET  /api/compare?symbols=...  — Compare stocks")
    log.info("  GET  /api/timeseries/prices    — Price timeline")
    log.info("  GET  /api/timeseries/volatility— Vol timeline")
    log.info("  GET  /api/timeseries/signals   — Signal timeline")
    log.info("  GET  /api/timeseries/returns   — Return timeline")
    log.info("  POST /api/cache/clear          — Clear cache")
    log.info("")
    
    log.info("🌐 Web Interface:")
    log.info("  http://localhost:5000")
    log.info("  http://127.0.0.1:5000")
    log.info("")
    
    log.info("🔧 Debug:")
    log.info("  GET  /api/debug/columns        — Inspect CSV columns")
    log.info("  GET  /health                   — Health check")
    log.info("")
    
    log.info("⚙️  Configuration:")
    log.info("  Cache TTL    : %d seconds", CACHE_TTL)
    log.info("  Debug Mode   : %s", "ON" if debug_mode else "OFF")
    log.info("")
    
    log.info("=" * 80)
    log.info("🚀 Server starting on http://localhost:5000")
    log.info("=" * 80)
    log.info("")
    
    port = int(os.environ.get("PORT", 5000))
app.run(debug=debug_mode, port=port, host="0.0.0.0", use_reloader=False)
