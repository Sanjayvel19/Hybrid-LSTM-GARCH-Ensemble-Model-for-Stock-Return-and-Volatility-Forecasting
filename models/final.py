# ==========================================
# COMPLETE PIPELINE - WITH ENSEMBLE PREDICTIONS
# ==========================================
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# CONFIGURATION
# ==========================================
RUN_LSTM_TRANSFORMER = True
RUN_ML_AFTER_LSTM = True
USE_EXISTING_LSTM_CSV = False   # Set False to use freshly trained LSTM
USE_RF = True                   # True = Random Forest, False = XGBoost

TIME_STEPS = 30
LSTM_EPOCHS = 30
BATCH_SIZE = 32

# ==========================================
# FILE PATHS – EDIT THESE
# ==========================================
PANEL_BANK_PATH   = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\ULITICS\final_simple_panel_data - BANK.csv"
PANEL_IT_PATH     = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\ULITICS\final_simple_panel_data - IT .csv"
PANEL_ENERGY_PATH = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\ULITICS\final_simple_panel_data - ENERGY.csv"
GARCH_PATH        = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\save_model\GARCH\volatility_forecast.csv"

OUTPUT_BASE = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\save_model"
LSTM_OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, "lstm_transformer_predictions")
ML_OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, "ml_after_lstm_predictions")
ENSEMBLE_OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, "ensemble_predictions")

for folder in [LSTM_OUTPUT_FOLDER, ML_OUTPUT_FOLDER, ENSEMBLE_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ==========================================
# LOAD DATA (same as before)
# ==========================================
def load_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1')

print("\nLoading data...")
df_bank = load_csv(PANEL_BANK_PATH)
df_it = load_csv(PANEL_IT_PATH)
df_energy = load_csv(PANEL_ENERGY_PATH)

df_bank['Sector'] = 'BANK'
df_it['Sector'] = 'IT'
df_energy['Sector'] = 'ENERGY'

df = pd.concat([df_bank, df_it, df_energy], ignore_index=True)
garch = load_csv(GARCH_PATH)

df.columns = df.columns.str.strip()
garch.columns = garch.columns.str.strip()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
garch['Date'] = pd.to_datetime(garch['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
garch = garch.dropna(subset=['Date'])

df = df.sort_values(['Sector', 'Stock', 'Date'])
df = df.merge(garch, on=['Date', 'Stock'], how='left')
df['Volatility'] = df['Volatility'].ffill().bfill()
df = df.dropna(subset=['Volatility'])

# Feature sets
sector_features = {
    'BANK': ['open','high','low','close','Return','Inflation_%','GDP_Growth_%',
             'NIFTY_Return_%','NIFTY_Volatility_%','USD_INR','PE_Ratio',
             'Revenue_Growth_%','EBITDA_Margin_%','Debt_Equity_Ratio','ROCE_%','Volatility'],
    'IT': ['open','high','low','close','Return','Inflation_%','GDP_Growth_%',
           'NIFTY_Return_%','NIFTY_Volatility_%','USD_INR','NC_Bn','CCG_%','TCV_Bn','AR_Bn','Volatility'],
    'ENERGY': ['open','high','low','close','Return','Inflation_%','GDP_Growth_%',
               'NIFTY_Return_%','NIFTY_Volatility_%','USD_INR','PE_Ratio',
               'Revenue_Growth_%','EBITDA_Margin_%','Debt_Equity_Ratio','ROCE_%','Volatility']
}

for sector, feats in sector_features.items():
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"WARNING: {sector} missing: {missing}")
        sector_features[sector] = [f for f in feats if f in df.columns]

# Convert to float32
all_feature_cols = set()
for cols in sector_features.values():
    all_feature_cols.update(cols)
for col in all_feature_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

# ==========================================
# LSTM MODEL (per-stock scaling)
# ==========================================
def prepare_sequences_per_stock(df, features, time_steps=30):
    X, y = [], []
    stock_scalers = {}
    for stock in df['Stock'].unique():
        stock_df = df[df['Stock'] == stock].sort_values('Date')
        if len(stock_df) <= time_steps:
            continue
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(stock_df[features].astype(np.float32))
        stock_scalers[stock] = scaler
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i])
            y.append(scaled_data[i, 3])
    if len(X) == 0:
        return np.array([]), np.array([]), stock_scalers
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), stock_scalers

def build_simple_lstm(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(32, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(16)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm_sector(df, sector, features):
    print(f"\n=== Training LSTM for {sector} ===")
    sector_df = df[df['Sector'] == sector].copy()
    features = [f for f in features if f in sector_df.columns]
    if len(features) == 0:
        return None, None, None
    X, y, stock_scalers = prepare_sequences_per_stock(sector_df, features, TIME_STEPS)
    if len(X) == 0:
        print(f"No sequences for {sector}")
        return None, None, None
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(features)}")
    model = build_simple_lstm((TIME_STEPS, len(features)))
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True),
                 ReduceLROnPlateau(factor=0.5, patience=3)]
    model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE,
              validation_split=0.1, callbacks=callbacks, verbose=1)
    y_pred = model.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{sector} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
    model.save(os.path.join(LSTM_OUTPUT_FOLDER, f"{sector}_lstm.keras"))
    joblib.dump(stock_scalers, os.path.join(LSTM_OUTPUT_FOLDER, f"{sector}_scalers.pkl"))
    joblib.dump(features, os.path.join(LSTM_OUTPUT_FOLDER, f"{sector}_features.pkl"))
    return model, stock_scalers, (rmse, r2)

def predict_lstm_for_stocks(df, sector, model, stock_scalers, features):
    sector_df = df[df['Sector'] == sector]
    predictions = {}
    for stock in sector_df['Stock'].unique():
        stock_df = sector_df[sector_df['Stock'] == stock].sort_values('Date')
        if len(stock_df) < TIME_STEPS or stock not in stock_scalers:
            continue
        scaler = stock_scalers[stock]
        latest = stock_df.tail(TIME_STEPS)
        X_raw = latest[features].ffill().bfill().astype(np.float32)
        X_scaled = scaler.transform(X_raw)
        X_seq = np.expand_dims(X_scaled, axis=0)
        pred_scaled = model.predict(X_seq, verbose=0)[0][0]
        dummy = np.zeros((1, len(features)))
        dummy[0, 3] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0, 3]
        current = latest['close'].iloc[-1]
        predictions[stock] = pred_price
        result = pd.DataFrame([{
            "Stock": stock,
            "Sector": sector,
            "Current_Price": round(current, 2),
            "LSTM_Prediction": round(pred_price, 2),
            "Change_%": round(((pred_price - current) / current) * 100, 2),
            "Signal": "BUY" if pred_price > current else "SELL"
        }])
        result.to_csv(os.path.join(LSTM_OUTPUT_FOLDER, f"{stock.replace('.', '_')}.csv"), index=False)
        print(f"  Saved LSTM for {stock}")
    return predictions

# ==========================================
# ML AFTER LSTM
# ==========================================
def train_ml_after_lstm(df_with_lstm, sector, base_features):
    print(f"\n=== ML after LSTM for {sector} ===")
    sector_df = df_with_lstm[df_with_lstm['Sector'] == sector].copy()
    ml_features = base_features + ['LSTM_Prediction']
    ml_features = [f for f in ml_features if f in sector_df.columns]
    sector_df = sector_df.sort_values(['Stock', 'Date'])
    X_list, y_list = [], []
    for stock in sector_df['Stock'].unique():
        stock_df = sector_df[sector_df['Stock'] == stock]
        for i in range(len(stock_df) - 1):
            X_list.append(stock_df[ml_features].iloc[i].values)
            y_list.append(stock_df['close'].iloc[i+1])
    if len(X_list) == 0:
        return None, None, None
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    split = int(0.8 * len(X))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    lstm_idx = ml_features.index('LSTM_Prediction')
    y_lstm = X_test[:, lstm_idx]
    base_rmse = np.sqrt(mean_squared_error(y_test, y_lstm))
    print(f"Baseline LSTM RMSE: {base_rmse:.4f}")
    if USE_RF:
        model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        model_name = "RF"
    else:
        model = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42)
        model_name = "XGB"
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ml_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{model_name} RMSE: {ml_rmse:.4f} (Improvement: {(base_rmse-ml_rmse)/base_rmse*100:.2f}%)")
    joblib.dump(model, os.path.join(ML_OUTPUT_FOLDER, f"{sector}_{model_name}_model.pkl"))
    joblib.dump(scaler, os.path.join(ML_OUTPUT_FOLDER, f"{sector}_scaler.pkl"))
    joblib.dump(ml_features, os.path.join(ML_OUTPUT_FOLDER, f"{sector}_features.pkl"))
    return model, scaler, ml_rmse

def predict_ml_for_stocks(df_with_lstm, sector, model, scaler, base_features):
    sector_df = df_with_lstm[df_with_lstm['Sector'] == sector].copy()
    ml_features = base_features + ['LSTM_Prediction']
    ml_features = [f for f in ml_features if f in sector_df.columns]
    results = {}
    for stock in sector_df['Stock'].unique():
        stock_df = sector_df[sector_df['Stock'] == stock]
        if len(stock_df) < 2:
            continue
        latest = stock_df.iloc[-1:]
        X = latest[ml_features].ffill().bfill().astype(np.float32)
        X_scaled = scaler.transform(X)
        ml_pred = model.predict(X_scaled)[0]
        results[stock] = ml_pred
        lstm_pred = latest['LSTM_Prediction'].iloc[0]
        current = latest['close'].iloc[0]
        result = pd.DataFrame([{
            "Stock": stock,
            "Sector": sector,
            "Current_Price": round(current, 2),
            "LSTM_Prediction": round(lstm_pred, 2),
            "ML_Prediction": round(ml_pred, 2),
            "ML_Change_%": round(((ml_pred - current) / current) * 100, 2),
            "Signal": "BUY" if ml_pred > current else "SELL"
        }])
        result.to_csv(os.path.join(ML_OUTPUT_FOLDER, f"{stock.replace('.', '_')}_enhanced.csv"), index=False)
        print(f"  Saved ML for {stock}")
    return results

# ==========================================
# ENSEMBLE (average of LSTM and ML predictions)
# ==========================================
def create_ensemble_predictions(df, sector, lstm_predictions, ml_predictions):
    print(f"\n=== Creating ensemble predictions for {sector} ===")
    sector_df = df[df['Sector'] == sector]
    results = []
    for stock in sector_df['Stock'].unique():
        if stock not in lstm_predictions or stock not in ml_predictions:
            continue
        lstm_pred = lstm_predictions[stock]
        ml_pred = ml_predictions[stock]
        ensemble_pred = (lstm_pred + ml_pred) / 2
        stock_df = sector_df[sector_df['Stock'] == stock]
        current = stock_df['close'].iloc[-1]
        ensemble_change = ((ensemble_pred - current) / current) * 100
        result = pd.DataFrame([{
            "Stock": stock,
            "Sector": sector,
            "Current_Price": round(current, 2),
            "LSTM_Prediction": round(lstm_pred, 2),
            "ML_Prediction": round(ml_pred, 2),
            "Ensemble_Prediction": round(ensemble_pred, 2),
            "Ensemble_Change_%": round(ensemble_change, 2),
            "Signal": "BUY" if ensemble_pred > current else "SELL"
        }])
        result.to_csv(os.path.join(ENSEMBLE_OUTPUT_FOLDER, f"{stock.replace('.', '_')}_ensemble.csv"), index=False)
        results.append(result)
        print(f"  Saved ensemble for {stock}")
    return pd.concat(results) if results else pd.DataFrame()

# ==========================================
# MAIN EXECUTION
# ==========================================
print("\n" + "="*70)
print("STARTING PIPELINE (with Ensemble)")
print("="*70)

lstm_predictions = {}
ml_predictions = {}

# 1. Train LSTM
if RUN_LSTM_TRANSFORMER:
    for sector, features in sector_features.items():
        if features:
            model, scalers, metrics = train_lstm_sector(df, sector, features)
            if model:
                preds = predict_lstm_for_stocks(df, sector, model, scalers, features)
                lstm_predictions.update(preds)

# 2. ML after LSTM
if RUN_ML_AFTER_LSTM:
    # Prepare dataframe with LSTM predictions
    df_with_lstm = df.copy()
    df_with_lstm['LSTM_Prediction'] = df_with_lstm['Stock'].map(lstm_predictions)
    df_with_lstm = df_with_lstm.dropna(subset=['LSTM_Prediction'])
    print(f"\nData with LSTM predictions: {len(df_with_lstm)} rows")
    
    if len(df_with_lstm) > 0:
        for sector, base_features in sector_features.items():
            if base_features and sector in df_with_lstm['Sector'].values:
                model, scaler, rmse = train_ml_after_lstm(df_with_lstm, sector, base_features)
                if model:
                    preds = predict_ml_for_stocks(df_with_lstm, sector, model, scaler, base_features)
                    ml_predictions.update(preds)

# 3. Ensemble (if both predictions exist)
if lstm_predictions and ml_predictions:
    print("\n" + "="*70)
    print("CREATING ENSEMBLE PREDICTIONS")
    print("="*70)
    all_ensemble = []
    for sector in sector_features.keys():
        ensemble_df = create_ensemble_predictions(df, sector, lstm_predictions, ml_predictions)
        if len(ensemble_df) > 0:
            all_ensemble.append(ensemble_df)
    if all_ensemble:
        final_ensemble = pd.concat(all_ensemble)
        final_ensemble.to_csv(os.path.join(ENSEMBLE_OUTPUT_FOLDER, "all_stocks_ensemble.csv"), index=False)
        print(f"\nEnsemble summary saved to {ENSEMBLE_OUTPUT_FOLDER}")
else:
    print("\nSkipping ensemble: missing LSTM or ML predictions.")

print("\n[OK] Pipeline finished successfully.")
print(f"Ensemble predictions saved in: {ENSEMBLE_OUTPUT_FOLDER}")
