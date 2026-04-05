# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add, Reshape
)
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy('float32')

# ==========================================
# 2. FILE PATHS – EDIT THESE FOUR PATHS
# ==========================================
PANEL_BANK_PATH   = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\ULITICS\final_simple_panel_data - BANK.csv"
PANEL_IT_PATH     = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\ULITICS\final_simple_panel_data - IT .csv"
PANEL_ENERGY_PATH = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\ULITICS\final_simple_panel_data - ENERGY.csv"
GARCH_PATH        = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\save_model\GARCH\volatility_forecast.csv"

SAVE_PATH = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\save_model"
PRED_PATH = os.path.join(SAVE_PATH, "predictions")

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(PRED_PATH, exist_ok=True)

# ==========================================
# 3. CHECK THAT ALL FILES EXIST
# ==========================================
for path, name in [(PANEL_BANK_PATH, "BANK panel"),
                   (PANEL_IT_PATH, "IT panel"),
                   (PANEL_ENERGY_PATH, "ENERGY panel"),
                   (GARCH_PATH, "GARCH volatility")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found: {path}")

# ==========================================
# 4. LOAD AND COMBINE PANEL DATA, ASSIGN SECTOR
# ==========================================
def load_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1')

df_bank   = load_csv(PANEL_BANK_PATH)
df_it     = load_csv(PANEL_IT_PATH)
df_energy = load_csv(PANEL_ENERGY_PATH)

df_bank['Sector']   = 'BANK'
df_it['Sector']     = 'IT'
df_energy['Sector'] = 'ENERGY'

df = pd.concat([df_bank, df_it, df_energy], ignore_index=True)
garch = load_csv(GARCH_PATH)

# ==========================================
# 5. CLEAN COLUMN NAMES
# ==========================================
df.columns = df.columns.str.strip()
garch.columns = garch.columns.str.strip()

# ==========================================
# 6. DATE CONVERSION
# ==========================================
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
garch['Date'] = pd.to_datetime(garch['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
garch = garch.dropna(subset=['Date'])

# ==========================================
# 7. SORT & MERGE GARCH VOLATILITY
# ==========================================
df = df.sort_values(['Sector', 'Stock', 'Date'])
df = df.merge(garch, on=['Date', 'Stock'], how='left')
df['Volatility'] = df['Volatility'].ffill().bfill()
df = df.dropna(subset=['Volatility'])

# ==========================================
# 8. FEATURE SETS (PER SECTOR) – KPI + MACRO
# ==========================================
sector_features = {
    'BANK': [
        'open', 'high', 'low', 'close', 'Return', 'Inflation_%', 'GDP_Growth_%',
        'NIFTY_Return_%', 'NIFTY_Volatility_%', 'USD_INR', 'PE_Ratio',
        'Revenue_Growth_%', 'EBITDA_Margin_%', 'Debt_Equity_Ratio', 'ROCE_%',
        'Volatility'
    ],
    'IT': [
        'open', 'high', 'low', 'close', 'Return', 'Inflation_%', 'GDP_Growth_%',
        'NIFTY_Return_%', 'NIFTY_Volatility_%', 'USD_INR', 'NC_Bn', 'CCG_%',
        'TCV_Bn', 'AR_Bn', 'Volatility'
    ],
    'ENERGY': [
        'open', 'high', 'low', 'close', 'Return', 'Inflation_%', 'GDP_Growth_%',
        'NIFTY_Return_%', 'NIFTY_Volatility_%', 'USD_INR', 'PE_Ratio',
        'Revenue_Growth_%', 'EBITDA_Margin_%', 'Debt_Equity_Ratio', 'ROCE_%',
        'Volatility'
    ]
}

# ==========================================
# 9. CHECK MISSING COLUMNS (WARNING ONLY)
# ==========================================
for sector, feats in sector_features.items():
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"WARNING: {sector} missing columns: {missing}")
        sector_features[sector] = [f for f in feats if f in df.columns]
        print(f"  -> Using only: {sector_features[sector]}")

# ==========================================
# 10. CONVERT ALL FEATURE COLUMNS TO FLOAT32
# ==========================================
all_feature_cols = set()
for cols in sector_features.values():
    all_feature_cols.update(cols)
for col in all_feature_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

# ==========================================
# 11. PREPARE DATA FUNCTION (unchanged)
# ==========================================
def prepare_data(df, features, time_steps=30):
    scaler = MinMaxScaler()
    df = df.copy()
    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("No valid features")
    df[features] = df[features].astype(np.float32)
    df[features] = scaler.fit_transform(df[features])

    X, y = [], []
    for stock in df['Stock'].unique():
        stock_df = df[df['Stock'] == stock]
        data = stock_df[features].values
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i])
            y.append(data[i, 3])  # close is the 4th feature (index 3)
    if len(X) == 0:
        return np.array([]), np.array([]), scaler
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler

# ==========================================
# 12. BUILD MODEL – LSTM + TRANSFORMER (ATTENTION)
# ==========================================
def transformer_encoder(inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.2):
    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    
    # Feed-forward network
    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn)

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # LSTM layer to capture sequential patterns
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Transformer encoder (2 layers)
    transformer_out = lstm_out
    for _ in range(2):
        transformer_out = transformer_encoder(transformer_out, head_size=32, num_heads=2, ff_dim=64, dropout=0.2)
    
    # Global pooling and regression head
    x = GlobalAveragePooling1D()(transformer_out)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# ==========================================
# 13. PREDICT AND SAVE FOR EACH STOCK (CSV OUTPUTS)
# ==========================================
def predict_and_save(df, sector, model, scaler, features):
    sector_df = df[df['Sector'] == sector]
    for stock in sector_df['Stock'].unique():
        stock_df = sector_df[sector_df['Stock'] == stock]
        if len(stock_df) < 30:
            continue
        latest = stock_df.tail(30)
        X = latest[features].ffill().bfill().astype(np.float32)
        X_scaled = scaler.transform(X)
        X_lstm = np.reshape(X_scaled, (1, 30, len(features))).astype(np.float32)
        pred = model.predict(X_lstm, verbose=0)[0][0]
        current = latest['close'].iloc[-1]
        change = ((pred - current) / current) * 100
        signal = "BUY" if change > 0 else "SELL"
        result = pd.DataFrame([{
            "Stock": stock,
            "Sector": sector,
            "Current": current,
            "Predicted": pred,
            "Change_%": change,
            "Signal": signal
        }])
        result.to_csv(os.path.join(PRED_PATH, f"{stock}.csv"), index=False)
        print(f"{stock} prediction saved")

# ==========================================
# 14. TRAIN SECTOR MODEL
# ==========================================
def train_sector(df, sector, features):
    print(f"\nTraining {sector}...")
    sector_df = df[df['Sector'] == sector].copy()
    features = [f for f in features if f in sector_df.columns]
    if len(features) == 0:
        print(f"No valid features for {sector}, skipping.")
        return None
    sector_df[features] = sector_df[features].astype(np.float32)
    if len(sector_df) < 50:
        print(f"Not enough data for {sector}")
        return None

    X, y, scaler = prepare_data(sector_df, features)
    if len(X) == 0:
        print(f"No sequences created for {sector}")
        return None

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    pred = model.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{sector} RMSE: {rmse:.4f}")

    model.save(os.path.join(SAVE_PATH, f"{sector}_model.h5"))
    joblib.dump(scaler, os.path.join(SAVE_PATH, f"{sector}_scaler.pkl"))
    joblib.dump(features, os.path.join(SAVE_PATH, f"{sector}_features.pkl"))

    predict_and_save(df, sector, model, scaler, features)
    return rmse

# ==========================================
# 15. RUN TRAINING FOR ALL SECTORS
# ==========================================
results = {}
for sector, features in sector_features.items():
    if len(features) > 0:
        rmse = train_sector(df, sector, features)
        results[sector] = rmse
    else:
        print(f"Skipping {sector} because no features available.")

# ==========================================
# 16. FINAL RESULTS
# ==========================================
print("\nFINAL RESULTS")
for k, v in results.items():
    print(f"{k} RMSE: {v}")