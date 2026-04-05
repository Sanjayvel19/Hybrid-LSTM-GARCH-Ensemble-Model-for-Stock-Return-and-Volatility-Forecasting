import pandas as pd
import numpy as np
from arch import arch_model
import os
import pickle

# ==============================
# FILE PATH
# ==============================
input_file = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\models\final_simple_panel_data.csv"
save_path = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\save_model"

os.makedirs(save_path, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(input_file)

print("Columns:", df.columns)

# ==============================
# CLEAN DATA
# ==============================
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ==============================
# DATE FORMAT
# ==============================
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# ==============================
# SORT DATA
# ==============================
df = df.sort_values(['Stock', 'Date'])

# ==============================
# CREATE RETURNS
# ==============================
price_col = 'close'

df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

df['Return'] = df.groupby('Stock')[price_col].pct_change() * 100
df = df.dropna(subset=['Return'])

# ==============================
# MACRO VARIABLES
# ==============================
macro_cols = ['Inflation_%', 'GDP_Growth_%', 'USD_INR']

for col in macro_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=macro_cols)

# ==============================
# STOCK LIST
# ==============================
stocks = df['Stock'].unique()
print("Total stocks:", len(stocks))

# ==============================
# STORAGE
# ==============================
summary = []
volatility_data = []

# ==============================
# LOOP
# ==============================
for stock in stocks:
    print("Processing:", stock)

    try:
        stock_df = df[df['Stock'] == stock].copy()

        if len(stock_df) < 50:
            print("Skipping:", stock)
            continue

        y = stock_df['Return']
        y = y - y.mean()  # stability

        X = stock_df[macro_cols]

        # ==============================
        # 🔥 GARCH MODEL
        # ==============================
        model = arch_model(
            y,
            x=X,
            mean='ARX',
            vol='Garch',   # ✅ changed from EGARCH
            p=1,
            q=1,
            dist='normal'
        )

        result = model.fit(disp='off')

        # ==============================
        # SAVE MODEL
        # ==============================
        model_file = os.path.join(save_path, f"{stock}_garch.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(result, f)

        # ==============================
        # 🔥 EXTRACT VOLATILITY (KEY FOR LSTM)
        # ==============================
        stock_df['Volatility'] = result.conditional_volatility.values

        volatility_data.append(stock_df[['Date', 'Stock', 'Volatility']])

        # ==============================
        # SUMMARY
        # ==============================
        summary.append({
            "Stock": stock,
            "Observations": len(y),
            "AIC": result.aic,
            "BIC": result.bic,
            "LogLikelihood": result.loglikelihood
        })

        print("Done:", stock)

    except Exception as e:
        print("Error in", stock, ":", e)

# ==============================
# SAVE OUTPUTS
# ==============================
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(save_path, "garch_summary.csv"), index=False)

# 🔥 COMBINE ALL VOLATILITY
vol_df = pd.concat(volatility_data)
vol_df.to_csv(os.path.join(save_path, "volatility_forecast.csv"), index=False)

print("\nGARCH Completed Successfully!")