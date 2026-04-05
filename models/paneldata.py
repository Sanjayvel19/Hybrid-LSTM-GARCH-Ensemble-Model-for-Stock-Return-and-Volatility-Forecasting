import pandas as pd
import yfinance as yf
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. The 12 Stocks
    tickers = [
        'COFORGE.NS', 'INFY.NS', 'TECHM.NS', 'WIPRO.NS',
        'AXISBANK.NS', 'FEDERALBNK.NS', 'HDFCBANK.NS', 'SBIN.NS',
        'GAIL.NS', 'ONGC.NS', 'RELIANCE.NS', 'TATAPOWER.NS'
    ]

    # 2. Download Data
    print("Downloading Open, High, Low, Close data for 12 stocks...")
    all_data = []
    
    for ticker in tickers:
        print(f" -> Fetching {ticker}...")
        # Note: end="2026-04-02" matches today's date
        df = yf.download(ticker, start="2016-01-01", end="2025-12-31", progress=False)
        
        if not df.empty:
            df = df.reset_index()
            df['Stock'] = ticker
            
            # Fix Yahoo Finance multi-index columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns]
            
            df.columns = df.columns.str.lower().str.strip()
            df.rename(columns={'stock': 'Stock', 'date': 'Date'}, inplace=True)
            
            all_data.append(df)

    if not all_data:
        print("No data downloaded. Check your internet connection or tickers.")
        return

    stock_df = pd.concat(all_data, ignore_index=True)

    # 3. Sort as Panel Data & Calculate Returns
    print("Calculating Returns...")
    stock_df = stock_df.sort_values(['Stock', 'Date'])
    stock_df['Return'] = stock_df.groupby('Stock')['close'].pct_change()
    
    # Create Year column (this creates an integer type)
    stock_df['Year'] = stock_df['Date'].dt.year
    
    # Keep only core columns
    stock_df = stock_df[['Date', 'Year', 'Stock', 'open', 'high', 'low', 'close', 'Return']]

    # 4. Combine with CSV
    csv_file = r"C:\Users\jack1\OneDrive\Desktop\MBA PROJECT\models\all_sectors_combined.csv"
    
    if os.path.exists(csv_file):
        print(f"Loading CSV and fixing type mismatches...")
        csv_df = pd.read_csv(csv_file)
        
        # FIX 1: Ensure 'Year' is an integer to match stock_df
        csv_df['Year'] = pd.to_numeric(csv_df['Year'], errors='coerce')
        
        # FIX 2: Clean up 'Stock' names in CSV (remove spaces)
        if 'Stock' in csv_df.columns:
            csv_df['Stock'] = csv_df['Stock'].astype(str).str.strip()
        
        # Drop any rows where Year couldn't be converted
        csv_df = csv_df.dropna(subset=['Year'])
        csv_df['Year'] = csv_df['Year'].astype(int)

        # Merge on Stock and Year
        final_df = pd.merge(stock_df, csv_df, on=['Stock', 'Year'], how='left')
    else:
        print(f"Warning: '{csv_file}' not found. Saving stock data only.")
        final_df = stock_df

    # 5. Save Output
    output_path = os.path.join(current_dir, "final_simple_panel_data.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Panel data created and saved to:\n{output_path}")

if __name__ == "__main__":
    main()