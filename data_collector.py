"""
data_collector.py
Collects stock data and calculates technical indicators for stock prediction.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time

# Configuration
TICKERS = ["AAPL", "TSLA", "NVDA"]
START_DATE = "2020-06-01"  # 5+ years of data for better training
END_DATE = None  # Defaults to today
DATA_DIR = "data"

def fetch_stock_data(ticker, start=START_DATE, end=END_DATE):
    """
    Fetch historical stock data
    """
    print(f"Downloading {ticker} data from {start} to {end or 'today'}...")
    
    try:
        # Make sure end date is not in the future
        if end is None:
            end = datetime.today().strftime('%Y-%m-%d')
            
        # Check if start date makes sense
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        if start_date > end_date:
            print(f"Warning: Start date {start} is after end date {end}, adjusting to one year before end date")
            start_date = end_date - pd.DateOffset(years=1)
            start = start_date.strftime('%Y-%m-%d')
            
        # Download the data
        df = yf.download(ticker, start=start, end=end)
        
        # Check if data was downloaded successfully
        if df.empty:
            print(f"Warning: No data downloaded for {ticker}")
            return None
        
        # Add ticker column for multi-ticker datasets
        df['Ticker'] = ticker
        
        # Clean data
        df = df.dropna().copy()
        
        print(f"Successfully downloaded {ticker} data with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def add_technical_indicators(df):
    """Add technical indicators required by the prediction model"""
    try:
        if df is None or len(df) == 0:
            print("Error: Empty DataFrame provided to add_technical_indicators")
            return df
            
        # Calculate SMA and EMA indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        
        # RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Add small epsilon to avoid division by zero
        rs = gain / (loss + 1e-10)  
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['20DaySTD'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (df['20DaySTD'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['20DaySTD'] * 2)
        
        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        # Next day target (for training)
        df['Next_Close'] = df['Close'].shift(-1)
        df['Direction'] = (df['Next_Close'] > df['Close']).astype(int)
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        return df

def save_to_csv(df, ticker):
    """Save processed data to CSV file"""
    try:
        if df is None or len(df) == 0:
            print(f"Error: Empty DataFrame provided to save_to_csv for {ticker}")
            return False
            
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        
        # Save to CSV
        df.to_csv(path)
        
        # Verify the file was created and has data
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            if file_size > 0:
                print(f"[OK] Saved {ticker} data with {len(df)} rows and {len(df.columns)} columns to {path}")
                return True
            else:
                print(f"Warning: {path} was created but is empty")
                return False
        else:
            print(f"Error: Failed to create {path}")
            return False
    except Exception as e:
        print(f"Error saving {ticker} data to CSV: {e}")
        return False

def verify_data_for_model(ticker):
    """Verify that the saved data is suitable for the prediction model"""
    try:
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(path):
            print(f"Error: Data file for {ticker} does not exist")
            return False
            
        # Load the saved data
        df = pd.read_csv(path)
        
        # Check if it has all required columns for the model
        required_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10',
            'RSI', 'MACD', 'Signal_Line',
            'Upper_Band', 'Lower_Band',
            'Momentum', 'OBV',
            'Next_Close', 'Direction'
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"Warning: Missing columns in saved data for {ticker}: {missing}")
            return False
            
        # Check for NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            print(f"Warning: NaN values found in columns: {nan_cols}")
            
        # Check row count
        if len(df) < 100:
            print(f"Warning: Only {len(df)} rows in dataset, which may be insufficient for training")
            
        print(f"Data verification for {ticker}: {len(df)} rows, all required columns present")
        return True
        
    except Exception as e:
        print(f"Error verifying data for {ticker}: {e}")
        return False

def main():
    """Main function to collect and process data"""
    print(f"Starting data collection for {TICKERS}")
    print(f"Data period: {START_DATE} to {END_DATE or 'today'}")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    for ticker in TICKERS:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}...")
            print(f"{'='*50}")
            
            # Download data
            df = fetch_stock_data(ticker)
            
            if df is not None and not df.empty:
                # Add technical indicators
                df = add_technical_indicators(df)
                
                # Save to CSV
                if save_to_csv(df, ticker):
                    # Verify data is suitable for model
                    verify_data_for_model(ticker)
            else:
                print(f"Error: Could not process {ticker} due to empty data")
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
        
        time.sleep(1)  # Small delay to avoid API rate limits
        
    print("\nData collection complete!")

if __name__ == "__main__":
    main()
