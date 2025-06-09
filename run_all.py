# run_all.py
import os
import sys
from datetime import datetime

# Ensure the current directory is in the Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main functions from your scripts
from data_collector import main as data_collector_main
from StockMarketPredictor import main as smp_main

# Configuration (can be moved to a config.py if preferred)
TICKERS = ["AAPL", "TSLA", "NVDA"]
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

def safe_makedirs(path):
    """Helper to create directories safely."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ensured directory exists: {path}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error creating directory {path}: {e}")
        sys.exit(1) # Exit if cannot create critical directory

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting full stock prediction pipeline...")

    # Ensure necessary directories exist on the host (via Docker volumes)
    safe_makedirs(DATA_DIR)
    safe_makedirs(MODELS_DIR)
    safe_makedirs(LOGS_DIR)

    # 1. Run data_collector.py
    print(f"\n--- Running Data Collection ---")
    try:
        data_collector_main() # Call the main function directly
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error during data collection: {e}")
        sys.exit(1) # Exit if data collection fails critically
    print(f"--- Data Collection Complete ---")

    # 2. Run StockMarketPredictor.py for each ticker
    for ticker in TICKERS:
        csv_file_path = os.path.join(DATA_DIR, f"{ticker}.csv")

        if not os.path.exists(csv_file_path):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warning: {csv_file_path} not found. Skipping model training for {ticker}.")
            continue

        print(f"\n--- Training and Evaluating Models for {ticker} ---")
        try:
            # Call the main function of StockMarketPredictor with the specific CSV file
            smp_main(csv_file_path)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Models for {ticker} processed successfully.")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error during model training/prediction for {ticker}: {e}")
            # Decide if you want to exit on first model failure or continue
            sys.exit(1) # Exiting on critical failure for any stock

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Full stock prediction pipeline completed.") 