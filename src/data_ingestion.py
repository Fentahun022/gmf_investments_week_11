# src/data_ingestion.py
import yfinance as yf
import pandas as pd

def fetch_financial_data(tickers, start_date, end_date):
    """
    Fetches historical financial data for given tickers from YFinance.

    Args:
        tickers (list): A list of stock/ETF tickers (e.g., ['TSLA', 'BND', 'SPY']).
        start_date (str): Start date for data in 'YYYY-MM-DD' format.
        end_date (str): End date for data in 'YYYY-MM-DD' format.
                        YFinance's end date is exclusive, so set to one day after desired end.

    Returns:
        pd.DataFrame: A DataFrame with multi-level columns (e.g., ('Adj Close', 'TSLA')).
                      Returns None if data fetching fails.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    try:
        # FIX: Add auto_adjust=False to ensure 'Adj Close' column is returned
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
        if data.empty:
            print("No data fetched. Check tickers or date range.")
            return None
        print("Data fetched successfully.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    tickers_list = ['TSLA', 'BND', 'SPY']
    start_dt = '2015-07-01'
    end_dt = '2025-08-01' # yfinance end date is exclusive, so set to one day after desired end

    raw_data = fetch_financial_data(tickers_list, start_dt, end_dt)

    if raw_data is not None:
        print("\nRaw Data Head:")
        print(raw_data.head())
        raw_data.info()
        # Save raw data for later use
        raw_data.to_csv('data/raw/financial_data.csv')
        print("Raw data saved to data/raw/financial_data.csv")