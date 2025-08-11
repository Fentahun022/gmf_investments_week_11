
import pandas as pd
import yfinance as yf
import os

from src.data_ingestion import fetch_financial_data
from src.data_preprocessing import clean_and_preprocess_data, scale_data
from src.eda import (
    plot_closing_prices, plot_daily_returns, plot_returns_distribution,
    plot_rolling_volatility, detect_outliers_and_print, run_adf_test,
    calculate_and_print_risk_metrics
)

TICKERS = ['TSLA', 'BND', 'SPY']
START_DATE = '2015-07-01'

END_DATE = '2025-08-01'
TSLA_FORECAST_TARGET_COLUMN = 'TSLA_Adj_Close' 

DATA_RAW_PATH = 'data/raw/financial_data.csv'
DATA_PROCESSED_PATH = 'data/processed/processed_financial_data.csv'
SCALER_PATH = 'data/processed/tsla_close_scaler.pkl'

def run_task1():
    print("--- Running Task 1: Preprocess and Explore the Data ---")
    raw_data_df = fetch_financial_data(TICKERS, START_DATE, END_DATE)
    if raw_data_df is None:
        print("Failed to fetch raw data. Exiting Task 1.")
        return None
    os.makedirs(os.path.dirname(DATA_RAW_PATH), exist_ok=True)
    raw_data_df.to_csv(DATA_RAW_PATH, header=True) 
    processed_df = clean_and_preprocess_data(raw_data_df)
    if processed_df.empty:
        print("Failed to preprocess data. Exiting Task 1.")
        return None
    os.makedirs(os.path.dirname(DATA_PROCESSED_PATH), exist_ok=True)
    processed_df.to_csv(DATA_PROCESSED_PATH)
    print(f"Processed data saved to {DATA_PROCESSED_PATH}")
    tsla_close_series = processed_df[TSLA_FORECAST_TARGET_COLUMN].copy()
    scaled_tsla_close, tsla_scaler = scale_data(tsla_close_series)
    import joblib
    joblib.dump(tsla_scaler, SCALER_PATH)
    print(f"TSLA Close Scaler saved to {SCALER_PATH}")
    processed_df[f'Scaled_{TSLA_FORECAST_TARGET_COLUMN}'] = scaled_tsla_close


    # 1.3 Exploratory Data Analysis (EDA)
    print("\n--- Performing EDA ---")
    plot_closing_prices(processed_df)
    plot_daily_returns(processed_df)
    plot_returns_distribution(processed_df, ticker='TSLA')
    plot_rolling_volatility(processed_df)
    detect_outliers_and_print(processed_df, ticker='TSLA')

    # 1.4 Seasonality and Trends (Stationarity Test)
    for ticker in TICKERS:
        run_adf_test(processed_df[f'{ticker}_Adj_Close'], f'{ticker} Adj Close Price')
        run_adf_test(processed_df[f'{ticker}_Daily_Return'], f'{ticker} Daily Returns')

    # 1.5 Analyze Volatility & Risk Metrics
    calculate_and_print_risk_metrics(processed_df, TICKERS)

    print("\n--- Task 1 Completed ---")
    return processed_df # Return processed DataFrame for subsequent tasks

if __name__ == '__main__':
    # For now, just run Task 1
    processed_data = run_task1()
    if processed_data is not None:
        print("\nProcessed Data Overview (after Task 1):")
        print(processed_data.head())
        print(processed_data.info())