# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_and_preprocess_data(data):
    """
    Cleans the fetched financial data and performs basic preprocessing.

    Args:
        data (pd.DataFrame): Raw DataFrame with multi-level columns from yfinance.

    Returns:
        pd.DataFrame: Cleaned and processed DataFrame.
    """
    if data is None or data.empty:
        print("No data to preprocess.")
        return pd.DataFrame()

    print("Cleaning and preprocessing data...")

    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Handle missing values (forward fill then backward fill for any leading NaNs)
    # This is applied to the raw multi-indexed dataframe
    data = data.ffill().bfill()
    if data.isnull().sum().sum() > 0:
        print(f"Warning: Still missing values after ffill/bfill in raw data: \n{data.isnull().sum()[data.isnull().sum() > 0]}")

    # Select Adj Close data and flatten multi-index columns to ticker names
    adj_close_data = data.xs('Adj Close', level=0, axis=1).copy()
    adj_close_data.columns = [col for col in adj_close_data.columns] # Ensure simple column names (e.g., 'TSLA')

    # Calculate daily returns
    daily_returns = adj_close_data.pct_change()

    # CRITICAL FIX: Convert inf/-inf to NaN and then drop NaNs immediately from daily_returns
    daily_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    daily_returns.dropna(inplace=True) # Drop rows where any daily return is NaN

    daily_returns = daily_returns.add_suffix('_Daily_Return')

    # Calculate rolling volatility (e.g., 20-day standard deviation of daily returns)
    # Ensure this is also clean and aligned
    rolling_volatility = daily_returns.iloc[:, [i for i, col in enumerate(daily_returns.columns) if '_Daily_Return' in col]].rolling(window=20).std()
    rolling_volatility = rolling_volatility.add_suffix('_Rolling_Vol_20D').rename(columns=lambda x: x.replace('_Daily_Return', ''))
    rolling_volatility.replace([np.inf, -np.inf], np.nan, inplace=True) # Clean infinities from rolling std
    rolling_volatility.dropna(inplace=True) # Drop NaNs introduced by rolling calculations or infinities

    # Merge all components. Use an inner join to ensure only dates common to all clean series remain.
    processed_data = adj_close_data.copy()
    processed_data = processed_data.add_suffix('_Adj_Close') # Add suffix to Adj_Close columns

    # Aligning and merging, ensuring common index
    # We explicitly merge each component to make sure they are aligned
    processed_data = processed_data.merge(daily_returns, left_index=True, right_index=True, how='inner')
    processed_data = processed_data.merge(rolling_volatility, left_index=True, right_index=True, how='inner')

    # FINAL, SUPER AGGRESSIVE CLEANING STEP:
    # After all merges, apply pd.to_numeric and dropna across the entire DataFrame
    print("Applying final rigorous cleaning to merged data...")
    for col in processed_data.columns:
        processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce') # Coerce non-numeric to NaN
    processed_data.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace any remaining infinities
    processed_data.dropna(inplace=True) # Drop any row with any NaN

    if processed_data.empty:
        raise ValueError("Processed DataFrame became empty after rigorous cleaning. Check raw data consistency.")


    print("Data preprocessing complete.")
    return processed_data

def scale_data(data_series, scaler=None):
    """
    Scales a single pandas Series using MinMaxScaler.
    If no scaler is provided, a new one is fitted.

    Args:
        data_series (pd.Series): The data series to scale.
        scaler (sklearn.preprocessing.MinMaxScaler, optional): An existing scaler to use. Defaults to None.

    Returns:
        tuple: A tuple containing the scaled data (pd.Series) and the fitted scaler.
    """
    if data_series.empty:
        return pd.Series(), scaler

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))
    else:
        scaled_data = scaler.transform(data_series.values.reshape(-1, 1))

    return pd.Series(scaled_data.flatten(), index=data_series.index), scaler

if __name__ == '__main__':
    # Example usage (assuming financial_data.csv exists)
    try:
        # Load the raw data with multi-level columns
        raw_data = pd.read_csv('data/raw/financial_data.csv', header=[0, 1], index_col=0, parse_dates=True)
        # Rename the level 0 of the MultiIndex to 'Metric' (optional but good for clarity)
        raw_data.columns.names = ['Metric', 'Ticker']
    except FileNotFoundError:
        print("Run data_ingestion.py first to create data/raw/financial_data.csv")
        exit()

    processed_df = clean_and_preprocess_data(raw_data)

    if not processed_df.empty:
        print("\nProcessed Data Head:")
        print(processed_df.head())
        print("\nProcessed Data Info:")
        processed_df.info()
        print("\nMissing values after processing:")
        print(processed_df.isnull().sum().sum())
        processed_df.to_csv('data/processed/processed_financial_data.csv')
        print("Processed data saved to data/processed/processed_financial_data.csv")

        # Example of scaling TSLA Adj Close
        tsla_close = processed_df['TSLA_Adj_Close']
        scaled_tsla_close, tsla_scaler = scale_data(tsla_close)
        print("\nScaled TSLA Adj Close Head:")
        print(scaled_tsla_close.head())
        # You'd save the scaler if you need to transform new data or inverse transform predictions
        import joblib
        joblib.dump(tsla_scaler, 'data/processed/tsla_close_scaler.pkl')
        print("TSLA Close Scaler saved to data/processed/tsla_close_scaler.pkl")