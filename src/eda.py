# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def plot_closing_prices(df, title='Adjusted Closing Prices Over Time'):
    """Plots the adjusted closing prices for all assets in the DataFrame."""
    plt.figure(figsize=(15, 7))
    # Select columns ending with '_Adj_Close'
    adj_close_cols = [col for col in df.columns if col.endswith('_Adj_Close')]
    if not adj_close_cols:
        print("No '_Adj_Close' columns found for plotting.")
        return
    df[adj_close_cols].plot(ax=plt.gca())
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price ($)')
    plt.legend([col.replace('_Adj_Close', '') for col in adj_close_cols], title='Ticker')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_daily_returns(df, title='Daily Percentage Change Over Time'):
    """Plots the daily percentage change for all assets."""
    plt.figure(figsize=(15, 7))
    # Select columns ending with '_Daily_Return'
    daily_return_cols = [col for col in df.columns if col.endswith('_Daily_Return')]
    if not daily_return_cols:
        print("No '_Daily_Return' columns found for plotting.")
        return
    df[daily_return_cols].plot(alpha=0.8, ax=plt.gca())
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.legend([col.replace('_Daily_Return', '') for col in daily_return_cols], title='Ticker')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_returns_distribution(df, ticker='TSLA', title='Distribution of Daily Returns'):
    """Plots a histogram and KDE for the daily returns of a specific ticker."""
    col_name = f'{ticker}_Daily_Return'
    if col_name not in df.columns:
        print(f"'{col_name}' not found in DataFrame.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col_name].dropna(), bins=50, kde=True)
    plt.title(f'{title} for {ticker}')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_rolling_volatility(df, window=20, title='Rolling Volatility (Standard Deviation)'):
    """Plots the rolling standard deviation of daily returns for all assets."""
    plt.figure(figsize=(15, 7))
    rolling_vol_cols = [col for col in df.columns if col.endswith('_Rolling_Vol_20D')]
    if not rolling_vol_cols:
        print("No '_Rolling_Vol_20D' columns found for plotting. Ensure data_preprocessing created them.")
        # Fallback if not precomputed, compute here (less efficient for EDA.py only)
        daily_return_cols = [col for col in df.columns if col.endswith('_Daily_Return')]
        if daily_return_cols:
            df[daily_return_cols].rolling(window=window).std().plot(ax=plt.gca())
            plt.legend([col.replace('_Daily_Return', '') for col in daily_return_cols], title='Ticker')
        else:
            print("No daily return columns to compute rolling volatility.")
            return

    else:
        df[rolling_vol_cols].plot(ax=plt.gca())
        plt.legend([col.replace('_Rolling_Vol_20D', '') for col in rolling_vol_cols], title='Ticker')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Standard Deviation')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def detect_outliers_and_print(df, ticker='TSLA', return_column_suffix='_Daily_Return', quantile_lower=0.005, quantile_upper=0.995):
    """Detects and prints extreme outliers for a given ticker's returns."""
    col_name = f'{ticker}{return_column_suffix}'
    if col_name not in df.columns:
        print(f"'{col_name}' not found in DataFrame for outlier detection.")
        return

    returns = df[col_name].dropna()
    if returns.empty:
        print(f"No returns data for {ticker} to detect outliers.")
        return

    outlier_threshold_upper = returns.quantile(quantile_upper)
    outlier_threshold_lower = returns.quantile(quantile_lower)

    print(f"\n{ticker} Daily Return Outlier Thresholds ({quantile_lower*100:.1f}% and {quantile_upper*100:.1f}%):")
    print(f"  Lower: {outlier_threshold_lower:.4f}")
    print(f"  Upper: {outlier_threshold_upper:.4f}")

    outliers_upper = returns[returns > outlier_threshold_upper]
    outliers_lower = returns[returns < outlier_threshold_lower]

    print(f"\nTop 5 positive outlier returns for {ticker}:")
    print(outliers_upper.sort_values(ascending=False).head())
    print(f"\nTop 5 negative outlier returns for {ticker}:")
    print(outliers_lower.sort_values().head())

def run_adf_test(series, name):
    """Performs and prints results of the Augmented Dickey-Fuller test."""
    print(f"\n--- ADF Test for {name} ---")
    if series.isnull().any():
        series = series.dropna()
        print(f"Warning: Dropped NaNs for ADF test on {name}.")
    if series.empty:
        print(f"Cannot run ADF test, series '{name}' is empty after dropping NaNs.")
        return

    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'P-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')

    if result[1] > 0.05:
        print(f"Conclusion: '{name}' is likely non-stationary. Differencing may be required.")
    else:
        print(f"Conclusion: '{name}' is likely stationary.")
    print("-" * (len(name) + 18))


def calculate_var(returns_series, confidence_level=0.95):
    """Calculates Historical Value at Risk (VaR)."""
    if returns_series.empty:
        print(f"No data for VaR calculation.")
        return np.nan
    # VaR is typically reported as a positive value for loss
    # (1 - confidence_level) * 100 gives the percentile, e.g., 5 for 95% confidence
    var = np.percentile(returns_series, (1 - confidence_level) * 100)
    return abs(var) # Return as a positive loss value

def calculate_sharpe_ratio(daily_returns_series, risk_free_rate_annual=0.01, trading_days_per_year=252):
    """Calculates the annualized Sharpe Ratio."""
    if daily_returns_series.empty:
        print(f"No data for Sharpe Ratio calculation.")
        return np.nan
    annualized_return = daily_returns_series.mean() * trading_days_per_year
    annualized_std = daily_returns_series.std() * np.sqrt(trading_days_per_year)
    if annualized_std == 0:
        return np.nan # Avoid division by zero
    sharpe = (annualized_return - risk_free_rate_annual) / annualized_std
    return sharpe

def calculate_and_print_risk_metrics(df, tickers, risk_free_rate_annual=0.01):
    """Calculates and prints VaR and Sharpe Ratio for specified tickers."""
    print("\n--- Risk Metrics ---")
    results = {}
    for ticker in tickers:
        returns_col = f'{ticker}_Daily_Return'
        if returns_col not in df.columns:
            print(f"Warning: {returns_col} not found for {ticker}.")
            continue

        returns = df[returns_col].dropna()
        if returns.empty:
            print(f"No valid returns data for {ticker} for risk metric calculation.")
            continue

        var_95 = calculate_var(returns, 0.95)
        var_99 = calculate_var(returns, 0.99)
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate_annual)

        results[ticker] = {
            'VaR_95': var_95,
            'VaR_99': var_99,
            'Sharpe_Ratio': sharpe
        }
        print(f"\n{ticker}:")
        print(f"  VaR (95% confidence): {var_95:.4f} or {var_95*100:.2f}%")
        print(f"  VaR (99% confidence): {var_99:.4f} or {var_99*100:.2f}%")
        print(f"  Sharpe Ratio (Annualized, RFR={risk_free_rate_annual*100}%): {sharpe:.4f}")
    print("--------------------")
    return results

if __name__ == '__main__':
    # Example usage
    try:
        processed_df = pd.read_csv('data/processed/processed_financial_data.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Run data_preprocessing.py first to create data/processed/processed_financial_data.csv")
        exit()

    print("\n--- Running EDA ---")
    tickers_list = ['TSLA', 'BND', 'SPY']

    plot_closing_prices(processed_df)
    plot_daily_returns(processed_df)
    plot_returns_distribution(processed_df, ticker='TSLA')
    plot_rolling_volatility(processed_df)
    detect_outliers_and_print(processed_df, ticker='TSLA')

    # Run ADF tests
    for ticker in tickers_list:
        run_adf_test(processed_df[f'{ticker}_Adj_Close'], f'{ticker} Adj Close Price')
        run_adf_test(processed_df[f'{ticker}_Daily_Return'], f'{ticker} Daily Returns')

    # Calculate and print risk metrics
    calculate_and_print_risk_metrics(processed_df, tickers_list)