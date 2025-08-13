# src/backtester.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# No direct PyPortfolioOpt imports here as we're doing a manual simulation for clarity

def run_backtest(df, recommended_weights, benchmark_weights,
                 backtest_start_date, backtest_end_date,
                 tickers=['TSLA', 'BND', 'SPY'],
                 risk_free_rate_annual=0.01):
    """
    Simulates portfolio performance over a historical period and compares it to a benchmark.

    Args:
        df (pd.DataFrame): Processed DataFrame with daily returns.
        recommended_weights (dict): Weights for the strategy portfolio (e.g., from Task 4).
        benchmark_weights (dict): Weights for the benchmark portfolio.
        backtest_start_date (str): Start date of the backtesting period.
        backtest_end_date (str): End date of the backtesting period.
        tickers (list): List of tickers in the portfolio.
        risk_free_rate_annual (float): Annual risk-free rate for Sharpe calculation.

    Returns:
        tuple: (strategy_cumulative_returns, benchmark_cumulative_returns,
                strategy_sharpe, benchmark_sharpe,
                strategy_total_return, benchmark_total_return)
                Returns None for all if data is empty.
    """
    print("\n--- Running Strategy Backtest ---")

    # Prepare daily returns for backtesting
    daily_returns_df = df[[f'{ticker}_Daily_Return' for ticker in tickers]].dropna()
    daily_returns_df.columns = tickers # Rename columns to just tickers

    backtest_data = daily_returns_df.loc[backtest_start_date:backtest_end_date].copy()

    if backtest_data.empty:
        print(f"No data for backtesting in the period {backtest_start_date} to {backtest_end_date}.")
        return None, None, None, None, None, None

    # Convert weights to pandas Series for element-wise multiplication
    # Ensure weights cover all tickers in `backtest_data` and align order
    rec_w_series = pd.Series(recommended_weights).reindex(tickers).fillna(0) # Reindex to ensure order and fill any missing
    bench_w_series = pd.Series(benchmark_weights).reindex(tickers).fillna(0)

    # Calculate daily returns of the strategy portfolio
    # (Simplified: assumes weights are held constant over the period for demonstration)
    # This is a fixed-weight backtest. For rebalancing, a loop would be needed.
    strategy_daily_returns = backtest_data.dot(rec_w_series)
    benchmark_daily_returns = backtest_data.dot(bench_w_series)

    # Calculate cumulative returns
    # Add 1 to returns to calculate compounding, then subtract 1 at the end
    strategy_cumulative_returns = (1 + strategy_daily_returns).cumprod()
    benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod()

    # Add an initial value of 1.0 (representing 0% cumulative return at start) for plotting
    # FIX: Use pd.concat() instead of .append()
    start_date_idx = backtest_data.index.min() - pd.Timedelta(days=1)
    if start_date_idx not in strategy_cumulative_returns.index: # Prevent adding if it already exists due to edge cases
        strategy_cumulative_returns = pd.concat([pd.Series([1.0], index=[start_date_idx]), strategy_cumulative_returns]).sort_index()
        benchmark_cumulative_returns = pd.concat([pd.Series([1.0], index=[start_date_idx]), benchmark_cumulative_returns]).sort_index()


    # Performance Metrics
    trading_days_per_year = 252

    # Strategy Performance
    strategy_mean_daily_return = strategy_daily_returns.mean()
    strategy_annual_return = strategy_mean_daily_return * trading_days_per_year
    strategy_annual_std = strategy_daily_returns.std() * np.sqrt(trading_days_per_year)
    strategy_sharpe = (strategy_annual_return - risk_free_rate_annual) / strategy_annual_std if strategy_annual_std != 0 else np.nan
    strategy_total_return = strategy_cumulative_returns.iloc[-1] - 1 # Subtract 1 because we added 1 for compounding

    # Benchmark Performance
    benchmark_mean_daily_return = benchmark_daily_returns.mean()
    benchmark_annual_return = benchmark_mean_daily_return * trading_days_per_year
    benchmark_annual_std = benchmark_daily_returns.std() * np.sqrt(trading_days_per_year)
    benchmark_sharpe = (benchmark_annual_return - risk_free_rate_annual) / benchmark_annual_std if benchmark_annual_std != 0 else np.nan
    benchmark_total_return = benchmark_cumulative_returns.iloc[-1] - 1 # Subtract 1

    print("\nBacktest Performance Summary:")
    print(f"Strategy Total Return: {strategy_total_return:.2%}")
    print(f"Benchmark Total Return: {benchmark_total_return:.2%}")
    print(f"Strategy Annualized Sharpe Ratio: {strategy_sharpe:.4f}")
    print(f"Benchmark Annualized Sharpe Ratio: {benchmark_sharpe:.4f}")

    # Plot cumulative returns (multiply by 100 for percentage)
    plt.figure(figsize=(14, 7))
    plt.plot(strategy_cumulative_returns.index, strategy_cumulative_returns.values * 100 - 100, label='Strategy Portfolio', color='green') # Adjust for 0% start
    plt.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns.values * 100 - 100, label='Benchmark Portfolio (60% SPY / 40% BND)', color='purple') # Adjust for 0% start
    plt.title('Cumulative Returns: Strategy vs. Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (strategy_cumulative_returns, benchmark_cumulative_returns,
            strategy_sharpe, benchmark_sharpe,
            strategy_total_return, benchmark_total_return)