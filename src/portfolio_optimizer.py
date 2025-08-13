# src/portfolio_optimizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns # Keep import for other methods or future use
# from pypfopt.plotting import plot_efficient_frontier # Not used directly as we custom plot
from pypfopt import DiscreteAllocation # Not strictly needed for this task's core output, but useful for real-world application

def calculate_expected_returns_and_covariance(df, tsla_forecast_price=None,
                                              tsla_current_price=None,
                                              tickers=['TSLA', 'BND', 'SPY'],
                                              risk_free_rate_annual=0.01):
    """
    Calculates expected returns and covariance matrix for portfolio optimization.

    Args:
        df (pd.DataFrame): Processed DataFrame containing '_Daily_Return' columns.
        tsla_forecast_price (float): The forecasted price for TSLA from Task 3.
        tsla_current_price (float): The latest actual price for TSLA from historical data.
        tickers (list): List of tickers to include in the portfolio.
        risk_free_rate_annual (float): Annual risk-free rate.

    Returns:
        tuple: (expected_annual_returns, sample_covariance_matrix)
    """
    print("\n--- Calculating Expected Returns and Covariance ---")

    # Get daily returns for all assets, ensuring a fresh copy
    daily_returns_df = df[[f'{ticker}_Daily_Return' for ticker in tickers]].copy()
    daily_returns_df.columns = tickers # Rename columns to just tickers for easier use

    # Apply the most robust cleaning steps directly to the DataFrame.
    # This block is *crucial* and should make the DataFrame numerically clean.
    daily_returns_df = daily_returns_df.apply(pd.to_numeric, errors='coerce') # Coerce any non-numeric to NaN
    daily_returns_df.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace explicit infinities
    daily_returns_df.dropna(inplace=True) # Drop any row with any NaN

    if daily_returns_df.empty:
        raise ValueError("Daily returns DataFrame is empty after comprehensive cleaning. Cannot perform portfolio optimization.")

    # Explicitly convert to float64, as sklearn often prefers this and it can mitigate subtle precision issues.
    daily_returns_df = daily_returns_df.astype(np.float64)

    print("\n--- Debug: Daily Returns Data before PyPortfolioOpt ---")
    print("Shape (DataFrame):", daily_returns_df.shape)
    print("NaNs after final cleaning:\n", daily_returns_df.isnull().sum())
    print("Are all values finite?", np.isfinite(daily_returns_df.values).all())
    print("DataFrame Dtype:", daily_returns_df.dtypes.unique()) # Add dtype check
    print("---------------------------------------")

    # FIX: Calculate historical average daily returns directly using pandas and numpy,
    # bypassing pypfopt.expected_returns.mean_historical_return to avoid its internal warnings/checks.
    daily_mean_returns = daily_returns_df.mean()
    expected_annual_returns = daily_mean_returns * 252 # Annualize daily mean returns

    # Override TSLA's expected return with the forecast if provided
    if tsla_forecast_price is not None and tsla_current_price is not None and tsla_current_price != 0:
        tsla_expected_future_return_period = (tsla_forecast_price - tsla_current_price) / tsla_current_price
        expected_annual_returns['TSLA'] = tsla_expected_future_return_period
        print(f"TSLA's expected annual return overridden with forecast: {expected_annual_returns['TSLA']:.2%}")
    else:
        print(f"TSLA's expected annual return using historical average: {expected_annual_returns['TSLA']:.2%}")

    # FIX: Use Ledoit-Wolf shrinkage with a try-except block.
    # If it fails, fall back to simple sample covariance.
    try:
        sample_covariance_matrix_annual = risk_models.risk_matrix(daily_returns_df, method="ledoit_wolf", frequency=252)
        print("Using Ledoit-Wolf shrinkage for covariance matrix.")
    except ValueError as e:
        print(f"Warning: Ledoit-Wolf shrinkage failed with error: {e}. Falling back to simple sample covariance.")
        # Fallback to simple sample covariance
        sample_covariance_matrix_annual = daily_returns_df.cov() * 252

    # Final check for NaN/Inf in the calculated matrices (should be clean now)
    if expected_annual_returns.isnull().any() or not np.isfinite(expected_annual_returns).all():
        raise ValueError("Expected annual returns (mu) contains NaN or Inf after calculation. This indicates a problem.")
    if sample_covariance_matrix_annual.isnull().any().any() or not np.isfinite(sample_covariance_matrix_annual).all().all():
        raise ValueError("Covariance matrix contains NaN or Inf after calculation. This indicates a problem.")


    print("Expected Annual Returns:")
    print(expected_annual_returns.apply(lambda x: f"{x:.2%}"))
    print("\nAnnualized Covariance Matrix:")
    print(sample_covariance_matrix_annual)

    return expected_annual_returns, sample_covariance_matrix_annual


def optimize_portfolio_mpt(expected_returns_vec, cov_matrix, risk_free_rate=0.01):
    """
    Runs MPT optimization to find Efficient Frontier, Max Sharpe, and Min Volatility portfolios.

    Args:
        expected_returns_vec (pd.Series): Annualized expected returns for assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix of asset returns.
        risk_free_rate (float): Annual risk-free rate for Sharpe Ratio calculation.

    Returns:
        tuple: (max_sharpe_weights, min_vol_weights, portfolios_df, max_sharpe_perf, min_vol_perf)
    """
    print("\n--- Running MPT Portfolio Optimization ---")

    # Final checks for NaN/Inf in the inputs to EfficientFrontier (should be clean now from previous function)
    if expected_returns_vec.isnull().any() or not np.isfinite(expected_returns_vec).all():
        raise ValueError("Expected returns vector contains NaN or Inf before optimization.")
    if cov_matrix.isnull().any().any() or not np.isfinite(cov_matrix).all().all():
        raise ValueError("Covariance matrix contains NaN or Inf before optimization.")


    ef = EfficientFrontier(expected_returns_vec, cov_matrix, weight_bounds=(0, 1))

    # Max Sharpe Ratio Portfolio
    max_sharpe_weights_raw = ef.max_sharpe(risk_free_rate=risk_free_rate)
    max_sharpe_weights = ef.clean_weights() # Clean weights to remove tiny values
    max_sharpe_performance = ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
    print("\nMaximum Sharpe Ratio Portfolio:")
    print("Weights:", {k: f"{v:.2%}" for k, v in max_sharpe_weights.items()})
    # ef.portfolio_performance already prints these, but for clarity:
    print(f"Expected annual return: {max_sharpe_performance[0]:.2%}")
    print(f"Annual volatility: {max_sharpe_performance[1]:.2%}")
    print(f"Sharpe Ratio: {max_sharpe_performance[2]:.4f}")

    # Minimum Volatility Portfolio
    ef_min_vol = EfficientFrontier(expected_returns_vec, cov_matrix, weight_bounds=(0, 1)) # Re-instantiate for min_vol
    min_vol_weights_raw = ef_min_vol.min_volatility()
    min_vol_weights = ef_min_vol.clean_weights()
    min_vol_performance = ef_min_vol.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
    print("\nMinimum Volatility Portfolio:")
    print("Weights:", {k: f"{v:.2%}" for k, v in min_vol_weights.items()})
    print(f"Expected annual return: {min_vol_performance[0]:.2%}")
    print(f"Annual volatility: {min_vol_performance[1]:.2%}")
    print(f"Sharpe Ratio: {min_vol_performance[2]:.4f}")

    # Generate Efficient Frontier points for plotting
    num_portfolios = 10000 # Increase for denser scatter plot
    results = np.zeros((3 + len(expected_returns_vec), num_portfolios)) # Return, Volatility, Sharpe, Weights

    # Randomly generate portfolios to plot the scatter
    for i in range(num_portfolios):
        weights = np.random.random(len(expected_returns_vec))
        weights /= np.sum(weights)
        ret = np.sum(weights * expected_returns_vec)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol

        results[0,i] = ret
        results[1,i] = vol
        results[2,i] = sharpe
        for j, w in enumerate(weights):
            results[j+3,i] = w

    portfolios_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'] + list(expected_returns_vec.index))


    print("\nPortfolio optimization complete.")
    return max_sharpe_weights, min_vol_weights, portfolios_df, max_sharpe_performance, min_vol_performance


def plot_efficient_frontier_and_portfolios(ef_data, max_sharpe_perf, min_vol_perf,
                                          title="Efficient Frontier with Optimal Portfolios"):
    """
    Plots the Efficient Frontier and marks key portfolios.

    Args:
        ef_data (pd.DataFrame): DataFrame containing portfolio returns, volatilities, and Sharpe ratios.
        max_sharpe_perf (tuple): (return, volatility, sharpe) for max Sharpe portfolio.
        min_vol_perf (tuple): (return, volatility, sharpe) for min volatility portfolio.
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(ef_data['Volatility'], ef_data['Return'], c=ef_data['Sharpe Ratio'],
                cmap='viridis', marker='o', s=10, alpha=0.6)
    plt.colorbar(label='Sharpe Ratio')
    plt.title(title)
    plt.xlabel('Volatility (Annualized)')
    plt.ylabel('Expected Return (Annualized)')
    plt.grid(True)

    # Plot Max Sharpe Ratio portfolio
    plt.scatter(max_sharpe_perf[1], max_sharpe_perf[0], marker='*', color='red', s=500, label='Max Sharpe Ratio')

    # Plot Minimum Volatility portfolio
    plt.scatter(min_vol_perf[1], min_vol_perf[0], marker='*', color='blue', s=500, label='Min Volatility')

    plt.legend()
    plt.tight_layout()
    plt.show()

def recommend_portfolio(max_sharpe_weights, min_vol_weights, max_sharpe_perf, min_vol_perf,
                        client_risk_profile="balanced"):
    """
    Recommends an optimal portfolio based on a hypothetical client risk profile.

    Args:
        max_sharpe_weights (dict): Weights of the max Sharpe portfolio.
        min_vol_weights (dict): Weights of the min volatility portfolio.
        max_sharpe_perf (tuple): Performance metrics of max Sharpe portfolio.
        min_vol_perf (tuple): Performance metrics of min volatility portfolio.
        client_risk_profile (str): 'conservative', 'balanced', or 'aggressive'.

    Returns:
        dict: Recommended portfolio weights and performance.
    """
    print("\n--- Portfolio Recommendation ---")
    recommended_portfolio = {}

    if client_risk_profile.lower() == "conservative":
        print("Prioritizing lower risk: Recommending Minimum Volatility Portfolio.")
        recommended_portfolio['weights'] = min_vol_weights
        recommended_portfolio['performance'] = {
            'return': min_vol_perf[0],
            'volatility': min_vol_perf[1],
            'sharpe_ratio': min_vol_perf[2]
        }
    elif client_risk_profile.lower() == "aggressive":
        print("Prioritizing higher risk-adjusted returns: Recommending Maximum Sharpe Ratio Portfolio.")
        recommended_portfolio['weights'] = max_sharpe_weights
        recommended_portfolio['performance'] = {
            'return': max_sharpe_perf[0],
            'volatility': max_sharpe_perf[1],
            'sharpe_ratio': max_sharpe_perf[2]
        }
    else: # Default to balanced, often the Max Sharpe
        print("Assuming a balanced risk profile: Recommending Maximum Sharpe Ratio Portfolio.")
        recommended_portfolio['weights'] = max_sharpe_weights
        recommended_portfolio['performance'] = {
            'return': max_sharpe_perf[0],
            'volatility': max_sharpe_perf[1],
            'sharpe_ratio': max_sharpe_perf[2]
        }

    print("\nFinal Recommended Portfolio:")
    print("Optimal Weights:")
    for ticker, weight in recommended_portfolio['weights'].items():
        print(f"  {ticker}: {weight:.2%}")
    print(f"Expected Annual Return: {recommended_portfolio['performance']['return']:.2%}")
    print(f"Annual Volatility: {recommended_portfolio['performance']['volatility']:.2%}")
    print(f"Sharpe Ratio: {recommended_portfolio['performance']['sharpe_ratio']:.4f}")

    return recommended_portfolio