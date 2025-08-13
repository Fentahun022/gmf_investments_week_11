# main.py
import pandas as pd
import numpy as np
import os
import joblib # For loading the scaler
import tensorflow as tf # Required for GPU memory management

# Ensure tensorflow does not pre-allocate all GPU memory
# This is important especially if running on a machine with a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow: GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"TensorFlow GPU setup error: {e}")

# --- Configuration ---
TICKERS = ['TSLA', 'BND', 'SPY']
START_DATE = '2015-07-01'
# YFinance end date is exclusive, so to get data *up to* July 31, 2025, set end_date to Aug 1, 2025
END_DATE = '2025-08-01'
TSLA_FORECAST_TARGET_COLUMN = 'TSLA_Adj_Close' # The column we'll forecast for TSLA

DATA_RAW_PATH = 'data/raw/financial_data.csv'
DATA_PROCESSED_PATH = 'data/processed/processed_financial_data.csv'
SCALER_PATH = 'data/processed/tsla_close_scaler.pkl'

# Task 2 & 3 Configuration
TRAIN_END_DATE_STR = '2023-12-31' # End of training for forecasting models
TEST_START_DATE_STR = '2024-01-01' # Start of testing for forecasting models
FORECAST_HORIZON_MONTHS = 9 # For Task 3, let's choose 9 months for demonstration
TRADING_DAYS_PER_MONTH = 21 # Approximate trading days in a month
FORECAST_STEPS = FORECAST_HORIZON_MONTHS * TRADING_DAYS_PER_MONTH

# Task 4 & 5 Configuration
RISK_FREE_RATE = 0.01 # 1% annual risk-free rate for Sharpe Ratio
BACKTEST_START_DATE = '2024-08-01'
BACKTEST_END_DATE = '2025-07-31' # Inclusive for backtest data
BENCHMARK_WEIGHTS = {'SPY': 0.60, 'BND': 0.40, 'TSLA': 0.00} # Static 60/40 SPY/BND

# --- Import custom modules ---
from src.data_ingestion import fetch_financial_data
from src.data_preprocessing import clean_and_preprocess_data, scale_data
from src.eda import (
    plot_closing_prices, plot_daily_returns, plot_returns_distribution,
    plot_rolling_volatility, detect_outliers_and_print, run_adf_test,
    calculate_and_print_risk_metrics
)
from src.forecasting_models import (
    train_arima_model, forecast_arima, build_lstm_model,
    create_sequences, train_lstm_model, forecast_lstm,
    evaluate_model, plot_forecast
)
from src.portfolio_optimizer import (
    calculate_expected_returns_and_covariance,
    optimize_portfolio_mpt,
    plot_efficient_frontier_and_portfolios,
    recommend_portfolio
)
from src.backtester import run_backtest

# --- Task 1 Function ---
def run_task1():
    print("--- Running Task 1: Preprocess and Explore the Data ---")

    # 1.1 Data Extraction
    raw_data_df = fetch_financial_data(TICKERS, START_DATE, END_DATE)
    if raw_data_df is None:
        print("Failed to fetch raw data. Exiting Task 1.")
        return None

    # Save raw data
    os.makedirs(os.path.dirname(DATA_RAW_PATH), exist_ok=True)
    raw_data_df.to_csv(DATA_RAW_PATH, header=True) # Ensure multi-index header is saved correctly

    # 1.2 Data Cleaning and Understanding
    processed_df = clean_and_preprocess_data(raw_data_df)
    if processed_df.empty:
        print("Failed to preprocess data. Exiting Task 1.")
        return None

    # Save processed data
    os.makedirs(os.path.dirname(DATA_PROCESSED_PATH), exist_ok=True)
    processed_df.to_csv(DATA_PROCESSED_PATH)
    print(f"Processed data saved to {DATA_PROCESSED_PATH}")

    # Scale TSLA Adj Close for later use in Deep Learning models
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
        # Check if the column exists before running ADF test
        if f'{ticker}_Adj_Close' in processed_df.columns:
            run_adf_test(processed_df[f'{ticker}_Adj_Close'], f'{ticker} Adj Close Price')
        else:
            print(f"Warning: {ticker} Adj Close Price column not found for ADF test.")
        if f'{ticker}_Daily_Return' in processed_df.columns:
            run_adf_test(processed_df[f'{ticker}_Daily_Return'], f'{ticker} Daily Returns')
        else:
            print(f"Warning: {ticker} Daily Return column not found for ADF test.")


    # 1.5 Analyze Volatility & Risk Metrics
    calculate_and_print_risk_metrics(processed_df, TICKERS)

    print("\n--- Task 1 Completed ---")
    return processed_df # Return processed DataFrame for subsequent tasks

# --- Task 2 Function ---
def run_task2(processed_df):
    print("\n--- Running Task 2: Develop Time Series Forecasting Models ---")

    if processed_df is None or processed_df.empty:
        print("Processed data not available. Please run Task 1 first.")
        return None, None, None, None # Return None for models and scaler

    tsla_close_data = processed_df[TSLA_FORECAST_TARGET_COLUMN]

    # Chronological Split
    train_data = tsla_close_data[tsla_close_data.index <= TRAIN_END_DATE_STR]
    test_data = tsla_close_data[tsla_close_data.index >= TEST_START_DATE_STR]

    if train_data.empty or test_data.empty:
        print(f"Error: Train or test data is empty. Check dates. Train end: {TRAIN_END_DATE_STR}, Test start: {TEST_START_DATE_STR}")
        print(f"TSLA data start: {tsla_close_data.index.min()}, end: {tsla_close_data.index.max()}")
        return None, None, None, None

    print(f"Train data period: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Test data period: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Train data points: {len(train_data)}, Test data points: {len(test_data)}")

    # Load the scaler (fitted on the whole TSLA_Adj_Close series during Task 1)
    try:
        tsla_scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"Scaler not found at {SCALER_PATH}. Run Task 1 first.")
        return None, None, None, None

    # Scale data for LSTM
    # Ensure train_data and test_data are passed as numpy arrays for scaling
    scaled_train_data = tsla_scaler.transform(train_data.values.reshape(-1, 1))
    scaled_test_data = tsla_scaler.transform(test_data.values.reshape(-1, 1))

    # --- ARIMA Model ---
    arima_model_fitted = train_arima_model(train_data.astype(float), auto_fit=True)

    # Forecast with ARIMA for the test period
    arima_forecast_steps = len(test_data)
    arima_predictions_values, arima_conf_int_values = forecast_arima(arima_model_fitted, arima_forecast_steps)
    arima_predictions_series = pd.Series(arima_predictions_values, index=test_data.index)
    arima_conf_int_df = pd.DataFrame(arima_conf_int_values, index=test_data.index, columns=['lower', 'upper'])


    # Evaluate ARIMA
    print("\nARIMA Model Evaluation on Test Set:")
    arima_metrics = evaluate_model(test_data.values, arima_predictions_series.values, "ARIMA")

    # Plot ARIMA forecast vs actual
    # Corrected: Use pd.concat and pass actual_test_data explicitly
    plot_forecast(historical_data=train_data.tail(200), # Show last part of train data
                  forecast_data=arima_predictions_series,
                  title="ARIMA Forecast vs Actual (TSLA)",
                  confidence_intervals=arima_conf_int_df,
                  actual_test_data=test_data) # Pass the actual test data for comparison


    # --- LSTM Model ---
    TIME_STEPS = 60 # Number of previous days to consider for prediction
    X_train_lstm, y_train_lstm = create_sequences(scaled_train_data, TIME_STEPS)
    X_test_lstm, y_test_lstm = create_sequences(scaled_test_data, TIME_STEPS)

    # Reshape for LSTM [samples, time_steps, features]
    X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
    X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

    lstm_model = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    lstm_model, history = train_lstm_model(lstm_model, X_train_lstm, y_train_lstm,
                                            epochs=100, batch_size=64, validation_split=0.2)

    # Predict on test set
    lstm_scaled_predictions = lstm_model.predict(X_test_lstm)
    lstm_predictions = tsla_scaler.inverse_transform(lstm_scaled_predictions).flatten()

    # Align LSTM predictions with test_data (adjusting for TIME_STEPS offset)
    if len(test_data) > TIME_STEPS:
        lstm_predictions_series = pd.Series(lstm_predictions, index=test_data.index[TIME_STEPS:])
        lstm_test_actual = test_data.iloc[TIME_STEPS:].values
    else:
        print("Test data too short for LSTM time steps. Skipping LSTM evaluation plot.")
        lstm_predictions_series = pd.Series([]) # Empty series
        lstm_test_actual = np.array([]) # Empty array


    # Evaluate LSTM (only if there are actual predictions)
    if not lstm_predictions_series.empty:
        print("\nLSTM Model Evaluation on Test Set:")
        lstm_metrics = evaluate_model(lstm_test_actual, lstm_predictions_series.values, "LSTM")

        # Plot LSTM forecast vs actual
        # Corrected: Use pd.concat for historical data and pass actual_test_data explicitly
        # Historical context for LSTM plot will be the train data up to the point of prediction start
        # The test_data part passed to `actual_test_data` will cover the actual values for comparison.
        plot_forecast(historical_data=train_data.tail(200),
                      forecast_data=lstm_predictions_series,
                      title="LSTM Forecast vs Actual (TSLA)",
                      actual_test_data=test_data.iloc[TIME_STEPS:])
    else:
        lstm_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}


    # --- Compare Models ---
    print("\n--- Model Comparison ---")
    print(f"ARIMA MAE: {arima_metrics['MAE']:.4f}, RMSE: {arima_metrics['RMSE']:.4f}, MAPE: {arima_metrics['MAPE']:.4f}%")
    print(f"LSTM MAE: {lstm_metrics['MAE']:.4f}, RMSE: {lstm_metrics['RMSE']:.4f}, MAPE: {lstm_metrics['MAPE']:.4f}%")

    best_model = None
    best_model_type = "None"

    if not np.isnan(arima_metrics['RMSE']) and (np.isnan(lstm_metrics['RMSE']) or arima_metrics['RMSE'] < lstm_metrics['RMSE']):
        print("ARIMA performed better in terms of RMSE (or LSTM data was insufficient).")
        best_model = arima_model_fitted
        best_model_type = "ARIMA"
    elif not np.isnan(lstm_metrics['RMSE']):
        print("LSTM performed better in terms of RMSE.")
        best_model = lstm_model
        best_model_type = "LSTM"
    else:
        print("Could not determine best model due to missing RMSE values.")

    print(f"\nRecommended model for forecasting: {best_model_type}")

    print("\n--- Task 2 Completed ---")
    return best_model, best_model_type, tsla_scaler, TIME_STEPS # Pass TIME_STEPS for LSTM for forecasting

# --- Task 3 Function ---
def run_task3(processed_df, best_model, best_model_type, tsla_scaler, time_steps_lstm):
    print("\n--- Running Task 3: Forecast Future Market Trends (TSLA) ---")

    if best_model is None:
        print("No best model provided. Exiting Task 3.")
        return None

    tsla_close_data = processed_df[TSLA_FORECAST_TARGET_COLUMN]

    # Use the entire available data for final model training before forecasting
    final_train_data = tsla_close_data.copy()

    forecast_values = None
    conf_int = None
    forecast_dates = None

    # Re-train the best model on the full dataset before generating the future forecast
    print(f"\nRe-training {best_model_type} model on full dataset for final forecasting...")
    if best_model_type == "ARIMA":
        # Re-fit auto_arima on the full data. This will re-evaluate best parameters for the whole history.
        final_arima_model = train_arima_model(final_train_data.astype(float), auto_fit=True)
        forecast_values, conf_int_array = forecast_arima(final_arima_model, FORECAST_STEPS)

        # Generate forecast dates starting from the day after the last historical data point
        forecast_start_date = final_train_data.index[-1] + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(start=forecast_start_date, periods=FORECAST_STEPS, freq='B') # Business day frequency
        conf_int = pd.DataFrame(conf_int_array, index=forecast_dates, columns=['lower', 'upper'])

    elif best_model_type == "LSTM":
        # Re-scale the entire historical data for LSTM
        scaled_final_train_data, _ = scale_data(final_train_data, tsla_scaler) # Use the existing scaler
        X_final_lstm, y_final_lstm = create_sequences(scaled_final_train_data.values.reshape(-1, 1), time_steps_lstm)
        X_final_lstm = X_final_lstm.reshape(X_final_lstm.shape[0], X_final_lstm.shape[1], 1)

        # Build a new LSTM model (or re-train the existing one) and train on all data
        final_lstm_model = build_lstm_model((X_final_lstm.shape[1], X_final_lstm.shape[2]))
        final_lstm_model, _ = train_lstm_model(final_lstm_model, X_final_lstm, y_final_lstm,
                                                epochs=100, batch_size=64, validation_split=0.1)

        forecast_values = forecast_lstm(final_lstm_model, scaled_final_train_data.values, time_steps_lstm, tsla_scaler, FORECAST_STEPS)
        # LSTM confidence intervals are more complex, often done with Monte Carlo or Quantile Regression.
        # For simplicity in this challenge, we won't generate them for LSTM here.
        conf_int = None
        forecast_start_date = final_train_data.index[-1] + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(start=forecast_start_date, periods=FORECAST_STEPS, freq='B') # Business day frequency

    else:
        print(f"No valid model type ({best_model_type}) to generate future forecast.")
        return None

    if forecast_values is None or forecast_dates is None:
        print("Forecast generation failed.")
        return None

    forecast_series = pd.Series(forecast_values, index=forecast_dates)

    # 3.1 Visualize the forecast
    plot_forecast(historical_data=final_train_data.tail(200), # Show last part of historical data
                  forecast_data=forecast_series,
                  title=f"TSLA {best_model_type} Forecast for Next {FORECAST_HORIZON_MONTHS} Months",
                  confidence_intervals=conf_int,
                  actual_test_data=None) # No actual test data for future forecast plot

    # 3.2 Interpret the Results
    print(f"\n--- {best_model_type} Forecast Analysis (Next {FORECAST_HORIZON_MONTHS} Months) ---")
    print(f"Forecasted Price Range: ${forecast_series.min():.2f} to ${forecast_series.max():.2f}")
    print(f"Initial Forecast Value: ${forecast_series.iloc[0]:.2f}")
    print(f"Final Forecast Value: ${forecast_series.iloc[-1]:.2f}")

    # Trend Analysis
    initial_price = final_train_data.iloc[-1]
    final_forecast_price = forecast_series.iloc[-1]
    if final_forecast_price > initial_price:
        print(f"Overall Trend: Upward trend expected. Price change: {(final_forecast_price - initial_price)/initial_price:.2%}")
    elif final_forecast_price < initial_price:
        print(f"Overall Trend: Downward trend expected. Price change: {(final_forecast_price - initial_price)/initial_price:.2%}")
    else:
        print("Overall Trend: Relatively stable.")

    # Volatility and Risk (from confidence intervals, if available)
    if conf_int is not None:
        initial_interval_width = conf_int.iloc[0, 1] - conf_int.iloc[0, 0]
        final_interval_width = conf_int.iloc[-1, 1] - conf_int.iloc[-1, 0]
        print(f"\nConfidence Interval Analysis (95% CI):")
        print(f"  Initial CI Width: ${initial_interval_width:.2f}")
        print(f"  Final CI Width: ${final_interval_width:.2f}")
        if final_interval_width > initial_interval_width:
            print(f"  Confidence intervals widen over time, indicating increasing uncertainty for longer-term forecasts.")
        else:
            print(f"  Confidence interval width remains relatively stable.")
        print(f"  This implies higher uncertainty and risk for predictions further into the future.")
    else:
        print("\nConfidence interval analysis not performed for LSTM (more complex to derive).")
        print("However, general principle: uncertainty increases with longer forecast horizons.")


    # Market Opportunities and Risks (Simplified based on forecast)
    print("\nPotential Market Opportunities & Risks (Based on Forecast):")
    if final_forecast_price > initial_price:
        print(f"- Opportunity: Potential for capital appreciation in TSLA given the expected upward trend.")
    else:
        print(f"- Risk: Potential for capital depreciation in TSLA if the downward trend materializes.")
    print(f"- Risk: High volatility of TSLA (as seen in historical data) means actual prices could deviate significantly from forecast.")
    print("- Risk: Forecast accuracy diminishes over longer horizons, especially for volatile assets like TSLA.")
    print("- Opportunity: If the forecast suggests a strong upward movement, considering a larger allocation to TSLA within risk limits.")


    print("\n--- Task 3 Completed ---")
    # Return the final forecasted price for TSLA for Task 4.
    return initial_price, final_forecast_price # Pass both current and forecasted price for accurate calculation in MPT

# --- Task 4 Function ---
def run_task4(processed_df, tsla_current_price, final_tsla_forecast_price):
    print("\n--- Running Task 4: Optimize Portfolio Based on Forecast ---")

    if processed_df is None:
        print("Processed data not available. Exiting Task 4.")
        return None

    # 4.1 Forecasted Asset (TSLA) and Historical Assets (BND, SPY)
    expected_returns_annual, cov_matrix_annual = \
        calculate_expected_returns_and_covariance(
            processed_df,
            tsla_forecast_price=final_tsla_forecast_price,
            tsla_current_price=tsla_current_price,
            tickers=TICKERS,
            risk_free_rate_annual=RISK_FREE_RATE
        )

    # 4.2 Generate Efficient Frontier
    max_sharpe_weights, min_vol_weights, portfolios_df, max_sharpe_perf, min_vol_perf = \
        optimize_portfolio_mpt(expected_returns_annual, cov_matrix_annual, RISK_FREE_RATE)

    # 4.3 Plot the Efficient Frontier
    plot_efficient_frontier_and_portfolios(portfolios_df, max_sharpe_perf, min_vol_perf)

    # 4.4 Select and Recommend an Optimal Portfolio
    recommended_portfolio = recommend_portfolio(max_sharpe_weights, min_vol_weights,
                                                max_sharpe_perf, min_vol_perf,
                                                client_risk_profile="balanced") # Can be changed to "conservative" or "aggressive"

    print("\n--- Task 4 Completed ---")
    return recommended_portfolio

# --- Task 5 Function ---
def run_task5(processed_df, recommended_portfolio):
    print("\n--- Running Task 5: Strategy Backtesting ---")

    if processed_df is None or recommended_portfolio is None:
        print("Processed data or recommended portfolio not available. Exiting Task 5.")
        return

    # Extract recommended weights
    strategy_weights = recommended_portfolio['weights']

    # Run the backtest simulation
    (strategy_cum_returns, benchmark_cum_returns,
     strategy_sharpe, benchmark_sharpe,
     strategy_total_return, benchmark_total_return) = \
        run_backtest(processed_df, strategy_weights, BENCHMARK_WEIGHTS,
                     BACKTEST_START_DATE, BACKTEST_END_DATE,
                     tickers=TICKERS, risk_free_rate_annual=RISK_FREE_RATE)

    # Conclusion
    print("\n--- Backtest Conclusion ---")
    if strategy_total_return is not None and benchmark_total_return is not None:
        if strategy_total_return > benchmark_total_return:
            print(f"Conclusion: Our strategy significantly outperformed the benchmark in the backtesting period.")
            print(f"Strategy total return: {strategy_total_return:.2%}, Benchmark total return: {benchmark_total_return:.2%}.")
        else:
            print(f"Conclusion: Our strategy did not outperform the benchmark in this backtesting period.")
            print(f"Strategy total return: {strategy_total_return:.2%}, Benchmark total return: {benchmark_total_return:.2%}.")

    if strategy_sharpe is not None and benchmark_sharpe is not None:
        if strategy_sharpe > benchmark_sharpe:
            print(f"Our strategy also delivered a better risk-adjusted return (Sharpe Ratio: {strategy_sharpe:.4f} vs Benchmark: {benchmark_sharpe:.4f}).")
        else:
            print(f"Our strategy had a lower risk-adjusted return (Sharpe Ratio: {strategy_sharpe:.4f} vs Benchmark: {benchmark_sharpe:.4f}).")

    print("\nInitial backtest suggestions about the viability of our model-driven approach: ")
    print("- The performance of the strategy depends heavily on the accuracy of the TSLA forecast.")
    print("- A single backtest period might not be representative of all market conditions.")
    print("- This simplified backtest assumes fixed weights throughout the period. In a real scenario, periodic rebalancing (e.g., monthly/quarterly) would be performed based on updated forecasts and market conditions.")
    print("- Transaction costs, tax implications, and market liquidity were not considered in this simulation.")
    print("- This simulation serves as a promising starting point but requires more rigorous testing (e.g., across multiple out-of-sample periods, with different rebalancing rules, and transaction cost modeling).")


    print("\n--- Task 5 Completed ---")


# --- Main Execution Block ---
if __name__ == '__main__':
    print("Starting GMF Investments Financial Analysis Project...")

    # Run Task 1
    processed_data_df = run_task1()

    if processed_data_df is not None:
        # Run Task 2
        best_model, best_model_type, tsla_scaler, time_steps_lstm = run_task2(processed_data_df)

        if best_model is not None:
            # Run Task 3
            tsla_current_price_for_forecast, final_tsla_forecast_price = run_task3(processed_df=processed_data_df,
                                                                                    best_model=best_model,
                                                                                    best_model_type=best_model_type,
                                                                                    tsla_scaler=tsla_scaler,
                                                                                    time_steps_lstm=time_steps_lstm)

            if tsla_current_price_for_forecast is not None and final_tsla_forecast_price is not None:
                # Run Task 4
                recommended_portfolio = run_task4(processed_data_df, tsla_current_price_for_forecast, final_tsla_forecast_price)

                if recommended_portfolio is not None:
                    # Run Task 5
                    run_task5(processed_data_df, recommended_portfolio)

    print("\nProject execution finished.")