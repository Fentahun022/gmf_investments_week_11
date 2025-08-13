# src/forecasting_models.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib # For saving/loading scalers

def create_sequences(data, time_steps):
    """
    Creates sequences for LSTM model training.

    Args:
        data (np.array): Input data (e.g., scaled TSLA Adj Close prices).
        time_steps (int): Number of previous time steps to use as input features.

    Returns:
        tuple: (X, y) where X is the 3D array of sequences and y is the 2D array of targets.
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def train_arima_model(train_data, order=None, seasonal_order=None, auto_fit=True):
    """
    Trains an ARIMA or SARIMA model.

    Args:
        train_data (pd.Series): Training data for the model.
        order (tuple): (p, d, q) order for ARIMA.
        seasonal_order (tuple): (P, D, Q, S) order for SARIMA.
        auto_fit (bool): If True, uses auto_arima to find best parameters.

    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper or pmdarima.arima.arima.ARIMA: Fitted ARIMA/SARIMA model.
    """
    print("\n--- Training ARIMA/SARIMA Model ---")
    if auto_fit:
        print("Using auto_arima to find best parameters...")
        model = auto_arima(train_data,
                           start_p=1, start_q=1,
                           test='adf',       # Use adftest to find optimal 'd'
                           max_p=5, max_q=5, # Maximum p and q
                           m=1,              # No seasonality if m=1 (daily data usually has no strong weekly seasonality for prices)
                           d=None,           # Let model determine 'd'
                           seasonal=False,   # No seasonality by default
                           stepwise=True,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           n_jobs=-1) # Use all available cores
        print(f"Auto-ARIMA best parameters: {model.order}")
    else:
        if order is None:
            raise ValueError("Order must be provided if auto_fit is False.")
        model = ARIMA(train_data, order=order, seasonal_order=seasonal_order)
        # For statsmodels ARIMA, you need to call .fit() separately
        model = model.fit()

    print("ARIMA/SARIMA model training complete.")
    # For auto_arima, the fitted model is directly returned.
    # For statsmodels, it's model.fit()
    return model

def forecast_arima(model, steps):
    """Generates forecasts from a fitted ARIMA model."""
    print(f"Generating ARIMA forecast for {steps} steps...")
    # pmdarima's auto_arima object has a predict method that includes confidence intervals
    # Ensure to return the forecast values and confidence intervals
    forecast_result, conf_int = model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
    return forecast_result, conf_int

def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model.

    Args:
        input_shape (tuple): Shape of the input sequences (time_steps, features).

    Returns:
        tf.keras.Model: Compiled Keras LSTM model.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("\nLSTM model built:")
    model.summary()
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
    """
    Trains the LSTM model with early stopping and learning rate reduction.
    """
    print("\n--- Training LSTM Model ---")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1)
    print("LSTM model training complete.")
    return model, history

def forecast_lstm(model, data, time_steps, scaler, forecast_steps):
    """
    Generates forecasts from a trained LSTM model.
    Assumes data is already scaled.
    """
    print(f"Generating LSTM forecast for {forecast_steps} steps...")
    # Use the last 'time_steps' of the historical data as the initial sequence
    last_sequence = data[-time_steps:].reshape(1, time_steps, 1)
    forecasts = []

    for _ in range(forecast_steps):
        next_pred_scaled = model.predict(last_sequence, verbose=0)[0, 0]
        forecasts.append(next_pred_scaled)
        # FIX: Reshape next_pred_scaled to (1, 1, 1) to match dimensions for concatenation
        new_timestep = np.array([[[next_pred_scaled]]])
        # Update the sequence by dropping the oldest timestep and adding the new prediction
        last_sequence = np.concatenate((last_sequence[:, 1:, :], new_timestep), axis=1)


    # Inverse transform the scaled forecasts
    forecasts_original_scale = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
    return forecasts_original_scale.flatten()


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluates model performance using MAE, RMSE, MAPE.
    Handles potential NaN/Inf in predictions by replacing them with a sensible value.

    Args:
        y_true (np.array): Actual values.
        y_pred (np.array): Predicted values.
        model_name (str): Name of the model for printing.
    """
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Identify non-finite values (NaN or Inf) in predictions
    non_finite_mask_pred = ~np.isfinite(y_pred)

    if np.any(non_finite_mask_pred):
        print(f"Warning: {model_name} predictions contain {np.sum(non_finite_mask_pred)} non-finite values (NaN/Inf).")
        finite_preds = y_pred[np.isfinite(y_pred)]
        if len(finite_preds) > 0:
            replacement_value = np.mean(finite_preds)
        else:
            # If all predictions are non-finite, fall back to mean of actuals (if finite) or 0
            finite_actuals = y_true[np.isfinite(y_true)]
            replacement_value = np.mean(finite_actuals) if len(finite_actuals) > 0 else 0.0
            print(f"Warning: All {model_name} predictions are non-finite. Using mean of actuals ({replacement_value:.4f}) as replacement.")

        y_pred[non_finite_mask_pred] = replacement_value
        print(f"Non-finite predictions in {model_name} replaced with {replacement_value:.4f}.")

    # Identify non-finite values in actuals (should ideally be handled by preprocessing)
    non_finite_mask_true = ~np.isfinite(y_true)
    if np.any(non_finite_mask_true):
        print(f"Warning: Actual values contain {np.sum(non_finite_mask_true)} non-finite values (NaN/Inf).")
        # For actuals, it's safer to remove the corresponding points than to impute.
        # This will create a consistent dataset for evaluation.
        valid_indices = ~non_finite_mask_true
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices] # Also filter predictions to match
        print(f"Non-finite actuals and corresponding predictions removed for evaluation.")

    if len(y_true) == 0:
        print("No valid data points remaining after handling non-finite values. Cannot calculate metrics.")
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}


    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Avoid division by zero for MAPE if y_true contains zero values (and filter out 0s if they appear in denominator)
    non_zero_true_mask = (y_true != 0)
    if np.any(non_zero_true_mask):
        mape = np.mean(np.abs((y_true[non_zero_true_mask] - y_pred[non_zero_true_mask]) / y_true[non_zero_true_mask])) * 100
    else:
        mape = np.nan # Cannot calculate MAPE if all true values are zero

    print(f"\n--- {model_name} Evaluation ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print("--------------------------")
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def plot_forecast(historical_data, forecast_data, title="Forecast vs Actual", confidence_intervals=None, actual_test_data=None):
    """
    Plots historical data, forecast data, and optionally actual test data for comparison.
    """
    plt.figure(figsize=(16, 8))
    plt.plot(historical_data.index, historical_data.values, label='Historical Data', color='blue')
    plt.plot(forecast_data.index, forecast_data.values, label='Forecast', color='red')

    if actual_test_data is not None and not actual_test_data.empty:
        plt.plot(actual_test_data.index, actual_test_data.values, label='Actual Test Data', color='green', linestyle='--')

    if confidence_intervals is not None:
        # Ensure confidence_intervals is a DataFrame with upper and lower bounds
        plt.fill_between(confidence_intervals.index,
                         confidence_intervals.iloc[:, 0], # Lower bound
                         confidence_intervals.iloc[:, 1], # Upper bound
                         color='red', alpha=0.1, label='Confidence Interval (95%)')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()