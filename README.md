# GMF Investments: Advanced Portfolio Management & Market Forecasting

## Project Overview

This project, developed for Guide Me in Finance (GMF) Investments, focuses on applying advanced time series forecasting and Modern Portfolio Theory (MPT) to enhance portfolio management strategies. As a forward-thinking financial advisory firm, GMF leverages cutting-edge technology and data-driven insights to provide clients with tailored investment strategies, aiming to predict market trends, optimize asset allocation, and enhance overall portfolio performance.

The core objective is to minimize risks and capitalize on market opportunities, recognizing the Efficient Market Hypothesis's practical implication that direct stock price prediction is challenging. Therefore, models are primarily used for volatility forecasting, identifying momentum factors, or as inputs into a broader decision-making framework.

## Business Need

As a Financial Analyst at GMF Investments, the primary goal is to:
*   Utilize YFinance data to extract historical financial information.
*   Preprocess and analyze data to identify trends and patterns.
*   Develop and evaluate time series forecasting models to predict future market movements (specifically for TSLA).
*   Use the insights from forecasts and historical data to recommend optimal portfolio adjustments.

## Key Assets Under Analysis

This project analyzes historical financial data for three key assets:

*   **Tesla (TSLA):** A high-growth, high-risk stock in the consumer discretionary sector (Automobile Manufacturing), offering potential for high returns with high volatility.
*   **Vanguard Total Bond Market ETF (BND):** A bond ETF tracking U.S. investment-grade bonds, providing stability and income, contributing to lower portfolio risk.
*   **S&P 500 ETF (SPY):** An ETF tracking the S&P 500 Index, offering broad U.S. market exposure with diversified, moderate risk.

The data covers the period from **July 1, 2015, to July 31, 2025**.

## Project Structure

The project is organized into modular components to ensure clarity, maintainability, and reusability.

gmf_investment_week_11/
├── data/
│ └── raw/ # Stores raw, downloaded financial data.
│ └── processed/ # Stores cleaned, preprocessed, and scaled data, along with saved scalers.
├── notebooks/ # Jupyter notebooks for exploratory analysis and iterative development.
├── src/
│ ├── data_ingestion.py # Handles fetching data from YFinance.
│ ├── data_preprocessing.py # Contains functions for data cleaning, feature engineering, and scaling.
│ ├── eda.py # Functions for exploratory data analysis, visualizations, and statistical tests.
│ ├── forecasting_models.py # Implementations for ARIMA/SARIMA and LSTM models, plus evaluation metrics.
│ ├── portfolio_optimizer.py# Logic for Modern Portfolio Theory (MPT), Efficient Frontier, and portfolio selection.
│ ├── backtester.py # Simulates portfolio performance against a benchmark.
│ └── init.py # Makes src a Python package.
├── tests/ # Placeholder for unit tests (optional but good practice).
├── reports/ # Output directory for interim reports, final memos, or visualizations.
├── requirements.txt # Lists all necessary Python libraries for the project.
├── .gitignore # Specifies files and directories to be ignored by Git.
├── README.md # This README file.
├── main.py # Orchestrates the entire workflow by calling functions from src/.


## Setup and Installation

Follow these steps to set up the project environment:

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/Fentahun022/gmf_investments_week_11.git 
    cd gmf_investments_week_11
 

2.  **Create a Virtual Environment (Recommended):**
    A virtual environment isolates your project's dependencies from your system's Python packages.
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    You should see `(venv)` prefixing your terminal prompt.

4.  **Install Dependencies:**
    Install all required Python libraries listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the entire financial analysis and portfolio optimization pipeline, execute the `main.py` script from the project's root directory:

```bash
python3 main.py