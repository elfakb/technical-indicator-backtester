# Quantitative Trading & Technical Indicator Backtester

This project is a  **Algorithmic Trading Backtesting Engine** built with Python. calculates complex technical indicators, and evaluate trading strategies using quantitative performance metrics.

The system focuses on **Borsa Istanbul (BIST)** stocks ('ASELS.IS', `THYAO.IS`, `GARAN.IS`) but supports any asset available on Yahoo Finance. It features a robust visualization pipeline that generates a 7-panel comprehensive dashboard for deep technical analysis.

## 🛠 Features

### 1. Technical Analysis Engine (`indicators.py`)
Calculates a wide range of indicators from scratch using `pandas` and `numpy`:
- **Trend:** SMA (50), EMA (20)
- **Momentum:** RSI (14), Stochastic Oscillator (%K, %D)
- **Volatility:** Bollinger Bands, ATR (Average True Range)
- **Trend Strength:** MACD (Moving Average Convergence Divergence)
- **Volume:** OBV (On-Balance Volume)

### 2. Quantitative Backtester (`backtester.py`)
Simulates trading scenarios over 10-year period data to calculate key performance indicators (KPIs):
- **CAGR** (Compound Annual Growth Rate)
- **Sharpe Ratio** (Risk-Adjusted Return)
- **Max Drawdown** (Downside Risk)
- **Profit Factor** & **Win Rate**

### 3. Strategy Logic
The current implementation tests a hybrid **Trend Following + Mean Reversion** strategy:
1.  **Trend Filter:** The stock must be in an uptrend (`SMA 50 > EMA 20`).
2.  **Entry Signal:** - **Condition A:** `RSI < 30` (Oversold Pullback) 
    - **OR**
    - **Condition B:** MACD Crossover (Momentum Shift)

**Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

**##Run the analysis:**
    Open `main.ipynb` in Jupyter Notebook or VS Code and run the cells.
    
    *To analyze a different stock, simply change the ticker in `main.ipynb`:*
    ```python
    TICKER = "GARAN.IS"  # Example: Garanti BBVA
    # TICKER = "THYAO.IS" # Example: Turkish Airlines
    ```

## Project Structure
- `main.ipynb`: The entry point and control center.
- `backtester.py`: Contains the `QuantEngine` class and visualization logic.
- `indicators.py`: Library of mathematical formulas for indicators.

## Disclaimer
This project is for educational and research purposes only. It does not constitute financial advice. Past performance is not indicative of future results.

## 📄esearch Paper & Documentation
**[Download the Full Project Report (PDF)](./BIST Hisseleri Üzerinde Teknik İndikatörlere Dayalı Algoritmik Ticaret Stratejilerinin Geliştirilmesi ve Backtest Analizi.pdf)** *Click above to view the detailed mathematical background, strategy analysis, and conclusion of this study.*