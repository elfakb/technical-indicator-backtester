import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from indicators import TechnicalIndicators  # indicators.py dosyasını çağırır

class PerformanceMetrics:
    
    @staticmethod
    def calculate_cagr(data):
        data = data.copy()
        data['daily_return'] = data['Close'].pct_change()
        data['cumulative_return'] = (1 + data['daily_return']).cumprod()

        n_years = len(data)/252  # assuming 252 trading days in a year
        cagr = (data['cumulative_return'].iloc[-1])**(1/n_years) - 1
        return cagr 
    
    @staticmethod
    def calculate_max_drawdown(data):
        data = data.copy()
        data['return'] = data['Close'].pct_change()
        data['cumulative_return'] = (1+data['return']).cumprod()
        data['rolling_max'] = data['cumulative_return'].cummax()
        data['drawdown'] = data['cumulative_return']/data['rolling_max']  -1
        max_drawdown = data['drawdown'].min()
        return max_drawdown
    
    @staticmethod
    def calculate_profit_factor(data):
        if 'Strategy_Return' not in data.columns:
            returns = data['Close'].pct_change()
        else:
            returns = data['Strategy_Return']
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0

        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_sharpe_ratio(data , risk_free_rate = 0.02):
        data = data.copy()

        data['return'] = data['Close'].pct_change()
        avg_daily_return = data['return'].mean()
        daily_volatility = data['return'].std()

        sharpe_ratio = ((avg_daily_return - risk_free_rate/252) / daily_volatility)* np.sqrt(252)

        return sharpe_ratio
    
    @staticmethod
    def calculate_win_rate(data):
        if 'Strategy_Return' not in data.columns:
            returns = data['Close'].pct_change()
        else:
            returns = data['Strategy_Return']
        wins = returns[returns > 0].count()
        total_trades = returns[returns != 0].count()

        if total_trades == 0:
            return 0.0

        win_rate = wins / total_trades
        return win_rate

class QuantEngine:
    def __init__(self, ticker, start, end, capital=10000):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.capital = capital
        self.data = None

    def fetch_data(self):
        print(f"Downloading data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
        
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.droplevel(1)
            
        # Column name check
        if 'Adj Close' in self.data.columns and 'Close' not in self.data.columns:
             self.data.rename(columns={'Adj Close': 'Close'}, inplace=True)
             
        self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        self.data.dropna(inplace=True)

    def run_pipeline(self):
        print("Calculating indicators...")
        self.data = TechnicalIndicators.add_sma(self.data)
        self.data = TechnicalIndicators.add_ema(self.data)
        self.data = TechnicalIndicators.add_rsi(self.data)
        self.data = TechnicalIndicators.add_macd(self.data)
        self.data = TechnicalIndicators.add_bollinger_bands(self.data)
        self.data = TechnicalIndicators.add_stochastic_oscillator(self.data)
        self.data = TechnicalIndicators.add_atr(self.data)
        self.data = TechnicalIndicators.add_obv(self.data)
        self.data.dropna(inplace=True)

        self.data['Signal'] = 0
       
        # 1. MACD Signal Line
        self.data['MACD_Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()

        # Conditions
        trend_up = self.data['SMA_50'] > self.data['EMA_20']  # Main Trend Filter
        rsi_oversold = self.data['RSI'] < 30                  # Oversold (Opportunity)
        
        # MACD Golden Cross
        macd_cross = (self.data['MACD'] > self.data['MACD_Signal_Line']) & \
                     (self.data['MACD'].shift(1) < self.data['MACD_Signal_Line'].shift(1))

        # BUY SIGNAL: Trend Up AND (RSI Dip OR MACD Cross)
        buy_cond = trend_up & (rsi_oversold | macd_cross)
        
        self.data['Signal'] = np.where(buy_cond, 1, 0)
        
        # Returns Calculation
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Strategy_Return'] = self.data['Daily_Return'] * self.data['Signal'].shift(1)
        self.data['Equity_Curve'] = (1 + self.data['Strategy_Return']).cumprod() * self.capital
        self.data['Benchmark_Curve'] = (1 + self.data['Daily_Return']).cumprod() * self.capital

    def show_results(self):
        print("\n" + "="*40)
        print(f" BACKTEST RESULTS: {self.ticker}")
        print("="*40)

        try:
            if self.data.empty: return
            final_equity = self.data['Equity_Curve'].iloc[-1]
            cagr = PerformanceMetrics.calculate_cagr(self.data)
            sharpe = PerformanceMetrics.calculate_sharpe_ratio(self.data)
            max_dd = PerformanceMetrics.calculate_max_drawdown(self.data)
            profit_factor = PerformanceMetrics.calculate_profit_factor(self.data)
            win_rate = PerformanceMetrics.calculate_win_rate(self.data)
        except Exception as e:
            print(f"Error: {e}")
            return

        print(f"Final Equity    : {final_equity:.2f} TRY")
        print(f"CAGR            : %{cagr*100:.2f}")
        print(f"Sharpe Ratio    : {sharpe:.2f}")
        print(f"Max Drawdown    : %{max_dd*100:.2f}")
        print(f"Profit Factor   : {profit_factor:.2f}")
        print(f"Win Rate        : %{win_rate*100:.2f}")
        

        # --- 7 PANEL PLOT (ENGLISH) ---
        fig, ax = plt.subplots(7, 1, figsize=(14, 25), sharex=True)

        # 1. Equity Curve
        ax[0].plot(self.data.index, self.data['Equity_Curve'], label='Strategy (Algo)', color='green', linewidth=2)
        ax[0].plot(self.data.index, self.data['Benchmark_Curve'], label='Benchmark (Buy & Hold)', color='gray', linestyle='--')
        ax[0].set_title("1. Equity Curve & Performance")
        ax[0].legend(loc="upper left")
        ax[0].grid(True, alpha=0.3)
        ax[0].set_ylabel('Capital')

        # 2. Price, Bollinger & Signals
        ax[1].plot(self.data.index, self.data['Close'], label='Close Price', color='black', alpha=0.6)
        if 'Bollinger_Upper_Band' in self.data.columns:
            ax[1].plot(self.data.index, self.data['Bollinger_Upper_Band'], color='red', linestyle=':', alpha=0.5, label='Upper Band')
            ax[1].plot(self.data.index, self.data['Bollinger_Lower_Band'], color='green', linestyle=':', alpha=0.5, label='Lower Band')
            ax[1].fill_between(self.data.index, self.data['Bollinger_Upper_Band'], self.data['Bollinger_Lower_Band'], color='gray', alpha=0.1)
        
        # BUY Signals
        buy_signals = self.data[self.data['Signal'] == 1]
        ax[1].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='BUY Signal', zorder=5)
        ax[1].set_title("2. Price Action, Bollinger Bands & Trades")
        ax[1].legend(loc="upper left")
        ax[1].grid(True, alpha=0.3)
        ax[1].set_ylabel('Price')

        # 3. Trend (SMA vs EMA)
        ax[2].plot(self.data.index, self.data['SMA_50'], label='SMA 50', color='orange')
        ax[2].plot(self.data.index, self.data['EMA_20'], label='EMA 20', color='blue')
        ax[2].set_title("3. Trend Indicators (SMA vs EMA)")
        ax[2].legend(loc="upper left")
        ax[2].grid(True, alpha=0.3)

        # 4. RSI
        ax[3].plot(self.data.index, self.data['RSI'], color='purple', label='RSI(14)')
        ax[3].axhline(70, color='red', linestyle='--', linewidth=1)
        ax[3].axhline(30, color='green', linestyle='--', linewidth=1)
        ax[3].fill_between(self.data.index, 70, 30, color='purple', alpha=0.05)
        ax[3].set_title("4. Relative Strength Index (RSI)")
        ax[3].set_ylim(0, 100)
        ax[3].set_ylabel('Value')
        ax[3].grid(True, alpha=0.3)

        # 5. MACD
        ax[4].plot(self.data.index, self.data['MACD'], label='MACD Line', color='blue')
        ax[4].plot(self.data.index, self.data['MACD_Signal_Line'], label='Signal Line', color='orange', linestyle='--')
        # Histogram
        hist = self.data['MACD'] - self.data['MACD_Signal_Line']
        ax[4].bar(self.data.index, hist, color=np.where(hist > 0, 'green', 'red'), alpha=0.3, label='Histogram')
        ax[4].set_title("5. MACD Momentum")
        ax[4].legend(loc="upper left")
        ax[4].grid(True, alpha=0.3)

        # 6. Stochastic
        ax[5].plot(self.data.index, self.data['%K'], label='%K Line', color='blue', linewidth=1)
        ax[5].plot(self.data.index, self.data['%D'], label='%D Line', color='orange', linestyle='--')
        ax[5].axhline(80, color='red', linestyle=':', linewidth=1)
        ax[5].axhline(20, color='green', linestyle=':', linewidth=1)
        ax[5].set_title("6. Stochastic Oscillator")
        ax[5].set_ylim(0, 100)
        ax[5].legend(loc="upper left")
        ax[5].grid(True, alpha=0.3)

        # 7. OBV
        ax[6].plot(self.data.index, self.data['OBV'], label='OBV', color='teal')
        ax[6].set_title("7. On-Balance Volume (OBV)")
        ax[6].grid(True, alpha=0.3)
        ax[6].set_xlabel('Date')

        plt.tight_layout()
        plt.show()