import pandas as pd
import numpy as np

class TechnicalIndicators:
    
    @staticmethod
    def add_sma(data):
        # window = 50 : 50 days simple moving average
        window = 50
        label = f"SMA_{window}"
        data[label] = data['Close'].rolling(window).mean()
        return data


    @staticmethod
    def add_ema(data):
        #calculate 20 days EMA(exponential moving average)
        window = 20
        label = f"EMA_{window}"
        data[label] = data['Close'].ewm(span = window, adjust= False).mean()
        return data
    
    @staticmethod
    def add_rsi(data):
        #calculate 14 days of RSI(Relative Strength Index)
        window = 14
        delta = data['Close'].diff()
        gain = delta.where(delta>0, 0)
        loss = -delta.where(delta<0,0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100/(1+rs))
        return data 
    
    @staticmethod
    def add_macd(data):
        #calculates MACD(Moving Average Convergence Divergence)
        ema_12 = data['Close'].ewm(span=12,adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26,adjust=False).mean()
        data['MACD'] = ema_12 - ema_26
        return data
    
    @staticmethod
    def add_bollinger_bands(data):
        #calculate Bollinger Bands
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()

        data['Bollinger_Upper_Band'] = sma_20+(2*std_20 )
        data['Bollinger_Lower_Band'] = sma_20-(2*std_20 )
        return data
    
    @staticmethod
    def add_stochastic_oscillator(data):
        #calculate Stochastic Oscillator
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()

        data["%K"] = (data['Close']- low_14) / (high_14 - low_14) *100
        data["%D"] = data["%K"].rolling(3).mean()
        return data
    
    @staticmethod
    def add_atr(data):
        #calculate ATR(Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        return data
    """
    @staticmethod
    def add_obv(data):
        #calculate OBV(On-Balance Volume)
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'][i] > data['Close'][i-1]:
                obv.append(obv[-1] + data['Volume'][i])
            elif data['Close'][i] < data['Close'][i-1]:
                obv.append(obv[-1] - data['Volume'][i])
            else:
                obv.append(obv[-1])
        data['OBV'] = obv
        return data"""
    
    @staticmethod
    def add_obv(data):
        # Vektörel Hızlandırma (Vectorization)
        # 1. Fiyat değişiminin yönünü bul (Signum fonksiyonu: +1, -1 veya 0)
        change_direction = np.sign(data['Close'].diff())
        
        # 2. Yön ile Hacmi çarp (Hacmi + veya - yapar)
        # NaN değerleri (ilk satır) 0 ile doldur
        volume_flow = change_direction * data['Volume']
        volume_flow = volume_flow.fillna(0)
        
        # 3. Kümülatif toplam al (Loop yerine cumsum)
        data['OBV'] = volume_flow.cumsum()
        return data

