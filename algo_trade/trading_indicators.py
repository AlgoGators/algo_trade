import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class marketIndicators: 
    
    #Trend Indicators

    #Simple Moving Average(SMA) Indicator
    @staticmethod
    def sma(data: pd.Series, length: int):
        """
        Purpose: Calculates simple moving average.
        Implementation: Uses pandas rolling window to find mean over 
        period length.
        Args:
        -data: Input time series data 
        -length: NUmber of periods in the moving average

        Returns: a pandas Series
        """
        return data.rolling(window=length).mean()
    
    #Exponential Moving Average (EMA) Indicator
    @staticmethod
    def ema(data: pd.Series, length: int):
        """
        Purpose: Calculates simple exponential moving average.
        Implementation: Uses pandas exponential weighted moving average function 
        for given length
        Args:
        -data: Input time series data 
        -length: NUmber of periods in the exponential moving average
        Returns: a pandas Series
        """
        return data.ewm(span=length, adjust = False).mean()
    

    #Double Exponential Moving Average (DEMA)
    @staticmethod
    def dema(data: pd.Series, length: int):
        """
        Purpose: Calculates doubled exponential weighted moving average, is doubly smoothed
        Implementation: Uses two different exponential weighted moving average functions.
        Args:
        -data: Input time series data 
        -length: NUmber of periods in the exponential moving average
        Returns: a pandas Series
        """
        ema = marketIndicators.ema(data,length)
        smooth_ema = marketIndicators.ema(ema,length)
        return 2*ema -smooth_ema
    
    #Triple Exponential Moving Average (TEMA)
    @staticmethod
    def tema(data: pd.Series, length: int):
        """
        Purpose: Calculates triple exponential weighted moving average, is triple smoothed
        Implementation: Uses three different exponential weighted moving average functions.
        Args:
        -data: Input time series data 
        -length: NUmber of periods in the exponential moving average
        Returns: a pandas Series
        """
        ema1 = marketIndicators.ema(data,length)
        ema2 = marketIndicators.ema(ema1,length)
        ema3 = marketIndicators.ema(ema2,length)
        return (3*ema1) - (3*ema2) + ema3
    
    #Moving Average Convergence Divergence (MACD)
    @staticmethod
    def macd(data: pd.Series, fast_length: int, slow_length: int, signal_smooth: int, ma_type: str):
        """
        Purpose: Identifies trend changes and momentum using short and long term moving averages. 
        Implementation: Uses difference in fast and slow moving averages. 
        Args:
        -data: Input time series data 
        -fast_length: Shorter period moving average
        -slow_length: Longer period moving average
        -signal_smooth: Period for smoothing the MACD line
        -ma_type = Type of moving average used for smoothing (sma or ema)
        Returns: Tuple of smoothed signal line, macd line, and macd histogram
        """
        if ma_type == 'ema':
            fast_ma = marketIndicators.ema(data,fast_length)
            slow_ma = marketIndicators.ema(data,slow_length)
            macd_line = fast_ma - slow_ma
            smooth_signal_line = marketIndicators.ema(macd_line,signal_smooth)
            macd_histogram = macd_line - smooth_signal_line
            return smooth_signal_line, macd_line, macd_histogram
        elif ma_type == 'sma':
            fast_ma = marketIndicators.sma(data,fast_length)
            slow_ma = marketIndicators.sma(data,slow_length)
            macd_line = fast_ma - slow_ma
            smooth_signal_line = marketIndicators.ema(macd_line,signal_smooth)
            macd_histogram = macd_line - smooth_signal_line
            return smooth_signal_line, macd_line, macd_histogram
    
    #Linear Regression Slope (LRS) 
    @staticmethod
    def lrs(data: pd.Series, length: int):
        """
        Purpose: Calculates the slope of a linear regression line for trend direction.
        Implementation: Uses scikit-learn LinearRegression to find slope 
        Args:
        -data: input time series data
        -length: Number of periods to use in regression
        Returns: a pandas series
        """
        model = LinearRegression()
        Close = data['Close'].values[-length:].reshape(-1,1)
        Close_Lagging = data['Close'].shift(1).values[-length:].reshape(-1, 1)
        x= np.arange(length).reshape(-1,1)
        updated_linear_reg = model.fit(x, Close).coef_[0][0]
        past_linear_reg = model.fit(x, Close_Lagging).coef_[0][0]
        LRC = (updated_linear_reg - past_linear_reg)/2
        return LRC

    #Ichimoku Cloud (IC)
    @staticmethod
    def ichimoku_cloud(data: pd.Series, conversion_line_length: int, base_line_length: int, leading_span_b_length: int,lagging_span:int):
        """
        Purpose: Identifies support/resistance, momentum, trend direction
        Implementation: Calculates multiple lines using different period highs and lows
        -Tenkan-sen: Short-term momentum
        -Kijun-sen: Medium-term trend
        Senkou Span A- Future support/resistance zones
        Senkou Span B - Future support/resistance zones
        Chickou Span: Lagging line comparing closes 
        Args:
        -data: Input time series data 
        -conversion_line_length: Period of Tenkan-sen
        base_line_length: Period for Kijun-sen
        leading span_b_length: Period for Senkou span B 
        lagging_span: Periods to shift Chikou Span
        Returns: Tuple of tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, and chikou_span
        """
        High = data['High']
        Low = data['Low']
        Close = data['Close']
        
        tenkan_sen = (High.rolling(window=conversion_line_length).max() + Low.rolling(window=conversion_line_length).min())/2
        kijun_sen = (High.rolling(window = base_line_length).max() + Low.rolling(window = base_line_length).min())/2
        senkou_span_a =(tenkan_sen+kijun_sen)/2
        senkou_span_b = (High.rolling(window = leading_span_b_length).max() + Low.rolling(window = leading_span_b_length).min())/2
        chikou_span = Close.shift(-lagging_span)
        return tenkan_sen, kijun_sen,senkou_span_a,senkou_span_b,chikou_span

    #Average Directional Index (ADX)
    @staticmethod
    def adx(data: pd.Series, adx_smoothing: int, di_length: int):
        """
        Purpose: Measures the strength of trends, determines direction (bullish,bearish,sideways)
        Implementation: Uses directional movement index, true range, directional index, and averages directional index
        Args:
        -data: Input time series data 
        -adx_smoothing: smoothing period for ADX
        -di_length: period for calculating Directional Indicators
        Returns: Tuple of DI+, DI-, ADX
        """
        High = data['High']
        Previous_High = High.shift(1)
        Low = data['Low']
        Previous_Low = Low.shift(1)
        Close_Yesterday = data['Close'].shift(1)
        true_range1 = High - Low
        true_range2 = abs(High - Close_Yesterday)
        true_range3 = abs(Low-Close_Yesterday)
        true_range = pd.DataFrame([true_range1,true_range2,true_range3]).max()

        directional_movement_plus = High - Previous_High
        directional_movement_minus = Previous_Low - Low

        directional_movement_plus = np.where(
        (directional_movement_plus > directional_movement_minus) & (directional_movement_plus > 0),
        directional_movement_plus,0)
        directional_movement_minus = np.where(
        (directional_movement_minus > directional_movement_plus) & (directional_movement_minus > 0),
        directional_movement_minus, 0) 
    
        smooth_true_range = true_range.rolling(window = di_length).mean()
        smoothed_directional_plus = pd.Series(directional_movement_plus).rolling(window=di_length).mean()
        smoothed_directional_minus = pd.Series(directional_movement_minus).rolling(window=di_length).mean()
        di_plus = (smoothed_directional_plus / smooth_true_range) * 100
        di_minus = (smoothed_directional_minus / smooth_true_range) * 100
        dx = (abs(di_plus - di_minus) / abs(di_plus + di_minus)) * 100
        adx = dx.rolling(window=adx_smoothing).mean()
        return di_plus, di_minus, adx



    #Momentum Indicators 

    #Relative Strength Index (RSI)
    @staticmethod
    def rsi(data: pd.Series, length: int, ma_type: str):
        """
        Purpose: Measures the speed and change of price movementsto determine overbought and oversold
        Implementation: Uses average gain and average loss to compute rsi. 
        Args:
        -data: Input time series data 
        -length: length of period 
        -ma_type: type of moving average for smoothing (ema or sma)
        Returns: A pandas series
        """
        Close = data['Close']
        delta = Close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0,0)
        if (ma_type == 'sma'):
            gain_average = marketIndicators.sma(gain,length)
            loss_average = marketIndicators.sma(loss,length)
            rs = gain_average/loss_average
            rsi = 100 - (100 / (1+rs))
            return rsi
        elif (ma_type == 'ema'):
            gain_average = marketIndicators.ema(gain,length)
            loss_average = marketIndicators.ema(loss,length)
            rs = gain_average/loss_average
            rsi = 100 - (100 / (1+rs))
            return rsi
    
    #Stochastic Indicator
    @staticmethod
    def stochastic(data:pd.Series, k_length: int, k_smoothing:int, d_smoothing:int):
        """
        Purpose: Measures momentum of price movements to determine overbought and oversold conditions
        Implementation: Uses low, high, and close values to find %k, smoothed %k, and %d values
        Args:
        -data: Input time series data 
        -k_length: Length of %k line 
        -k_smoothing: Length of smoothed k line 
        d_smoothing: Length of d smoothed line. 
        Returns: Tuple of percent_k, smoothed_percent_k, d_smoothing
        """
        Close_Recent = data['Close']
        High = data['High']
        Low = data['Low']
        highest_high = High.rolling(window=k_length).max()
        lowest_low = Low.rolling(window=k_length).min()
        percent_k = (Close_Recent - lowest_low)/(highest_high-lowest_low) * 100
        smoothed_percent_k = marketIndicators.sma(percent_k,k_smoothing)
        d_smoothing = marketIndicators.sma(smoothed_percent_k,d_smoothing)
        return percent_k, smoothed_percent_k, d_smoothing

    #Rate of Change 
    @staticmethod
    def roc(data:pd.Series, length: int):
        """
        Purpose: Measures the rate of change of price movements
        Implementation: Finds rate of change of current close and a close value n days ago
        Args:
        -data: Input time series data 
        -length: Close of period length n ago  
        Returns: pandas series of rate of change
        """
        Current_Close = data['Close']
        Close_Length_Ago = data['Close'].shift(length)
        roc = (Current_Close - Close_Length_Ago)/Close_Length_Ago *100
        return roc
    
    #Momentum (MOM)
    @staticmethod
    def mom(data:pd.Series, length: int):
        """
        Purpose: Measures trend direction and strength
        Implementation: Finds difference in closes to find momentum
        Args:
        -data: Input time series data 
        -length: Close of period length n ago  
        Returns: pandas series
        """
        Current_Close = data['Close']
        Close_Length_Ago = data['Close'].shift(length)
        mom = Current_Close - Close_Length_Ago
        return mom
    
    #Williams %R 
    @staticmethod
    def williams_r(data:pd.Series, length: int):
        """
        Purpose: Uses closing price in relation to High and Low after a period to find momentum
        Implementation: Uses High and Low values of a period with current close to find percent_r
        Args:
        -data: Input time series data 
        -length: Length of the period chosen 
        Returns: pandas data series
        """
        Current_Close = data['Close']
        High = data['High']
        Low = data['Low']
        Highest_high = High.rolling(window=length).max()
        lowest_low = Low.rolling(window=length).min()
        percent_r = (Highest_high - Current_Close)/(Highest_high-lowest_low) * (-100)
        return percent_r
    
    #Commodity Channel Index (CCI)
    @staticmethod
    def cci(data:pd.Series, length_cci: int,length_ma:int):
        """
        Purpose: Identifies potential price reversals
        Implementation: Uses typical price and mean deviation
        Args:
        -data:Input time series data
        -length_cci: length of period of CCI
        -length_ma: Length of period of mean deviation to measure in the market 
        Returns: pandas data series
        """
        High = data['High']
        Low = data['Low']
        Close = data['Close']
        typical_price = (High + Low + Close)/3
        mean_deviation = marketIndicators.sma(typical_price - marketIndicators.sma(typical_price, length_ma),length_cci)
        cci = (typical_price - marketIndicators.sma(typical_price, length_ma))/(.015 * mean_deviation)
        return cci

    #Money Flow Index 
    @staticmethod
    def mfl(data:pd.Series, length:int):
        """
        Purpose: Identifies potential price reversals using price and volume
        Implementation: Uses volume and typical price to find positive and negative flow
        Args:
        -data: Input time series data
        -length: length of period to posiive and negative money flow
        Returns: pandas data series
        """
        High = data['High']
        Volume = data['Volume']
        Low = data['Low']
        Close = data['Close']

        typical_price = (High+Low+Close)/3
        money_flow = typical_price * Volume

        positive_flow = money_flow.where(typical_price >typical_price.shift(1),0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1),0)

        positive_money_flow = positive_flow.rolling(window=length).sum()
        negative_money_flow = negative_flow.rolling(window=length).sum()
        mfi = 100 - (100/(1+positive_money_flow/negative_money_flow))
        return mfi
    
    #True Strength Index (TSI)
    @staticmethod
    def tsi(data:pd.Series,n_period:int, long_length: int, short_length:int,signal_length:int):
        """
        Purpose: Identifies momentum of price changes and is less laggy by using double smoothing
        Implementation:  Finds smoothed mom indicator, double smooths the mom indicator, finds absolute value of double smoothed
        indicator, calculates tsi and tsi_signal
        Args:
        -data: Input time series data
        -n_period: used for close for n length ago 
        -long_length: used for single smoothed mom 
        -short_length: used for double smoothed mom
        -signal_length: used for ema of calculated tsi
        Returns: Tuple of tsi, tsi_signal
        """
        Close = data['Close']
        price_momentum = Current_Close = data['Close']
        Close_Length_Ago = data['Close'].shift(n_period)
        mom = Current_Close - Close_Length_Ago
        single_smooth_price_momentum = marketIndicators.ema(mom,long_length)
        second_smooth_price_momentum = marketIndicators.ema(single_smooth_price_momentum,short_length)
        absolute_value_mom = abs(Current_Close - Close_Length_Ago)
        absolute_single_smooth_price_momentum = marketIndicators.ema(absolute_value_mom,long_length)
        absolute_second_smooth_price_momentum = marketIndicators.ema(absolute_single_smooth_price_momentum,short_length)
        tsi = 100 *(second_smooth_price_momentum/absolute_second_smooth_price_momentum)
        tsi_signal = marketIndicators.ema(tsi,signal_length)
        return tsi, tsi_signal
    
    #Ultimate Oscillator
    @staticmethod
    def ultimate_oscillator(data:pd.Series,fast_length:int,middle_length:int,slow_length:int):
        """
        Purpose: Meausres price momentum using different time period to reduce false signals
        Implementation: Uses true range to calculate different time period rolling means.
        Args:
        -data: Input time series data
        -fast_length: Time period for finding rolling mean for short bp  and true range
        -middle_length: Time period for finding rolling mean for medium bp and true range
        slow_length: Time period for finding rolling mean for long bp and true range 
        Returns: pandas data series
        """
        Close = data['Close']
        High = data['High']
        Previous_Close = data['Close'].shift(1)
        Low = data['Low']
        BP = Close - min(Low,Previous_Close)
        true_range1 = High - Low
        true_range2 = abs(High - Previous_Close)
        true_range3 = abs(Low-Previous_Close)
        true_range = pd.DataFrame([true_range1,true_range2,true_range3]).max()

        short_bp = BP.rolling(window=fast_length).mean()
        middle_bp = BP.rolling(window=middle_length).mean()
        long_bp = BP.rolling(window=slow_length).mean()

        short_true_range = true_range.rolling(window=fast_length).mean()
        middle_true_range =true_range.rolling(window=middle_length).mean()
        long_true_range = true_range.rolling(window=slow_length).mean()

        ultimate_oscillator = ((4*short_bp) + (2*middle_bp) + long_bp)/((4*short_true_range) +(2*middle_true_range) + long_true_range) * 100
        return ultimate_oscillator
    
    @staticmethod
    #Balance of Power (BOP)
    def bop(data:pd.Series):
        """
        Purpose: Gauges market sentiment by measuring buying and selling pressure
        Implementation: Close - Open price / High - Low Price 
        Args:
        -data: Input time series data
        Returns: pandas data series
        """
        Close = data['Close']
        Open = data['Open']
        High = data['High']
        Low = data['Low']
        BOP = (Close-Open)/(High-Low)
        return BOP

    #Volatility Indicators

    #Average True Range (ATR)
    @staticmethod
    def atr(data: pd.Series, length: int, ma_type: str):
        """
        Purpose: Measures market volatility using average range of price in n period
        Implementation: True range for period length using sma or ema
        Args:
        -data: Input time series data
        -length: length of data period
        -ma_type: ema or sma for computing rolling mean
        Returns: pandas data series
        """
        High = data['High']
        Low = data['Low']
        Close_Yesterday = data['Close'].shift(1)
        true_range1 = High - Low
        true_range2 = abs(High - Close_Yesterday)
        true_range3 = abs(Low-Close_Yesterday)
        true_rangemax = pd.DataFrame[(true_range1,true_range2,true_range3)].max()

        if ma_type == 'sma':
            atr = marketIndicators.sma(true_rangemax, length)
        elif ma_type == 'ema':
            atr = marketIndicators.ema(true_rangemax, length)
    
    #Standard Deviation
    @staticmethod
    def std(data: pd.Series, length: int):
        """
        Purpose: Measures volatility from mean
        Implementation: Finds standard deviation of given period
        Args:
        -data: Input time series data
        -length: length of the period that is computed
        Returns: pandas data series
        """
        std = data.rolling(window=length).std()
        return std
    
    #Normalized Average True Range (NATR)
    @staticmethod
    def natr(data: pd.Series, length: int, ma_type: str):
        """
        Purpose: Measures market volatility using average range of price in n period
        Implementation: True range for period length using sma or ema and normalizes it 
        Args:
        -data: Input time series data
        -length: length of data period
        -ma_type: ema or sma for computing rolling mean
        Returns: pandas data series
        """
        High = data['High']
        Low = data['Low']
        Close_Yesterday = data['Close'].shift(1)
        true_range1 = High - Low
        true_range2 = abs(High - Close_Yesterday)
        true_range3 = abs(Low-Close_Yesterday)
        true_rangemax = np.maximum.reduce[(true_range1,true_range2,true_range3)]

        if ma_type == 'sma':
            atr = marketIndicators.sma(true_rangemax, length)
            narm = (atr/Close_Yesterday) * 100
        elif ma_type == 'ema':
            atr = marketIndicators.ema(true_rangemax, length)
            natr = (atr/Close_Yesterday) * 100
        return natr

    #Bollinger Bands (BB)
    @staticmethod
    def bollingerbands(data: pd.Series, length: int, stdDevInput: float, ma_type: str):
        """
        Purpose: Finds volatility expansion and determine overextendedness
        Implementation: Computes chosen ma_type and then computes middle_band, lower_band, and upper_band
        using standard deviation amount
        Args:
        -data: Input time series data
        -length: length of data period
        -ma_type: ema or sma for computing rolling mean
        Returns: Tuple of middle_band, upper_band, lower_band
        """
        std = marketIndicators.std(data,length)
        if ma_type == 'sma':
            middle_band = marketIndicators.sma(data,length)
            upper_band = middle_band + (stdDevInput *std)
            lower_band = middle_band - (stdDevInput*std)
            return middle_band, upper_band, lower_band
        elif ma_type == 'ema':
            middle_band = marketIndicators.ema(data,length)
            upper_band = middle_band + (stdDevInput *std)
            lower_band = middle_band - (stdDevInput*std)
            return middle_band, upper_band, lower_band
    
    #Keltner Channel (KC)
    @staticmethod
    def keltnerchannel(data: pd.Series, length: int, multiplier: float, atr_length: int):
        """
        Purpose: Identify uptrend or downtrend and volatility. 
        Implementation: Uses ema and finds upper and lower bands using multiplier of atr tracking ema
        Args:
        -data: Input time series data
        -length: length of data period
        -multiplier: adjusted atr multiplier 
        -atr_length: length of period of atr
        Returns: Tuple of kc_middle, kc_upper, kc_lower 
        """
        kc_middle = marketIndicators.ema(data,length)
        kc_upper = kc_middle + (multiplier*marketIndicators.atr(data,atr_length,'ema'))
        kc_lower = kc_middle - (multiplier*marketIndicators.atr(data,atr_length,'ema'))
        return kc_middle, kc_upper, kc_lower
    
    #Donchian Channel (DC)
    @staticmethod
    def donchianchannel(data: pd.Series,length: int, offset:int):
        """
        Purpose: Identifies uptrend, downtrend and breakout. 
        Implementation: Highest High and Lowest Low to compute the middle channel. 
        Args:
        -data: Input time series data
        -length: length of data period
        -offset: finds a shifted moving average for n periods ago
        Returns: Tuple of middle_channel, highest_high, lowest_low
        """
        High = data['High']
        Low = data['Low']
        highest_high = High.rolling(window=length).max().shift(offset)
        lowest_low = Low.rolling(window=length).min().shift(offset)
        middle_channel = (highest_high+lowest_low)/2
        return middle_channel, highest_high, lowest_low

    #Volume Indicators

    #Volume Weighted Average Price (VWAP)
    @staticmethod
    def vwap(data:pd.Series,period: int):
        """
        Purpose: Determine average market price
        Implementation: Finds sum of typical price times volume for n length period. 
        Divides by sum of volume over n period
        Args:
        -data: Input time series data
        -period: length of data period
        Returns: pandas data series
        """
        Close= data['Close']
        High = data['High']
        Low = data['Low']
        Volume = data['Volume']
        Price = (High + Low + Close)/3
        numerator = (Price * Volume).rolling(window=period).sum()
        denominator = Volume.rolling(window=period).sum()
        vwap = numerator/denominator
        return vwap

    #Accumulation/Distribution Line (A/D)
    @staticmethod
    def adl (data:pd.Series):
        """
        Purpose: Determine money flow into a security
        Implementation: Finds money flow multiplier times volume. Cumulative sum of it
        Args:
        -data: Input time series data
        Returns: pandas data series
        """
        Close = data['Close']
        Low = data['Low']
        High = data['High']
        Volume = data['Volume']
        Money_Flow_Multiplier = ((Close-Low)-(High-Close))/(High-Low)
        Money_Flow_Volume = Money_Flow_Multiplier * Volume
        adl = Money_Flow_Volume.cumsum()
        return adl 
    
    #Chaikin Money Flow (CMF)
    @staticmethod
    def cmf(data:pd.Series, period_length:int):
        """
        Purpose: Determine buying and selling pressure for period length 
        Implementation: Determines money flow multiplier times volume. Finds sma of money
        flow volume divided by sma of volume. 
        Args:
        -data: Input time series data
        -period_length: length of data period
        Returns: pandas data series
        """
        Close = data['Close']
        Low = data['Low']
        High = data['High']
        Volume = data['Volume']
        Money_Flow_Multiplier = ((Close-Low)-(High-Close))/(High-Low)
        Money_Flow_Volume = Money_Flow_Multiplier * Volume
        numerator = Money_Flow_Volume.rolling(window=period_length).sum()
        denominator = Volume.rolling(window=period_length).sum()
        cmf = numerator/denominator
        return cmf
    
    #Volume Rate of Change (VRC)
    @staticmethod
    def vrc (data:pd.Series, period_length: int): 
        """
        Purpose: Determine change in volume over period of length l.
        Implementation: Rate of change formula for volume - volume n periods ago 
        Args:
        -data: Input time series data
        -period_length: length of data period
        Returns: pandas data series
        """
        Volume = data['Volume']
        Volume_Past_period = data['Volume'].shift(period_length)
        vrc = (Volume-Volume_Past_period)/Volume_Past_period *100 
        return vrc
    
    #Ease of Movement (EMV)
    @staticmethod
    def emv(data:pd.Series,period_length):
        """
        Purpose: Determine friction of price movement in relation to volume 
        Implementation: Using midpoint prices to find distances between points. 
        Using distances and volume range to make emv. Fnds sma of emv 
        Args:
        -data: Input time series data
        -period_length: length of data period
        Returns: pandas data series
        """
        High = data['High']
        Low = data['Low']
        Volume = data['Volume']
        midpoint_price = (High+Low)/2
        distance = midpoint_price - midpoint_price.shift(1)
        volume_range_ratio = Volume/(High-Low)
        unsmooth_emv = distance/volume_range_ratio * 100000
        emv = unsmooth_emv.rolling(window=period_length).mean()
        return emv 

    #Force Index 
    @staticmethod
    def forceindex(data:pd.Series,period_length:int):
        """
        Purpose: Use price-weighted volume to identify market strength
        Implementation: Difference in closes times volume
        Args:
        -data: Input time series data
        -period_length: length of data period
        Returns: pandas data series
        """
        Close = data['Close']
        Close_period = data['Close'].shift(period_length)
        Volume = data['Volume']
        fi = (Close-Close_period) * Volume
        return fi
    
    #Money Flow Volume 
    @staticmethod
    def moneyflowvolume(data:pd.Series,period_length:int):
        """
        Purpose: Determines buying and selling pressure
        Implementation: Difference in closes, low, high data. Multiplied by volume 
        Args:
        -data: Input time series data
        -period_length: length of data period
        Returns: pandas data series
        """
        Close = data['Close']
        Low = data['Low']
        High = data['High']
        Volume = data['Volume']
        Money_Flow_Multiplier = ((Close-Low)-(High-Close))/(High-Low)
        Money_Flow_Volume = Money_Flow_Multiplier * Volume
        return Money_Flow_Volume


'''
Use examples
macd_signal, macd_line, macd_histogram = marketIndicators.macd(data['Close'], 26, 12, 9, 'ema')
data['macd'] = macd_line
data['macd_signal'] = macd_signal
data['macd_histogram'] = macd_histogram

data['sma10']= marketIndicators.sma(data['Close'],10)
data['ema20'] = marketIndicators.ema(data['Close'],20)

'''
