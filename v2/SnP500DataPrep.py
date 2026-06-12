"""
S&P 500 Data Preparation for ML Models
=======================================
This script handles:
1. Data downloading from Yahoo Finance
2. Technical indicator calculation
3. Feature engineering
4. Proper chronological splitting (no data leakage)
5. Normalization (fit on train only)
6. Sequence creation for LSTM/Transformer

Author: Ronald
Date: 2024
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Technical Analysis library
import ta


class SP500DataPreparation:
    """
    Comprehensive data preparation pipeline for S&P 500 prediction
    """
    
    def __init__(self, 
                 ticker='^GSPC',
                 start_date='2000-01-01',
                 end_date='2024-11-01',
                 sequence_length=60,
                 train_ratio=0.70,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 normalization='standard'):
        """
        Initialize data preparation pipeline
        
        Parameters:
        -----------
        ticker : str
            Yahoo Finance ticker (^GSPC for S&P 500)
        start_date : str
            Start date for data collection
        end_date : str
            End date for data collection
        sequence_length : int
            Number of days to look back for sequence models
        train_ratio : float
            Proportion of data for training (0.70 = 70%)
        val_ratio : float
            Proportion of data for validation (0.15 = 15%)
        test_ratio : float
            Proportion of data for testing (0.15 = 15%)
        normalization : str
            'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalization = normalization
        
        # Will be set during processing
        self.raw_data = None
        self.processed_data = None
        self.scaler = None
        self.feature_columns = None
        
    def download_data(self):
        """
        Download S&P 500 data from Yahoo Finance
        """
        print(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")
        
        data = yf.download(self.ticker, 
                          start=self.start_date, 
                          end=self.end_date,
                          progress=False)
        
        if data.empty:
            raise ValueError("No data downloaded. Check ticker symbol and date range.")
        
        # Flatten MultiIndex columns if present (yfinance sometimes returns MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Ensure we have the required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Available: {data.columns.tolist()}")
        
        print(f"Downloaded {len(data)} days of data")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        self.raw_data = data
        return data
    
    def calculate_technical_indicators(self, df):
        """
        Calculate comprehensive technical indicators
        
        Returns DataFrame with all technical indicators added
        """
        print("Calculating technical indicators...")
        
        data = df.copy()
        
        # ========================================
        # PRICE-BASED FEATURES
        # ========================================
        
        # Returns (different timeframes)
        data['return_1d'] = data['Close'].pct_change(1)
        data['return_5d'] = data['Close'].pct_change(5)
        data['return_20d'] = data['Close'].pct_change(20)
        
        # Log returns (alternative to simple returns)
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # ========================================
        # TREND INDICATORS
        # ========================================
        
        # Simple Moving Averages (multiple timeframes)
        data['sma_10'] = ta.trend.sma_indicator(data['Close'], window=10)
        data['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['sma_100'] = ta.trend.sma_indicator(data['Close'], window=100)
        data['sma_200'] = ta.trend.sma_indicator(data['Close'], window=200)
        
        # Exponential Moving Averages (traditional + variations)
        data['ema_12'] = ta.trend.ema_indicator(data['Close'], window=12)
        data['ema_26'] = ta.trend.ema_indicator(data['Close'], window=26)
        data['ema_50'] = ta.trend.ema_indicator(data['Close'], window=50)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()
        
        # Price position relative to moving averages (safe division)
        data['price_to_sma20'] = np.where(
            data['sma_20'] > 0,
            data['Close'] / data['sma_20'] - 1,
            0.0
        )
        data['price_to_sma50'] = np.where(
            data['sma_50'] > 0,
            data['Close'] / data['sma_50'] - 1,
            0.0
        )
        
        # ========================================
        # MOMENTUM INDICATORS
        # ========================================
        
        # RSI (Relative Strength Index) - multiple timeframes
        data['rsi_7'] = ta.momentum.rsi(data['Close'], window=7)
        data['rsi_14'] = ta.momentum.rsi(data['Close'], window=14)
        data['rsi_21'] = ta.momentum.rsi(data['Close'], window=21)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], 
                                                  window=14, smooth_window=3)
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        # Rate of Change (ROC)
        data['roc_12'] = ta.momentum.roc(data['Close'], window=12)
        
        # Williams %R
        data['williams_r'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'], 
                                                     lbp=14)
        
        # ========================================
        # VOLATILITY INDICATORS
        # ========================================
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['bb_high'] = bollinger.bollinger_hband()
        data['bb_mid'] = bollinger.bollinger_mavg()
        data['bb_low'] = bollinger.bollinger_lband()
        data['bb_width'] = bollinger.bollinger_wband()
        data['bb_pband'] = bollinger.bollinger_pband()  # Price position in bands
        
        # Average True Range (ATR)
        data['atr_14'] = ta.volatility.average_true_range(data['High'], data['Low'], 
                                                           data['Close'], window=14)
        
        # Historical Volatility (rolling standard deviation of returns)
        data['volatility_20'] = data['return_1d'].rolling(window=20).std()
        data['volatility_50'] = data['return_1d'].rolling(window=50).std()
        
        # ========================================
        # VOLUME INDICATORS
        # ========================================
        
        # Volume changes
        data['volume_change'] = data['Volume'].pct_change(1)
        data['volume_sma_20'] = ta.trend.sma_indicator(data['Volume'], window=20)
        # Safe division to avoid inf
        data['volume_ratio'] = np.where(
            data['volume_sma_20'] > 0,
            data['Volume'] / data['volume_sma_20'],
            1.0  # Default to 1.0 when denominator is 0
        )
        
        # On-Balance Volume (OBV)
        data['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        data['obv_sma'] = ta.trend.sma_indicator(data['obv'], window=20)
        
        # Volume Weighted Average Price (VWAP) - approximation
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        cum_volume = data['Volume'].cumsum()
        cum_tp_volume = (data['Volume'] * typical_price).cumsum()
        # Safe division to avoid inf
        data['vwap'] = np.where(
            cum_volume > 0,
            cum_tp_volume / cum_volume,
            typical_price  # Fallback to typical price
        )
        
        # ========================================
        # TIME-BASED FEATURES (cyclical encoding)
        # ========================================
        
        # Day of week (0 = Monday, 4 = Friday for trading days)
        data['day_of_week'] = data['Date'].dt.dayofweek
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 5)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 5)
        
        # Month (seasonal patterns)
        data['month'] = data['Date'].dt.month
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Quarter
        data['quarter'] = data['Date'].dt.quarter
        
        print(f"Calculated {len([col for col in data.columns if col not in df.columns])} technical indicators")
        
        return data
    
    def create_target_variable(self, df, target_type='return', horizon=1):
        """
        Create target variable for prediction
        
        Parameters:
        -----------
        target_type : str
            'return' - predict next day return (%)
            'price' - predict next day closing price
            'direction' - predict direction (1=up, 0=down)
        horizon : int
            Number of days ahead to predict (1 = next day)
        """
        data = df.copy()
        
        if target_type == 'return':
            # Predict next day's return
            data['target'] = data['Close'].pct_change(horizon).shift(-horizon)
            print(f"Target: {horizon}-day ahead return")
            
        elif target_type == 'price':
            # Predict next day's price
            data['target'] = data['Close'].shift(-horizon)
            print(f"Target: {horizon}-day ahead closing price")
            
        elif target_type == 'direction':
            # Predict direction (binary classification)
            future_return = data['Close'].pct_change(horizon).shift(-horizon)
            data['target'] = (future_return > 0).astype(int)
            print(f"Target: {horizon}-day ahead direction (1=up, 0=down)")
            
        else:
            raise ValueError("target_type must be 'return', 'price', or 'direction'")
        
        return data
    
    def prepare_features(self, df):
        """
        Select and prepare feature columns
        """
        # Remove non-feature columns
        exclude_columns = ['Date', 'target', 'Adj Close']
        
        # Get all numeric columns except excluded ones
        feature_cols = [col for col in df.columns 
                       if col not in exclude_columns and df[col].dtype in ['float64', 'int64']]
        
        # Remove columns with too many NaN values
        valid_features = []
        for col in feature_cols:
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio < 0.1:  # Keep features with less than 10% NaN
                valid_features.append(col)
            else:
                print(f"Removing {col}: {nan_ratio*100:.1f}% missing values")
        
        self.feature_columns = valid_features
        print(f"\nFinal feature count: {len(self.feature_columns)}")
        
        return valid_features
    
    def split_data_chronologically(self, df):
        """
        Split data chronologically into train/val/test sets
        
        Returns:
        --------
        train_df, val_df, test_df
        """
        # Remove NaN rows (from indicator calculation)
        df_clean = df.dropna().reset_index(drop=True)
        
        n = len(df_clean)
        train_size = int(n * self.train_ratio)
        val_size = int(n * self.val_ratio)
        
        # Chronological split
        train_df = df_clean.iloc[:train_size].copy()
        val_df = df_clean.iloc[train_size:train_size + val_size].copy()
        test_df = df_clean.iloc[train_size + val_size:].copy()
        
        print("\n" + "="*60)
        print("CHRONOLOGICAL DATA SPLIT")
        print("="*60)
        print(f"Training Set:")
        print(f"  Samples: {len(train_df)}")
        print(f"  Date Range: {train_df['Date'].min()} to {train_df['Date'].max()}")
        print(f"  Percentage: {len(train_df)/n*100:.1f}%")
        print(f"\nValidation Set:")
        print(f"  Samples: {len(val_df)}")
        print(f"  Date Range: {val_df['Date'].min()} to {val_df['Date'].max()}")
        print(f"  Percentage: {len(val_df)/n*100:.1f}%")
        print(f"\nTest Set:")
        print(f"  Samples: {len(test_df)}")
        print(f"  Date Range: {test_df['Date'].min()} to {test_df['Date'].max()}")
        print(f"  Percentage: {len(test_df)/n*100:.1f}%")
        print("="*60 + "\n")
        
        return train_df, val_df, test_df
    
    def normalize_features(self, train_df, val_df, test_df):
        """
        Normalize features using only training data statistics
        CRITICAL: Fit scaler on training data only!
        """
        print("Normalizing features...")
        
        # Initialize scaler
        if self.normalization == 'standard':
            self.scaler = StandardScaler()
            print("Using StandardScaler (mean=0, std=1)")
        elif self.normalization == 'minmax':
            self.scaler = MinMaxScaler()
            print("Using MinMaxScaler (range 0 to 1)")
        else:
            raise ValueError("normalization must be 'standard' or 'minmax'")
        
        # Extract features
        X_train = train_df[self.feature_columns].values
        X_val = val_df[self.feature_columns].values
        X_test = test_df[self.feature_columns].values
        
        # Calculate column medians from training data (for imputing inf values)
        print("Calculating feature statistics from training data...")
        col_medians = np.nanmedian(X_train, axis=0)
        
        # Replace inf values with NaN, then fill with column median
        print("Checking for infinity values...")
        for X, name in [(X_train, 'train'), (X_val, 'val'), (X_test, 'test')]:
            inf_mask = ~np.isfinite(X)
            if inf_mask.any():
                inf_count = inf_mask.sum()
                print(f"  {name}: Found {inf_count} inf/extreme values, replacing with column median")
                
                # Replace inf with NaN
                X[inf_mask] = np.nan
                
                # Fill NaN values in each column with its median
                for col_idx in range(X.shape[1]):
                    col_mask = np.isnan(X[:, col_idx])
                    if col_mask.any():
                        X[col_mask, col_idx] = col_medians[col_idx]
        
        # Fit on training data ONLY
        self.scaler.fit(X_train)
        
        # Transform all sets using training statistics
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get targets
        y_train = train_df['target'].values
        y_val = val_df['target'].values
        y_test = test_df['target'].values
        
        print(f"Feature shape: {X_train_scaled.shape[1]} features")
        print(f"Training samples: {len(X_train_scaled)}")
        print(f"Validation samples: {len(X_val_scaled)}")
        print(f"Test samples: {len(X_test_scaled)}")
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
    
    def create_sequences(self, X, y, sequence_length):
        """
        Create sequences for LSTM/Transformer models
        
        Converts:
        X: (samples, features) -> (samples - sequence_length, sequence_length, features)
        y: (samples,) -> (samples - sequence_length,)
        
        Example with sequence_length=60:
        - Use days [0:60] to predict day 60
        - Use days [1:61] to predict day 61
        - etc.
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_sequences(self, train_data, val_data, test_data):
        """
        Create sequences for all datasets
        """
        print(f"\nCreating sequences with lookback window: {self.sequence_length} days")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, self.sequence_length)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, self.sequence_length)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, self.sequence_length)
        
        print(f"\nSequence shapes:")
        print(f"X_train: {X_train_seq.shape} -> (samples, sequence_length, features)")
        print(f"X_val: {X_val_seq.shape}")
        print(f"X_test: {X_test_seq.shape}")
        print(f"y_train: {y_train_seq.shape}")
        print(f"y_val: {y_val_seq.shape}")
        print(f"y_test: {y_test_seq.shape}")
        
        return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq)
    
    def run_full_pipeline(self, target_type='return', horizon=1, save_path='data/'):
        """
        Execute complete data preparation pipeline
        
        Returns:
        --------
        Dictionary with all prepared data
        """
        print("\n" + "="*80)
        print("S&P 500 DATA PREPARATION PIPELINE")
        print("="*80 + "\n")
        
        # Step 1: Download data
        df = self.download_data()
        
        # Step 2: Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Step 3: Create target variable
        df = self.create_target_variable(df, target_type=target_type, horizon=horizon)
        
        # Step 4: Prepare features
        self.prepare_features(df)
        
        # Step 5: Split data chronologically
        train_df, val_df, test_df = self.split_data_chronologically(df)
        
        # Step 6: Normalize features
        train_data, val_data, test_data = self.normalize_features(train_df, val_df, test_df)
        
        # Step 7: Create sequences for neural networks
        train_seq, val_seq, test_seq = self.prepare_sequences(train_data, val_data, test_data)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80 + "\n")
        
        # Prepare return dictionary
        result = {
            'train': {
                'X': train_seq[0],
                'y': train_seq[1],
                'dates': train_df['Date'].values[self.sequence_length:]
            },
            'val': {
                'X': val_seq[0],
                'y': val_seq[1],
                'dates': val_df['Date'].values[self.sequence_length:]
            },
            'test': {
                'X': test_seq[0],
                'y': test_seq[1],
                'dates': test_df['Date'].values[self.sequence_length:]
            },
            'metadata': {
                'feature_columns': self.feature_columns,
                'n_features': len(self.feature_columns),
                'sequence_length': self.sequence_length,
                'scaler': self.scaler,
                'target_type': target_type,
                'normalization': self.normalization
            }
        }
        
        # Save data
        import os
        os.makedirs(save_path, exist_ok=True)
        
        np.save(f'{save_path}X_train.npy', train_seq[0])
        np.save(f'{save_path}y_train.npy', train_seq[1])
        np.save(f'{save_path}X_val.npy', val_seq[0])
        np.save(f'{save_path}y_val.npy', val_seq[1])
        np.save(f'{save_path}X_test.npy', test_seq[0])
        np.save(f'{save_path}y_test.npy', test_seq[1])
        
        # Save metadata
        import pickle
        with open(f'{save_path}metadata.pkl', 'wb') as f:
            pickle.dump(result['metadata'], f)
        
        print(f"Data saved to: {save_path}")
        print(f"  - X_train.npy, y_train.npy")
        print(f"  - X_val.npy, y_val.npy")
        print(f"  - X_test.npy, y_test.npy")
        print(f"  - metadata.pkl")
        
        return result


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize pipeline
    pipeline = SP500DataPreparation(
        ticker='^GSPC',              # S&P 500
        start_date='2001-01-01',     # 20 years of data
        end_date='2024-11-01',
        sequence_length=60,          # 60-day lookback window
        train_ratio=0.70,            # 70% training
        val_ratio=0.15,              # 15% validation
        test_ratio=0.15,             # 15% test
        normalization='standard'     # StandardScaler
    )
    
    # Run complete pipeline
    data = pipeline.run_full_pipeline(
        target_type='return',        # Predict next-day return
        horizon=1,                   # 1-day ahead prediction
        save_path='data/'
    )
    
    # Access prepared data
    X_train = data['train']['X']
    y_train = data['train']['y']
    X_val = data['val']['X']
    y_val = data['val']['y']
    X_test = data['test']['X']
    y_test = data['test']['y']
    
    print("\nData ready for model training!")
    print(f"Training sequences: {X_train.shape}")
    print(f"Feature count: {data['metadata']['n_features']}")