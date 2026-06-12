"""
S&P 500 Prediction Algorithm
Comparing LSTM vs XGBoost with Technical Indicators
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For LSTM
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Installing...")
    KERAS_AVAILABLE = False

# For XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Will install...")
    XGBOOST_AVAILABLE = False


class TechnicalIndicators:
    """Calculate common technical indicators used in trading algorithms"""
    
    @staticmethod
    def calculate_sma(data, window):
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data, window):
        """Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data, window=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    @staticmethod
    def calculate_bollinger_bands(data, window=20, num_std=2):
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_atr(high, low, close, window=14):
        """Average True Range (volatility indicator)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    @staticmethod
    def calculate_obv(close, volume):
        """On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv


class SP500Predictor:
    """S&P 500 prediction using multiple ML models"""
    
    def __init__(self, ticker='^GSPC', start_date='2015-01-01', end_date=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch S&P 500 data from Yahoo Finance"""
        print(f"Fetching {self.ticker} data from {self.start_date} to {self.end_date}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(f"Downloaded {len(self.data)} days of data")
        return self.data
    
    def add_technical_indicators(self):
        """Add technical indicators to the dataset"""
        print("\nAdding technical indicators...")
        df = self.data.copy()
        
        # Price-based indicators
        df['SMA_20'] = TechnicalIndicators.calculate_sma(df['Close'], 20)
        df['SMA_50'] = TechnicalIndicators.calculate_sma(df['Close'], 50)
        df['EMA_12'] = TechnicalIndicators.calculate_ema(df['Close'], 12)
        df['EMA_26'] = TechnicalIndicators.calculate_ema(df['Close'], 26)
        
        # RSI
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'] = TechnicalIndicators.calculate_macd(df['Close'])
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR (Average True Range)
        df['ATR'] = TechnicalIndicators.calculate_atr(df['High'], df['Low'], df['Close'])
        
        # OBV (On-Balance Volume)
        df['OBV'] = TechnicalIndicators.calculate_obv(df['Close'], df['Volume'])
        
        # Price momentum
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_20d'] = df['Close'].pct_change(20)
        
        # Volume indicators
        df['Volume_SMA_20'] = TechnicalIndicators.calculate_sma(df['Volume'], 20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Price range
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Lag features (previous days)
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        
        # Target: next day's closing price
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        self.data = df
        print(f"Added {len(df.columns) - len(self.data.columns) + len(df.columns)} features")
        print(f"Final dataset shape: {df.shape}")
        return df
    
    def prepare_data(self, lookback=60, test_size=0.2):
        """
        Prepare data for training with CHRONOLOGICAL split
        
        CRITICAL: We use chronological splitting for time series!
        Random splitting would introduce look-ahead bias.
        """
        print(f"\nPreparing data with {lookback}-day lookback...")
        
        # Select features (excluding target and date-related)
        feature_columns = [col for col in self.data.columns 
                          if col not in ['Target', 'Close'] and not col.startswith('Adj')]
        
        features = self.data[feature_columns].values
        target = self.data['Target'].values
        
        # CHRONOLOGICAL SPLIT (not random!)
        split_idx = int(len(features) * (1 - test_size))
        
        X_train = features[:split_idx]
        y_train = target[:split_idx]
        X_test = features[split_idx:]
        y_test = target[split_idx:]
        
        print(f"Train set: {len(X_train)} samples ({self.data.index[0]} to {self.data.index[split_idx-1].strftime('%Y-%m-%d')})")
        print(f"Test set: {len(X_test)} samples ({self.data.index[split_idx].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')})")
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Scale target
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1))
        
        # For LSTM: create sequences
        X_train_lstm, y_train_lstm = self.create_sequences(X_train_scaled, y_train_scaled, lookback)
        X_test_lstm, y_test_lstm = self.create_sequences(X_test_scaled, y_test_scaled, lookback)
        
        # For XGBoost: use the scaled data directly (but align with LSTM)
        X_train_xgb = X_train_scaled[lookback:]
        y_train_xgb = y_train_scaled[lookback:]
        X_test_xgb = X_test_scaled[lookback:]
        y_test_xgb = y_test_scaled[lookback:]
        
        return {
            'lstm': (X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm),
            'xgboost': (X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb),
            'feature_names': feature_columns
        }
    
    def create_sequences(self, X, y, lookback):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train LSTM model"""
        print("\n" + "="*50)
        print("Training LSTM Model")
        print("="*50)
        
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        return model, history
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train.ravel(),
            eval_set=[(X_test, y_test.ravel())],
            verbose=50
        )
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name, model_type='lstm'):
        """Evaluate model performance"""
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print('='*50)
        
        # Make predictions
        if model_type == 'lstm':
            predictions_scaled = model.predict(X_test, verbose=0)
        else:  # xgboost
            predictions_scaled = model.predict(X_test).reshape(-1, 1)
        
        # Inverse transform to get actual prices
        predictions = self.scaler.inverse_transform(predictions_scaled)
        actuals = self.scaler.inverse_transform(y_test)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Calculate directional accuracy
        actual_direction = np.diff(actuals.flatten()) > 0
        pred_direction = np.diff(predictions.flatten()) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def plot_comparison(self, results_dict):
        """Plot comparison between models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot predictions vs actuals for each model
        for idx, (model_name, results) in enumerate(results_dict.items()):
            ax = axes[idx, 0]
            ax.plot(results['actuals'], label='Actual', linewidth=2)
            ax.plot(results['predictions'], label='Predicted', linewidth=2, alpha=0.7)
            ax.set_title(f'{model_name} - Predictions vs Actuals')
            ax.set_xlabel('Days')
            ax.set_ylabel('S&P 500 Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot prediction errors
        for idx, (model_name, results) in enumerate(results_dict.items()):
            ax = axes[idx, 1]
            errors = results['predictions'] - results['actuals']
            ax.plot(errors, label='Prediction Error', color='red', alpha=0.6)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f'{model_name} - Prediction Errors')
            ax.set_xlabel('Days')
            ax.set_ylabel('Error ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/sp500_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved to outputs folder")
        
    def plot_metrics_comparison(self, results_dict):
        """Plot bar chart comparing model metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(results_dict.keys())
        metrics = ['rmse', 'mae', 'r2', 'directional_accuracy']
        titles = ['RMSE ($)', 'MAE ($)', 'R² Score', 'Directional Accuracy (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            values = [results_dict[model][metric] for model in models]
            bars = ax.bar(models, values, color=['blue', 'green'], alpha=0.7)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/sp500_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Metrics comparison saved to outputs folder")


def main():
    """Main execution function"""
    print("="*70)
    print("S&P 500 PREDICTION ALGORITHM - LSTM vs XGBoost")
    print("="*70)
    
    # Initialize predictor
    predictor = SP500Predictor(start_date='2015-01-01')
    
    # Fetch and prepare data
    predictor.fetch_data()
    predictor.add_technical_indicators()
    
    # Prepare data with chronological split
    data_dict = predictor.prepare_data(lookback=60, test_size=0.2)
    
    # Train both models
    X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm = data_dict['lstm']
    X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb = data_dict['xgboost']
    
    # Train LSTM
    lstm_model, lstm_history = predictor.train_lstm(
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
        epochs=100, batch_size=32
    )
    
    # Train XGBoost
    xgb_model = predictor.train_xgboost(
        X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb
    )
    
    # Evaluate both models
    lstm_results = predictor.evaluate_model(
        lstm_model, X_test_lstm, y_test_lstm, "LSTM", model_type='lstm'
    )
    
    xgb_results = predictor.evaluate_model(
        xgb_model, X_test_xgb, y_test_xgb, "XGBoost", model_type='xgboost'
    )
    
    # Compare results
    results_dict = {
        'LSTM': lstm_results,
        'XGBoost': xgb_results
    }
    
    # Generate comparison plots
    predictor.plot_comparison(results_dict)
    predictor.plot_metrics_comparison(results_dict)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} {'LSTM':<20} {'XGBoost':<20} {'Winner'}")
    print("-"*70)
    
    metrics = [
        ('RMSE', 'rmse', 'lower'),
        ('MAE', 'mae', 'lower'),
        ('R² Score', 'r2', 'higher'),
        ('Directional Accuracy', 'directional_accuracy', 'higher')
    ]
    
    for metric_name, metric_key, better in metrics:
        lstm_val = lstm_results[metric_key]
        xgb_val = xgb_results[metric_key]
        
        if better == 'lower':
            winner = 'LSTM' if lstm_val < xgb_val else 'XGBoost'
        else:
            winner = 'LSTM' if lstm_val > xgb_val else 'XGBoost'
        
        print(f"{metric_name:<25} {lstm_val:<20.4f} {xgb_val:<20.4f} {winner}")
    
    print("="*70)
    print("\nAnalysis complete! Check the outputs folder for visualizations.")
    
    return predictor, lstm_model, xgb_model, results_dict


if __name__ == "__main__":
    # Check and install dependencies
    import subprocess
    import sys
    
    if not KERAS_AVAILABLE:
        print("Installing TensorFlow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "tensorflow", "--break-system-packages", "-q"])
    
    if not XGBOOST_AVAILABLE:
        print("Installing XGBoost...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "xgboost", "--break-system-packages", "-q"])
    
    predictor, lstm_model, xgb_model, results = main()