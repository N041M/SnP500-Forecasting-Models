import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow 2.19.0 + Keras 3.10.0 compatible imports
import tensorflow as tf

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SP500LSTMPredictor:
    
    def __init__(self, config=None):
        """
        Initialize the predictor with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        self.data = None
        
        # Print TensorFlow info
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {tf.keras.__version__}")
        
        # Enable mixed precision for efficiency
        if self.config['mixed_precision']:
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision enabled")
            except:
                print("Mixed precision not available")
    
    def _get_default_config(self):
        """Default configuration parameters."""
        return {
            'symbol': '^GSPC',  # S&P 500 symbol
            'period': '10y',    # Data period
            'sequence_length': 360,  # Number of days to look back
            'train_split': 0.8,
            'validation_split': 0.1,
            'batch_size': 16,
            'epochs': 200,
            'learning_rate': 0.00001,
            'lstm_units': [256, 128, 64],  # LSTM layers configuration
            'dropout_rate': 0.4,
            'patience': 50,
            'mixed_precision': True,
            'features': ['Close', 'Volume', 'High', 'Low'],  # Features to use
            'target': 'Close'  # Target variable
        }
    
    def fetch_data(self):
        """
        Fetch S&P 500 data efficiently using yfinance.
        
        Returns:
            pd.DataFrame: Raw stock data
        """
        print(f"Fetching {self.config['symbol']} data for {self.config['period']}...")
        
        ticker = yf.Ticker(self.config['symbol'])
        data = ticker.history(period=self.config['period'])
        
        if data.empty:
            raise ValueError("No data fetched. Check symbol and period.")
        
        # Calculate additional technical indicators for better predictions
        data = self._add_technical_indicators(data)
        
        self.data = data
        print(f"Data fetched: {len(data)} records from {data.index[0]} to {data.index[-1]}")
        return data
    
    def _add_technical_indicators(self, data):
        """
        Add technical indicators with robust error handling.
        
        Args:
            data (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        # Moving averages
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI) - with protection against division by zero
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)  # Add small epsilon to prevent division by zero
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Clip RSI to valid range [0, 100]
        data['RSI'] = np.clip(data['RSI'], 0, 100)
        
        # Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = rolling_mean + (rolling_std * 2)
        data['BB_lower'] = rolling_mean - (rolling_std * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        
        # Price changes - with outlier protection
        price_change = data['Close'].pct_change()
        data['Price_Change'] = np.clip(price_change, -1, 1)  # Clip extreme changes
        
        # Volume changes - with outlier protection
        volume_change = data['Volume'].pct_change()
        data['Volume_Change'] = np.clip(volume_change, -5, 5)  # Clip extreme volume changes
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        # Clean data - remove NaN, inf, and extreme values
        data = data.dropna()
        
        # Replace any remaining infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        print(f"Data after cleaning: {len(data)} records")
        return data
    
    def prepare_data(self, data=None):
        """
        Prepare data for LSTM training with efficient preprocessing.
        
        Args:
            data (pd.DataFrame): Stock data (uses self.data if None)
            
        Returns:
            tuple: Training, validation, and test datasets
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        print("Preparing data for LSTM training...")
        
        # Select features dynamically
        available_features = [f for f in self.config['features'] if f in data.columns]
        if not available_features:
            raise ValueError("No specified features found in data.")
        
        # Add technical indicators to features if available
        tech_indicators = ['MA_5', 'MA_20', 'MA_50', 'RSI', 'BB_width', 
                          'Price_Change', 'Volume_Change', 'Volatility']
        available_tech = [f for f in tech_indicators if f in data.columns]
        all_features = available_features + available_tech
        
        # Prepare feature matrix
        feature_data = data[all_features].values
        target_data = data[self.config['target']].values.reshape(-1, 1)
        
        # Additional data validation and cleaning
        print("Validating data quality...")
        
        # Check for infinite values
        inf_mask = np.isinf(feature_data)
        if np.any(inf_mask):
            print(f"Warning: Found {np.sum(inf_mask)} infinite values, replacing with NaN")
            feature_data[inf_mask] = np.nan
        
        # Check for very large values (potential outliers)
        large_values = np.abs(feature_data) > 1e6
        if np.any(large_values):
            print(f"Warning: Found {np.sum(large_values)} very large values, clipping")
            feature_data = np.clip(feature_data, -1e6, 1e6)
        
        # Check for NaN values and handle them
        nan_mask = np.isnan(feature_data)
        if np.any(nan_mask):
            print(f"Warning: Found {np.sum(nan_mask)} NaN values, using forward fill")
            # Convert back to DataFrame for easier handling
            temp_df = pd.DataFrame(feature_data, columns=all_features)
            temp_df = temp_df.ffill().bfill()  # Updated pandas syntax
            feature_data = temp_df.values
        
        # Final check
        if np.any(np.isnan(feature_data)) or np.any(np.isinf(feature_data)):
            print("Error: Still have invalid values after cleaning")
            # Use more aggressive cleaning
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print("✅ Data validation complete")
        
        # Scale features and target separately for better performance
        scaled_features = self.scaler.fit_transform(feature_data)
        scaled_target = MinMaxScaler().fit_transform(target_data)
        
        # Store target scaler for inverse transformation
        self.target_scaler = MinMaxScaler().fit(target_data)
        
        # Create sequences efficiently using vectorized operations
        X, y = self._create_sequences(scaled_features, scaled_target.ravel())
        
        # Split data
        train_size = int(len(X) * self.config['train_split'])
        val_size = int(len(X) * self.config['validation_split'])
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"Data prepared - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _create_sequences(self, features, target):
        """
        Create sequences for LSTM training using efficient vectorization.
        
        Args:
            features (np.array): Scaled features
            target (np.array): Scaled target values
            
        Returns:
            tuple: (X, y) sequences
        """
        seq_len = self.config['sequence_length']
        n_features = features.shape[1]
        
        # Pre-allocate arrays for efficiency
        X = np.zeros((len(features) - seq_len, seq_len, n_features))
        y = np.zeros(len(features) - seq_len)
        
        # Vectorized sequence creation
        for i in range(len(features) - seq_len):
            X[i] = features[i:i + seq_len]
            y[i] = target[i + seq_len]
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build LSTM model with configurable architecture.
        
        Args:
            input_shape (tuple): Input shape (sequence_length, n_features)
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = tf.keras.models.Sequential(name='SP500_LSTM_Predictor')
        
        # Input layer with first LSTM
        model.add(tf.keras.layers.LSTM(
            self.config['lstm_units'][0], 
            return_sequences=len(self.config['lstm_units']) > 1,
            input_shape=input_shape,
            name='lstm_1'
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(self.config['dropout_rate']))
        
        # Additional LSTM layers
        for i, units in enumerate(self.config['lstm_units'][1:], 2):
            return_sequences = i < len(self.config['lstm_units'])
            model.add(tf.keras.layers.LSTM(
                units, 
                return_sequences=return_sequences,
                name=f'lstm_{i}'
            ))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.config['dropout_rate']))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1, activation='linear', name='output'))
        
        # Compile with adaptive learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print(f"Model built with {model.count_params():,} parameters")
        return model
    
    def train(self, train_data, val_data):
        """
        Train the LSTM model with advanced callbacks.
        
        Args:
            train_data (tuple): (X_train, y_train)
            val_data (tuple): (X_val, y_val)
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Advanced callbacks for efficient training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['patience'] // 2,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_sp500_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (np.array): Input sequences
            
        Returns:
            np.array: Predictions in original scale
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, batch_size=self.config['batch_size'])
        return self.target_scaler.inverse_transform(predictions)
    
    def evaluate(self, test_data):
        """
        Comprehensive model evaluation.
        
        Args:
            test_data (tuple): (X_test, y_test)
            
        Returns:
            dict: Evaluation metrics
        """
        X_test, y_test = test_data
        
        # Get predictions
        predictions = self.predict(X_test)
        y_test_actual = self.target_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, predictions)
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)
        
        return metrics, predictions, y_test_actual
    
    def plot_results(self, predictions, actual, title="S&P 500 LSTM Predictions"):
        """
        Plot prediction results.
        
        Args:
            predictions (np.array): Model predictions
            actual (np.array): Actual values
            title (str): Plot title
        """
        plt.figure(figsize=(15, 8))
        
        # Plot last 200 points for clarity
        n_points = min(200, len(predictions))
        
        plt.plot(actual[-n_points:], label='Actual', linewidth=2, alpha=0.8)
        plt.plot(predictions[-n_points:], label='Predicted', linewidth=2, alpha=0.8)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and complete workflow
def main():
    """
    Example usage of the SP500LSTMPredictor class.
    """
    # Custom configuration
    config = {
        'sequence_length': 60,
        'lstm_units': [128, 64, 32],
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    # Initialize predictor
    predictor = SP500LSTMPredictor(config)
    
    # Fetch and prepare data
    data = predictor.fetch_data()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = predictor.prepare_data()
    
    # Build and train model
    predictor.build_model((X_train.shape[1], X_train.shape[2]))
    predictor.train((X_train, y_train), (X_val, y_val))
    
    # Evaluate model
    metrics, predictions, actual = predictor.evaluate((X_test, y_test))
    
    # Plot results
    predictor.plot_training_history()
    predictor.plot_results(predictions, actual)
    
    # Save model
    predictor.save_model('sp500_lstm_model.h5')
    
    return predictor, metrics

if __name__ == "__main__":
    predictor, metrics = main()