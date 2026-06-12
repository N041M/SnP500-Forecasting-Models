import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from pathlib import Path
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 🎯 CONFIGURATION SELECTION - MODIFY THIS SECTION TO CHOOSE YOUR SETTINGS
# ============================================================================

"""
EASY CONFIGURATION SELECTION:
Simply change CONFIG_CHOICE to one of these options:

"baseline"                    → Proven working config (R² = 77%, MAPE = 1.47%)
"conservative_improvement"    → Very gentle changes to your best config (recommended)
"gentle_directional"          → Balanced directional improvements
"directional_focus"           → Moderate directional focus (fixed from aggressive version)
"hybrid_best"                 → Combines your best-performing parameters
"optimized"                   → Improved architecture with lessons learned
"quick_test"                  → Fast training for testing (30 epochs)
"custom_stock"                → Analyze any stock (set CUSTOM_STOCK_SYMBOL below)
"from_file"                   → Load from JSON file (set CONFIG_FILE_PATH below)

RECOMMENDATION AFTER YOUR TESTS:
Your best config achieved R² = 82.1%, MAPE = 1.37% - try "conservative_improvement" 
to gently enhance it while maintaining performance.
"""

# Choose your configuration:
CONFIG_CHOICE = "from_file"  # Recommended: gentle improvements to your best config  

# For custom stock analysis:
CUSTOM_STOCK_SYMBOL = "AAPL"  

"""
POPULAR STOCK SYMBOLS:
Tech Stocks:    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA"
Finance:        "JPM", "BAC", "WFC", "GS", "MS", "C"
Indices:        "^GSPC" (S&P 500), "^DJI" (Dow), "^IXIC" (NASDAQ)
Crypto:         "BTC-USD", "ETH-USD", "DOGE-USD", "ADA-USD"
Commodities:    "GLD" (Gold), "SLV" (Silver), "USO" (Oil)
Others:         "^VIX" (Volatility), "^TNX" (10-Year Treasury)
"""

# For loading from JSON file:
CONFIG_FILE_PATH = "config.json"  # Path to your custom JSON configuration file

"""
CREATING CUSTOM JSON CONFIG:
Create a file named 'my_config.json' with this structure:

{
  "symbol": "TSLA",
  "period": "3y", 
  "sequence_length": 75,
  "train_split": 0.75,
  "validation_split": 0.15,
  "batch_size": 24,
  "epochs": 200,
  "learning_rate": 0.0008,
  "lstm_units": [72, 36],
  "dropout_rate": 0.25,
  "patience": 22
}

Then set: CONFIG_CHOICE = "from_file"
"""

# ============================================================================


@dataclass
class ModelConfig:
    """Configuration for S&P 500 LSTM Predictor - all parameters must be explicitly set."""
    
    # Data parameters
    symbol: str
    period: str  
    sequence_length: int
    
    # Training parameters
    train_split: float
    validation_split: float
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    
    # Model parameters
    lstm_units: List[int]
    dropout_rate: float
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        config_dict = {
            'symbol': self.symbol,
            'period': self.period,
            'sequence_length': self.sequence_length,
            'train_split': self.train_split,
            'validation_split': self.validation_split,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'patience': self.patience
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ModelConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls(**config_dict)


class SP500LSTMPredictor:
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.data = None
        
        logger.info(f"Predictor initialized for {self.config.symbol}")
        logger.info(f"TensorFlow version: {tf.__version__}")
    
    def fetch_data(self) -> pd.DataFrame:
        logger.info(f"Fetching {self.config.symbol} data for {self.config.period}...")
        
        try:
            ticker = yf.Ticker(self.config.symbol)
            data = ticker.history(period=self.config.period)
            
            if data.empty:
                raise ValueError(f"No data fetched for {self.config.symbol}")
            
            # Keep only Close price for simplicity
            data = data[['Close']].copy()
            
            # Remove any NaN values
            data = data.dropna()
            
            # Simple outlier removal (remove extreme values)
            q1 = data['Close'].quantile(0.01)
            q99 = data['Close'].quantile(0.99)
            data = data[(data['Close'] >= q1) & (data['Close'] <= q99)]
            
            self.data = data
            logger.info(f"Data prepared: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
            logger.info(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            
            return data
            
        except Exception as e:
            logger.error(f"Data fetching failed: {e}")
            raise
    
    def prepare_data(self, data: pd.DataFrame = None) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        if data is None:
            data = self.data
        
        logger.info("Preparing data for LSTM training...")
        
        # Use only Close price
        dataset = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        dataset_scaled = self.scaler.fit_transform(dataset)
        
        # Create sequences
        X, y = self._create_sequences(dataset_scaled)
        
        # Split data temporally (important for time series)
        train_size = int(len(X) * self.config.train_split)
        val_size = int(len(X) * self.config.validation_split)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Sequence shape: {X_train.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        
        for i in range(self.config.sequence_length, len(data)):
            X.append(data[i-self.config.sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int,]) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                self.config.lstm_units[0],
                return_sequences=False,
                input_shape=(input_shape[0], 1)
            ),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Model built with {model.count_params():,} parameters")
        return model
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray], 
              val_data: Tuple[np.ndarray, np.ndarray]) -> tf.keras.callbacks.History:
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        
        if self.model is None:
            self.build_model(X_train.shape[1:])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-5
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        logger.info("Starting training...")
        logger.info(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        logger.info("Training completed!")
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Reshape for LSTM
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Make predictions
        predictions_scaled = self.model.predict(X_reshaped, verbose=0)
        
        # Inverse transform to get actual prices
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        X_test, y_test = test_data
        
        # Get predictions
        predictions = self.predict(X_test)
        
        # Inverse transform actual values
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, predictions)
        
        # Calculate MAPE
        non_zero_mask = y_test_actual != 0
        mape = np.mean(np.abs((y_test_actual[non_zero_mask] - predictions[non_zero_mask]) / 
                             y_test_actual[non_zero_mask])) * 100
        
        # Directional accuracy
        if len(predictions) > 1:
            directional_accuracy = np.mean(
                np.sign(np.diff(predictions)) == np.sign(np.diff(y_test_actual))
            ) * 100
        else:
            directional_accuracy = 0.0
        
        metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2),
            'MAPE': float(mape),
            'Directional_Accuracy': float(directional_accuracy)
        }
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        for metric, value in metrics.items():
            if metric == 'R²':
                if value > 0.7:
                    status = "Excellent"
                elif value > 0.4:
                    status = "Good"
                elif value > 0:
                    status = "Fair"
                else:
                    status = "Poor"
                logger.info(f"{metric}: {value:.4f} ({status})")
            else:
                logger.info(f"{metric}: {value:.4f}")
        logger.info("="*60)
        
        return metrics, predictions, y_test_actual
    
    def plot_results(self, predictions: np.ndarray, actual: np.ndarray, 
                    title: str = "S&P 500 Predictions", n_points: int = 200):
        
        plt.figure(figsize=(15, 8))
        
        # Plot predictions vs actual
        n_points = min(n_points, len(predictions))
        
        plt.plot(actual[-n_points:], label='Actual Price', color='blue', linewidth=2)
        plt.plot(predictions[-n_points:], label='Predicted Price', color='red', linewidth=2, alpha=0.8)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Time Steps')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: Union[str, Path]):
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = str(filepath)
        if not filepath.endswith('.keras'):
            filepath += '.keras'
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]):
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


def create_baseline_config() -> ModelConfig:
    """Proven baseline configuration (R² = 77%)"""
    return ModelConfig(
        symbol='^GSPC',
        period='5y',
        sequence_length=50,
        train_split=0.8,
        validation_split=0.1,
        batch_size=32,
        epochs=150,
        learning_rate=0.001,
        patience=20,
        lstm_units=[50],
        dropout_rate=0.2
    )

def create_optimized_config() -> ModelConfig:
    """Optimized configuration combining best aspects of previous tests"""
    return ModelConfig(
        symbol='^GSPC',
        period='5y',               # Back to 5y (10y showed worse performance)
        sequence_length=60,        # Sweet spot between 50-75
        train_split=0.8,           # Keep successful split
        validation_split=0.1,
        batch_size=48,             # Between 32-64 for optimization
        epochs=200,                # Reduced from 500
        learning_rate=0.0005,      # Higher than 0.0001, lower than 0.001
        patience=20,               # Much lower than 100
        lstm_units=[64, 32],       # Simpler than 4 layers, more than single
        dropout_rate=0.25          # Slightly higher for regularization
    )

def create_directional_focus_config() -> ModelConfig:
    """Balanced approach for better directional accuracy without sacrificing too much"""
    return ModelConfig(
        symbol='^GSPC',
        period='5y',
        sequence_length=80,        # Moderately longer for trends
        train_split=0.75,          # Slightly more validation
        validation_split=0.15,
        batch_size=32,             # Balanced batch size
        epochs=150,                # Sufficient training time
        learning_rate=0.0008,      # Moderate learning rate
        patience=25,               # More patience to avoid underfitting
        lstm_units=[70, 35],       # Balanced architecture
        dropout_rate=0.28          # Moderate dropout
    )

def create_gentle_directional_config() -> ModelConfig:
    """Gentle improvements to directional accuracy while preserving performance"""
    return ModelConfig(
        symbol='^GSPC',
        period='5y',
        sequence_length=65,        # Slightly longer than your best (55)
        train_split=0.8,           # Keep successful split
        validation_split=0.1,
        batch_size=48,             # Between your successful 64 and 32
        epochs=200,                # Adequate training
        learning_rate=0.0007,      # Slightly lower than your 0.001
        patience=30,               # Enough patience for convergence
        lstm_units=[65, 35],       # Moderate complexity
        dropout_rate=0.25          # Gentle regularization increase
    )

def create_conservative_improvement_config() -> ModelConfig:
    """Very conservative changes to your best performing config"""
    return ModelConfig(
        symbol='^GSPC',
        period='5y',
        sequence_length=60,        # Slightly longer than your best 55
        train_split=0.8,
        validation_split=0.1,
        batch_size=64,             # Keep what worked
        epochs=300,                # More training time
        learning_rate=0.0008,      # Slightly lower for stability
        patience=40,               # More patience
        lstm_units=[75, 45],       # Slightly more complex
        dropout_rate=0.23          # Minimal increase
    )

def create_momentum_config() -> ModelConfig:
    """Focused on learning price momentum and direction changes"""
    return ModelConfig(
        symbol='^GSPC',
        period='5y',
        sequence_length=90,        # Long enough for momentum patterns
        train_split=0.75,
        validation_split=0.15,
        batch_size=24,             # Medium batch size
        epochs=80,                 # Shorter training to avoid price-level overfitting
        learning_rate=0.0015,      # Higher learning rate for momentum
        patience=12,               # Aggressive early stopping
        lstm_units=[64, 32],       # Simpler architecture
        dropout_rate=0.35          # High regularization
    )

def create_trend_detection_config() -> ModelConfig:
    """Specialized for trend and direction detection"""
    return ModelConfig(
        symbol='^GSPC',
        period='3y',               # Shorter period for consistent market regime
        sequence_length=100,       # Long sequences for trend patterns
        train_split=0.7,
        validation_split=0.2,
        batch_size=20,             # Small batches for detailed learning
        epochs=60,                 # Short training to focus on trends not prices
        learning_rate=0.002,       # High learning rate
        patience=10,               # Very early stopping
        lstm_units=[120, 60, 30],  # Deeper but aggressive dropout
        dropout_rate=0.5           # Very high dropout for generalization
    )

def create_hybrid_best_config() -> ModelConfig:
    """Hybrid of your best-performing configurations"""
    return ModelConfig(
        symbol='^GSPC',
        period='5y',               # Proven to work better
        sequence_length=55,        # Between your successful 50 and current 75
        train_split=0.8,
        validation_split=0.1,
        batch_size=56,             # Between your successful 64 and current 32
        epochs=250,                # Moderate epochs
        learning_rate=0.0007,      # Between 0.001 (too high) and 0.0001 (too low)
        patience=25,               # Balanced patience
        lstm_units=[60, 40],       # Simpler than 4 layers, optimized size
        dropout_rate=0.22          # Fine-tuned dropout
    )

def create_quick_test_config() -> ModelConfig:
    """Fast training for testing"""
    return ModelConfig(
        symbol='^GSPC',
        period='2y',
        sequence_length=30,
        train_split=0.8,
        validation_split=0.1,
        batch_size=64,
        epochs=50,
        learning_rate=0.002,
        patience=10,
        lstm_units=[32],
        dropout_rate=0.2
    )

def create_custom_stock_config(symbol: str) -> ModelConfig:
    """Custom configuration optimized for different stock types"""
    
    # Adjust parameters based on stock type
    if symbol in ['BTC-USD', 'ETH-USD', 'DOGE-USD']:  # Crypto
        return ModelConfig(
            symbol=symbol,
            period='2y',            # Shorter period for volatile crypto
            sequence_length=30,     # Shorter sequences for crypto
            train_split=0.75,
            validation_split=0.15,
            batch_size=16,
            epochs=150,
            learning_rate=0.001,
            patience=20,
            lstm_units=[64, 32],    # More complex for volatile assets
            dropout_rate=0.3        # Higher dropout for crypto volatility
        )
    elif symbol in ['^VIX', '^TNX']:  # Volatility/Interest rates
        return ModelConfig(
            symbol=symbol,
            period='5y',
            sequence_length=60,
            train_split=0.8,
            validation_split=0.1,
            batch_size=32,
            epochs=150,
            learning_rate=0.0005,
            patience=25,
            lstm_units=[80, 40],
            dropout_rate=0.25
        )
    else:  # Regular stocks (AAPL, MSFT, TSLA, etc.)
        return ModelConfig(
            symbol=symbol,
            period='5y',
            sequence_length=60,
            train_split=0.8,
            validation_split=0.1,
            batch_size=32,
            epochs=150,
            learning_rate=0.001,
            patience=20,
            lstm_units=[64, 32],
            dropout_rate=0.25
        )


def main():
    """
    Main execution function - automatically uses the configuration selected at the top of the script.
    """
    
    # === AUTOMATIC CONFIGURATION SELECTION ===
    # Based on CONFIG_CHOICE set at the top of the script
    
    if CONFIG_CHOICE == "baseline":
        config = create_baseline_config()
        logger.info("🎯 Using BASELINE configuration (Proven: R² = 77%)")
        
    elif CONFIG_CHOICE == "conservative_improvement":
        config = create_conservative_improvement_config()
        logger.info("🔧 Using CONSERVATIVE IMPROVEMENT configuration (Gentle enhancements)")
        
    elif CONFIG_CHOICE == "gentle_directional":
        config = create_gentle_directional_config()
        logger.info("📈 Using GENTLE DIRECTIONAL configuration (Balanced approach)")
        
    elif CONFIG_CHOICE == "directional_focus":
        config = create_directional_focus_config()
        logger.info("🎯 Using DIRECTIONAL FOCUS configuration (Moderate directional improvements)")
        
    elif CONFIG_CHOICE == "optimized":
        config = create_optimized_config()
        logger.info("🚀 Using OPTIMIZED configuration (Improved architecture)")
        
    elif CONFIG_CHOICE == "hybrid_best":
        config = create_hybrid_best_config()
        logger.info("🏆 Using HYBRID BEST configuration (Combines your successful parameters)")
        
    elif CONFIG_CHOICE == "quick_test":
        config = create_quick_test_config()
        logger.info("⚡ Using QUICK TEST configuration")
        
    elif CONFIG_CHOICE == "custom_stock":
        config = create_custom_stock_config(CUSTOM_STOCK_SYMBOL)
        logger.info(f"📈 Using CUSTOM STOCK configuration for {CUSTOM_STOCK_SYMBOL}")
        
    elif CONFIG_CHOICE == "from_file":
        try:
            config = ModelConfig.load(CONFIG_FILE_PATH)
            logger.info(f"📁 Loading configuration from {CONFIG_FILE_PATH}")
        except FileNotFoundError:
            logger.error(f"❌ Config file {CONFIG_FILE_PATH} not found! Using baseline instead.")
            config = create_baseline_config()
        except Exception as e:
            logger.error(f"❌ Error loading config file: {e}. Using baseline instead.")
            config = create_baseline_config()
            
    else:
        logger.warning(f"❌ Unknown CONFIG_CHOICE: {CONFIG_CHOICE}. Using baseline configuration.")
        config = create_baseline_config()
    
    # Save current config for reference
    config.save('current_run_config.json')
    
    # Log current configuration
    logger.info("=" * 70)
    logger.info("📊 LSTM PREDICTOR CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"📈 Symbol: {config.symbol}")
    logger.info(f"📅 Data Period: {config.period}")
    logger.info(f"🔢 Sequence Length: {config.sequence_length} days")
    logger.info(f"🧠 LSTM Architecture: {config.lstm_units}")
    logger.info(f"🎯 Max Epochs: {config.epochs}")
    logger.info(f"⏱️  Early Stopping Patience: {config.patience}")
    logger.info(f"📊 Learning Rate: {config.learning_rate}")
    logger.info(f"📦 Batch Size: {config.batch_size}")
    logger.info(f"💧 Dropout Rate: {config.dropout_rate}")
    logger.info("=" * 70)
    
    predictor = SP500LSTMPredictor(config)
    
    try:
        # Fetch and prepare data
        data = predictor.fetch_data()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = predictor.prepare_data()
        
        # Train model
        predictor.train((X_train, y_train), (X_val, y_val))
        
        # Evaluate model
        metrics, predictions, actual = predictor.evaluate((X_test, y_test))
        
        # Plot results
        predictor.plot_training_history()
        predictor.plot_results(predictions, actual)
        
        # Save model
        predictor.save_model('sp500_lstm_baseline.keras')
        
        return predictor, metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    predictor, metrics = main()