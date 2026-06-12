import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import json
import ta  # Technical Analysis library
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LSTM Stock Predictor."""
    
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
    
    # Feature parameters
    features: Optional[List[str]] = None
    use_volume: bool = True
    use_technical_indicators: bool = True
    use_price_features: bool = True
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['close', 'volume', 'returns', 'ma_7', 'ma_21', 'rsi', 'macd', 'bb_high', 'bb_low']
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ModelConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls(**config_dict)


class RunTracker:
    """Track and manage experiment runs."""
    
    def __init__(self, tracker_file: str = "experiment_runs.json"):
        self.tracker_file = Path(tracker_file)
        self.runs = self._load_runs()
    
    def _load_runs(self) -> List[Dict]:
        """Load existing runs from file."""
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_runs(self):
        """Save runs to file."""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.runs, f, indent=2)
    
    def add_run(self, config: ModelConfig, metrics: Dict[str, float], 
                training_time: float, notes: str = ""):
        """Add a new run to the tracker."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a hash of the config for easy comparison
        config_str = json.dumps(config.__dict__, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config_hash": config_hash,
            "config": config.__dict__,
            "metrics": metrics,
            "training_time_seconds": training_time,
            "notes": notes
        }
        
        self.runs.append(run_data)
        self._save_runs()
        
        logger.info(f"Run {run_id} saved to tracker")
        return run_id
    
    def get_best_runs(self, metric: str = "R²", top_n: int = 5) -> List[Dict]:
        """Get the best runs based on a specific metric."""
        if not self.runs:
            return []
        
        # Sort runs by metric (descending for R², ascending for errors)
        reverse = metric in ["R²", "Directional_Accuracy", "Profitable_Trades_%"]
        sorted_runs = sorted(
            [r for r in self.runs if metric in r.get("metrics", {})],
            key=lambda x: x["metrics"][metric],
            reverse=reverse
        )
        
        return sorted_runs[:top_n]
    
    def compare_runs(self, run_ids: List[str] = None) -> pd.DataFrame:
        """Compare multiple runs in a DataFrame."""
        if run_ids:
            runs = [r for r in self.runs if r["run_id"] in run_ids]
        else:
            runs = self.runs[-5:]  # Last 5 runs
        
        if not runs:
            return pd.DataFrame()
        
        data = []
        for run in runs:
            row = {
                "run_id": run["run_id"],
                "timestamp": run["timestamp"][:19],
                "symbol": run["config"]["symbol"],
                "features": len(run["config"].get("features", [])) if run["config"].get("features") else "auto",
                "lstm_units": str(run["config"]["lstm_units"]),
                "epochs": run["config"]["epochs"],
                "batch_size": run["config"]["batch_size"],
                "learning_rate": run["config"]["learning_rate"],
                **run["metrics"]
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def plot_run_history(self, metric: str = "R²"):
        """Plot the progression of a metric over runs."""
        if not self.runs:
            logger.warning("No runs to plot")
            return
        
        runs_with_metric = [r for r in self.runs if metric in r.get("metrics", {})]
        if not runs_with_metric:
            logger.warning(f"No runs with metric {metric}")
            return
        
        plt.figure(figsize=(12, 6))
        
        x = list(range(len(runs_with_metric)))
        y = [r["metrics"][metric] for r in runs_with_metric]
        timestamps = [r["timestamp"][:10] for r in runs_with_metric]
        
        plt.plot(x, y, 'bo-', markersize=8)
        plt.xlabel("Run Number")
        plt.ylabel(metric)
        plt.title(f"{metric} Progression Over Runs")
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.annotate(f'{yi:.3f}', (xi, yi), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        # Add best run marker
        best_idx = np.argmax(y) if metric in ["R²", "Directional_Accuracy"] else np.argmin(y)
        plt.plot(best_idx, y[best_idx], 'r*', markersize=15, label=f'Best: {y[best_idx]:.3f}')
        
        plt.legend()
        plt.xticks(x[::max(1, len(x)//10)], timestamps[::max(1, len(x)//10)], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def export_summary(self, filename: str = "experiment_summary.csv"):
        """Export all runs to a CSV file."""
        df = self.compare_runs(run_ids=None)  # Get all runs
        if not df.empty:
            df.to_csv(filename, index=False)
            logger.info(f"Experiment summary exported to {filename}")


class LSTMStockPredictor:
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scalers = {}  # Multiple scalers for different features
        self.feature_columns = []
        self.history = None
        self.data = None
        
        logger.info(f"Predictor initialized for {self.config.symbol}")
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        logger.info(f"Fetching {self.config.symbol} data for {self.config.period}...")
        
        ticker = yf.Ticker(self.config.symbol)
        data = ticker.history(period=self.config.period)
        
        if data.empty:
            raise ValueError(f"No data fetched for {self.config.symbol}")
        
        # Keep all price and volume data
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.columns = data.columns.str.lower()
        data = data.dropna()
        
        # Remove outliers from close price
        q1 = data['close'].quantile(0.01)
        q99 = data['close'].quantile(0.99)
        data = data[(data['close'] >= q1) & (data['close'] <= q99)]
        
        self.data = data
        logger.info(f"Data prepared: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and additional features."""
        logger.info("Engineering features...")
        
        df = data.copy()
        
        # Price-based features
        if self.config.use_price_features:
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_change'] = df['close'] - df['open']
            df['high_low_spread'] = df['high'] - df['low']
            df['close_to_high'] = df['close'] / df['high']
            df['close_to_low'] = df['close'] / df['low']
        
        # Volume features
        if self.config.use_volume:
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Technical indicators
        if self.config.use_technical_indicators:
            # Moving averages
            df['ma_7'] = ta.trend.sma_indicator(df['close'], window=7)
            df['ma_21'] = ta.trend.sma_indicator(df['close'], window=21)
            df['ma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # Price relative to moving averages
            df['close_to_ma7'] = df['close'] / df['ma_7']
            df['close_to_ma21'] = df['close'] / df['ma_21']
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['rsi_ma'] = df['rsi'].rolling(window=5).mean()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_width'] = df['bb_high'] - df['bb_low']
            df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
            
            # ATR (Average True Range)
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # OBV (On Balance Volume)
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['obv_ma'] = df['obv'].rolling(window=10).mean()
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Support and Resistance levels (simplified)
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            df['close_to_resistance'] = df['close'] / df['resistance']
            df['close_to_support'] = df['close'] / df['support']
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        # Select features based on config
        if self.config.features:
            available_features = [col for col in self.config.features if col in df.columns]
            if len(available_features) < len(self.config.features):
                missing = set(self.config.features) - set(available_features)
                logger.warning(f"Missing features: {missing}")
            self.feature_columns = available_features
        else:
            # Use all numeric columns except the target
            self.feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                                   if col != 'close']
        
        logger.info(f"Features engineered. Total features: {len(self.feature_columns) + 1}")
        logger.info(f"Feature columns: {self.feature_columns}")
        
        return df
    
    def prepare_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """Prepare data for LSTM training."""
        logger.info("Preparing data for LSTM training...")
        
        # Engineer features
        df = self.engineer_features(self.data)
        
        # Prepare feature matrix (all features including close)
        all_features = ['close'] + self.feature_columns
        feature_data = df[all_features].values
        
        # Scale features independently
        scaled_features = np.zeros_like(feature_data)
        for i, feature in enumerate(all_features):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features[:, i] = scaler.fit_transform(feature_data[:, i].reshape(-1, 1)).flatten()
            self.scalers[feature] = scaler
        
        # Create sequences
        X, y = self._create_sequences(scaled_features)
        
        # Split data
        train_size = int(len(X) * self.config.train_split)
        val_size = int(len(X) * self.config.validation_split)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Input shape: {X_train.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        X, y = [], []
        
        for i in range(self.config.sequence_length, len(data)):
            X.append(data[i-self.config.sequence_length:i])
            y.append(data[i, 0])  # Predict close price (index 0)
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model."""
        layers = []
        
        # Input layer
        layers.append(tf.keras.layers.Input(shape=input_shape))
        
        # First LSTM layer
        layers.append(tf.keras.layers.LSTM(
            self.config.lstm_units[0],
            return_sequences=len(self.config.lstm_units) > 1,
            input_shape=input_shape
        ))
        layers.append(tf.keras.layers.Dropout(self.config.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(self.config.lstm_units)):
            layers.append(tf.keras.layers.LSTM(
                self.config.lstm_units[i],
                return_sequences=i < len(self.config.lstm_units) - 1
            ))
            layers.append(tf.keras.layers.Dropout(self.config.dropout_rate))
        
        # Dense layers
        layers.append(tf.keras.layers.Dense(50, activation='relu'))
        layers.append(tf.keras.layers.Dropout(self.config.dropout_rate / 2))
        layers.append(tf.keras.layers.Dense(25, activation='relu'))
        layers.append(tf.keras.layers.Dense(1))
        
        model = tf.keras.Sequential(layers)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Model built with {model.count_params():,} parameters")
        model.summary()
        return model
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray], 
              val_data: Tuple[np.ndarray, np.ndarray]) -> tf.keras.callbacks.History:
        """Train the LSTM model."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
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
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_checkpoint.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        logger.info("Starting training...")
        
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
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions = self.scalers['close'].inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate model performance."""
        X_test, y_test = test_data
        
        predictions = self.predict(X_test)
        y_test_actual = self.scalers['close'].inverse_transform(y_test.reshape(-1, 1)).flatten()
        
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
        
        # Calculate profit if trading based on predictions
        if len(predictions) > 1:
            predicted_returns = np.diff(predictions) / predictions[:-1]
            actual_returns = np.diff(y_test_actual) / y_test_actual[:-1]
            
            # Simple strategy: buy when predict price increase
            strategy_returns = predicted_returns * actual_returns
            profitable_trades = np.sum(strategy_returns > 0) / len(strategy_returns) * 100
        else:
            profitable_trades = 0.0
        
        metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2),
            'MAPE': float(mape),
            'Directional_Accuracy': float(directional_accuracy),
            'Profitable_Trades_%': float(profitable_trades)
        }
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("="*60)
        
        return metrics, predictions, y_test_actual
    
    def plot_results(self, predictions: np.ndarray, actual: np.ndarray, 
                    title: str = None, n_points: int = 200):
        """Plot prediction results."""
        if title is None:
            title = f"{self.config.symbol} Price Predictions"
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Price predictions
        n_points = min(n_points, len(predictions))
        
        ax1.plot(actual[-n_points:], label='Actual Price', color='blue', linewidth=2)
        ax1.plot(predictions[-n_points:], label='Predicted Price', color='red', linewidth=2, alpha=0.8)
        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Prediction error
        errors = predictions[-n_points:] - actual[-n_points:]
        ax2.plot(errors, label='Prediction Error', color='green', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Prediction Error Over Time')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Error ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """Plot training history."""
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
    
    def plot_feature_importance(self):
        """Analyze feature importance using permutation."""
        # This is a simplified version - for full implementation, 
        # you'd want to use permutation importance or SHAP values
        logger.info("Feature importance analysis would require additional implementation")
    
    def save_model(self, filepath: Union[str, Path]):
        """Save the trained model and scalers."""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = Path(filepath)
        if not str(filepath).endswith('.keras'):
            filepath = filepath.with_suffix('.keras')
        
        self.model.save(filepath)
        
        # Save scalers
        scaler_path = filepath.with_suffix('.scalers')
        with open(scaler_path, 'wb') as f:
            import pickle
            pickle.dump(self.scalers, f)
        
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Scalers saved to {scaler_path}")
    
    def load_model(self, filepath: Union[str, Path]):
        """Load a trained model and scalers."""
        filepath = Path(filepath)
        self.model = tf.keras.models.load_model(filepath)
        
        # Load scalers
        scaler_path = filepath.with_suffix('.scalers')
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                import pickle
                self.scalers = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")


def main(config_path: str = "config.json", notes: str = ""):
    """
    Main execution function.
    
    Args:
        config_path: Path to JSON configuration file
        notes: Optional notes about this run
    """
    start_time = datetime.now()
    
    try:
        # Initialize run tracker
        tracker = RunTracker()
        
        # Load configuration
        config = ModelConfig.load(config_path)
        
        # Log configuration
        logger.info("=" * 70)
        logger.info("📊 ENHANCED LSTM PREDICTOR CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"📈 Symbol: {config.symbol}")
        logger.info(f"📅 Data Period: {config.period}")
        logger.info(f"🔢 Sequence Length: {config.sequence_length} days")
        logger.info(f"🧠 LSTM Architecture: {config.lstm_units}")
        logger.info(f"📊 Features: {config.features if hasattr(config, 'features') else 'Auto-select'}")
        logger.info(f"🎯 Max Epochs: {config.epochs}")
        logger.info(f"⏱️  Early Stopping Patience: {config.patience}")
        logger.info(f"📊 Learning Rate: {config.learning_rate}")
        logger.info(f"📦 Batch Size: {config.batch_size}")
        logger.info(f"💧 Dropout Rate: {config.dropout_rate}")
        logger.info("=" * 70)
        
        # Initialize predictor
        predictor = LSTMStockPredictor(config)
        
        # Fetch and prepare data
        data = predictor.fetch_data()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = predictor.prepare_data()
        
        # Train model
        predictor.train((X_train, y_train), (X_val, y_val))
        
        # Evaluate model
        metrics, predictions, actual = predictor.evaluate((X_test, y_test))
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Add run to tracker
        run_id = tracker.add_run(config, metrics, training_time, notes)
        
        # Show comparison with recent runs
        logger.info("\n" + "=" * 70)
        logger.info("📊 COMPARISON WITH RECENT RUNS")
        logger.info("=" * 70)
        comparison_df = tracker.compare_runs()
        if not comparison_df.empty:
            print(comparison_df.tail(5).to_string())
        
        # Show best runs
        best_runs = tracker.get_best_runs(metric="R²", top_n=3)
        if best_runs:
            logger.info("\n" + "=" * 70)
            logger.info("🏆 TOP 3 RUNS BY R²")
            logger.info("=" * 70)
            for i, run in enumerate(best_runs, 1):
                logger.info(f"{i}. Run {run['run_id']}: R² = {run['metrics']['R²']:.4f}, "
                          f"MAPE = {run['metrics']['MAPE']:.2f}%, "
                          f"Config: {run['config']['lstm_units']}")
        
        # Plot results
        predictor.plot_training_history()
        predictor.plot_results(predictions, actual)
        
        # Plot run history
        tracker.plot_run_history("R²")
        
        # Save model with run ID
        model_name = f"{config.symbol.replace('^', '')}_{run_id}_lstm.keras"
        predictor.save_model(model_name)
        
        # Save detailed results
        results = {
            'run_id': run_id,
            'config': config.__dict__,
            'metrics': metrics,
            'feature_columns': predictor.feature_columns,
            'training_time_seconds': training_time,
            'notes': notes
        }
        
        with open(f'results_{run_id}.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to results_{run_id}.json")
        
        # Export summary
        tracker.export_summary()
        
        return predictor, metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    # Get config file path from command line or use default
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    notes = sys.argv[2] if len(sys.argv) > 2 else ""
    
    predictor, metrics = main(config_file, notes)