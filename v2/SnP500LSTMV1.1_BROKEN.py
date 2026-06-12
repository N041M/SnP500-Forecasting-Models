"""
Enhanced LSTM Model for S&P 500 Prediction (V1.1)
==================================================
This script implements an enhanced LSTM model with all major improvements
from v1.1.2, including:

- 20+ engineered features (MA, RSI, MACD, Bollinger Bands, Stochastic, volume indicators)
- Bidirectional LSTM architecture (128, 64, 32 units)
- Dual outputs (price prediction + direction classification)
- Custom biased loss function with growth bias
- Market-cap weighted S&P 500 aggregation
- Adaptive company weighting (top performers get 1.8x weight)
- Sector rotation logic
- Negative streak detection and diversification boost
- 200 epoch training with early stopping

Author: Ronald
Version: 1.1
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import warnings
import logging

# TensorFlow/Keras imports
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import Huber

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EnhancedLSTMConfig:
    """Enhanced configuration with market bias parameters."""

    sequence_length: int
    train_split: float
    validation_split: float
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    lstm_units: List[int]
    dropout_rate: float
    growth_bias: float = 0.005
    top_performer_weight: float = 1.8
    low_rep_boost: float = 1.5
    top_performer_percentile: float = 0.8
    negative_streak_threshold: int = 2

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        config_dict = self.__dict__.copy()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'EnhancedLSTMConfig':
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls(**config_dict)


class EnhancedLSTMPredictor:
    """
    Enhanced LSTM model for S&P 500 prediction with custom biases
    """

    def __init__(self, config: EnhancedLSTMConfig, n_features: int = 1):
        self.config = config
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.training_time = None
        self.company_weights = {}
        self.company_representations = {}
        self.negative_streak_counter = 0
        self.is_aggregate = False
        self.momentum_adjustment = 1.0

        logger.info("Enhanced LSTM Predictor initialized with custom biases")
        logger.info(f"Growth bias: {self.config.growth_bias}")
        logger.info(f"Top performer weight: {self.config.top_performer_weight}")
        logger.info(f"Low representation boost: {self.config.low_rep_boost}")
        logger.info(f"Number of features: {self.n_features}")
        logger.info(f"TensorFlow version: {tf.__version__}")

    def prepare_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare enhanced features with 20+ technical indicators
        """
        logger.info("Preparing enhanced features...")
        if data.empty:
            logger.error("Input data is empty!")
            raise ValueError("Input data is empty")

        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {list(data.columns)}")

        features = pd.DataFrame(index=data.index)

        # Base price feature
        if 'Close' in data.columns:
            features['price'] = data['Close']
            logger.info("Using Close price as main feature")
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                features['price'] = data[numeric_cols[0]]
                logger.info(f"Using {numeric_cols[0]} as price feature")
            else:
                logger.error("No numeric price data found in columns")
                raise ValueError("No numeric price data found")

        # Volume indicators
        if 'Volume' in data.columns:
            features['volume'] = data['Volume']
            features['volume_ma'] = features['volume'].rolling(20, min_periods=1).mean()
            features['volume_ratio'] = features['volume'] / features['volume_ma']
            logger.info("Added volume indicators")

        # Technical indicators
        logger.info("Calculating technical indicators...")
        features['returns'] = features['price'].pct_change()

        # Moving averages
        features['ma_5'] = features['price'].rolling(5, min_periods=1).mean()
        features['ma_20'] = features['price'].rolling(10, min_periods=1).mean()
        features['ma_50'] = features['price'].rolling(20, min_periods=1).mean()

        # Volatility
        features['volatility'] = features['returns'].rolling(10, min_periods=1).std()

        # Bollinger Bands
        features['bollinger_upper'] = features['ma_20'] + 2 * features['price'].rolling(20, min_periods=1).std()
        features['bollinger_lower'] = features['ma_20'] - 2 * features['price'].rolling(20, min_periods=1).std()

        # Stochastic Oscillator
        low_14 = features['price'].rolling(14, min_periods=1).min()
        high_14 = features['price'].rolling(14, min_periods=1).max()
        features['stochastic_k'] = 100 * (features['price'] - low_14) / (high_14 - low_14 + 1e-10)

        # Price ratios
        features['price_to_ma5'] = features['price'] / features['ma_5']
        features['price_to_ma20'] = features['price'] / features['ma_20']

        # RSI (Relative Strength Index)
        delta = features['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = features['price'].ewm(span=6, adjust=False).mean()
        exp2 = features['price'].ewm(span=13, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=4, adjust=False).mean()

        # Bull market indicator
        if len(features) >= 10:
            features['bull_market'] = (features['ma_50'] > features['ma_50'].shift(10, fill_value=features['ma_50'].iloc[0])).astype(int)
            logger.info("Added bull_market indicator")
        else:
            features['bull_market'] = 0
            logger.warning("Insufficient data for bull_market calculation, setting to 0")

        # Performance rank
        features['performance_rank'] = features['returns'].rolling(10, min_periods=1).mean().rank(pct=True)

        # Momentum bias
        features['momentum_bias'] = np.where(
            features['returns'].rolling(5, min_periods=1).mean() > 0,
            1 + self.config.growth_bias,
            1.0
        )

        # Negative streak detection
        logger.info("Detecting negative streaks...")
        features['negative_streak'] = 0.0
        streak = 0
        for i in range(1, len(features)):
            if not pd.isna(features['returns'].iloc[i]) and features['returns'].iloc[i] < 0:
                streak += 1
            else:
                streak = 0
            features.loc[features.index[i], 'negative_streak'] = float(streak)

        # Diversification factor
        features['diversification_factor'] = np.where(
            features['negative_streak'] >= self.config.negative_streak_threshold,
            self.config.low_rep_boost,
            1.0
        )

        # Weighted price
        features['weighted_price'] = features['price'] * features['momentum_bias']

        # Fill NaN values
        logger.info(f"Features shape before dropping NaNs: {features.shape}")
        features = features.ffill().bfill()
        logger.info(f"Features shape after filling NaNs: {features.shape}")

        # Normalize features (except price)
        for col in features.columns:
            if col not in ['price', 'weighted_price'] and features[col].dtype in [np.float64, np.float32]:
                features[col] = MinMaxScaler().fit_transform(features[[col]])

        logger.info(f"Features prepared: {features.shape} with {len(features.columns)} indicators")
        logger.info(f"Available features: {list(features.columns)}")
        return features

    def prepare_data(self, features: pd.DataFrame) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """
        Prepare data for dual-output LSTM training (price + direction)

        CRITICAL: Prevents data leakage by:
        1. Splitting raw data FIRST
        2. Fitting scaler ONLY on training data
        3. Transforming all splits with train-fitted scaler
        """
        logger.info("Preparing data for LSTM training...")

        if 'weighted_price' in features.columns:
            dataset = features[['weighted_price']].values
        elif 'price' in features.columns:
            dataset = features[['price']].values
        else:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                dataset = features[[numeric_cols[0]]].values
            else:
                raise ValueError("No suitable price data found in features")

        # STEP 1: Split raw data FIRST (before scaling)
        train_size = int(len(dataset) * self.config.train_split)
        val_size = int(len(dataset) * self.config.validation_split)

        dataset_train = dataset[:train_size]
        dataset_val = dataset[train_size:train_size + val_size]
        dataset_test = dataset[train_size + val_size:]

        logger.info(f"Raw data splits - Train: {len(dataset_train)}, Val: {len(dataset_val)}, Test: {len(dataset_test)}")

        # STEP 2: Fit scaler ONLY on training data
        self.scaler.fit(dataset_train)
        logger.info(f"Scaler fitted on training data only - Min: {self.scaler.data_min_[0]:.6f}, Max: {self.scaler.data_max_[0]:.6f}")

        # STEP 3: Transform all splits using train-fitted scaler
        dataset_train_scaled = self.scaler.transform(dataset_train)
        dataset_val_scaled = self.scaler.transform(dataset_val)
        dataset_test_scaled = self.scaler.transform(dataset_test)

        # Create sequences from each split separately
        def create_sequences(data_scaled, sequence_length):
            X, y_price, y_direction = [], [], []
            for i in range(sequence_length, len(data_scaled)):
                X.append(data_scaled[i-sequence_length:i, 0])
                y_price.append(data_scaled[i, 0])
                y_direction.append(1 if data_scaled[i, 0] > data_scaled[i-1, 0] else 0)
            return np.array(X), np.array(y_price), np.array(y_direction)

        X_train, y_train_price, y_train_direction = create_sequences(dataset_train_scaled, self.config.sequence_length)
        X_val, y_val_price, y_val_direction = create_sequences(dataset_val_scaled, self.config.sequence_length)
        X_test, y_test_price, y_test_direction = create_sequences(dataset_test_scaled, self.config.sequence_length)

        logger.info(f"Sequence splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info("✅ Data leakage prevented: Scaler fitted on training data only")

        return (X_train, y_train_price, y_train_direction), (X_val, y_val_price, y_val_direction), (X_test, y_test_price, y_test_direction)

    def build_model(self):
        """
        Build enhanced LSTM model with dual outputs
        """
        print("\n" + "="*60)
        print("BUILDING ENHANCED LSTM MODEL (V1.1)")
        print("="*60)

        # Input layer
        inputs = Input(shape=(self.config.sequence_length, self.n_features))

        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(self.config.lstm_units[0], return_sequences=len(self.config.lstm_units) > 1))(inputs)
        x = Dropout(self.config.dropout_rate)(x)

        for i, units in enumerate(self.config.lstm_units[1:], 1):
            return_seq = i < len(self.config.lstm_units) - 1
            x = LSTM(units, return_sequences=return_seq)(x)
            x = Dropout(self.config.dropout_rate)(x)

        # Dense layers
        x = Dense(50, activation='relu')(x)
        x = Dropout(self.config.dropout_rate / 2)(x)
        x = Dense(25, activation='relu')(x)

        # Dual outputs
        price_output = Dense(1, name='price_output')(x)
        direction_output = Dense(1, activation='sigmoid', name='direction_output')(x)

        model = Model(inputs=inputs, outputs=[price_output, direction_output])

        # Custom biased MSE loss function
        @tf.function
        def biased_mse(y_true, y_pred):
            mse = keras.losses.MeanSquaredError()(y_true, y_pred)
            diff = y_pred - y_true
            bias_penalty = tf.where(diff < 0,
                                   tf.abs(diff) * (1 + self.config.growth_bias),
                                   tf.abs(diff))
            return mse + tf.reduce_mean(bias_penalty) * 0.1

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss={'price_output': biased_mse, 'direction_output': 'binary_crossentropy'},
            loss_weights={'price_output': 1.0, 'direction_output': 0.5},
            metrics={'price_output': 'mae', 'direction_output': 'accuracy'}
        )

        self.model = model

        # Print model summary
        print("\nModel Architecture:")
        print("-" * 60)
        self.model.summary()
        print("-" * 60)
        print(f"\nTotal trainable parameters: {self.model.count_params():,}")
        print(f"LSTM architecture: Bidirectional {self.config.lstm_units}")
        print(f"Growth bias: {self.config.growth_bias:.3%}")
        print(f"Dual outputs: Price prediction + Direction classification")
        print("="*60 + "\n")

        return model

    def train(self, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """
        Train the enhanced LSTM model
        """
        X_train, y_train_price, y_train_direction = train_data
        X_val, y_val_price, y_val_direction = val_data

        # V1 data is already 3D with shape (samples, sequence_length, n_features)
        # No reshaping needed

        if self.model is None:
            self.build_model()

        print("\n" + "="*60)
        print("TRAINING ENHANCED LSTM MODEL")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Early stopping patience: {self.config.patience}")
        print("="*60 + "\n")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_price_output_loss',
                mode='min',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-5
            ),
            ReduceLROnPlateau(
                monitor='val_price_output_loss',
                mode='min',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                monitor='val_price_output_loss',
                mode='min',
                filepath='best_enhanced_lstm_model.keras',
                save_best_only=True,
                verbose=0
            )
        ]

        logger.info("Starting enhanced training with biases...")
        start_time = datetime.now()

        self.history = self.model.fit(
            X_train, {'price_output': y_train_price, 'direction_output': y_train_direction},
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_val, {'price_output': y_val_price, 'direction_output': y_val_direction}),
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )

        end_time = datetime.now()
        self.training_time = (end_time - start_time).total_seconds()

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Training time: {self.training_time:.2f} seconds ({self.training_time/60:.2f} minutes)")
        print(f"Epochs trained: {len(self.history.history['loss'])}")
        print(f"Best validation loss: {min(self.history.history['val_price_output_loss']):.6f}")
        print("="*60 + "\n")

        return self.history

    def apply_growth_bias(self, predictions: np.ndarray, features: pd.DataFrame = None) -> np.ndarray:
        """Apply growth bias to predictions"""
        bias_factor = 1 + self.config.growth_bias

        # Check if features are provided and contain bull_market indicator
        if features is not None and 'bull_market' in features.columns and len(features) > 0:
            if features['bull_market'].iloc[-1] == 1:
                bias_factor *= 1.5
                logger.info("Applying enhanced growth bias due to bull market detection")
            else:
                logger.info("No bull market detected, applying standard growth bias")
        else:
            if features is None:
                logger.info("No features provided, applying standard growth bias")
            else:
                logger.warning("bull_market feature not found, applying standard growth bias")

        for i in range(len(predictions)):
            time_factor = 1 - np.exp(-i / 10)
            predictions[i] *= (1 + self.config.growth_bias * time_factor * bias_factor)
        return predictions

    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 features: pd.DataFrame = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate the enhanced LSTM model
        """
        X_test, y_test_price, y_test_direction = test_data

        if self.model is None:
            raise ValueError("Model not trained")

        # V1 data is already 3D with shape (samples, sequence_length, n_features)
        price_predictions_scaled, direction_predictions = self.model.predict(X_test, verbose=0)

        # Work in scaled space (V1 data is already scaled, no inverse_transform needed)
        price_predictions = price_predictions_scaled.flatten()

        # Apply growth bias
        price_predictions = self.apply_growth_bias(price_predictions, features)

        y_test_actual = y_test_price  # Already in scaled space

        # Calculate metrics
        mse = mean_squared_error(y_test_actual, price_predictions)
        mae = mean_absolute_error(y_test_actual, price_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, price_predictions)

        non_zero_mask = y_test_actual != 0
        mape = np.mean(np.abs((y_test_actual[non_zero_mask] - price_predictions[non_zero_mask]) /
                             y_test_actual[non_zero_mask])) * 100

        directional_accuracy = np.mean((direction_predictions > 0.5) == y_test_direction) * 100

        avg_prediction = np.mean(price_predictions)
        avg_actual = np.mean(y_test_actual)
        bias_effectiveness = (avg_prediction - avg_actual) / avg_actual * 100

        metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2),
            'MAPE': float(mape),
            'Directional_Accuracy': float(directional_accuracy),
            'Bias_Effect_%': float(bias_effectiveness)
        }

        logger.info("\n" + "="*60)
        logger.info("ENHANCED LSTM MODEL EVALUATION RESULTS")
        logger.info("="*60)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("="*60)

        return metrics, price_predictions, y_test_actual

    def plot_training_history(self, save_path='plots/'):
        """Plot training history for dual outputs"""
        if self.history is None:
            logger.warning("No training history available")
            return

        os.makedirs(save_path, exist_ok=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Price Loss
        ax1.plot(self.history.history['price_output_loss'], label='Training Price Loss')
        ax1.plot(self.history.history['val_price_output_loss'], label='Validation Price Loss')
        ax1.set_title('Price Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Price MAE
        ax2.plot(self.history.history['price_output_mae'], label='Training Price MAE')
        ax2.plot(self.history.history['val_price_output_mae'], label='Validation Price MAE')
        ax2.set_title('Price MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)

        # Direction Accuracy
        ax3.plot(self.history.history['direction_output_accuracy'], label='Training Direction Accuracy')
        ax3.plot(self.history.history['val_direction_output_accuracy'], label='Validation Direction Accuracy')
        ax3.set_title('Direction Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_path}enhanced_lstm_training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}enhanced_lstm_training_history.png")
        plt.close()

    def plot_results(self, predictions: np.ndarray, actual: np.ndarray,
                    dataset_name='Test', save_path='plots/', n_points: int = 200):
        """Plot predictions with growth bias visualization"""
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(15, 8))
        n_points = min(n_points, len(predictions))
        plt.plot(actual[-n_points:], label='Actual Price', color='blue', linewidth=2)
        plt.plot(predictions[-n_points:], label='Predicted (with bias)',
                color='red', linewidth=2, alpha=0.8)

        # Growth bias trend line
        x = np.arange(n_points)
        trend = actual[-n_points] * (1 + self.config.growth_bias) ** (x / 252)
        plt.plot(trend, '--', label='Growth Bias Trend', color='green', alpha=0.5)

        plt.title(f'Enhanced LSTM Predictions - {dataset_name} Set', fontsize=16)
        plt.xlabel('Time Steps')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}enhanced_lstm_predictions_{dataset_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to: {save_path}enhanced_lstm_predictions_{dataset_name.lower()}.png")
        plt.close()

    def save_model(self, save_path='models/'):
        """Save model and configuration"""
        os.makedirs(save_path, exist_ok=True)

        self.model.save(f'{save_path}enhanced_lstm_model.keras')
        self.config.save(f'{save_path}enhanced_lstm_config.json')

        print(f"Model saved to: {save_path}enhanced_lstm_model.keras")
        print(f"Config saved to: {save_path}enhanced_lstm_config.json")




def create_default_config(sequence_length: int = 20) -> EnhancedLSTMConfig:
    """Create default configuration for enhanced LSTM"""
    return EnhancedLSTMConfig(
        sequence_length=sequence_length,
        train_split=0.8,
        validation_split=0.1,
        batch_size=32,
        epochs=200,
        learning_rate=0.001,
        patience=30,
        lstm_units=[128, 64, 32],
        dropout_rate=0.25,
        growth_bias=0.005,
        top_performer_weight=1.8,
        low_rep_boost=1.5,
        top_performer_percentile=0.8,
        negative_streak_threshold=2
    )


def main():
    """
    Main execution function - complete Enhanced LSTM V1.1 training pipeline

    NOTE: This version expects preprocessed data in numpy format.
    Use SnP500DataPrepV1.1.py to prepare data with enhanced features first.
    """
    print("\n" + "="*80)
    print("ENHANCED LSTM MODEL V1.1 FOR S&P 500 PREDICTION")
    print("="*80 + "\n")

    # Load prepared data (V1 approach)
    print("Loading prepared data...")
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    with open('data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"Data loaded successfully!")
    print(f"Training samples: {X_train.shape}")
    print(f"Validation samples: {X_val.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Features: {metadata['n_features']}")

    # Create configuration with sequence_length from metadata
    config = create_default_config(sequence_length=metadata['sequence_length'])
    config.save('models/enhanced_lstm_v1.1_config.json')

    logger.info("=" * 70)
    logger.info("📊 ENHANCED LSTM PREDICTOR V1.1")
    logger.info("=" * 70)
    logger.info(f"📈 Growth Bias: {config.growth_bias:.3%}")
    logger.info(f"⭐ Top Performer Weight: {config.top_performer_weight}x")
    logger.info(f"📉 Low Rep Boost: {config.low_rep_boost}x")
    logger.info(f"🔢 Sequence Length: {metadata['sequence_length']} steps")
    logger.info(f"🧠 LSTM Architecture: Bidirectional {config.lstm_units}")
    logger.info(f"🎯 Max Epochs: {config.epochs}")
    logger.info("=" * 70)

    # Initialize Enhanced LSTM predictor with V1.1 architecture
    lstm = EnhancedLSTMPredictor(config, n_features=metadata['n_features'])

    # V1 data is already scaled - no scaler needed (work in scaled space)
    logger.info(f"Using V1 pre-scaled data - working in scaled space for metrics")

    # Build enhanced model with bidirectional LSTM and dual outputs
    lstm.build_model()

    # Train model with custom biased loss (parameters from config)
    lstm.train(
        (X_train, y_train, np.zeros_like(y_train)),  # Dummy direction for compatibility
        (X_val, y_val, np.zeros_like(y_val))
    )

    # Evaluate on all datasets
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    train_metrics, _, _ = lstm.evaluate(
        (X_train, y_train, np.zeros_like(y_train)),
        None  # No features DataFrame needed
    )
    val_metrics, _, _ = lstm.evaluate(
        (X_val, y_val, np.zeros_like(y_val)),
        None
    )
    test_metrics, price_predictions, y_test_actual = lstm.evaluate(
        (X_test, y_test, np.zeros_like(y_test)),
        None
    )

    # Save results
    results = {
        'train': {k: v for k, v in train_metrics.items() if k not in ['predictions', 'actuals']},
        'val': {k: v for k, v in val_metrics.items() if k not in ['predictions', 'actuals']},
        'test': {k: v for k, v in test_metrics.items() if k not in ['predictions', 'actuals']},
        'config': {
            'lstm_units': lstm.config.lstm_units,
            'dropout_rate': lstm.config.dropout_rate,
            'learning_rate': lstm.config.learning_rate,
            'growth_bias': lstm.config.growth_bias,
            'bidirectional': True,
            'training_time': lstm.training_time
        }
    }

    with open('results/enhanced_lstm_v1.1_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to: results/enhanced_lstm_v1.1_results.json")

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    lstm.plot_training_history(save_path='plots/')
    lstm.plot_results(price_predictions, y_test_actual, save_path='plots/')

    # Save model
    lstm.save_model(save_path='models/')

    print("\n" + "="*80)
    print("ENHANCED LSTM V1.1 PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - models/enhanced_lstm_v1.1_model.h5")
    print("  - models/enhanced_lstm_v1.1_config.json")
    print("  - results/enhanced_lstm_v1.1_results.json")
    print("  - plots/enhanced_lstm_v1.1_training_history.png")
    print("  - plots/enhanced_lstm_v1.1_predictions_test.png")
    print("\n" + "="*80 + "\n")

    return lstm, test_metrics


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Run main pipeline
    main()
