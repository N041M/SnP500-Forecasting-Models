"""
Enhanced Hybrid LSTM + Transformer Model for S&P 500 Prediction V3.0
=====================================================================
This model consumes preprocessed data from SnP500DataPrepV3.py

Features:
- Extended input features (technical, fundamental, macro, inequality, sector)
- Bidirectional LSTM layers for local pattern capture
- Transformer layers with multi-head attention for long-range dependencies
- Feature attention mechanism to track feature importance
- Dual outputs (return prediction + direction classification)
- Custom biased loss function with growth bias
- Covariance-based feature importance validation

Architecture Flow:
Input → Feature Attention → Bidirectional LSTM → Transformer → Dense → Dual Outputs

Author: Ronald
Version: 3.0
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
import pickle
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import warnings
import logging
from scipy import stats

# TensorFlow/Keras imports
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization, LSTM, Bidirectional,
    MultiHeadAttention, GlobalAveragePooling1D, Add, Multiply,
    Concatenate, Permute, Reshape, Lambda, Softmax
)
from keras.optimizers import Adam
from keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler, TensorBoard
)
from keras.losses import Huber
from keras.regularizers import l2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfigV3:
    """Configuration for V3 Hybrid model"""

    # Data parameters
    sequence_length: int = 60
    n_features: int = 50

    # Training parameters
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 0.0005
    patience: int = 30
    warmup_epochs: int = 10

    # LSTM parameters
    lstm_units: List[int] = field(default_factory=lambda: [128, 64])
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.1

    # Transformer parameters
    d_model: int = 64
    num_heads: int = 4
    num_transformer_layers: int = 3
    dff: int = 256
    transformer_dropout: float = 0.1

    # Dense parameters
    dense_units: List[int] = field(default_factory=lambda: [64, 32])
    dense_dropout: float = 0.3

    # Regularization
    l2_reg: float = 0.001

    # Bias parameters
    growth_bias: float = 0.005
    under_prediction_weight: float = 2.0

    # Feature attention
    use_feature_attention: bool = True

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        config_dict = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'warmup_epochs': self.warmup_epochs,
            'lstm_units': self.lstm_units,
            'lstm_dropout': self.lstm_dropout,
            'lstm_recurrent_dropout': self.lstm_recurrent_dropout,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_transformer_layers': self.num_transformer_layers,
            'dff': self.dff,
            'transformer_dropout': self.transformer_dropout,
            'dense_units': self.dense_units,
            'dense_dropout': self.dense_dropout,
            'l2_reg': self.l2_reg,
            'growth_bias': self.growth_bias,
            'under_prediction_weight': self.under_prediction_weight,
            'use_feature_attention': self.use_feature_attention
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ModelConfigV3':
        """Load from flat config (saved model config)"""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'ModelConfigV3':
        """Load from hierarchical config file (config/model_config.json)"""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            cfg = json.load(f)

        return cls(
            # Data
            sequence_length=cfg['data']['sequence_length'],
            n_features=cfg['data']['n_features'],
            # Training
            batch_size=cfg['training']['batch_size'],
            epochs=cfg['training']['epochs'],
            learning_rate=cfg['training']['learning_rate'],
            patience=cfg['training']['patience'],
            warmup_epochs=cfg['training']['warmup_epochs'],
            # LSTM
            lstm_units=cfg['lstm']['units'],
            lstm_dropout=cfg['lstm']['dropout'],
            lstm_recurrent_dropout=cfg['lstm']['recurrent_dropout'],
            # Transformer
            d_model=cfg['transformer']['d_model'],
            num_heads=cfg['transformer']['num_heads'],
            num_transformer_layers=cfg['transformer']['num_layers'],
            dff=cfg['transformer']['dff'],
            transformer_dropout=cfg['transformer']['dropout'],
            # Dense
            dense_units=cfg['dense']['units'],
            dense_dropout=cfg['dense']['dropout'],
            # Regularization
            l2_reg=cfg['regularization']['l2'],
            # Bias
            growth_bias=cfg['bias']['growth_bias'],
            under_prediction_weight=cfg['bias']['under_prediction_weight'],
            # Features
            use_feature_attention=cfg['features']['use_feature_attention']
        )


class PositionalEncoding(keras.layers.Layer):
    """Positional encoding layer for Transformer component"""

    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(sequence_length, d_model)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _positional_encoding(self, sequence_length, d_model):
        angle_rads = self._get_angles(
            np.arange(sequence_length)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config


class TransformerBlock(keras.layers.Layer):
    """Transformer encoder block with pre-norm architecture"""

    def __init__(self, d_model: int, num_heads: int, dff: int,
                 dropout_rate: float = 0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            Dense(dff, activation='gelu'),
            Dropout(dropout_rate),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        # Pre-norm architecture (better gradient flow)
        x_norm = self.layernorm1(inputs)
        attn_output = self.att(x_norm, x_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output

        x_norm = self.layernorm2(out1)
        ffn_output = self.ffn(x_norm)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config


class FeatureAttention(keras.layers.Layer):
    """
    Feature attention layer to learn feature importance

    Computes attention weights over features at each timestep,
    allowing the model to focus on the most relevant features.
    """

    def __init__(self, n_features: int, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)
        self.n_features = n_features
        self.attention_dense = Dense(n_features, activation='tanh')
        self.attention_weights = Dense(n_features, activation='softmax')

    def call(self, inputs):
        # inputs: (batch, sequence, features)
        attention_scores = self.attention_dense(inputs)
        attention_weights = self.attention_weights(attention_scores)

        # Apply attention
        attended = inputs * attention_weights

        return attended, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({'n_features': self.n_features})
        return config


class FeatureImportanceCallback(keras.callbacks.Callback):
    """
    Callback to track feature importance during training

    Extracts attention weights from FeatureAttention layer
    to monitor which features the model focuses on.
    """

    def __init__(self, validation_data: Tuple[np.ndarray, np.ndarray],
                 feature_names: List[str], log_interval: int = 10):
        super().__init__()
        self.validation_data = validation_data
        self.feature_names = feature_names
        self.log_interval = log_interval
        self.importance_history = []
        self._attention_model = None  # Cache the extraction model

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval != 0:
            return

        # Create attention extraction model once and cache it
        if self._attention_model is None:
            attention_layer = None
            for layer in self.model.layers:
                if isinstance(layer, FeatureAttention):
                    attention_layer = layer
                    break

            if attention_layer is None:
                return

            self._attention_model = Model(
                inputs=self.model.input,
                outputs=attention_layer.output[1]  # attention_weights
            )

        # Get attention weights using cached model
        X_val = self.validation_data[0][:100]  # Sample
        attention_weights = self._attention_model.predict(X_val, verbose=0)

        # Average over samples and timesteps
        mean_attention = np.mean(attention_weights, axis=(0, 1))

        self.importance_history.append({
            'epoch': epoch,
            'importance': dict(zip(self.feature_names, mean_attention))
        })


class HybridModelV3:
    """
    Enhanced Hybrid LSTM+Transformer Model V3.0

    Key improvements:
    - Feature attention for interpretability
    - Pre-norm Transformer architecture
    - GELU activation
    - Gradient clipping
    - Cosine learning rate schedule
    """

    def __init__(self, config: ModelConfigV3):
        self.config = config
        self.model = None
        self.history = None
        self.training_time = None
        self.feature_importance = {}
        self.attention_weights = None

        logger.info("HybridModelV3 initialized")
        logger.info(f"  Sequence length: {config.sequence_length}")
        logger.info(f"  Features: {config.n_features}")
        logger.info(f"  LSTM units: {config.lstm_units}")
        logger.info(f"  Transformer: {config.num_transformer_layers} layers, {config.num_heads} heads")

    def build_model(self) -> Model:
        """Build the enhanced hybrid model"""
        print("\n" + "=" * 60)
        print("BUILDING HYBRID LSTM+TRANSFORMER MODEL V3.0")
        print("=" * 60)

        # Input layer
        inputs = Input(
            shape=(self.config.sequence_length, self.config.n_features),
            name='input'
        )

        x = inputs

        # ============================================
        # FEATURE ATTENTION (Optional)
        # ============================================
        if self.config.use_feature_attention:
            x, attention_weights = FeatureAttention(
                n_features=self.config.n_features,
                name='feature_attention'
            )(x)
            print(f"  Feature Attention: enabled")

        # ============================================
        # BIDIRECTIONAL LSTM COMPONENT
        # ============================================
        for i, units in enumerate(self.config.lstm_units):
            x = Bidirectional(
                LSTM(
                    units,
                    return_sequences=True,
                    dropout=self.config.lstm_dropout,
                    recurrent_dropout=self.config.lstm_recurrent_dropout,
                    kernel_regularizer=l2(self.config.l2_reg)
                ),
                name=f'bilstm_{i+1}'
            )(x)
            x = LayerNormalization(name=f'lstm_norm_{i+1}')(x)

        print(f"  Bidirectional LSTM: {self.config.lstm_units}")

        # ============================================
        # PROJECTION TO TRANSFORMER DIMENSION
        # ============================================
        lstm_output_dim = self.config.lstm_units[-1] * 2  # Bidirectional

        if lstm_output_dim != self.config.d_model:
            x = Dense(
                self.config.d_model,
                kernel_regularizer=l2(self.config.l2_reg),
                name='projection'
            )(x)

        # ============================================
        # TRANSFORMER COMPONENT
        # ============================================
        x = PositionalEncoding(
            self.config.sequence_length,
            self.config.d_model,
            name='positional_encoding'
        )(x)

        for i in range(self.config.num_transformer_layers):
            x = TransformerBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                dff=self.config.dff,
                dropout_rate=self.config.transformer_dropout,
                name=f'transformer_{i+1}'
            )(x)

        print(f"  Transformer: d_model={self.config.d_model}, heads={self.config.num_heads}, layers={self.config.num_transformer_layers}")

        # ============================================
        # AGGREGATION
        # ============================================
        x = GlobalAveragePooling1D(name='global_pool')(x)

        # ============================================
        # DENSE LAYERS
        # ============================================
        for i, units in enumerate(self.config.dense_units):
            x = Dense(
                units,
                activation='gelu',
                kernel_regularizer=l2(self.config.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = Dropout(self.config.dense_dropout, name=f'dense_dropout_{i+1}')(x)

        print(f"  Dense layers: {self.config.dense_units}")

        # ============================================
        # DUAL OUTPUTS
        # ============================================
        # Price/Return prediction
        price_output = Dense(1, name='price_output')(x)

        # Direction classification
        direction_output = Dense(1, activation='sigmoid', name='direction_output')(x)

        print(f"  Outputs: price (regression), direction (classification)")

        # Create model
        self.model = Model(
            inputs=inputs,
            outputs=[price_output, direction_output],
            name='HybridLSTMTransformer_V3'
        )

        # Custom loss function with under-prediction penalty
        def biased_mse(y_true, y_pred):
            error = y_true - y_pred
            weights = tf.where(
                error > 0,
                self.config.under_prediction_weight,
                1.0
            )
            return tf.reduce_mean(weights * tf.square(error))

        # Compile with gradient clipping
        optimizer = Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0
        )

        self.model.compile(
            optimizer=optimizer,
            loss={
                'price_output': biased_mse,
                'direction_output': 'binary_crossentropy'
            },
            loss_weights={
                'price_output': 1.0,
                'direction_output': 0.3
            },
            metrics={
                'price_output': ['mae', 'mse'],
                'direction_output': ['accuracy']
            }
        )

        # Print summary
        print("\n" + "-" * 60)
        self.model.summary()
        print("-" * 60)
        print(f"Total parameters: {self.model.count_params():,}")
        print("=" * 60 + "\n")

        return self.model

    def _cosine_lr_schedule(self, epoch: int, total_epochs: int,
                            warmup_epochs: int, base_lr: float) -> float:
        """Cosine learning rate schedule with warmup"""
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              feature_names: List[str] = None) -> keras.callbacks.History:
        """Train the model"""

        if self.model is None:
            self.build_model()

        # Derive direction labels
        y_train_direction = (y_train > 0).astype(np.float32)
        y_val_direction = (y_val > 0).astype(np.float32)

        print("\n" + "=" * 60)
        print("TRAINING HYBRID MODEL V3.0")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Features: {X_train.shape[-1]}")
        print(f"Sequence length: {X_train.shape[1]}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Direction dist (train): {y_train_direction.mean()*100:.1f}% up")
        print("=" * 60 + "\n")

        # Learning rate schedule
        def lr_schedule(epoch):
            return self._cosine_lr_schedule(
                epoch,
                self.config.epochs,
                self.config.warmup_epochs,
                self.config.learning_rate
            )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_price_output_loss',
                mode='min',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1
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
                filepath='models/best_model_v3.keras',
                monitor='val_price_output_loss',
                mode='min',
                save_best_only=True,
                verbose=0
            ),
            LearningRateScheduler(lr_schedule, verbose=0)
        ]

        # Feature importance callback
        if feature_names and self.config.use_feature_attention:
            fi_callback = FeatureImportanceCallback(
                validation_data=(X_val, y_val),
                feature_names=feature_names,
                log_interval=10
            )
            callbacks.append(fi_callback)

        start_time = datetime.now()

        self.history = self.model.fit(
            X_train,
            {'price_output': y_train, 'direction_output': y_train_direction},
            validation_data=(
                X_val,
                {'price_output': y_val, 'direction_output': y_val_direction}
            ),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )

        end_time = datetime.now()
        self.training_time = (end_time - start_time).total_seconds()

        # Extract feature importance if available
        if feature_names and self.config.use_feature_attention:
            self._extract_feature_importance(X_val, feature_names)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Training time: {self.training_time:.1f}s ({self.training_time/60:.1f}min)")
        print(f"Epochs trained: {len(self.history.history['loss'])}")
        print(f"Best val loss: {min(self.history.history['val_price_output_loss']):.6f}")
        print("=" * 60 + "\n")

        return self.history

    def _extract_feature_importance(self, X_val: np.ndarray,
                                     feature_names: List[str]):
        """Extract feature importance from attention weights"""
        # Find feature attention layer
        attention_layer = None
        for layer in self.model.layers:
            if isinstance(layer, FeatureAttention):
                attention_layer = layer
                break

        if attention_layer is None:
            return

        # Create extraction model
        attention_model = Model(
            inputs=self.model.input,
            outputs=attention_layer.output[1]
        )

        # Get weights
        attention_weights = attention_model.predict(X_val[:500], verbose=0)
        mean_attention = np.mean(attention_weights, axis=(0, 1))

        # Store
        self.feature_importance = dict(zip(feature_names, mean_attention))
        self.attention_weights = attention_weights

        # Log top features
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info("Top 10 features by attention:")
        for name, weight in sorted_features[:10]:
            logger.info(f"  {name}: {weight:.4f}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained")

        y_test_direction = (y_test > 0).astype(np.float32)

        # Predictions
        price_pred, direction_pred = self.model.predict(X_test, verbose=0)
        price_pred = price_pred.flatten()
        direction_pred = (direction_pred.flatten() > 0.5).astype(int)

        # Apply growth bias
        price_pred_biased = price_pred * (1 + self.config.growth_bias)

        # Metrics
        mse = mean_squared_error(y_test, price_pred_biased)
        mae = mean_absolute_error(y_test, price_pred_biased)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, price_pred_biased)

        # MAPE (handle zeros)
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - price_pred_biased[mask]) / y_test[mask])) * 100

        # Direction accuracy
        direction_accuracy = accuracy_score(y_test_direction, direction_pred) * 100

        metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MAPE': float(mape),
            'Direction_Accuracy': float(direction_accuracy),
            'Growth_Bias': self.config.growth_bias
        }

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print("=" * 60 + "\n")

        return metrics, price_pred_biased, y_test

    def plot_training_history(self, save_path: str = 'plots/'):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return

        os.makedirs(save_path, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Price loss
        axes[0, 0].plot(self.history.history['price_output_loss'], label='Train')
        axes[0, 0].plot(self.history.history['val_price_output_loss'], label='Val')
        axes[0, 0].set_title('Price Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Price MAE
        axes[0, 1].plot(self.history.history['price_output_mae'], label='Train')
        axes[0, 1].plot(self.history.history['val_price_output_mae'], label='Val')
        axes[0, 1].set_title('Price MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Direction loss
        axes[1, 0].plot(self.history.history['direction_output_loss'], label='Train')
        axes[1, 0].plot(self.history.history['val_direction_output_loss'], label='Val')
        axes[1, 0].set_title('Direction Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Direction accuracy
        axes[1, 1].plot(self.history.history['direction_output_accuracy'], label='Train')
        axes[1, 1].plot(self.history.history['val_direction_output_accuracy'], label='Val')
        axes[1, 1].set_title('Direction Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}training_history_v3.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Training history saved to {save_path}training_history_v3.png")

    def plot_predictions(self, predictions: np.ndarray, actuals: np.ndarray,
                         save_path: str = 'plots/', n_points: int = 200):
        """Plot predictions vs actuals"""
        os.makedirs(save_path, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        n = min(n_points, len(predictions))

        # Time series comparison
        axes[0].plot(actuals[-n:], label='Actual', color='blue', linewidth=1.5)
        axes[0].plot(predictions[-n:], label='Predicted', color='red',
                    linewidth=1.5, alpha=0.8)
        axes[0].set_title('Predictions vs Actuals (Last {} points)'.format(n))
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Returns')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Scatter plot
        axes[1].scatter(actuals, predictions, alpha=0.5, s=10)
        axes[1].plot([actuals.min(), actuals.max()],
                    [actuals.min(), actuals.max()],
                    'r--', linewidth=2, label='Perfect prediction')
        axes[1].set_title('Prediction Scatter Plot')
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}predictions_v3.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Predictions plot saved to {save_path}predictions_v3.png")

    def plot_feature_importance(self, save_path: str = 'plots/', top_n: int = 30):
        """Plot feature importance from attention weights"""
        if not self.feature_importance:
            logger.warning("No feature importance available")
            return

        os.makedirs(save_path, exist_ok=True)

        # Sort features
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        names = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        fig, ax = plt.subplots(figsize=(12, 10))

        y_pos = np.arange(len(names))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Attention Weight')
        ax.set_title(f'Top {top_n} Features by Attention Weight')
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{save_path}feature_importance_v3.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature importance plot saved to {save_path}feature_importance_v3.png")

    # =========================================================================
    # POST-TRAINING ANALYSIS METHODS
    # =========================================================================

    def evaluate_split(self, X: np.ndarray, y: np.ndarray,
                       split_name: str = 'split') -> Dict:
        """Detailed evaluation on a data split. Returns metrics + raw arrays."""
        if self.model is None:
            raise ValueError("Model not trained")

        y_direction = (y > 0).astype(np.float32)
        price_pred_raw, direction_pred_raw = self.model.predict(X, verbose=0)
        price_pred_raw = price_pred_raw.flatten()
        direction_pred_raw = direction_pred_raw.flatten()

        price_pred = price_pred_raw * (1 + self.config.growth_bias)
        direction_pred_binary = (direction_pred_raw > 0.5).astype(int)
        y_direction_int = y_direction.astype(int)

        errors = price_pred - y
        mse  = mean_squared_error(y, price_pred)
        mae  = mean_absolute_error(y, price_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y, price_pred)
        mask = y != 0
        mape = np.mean(np.abs((y[mask] - price_pred[mask]) / y[mask])) * 100

        error_skew = float(stats.skew(errors))
        error_kurt = float(stats.kurtosis(errors))
        abs_errors = np.abs(errors)
        within_05pct = float(np.mean(abs_errors < 0.005) * 100)
        within_1pct  = float(np.mean(abs_errors < 0.010) * 100)
        within_2pct  = float(np.mean(abs_errors < 0.020) * 100)

        dir_accuracy = accuracy_score(y_direction_int, direction_pred_binary) * 100
        cm = confusion_matrix(y_direction_int, direction_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        up_precision   = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        up_recall      = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        down_precision = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0.0
        down_recall    = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0

        # ── Collapse / health diagnostics ──────────────────────────────────
        pred_mean = float(np.mean(price_pred))
        pred_std  = float(np.std(price_pred))
        act_mean  = float(np.mean(y))
        act_std   = float(np.std(y))
        # Variance ratio: how much of actual variance does the model capture?
        # A well-behaved model should have pred_std in the same order as act_std.
        variance_ratio = pred_std / act_std if act_std > 0 else 0.0
        # Model is considered collapsed if its std is < 5 % of actual std
        is_collapsed = variance_ratio < 0.05

        # Naive benchmark: always predict the actual mean of this split
        naive_pred = np.full_like(y, act_mean)
        naive_mae  = float(mean_absolute_error(y, naive_pred))
        naive_r2   = float(r2_score(y, naive_pred))          # will be 0 by definition
        naive_dir  = float(accuracy_score(y_direction_int,
                                          (naive_pred > 0).astype(int)) * 100)
        # Skill score: how much better than naive? (positive = better)
        mae_skill = float((naive_mae - mae) / naive_mae * 100) if naive_mae > 0 else 0.0

        return {
            'split_name':          split_name,
            'n_samples':           int(len(y)),
            'MSE':                 float(mse),
            'MAE':                 float(mae),
            'RMSE':                float(rmse),
            'R2':                  float(r2),
            'MAPE':                float(mape),
            'error_mean':          float(np.mean(errors)),
            'error_std':           float(np.std(errors)),
            'error_skew':          error_skew,
            'error_kurt':          error_kurt,
            'within_05pct':        within_05pct,
            'within_1pct':         within_1pct,
            'within_2pct':         within_2pct,
            'Direction_Accuracy':  float(dir_accuracy),
            'Up_Precision':        float(up_precision),
            'Up_Recall':           float(up_recall),
            'Down_Precision':      float(down_precision),
            'Down_Recall':         float(down_recall),
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            # Prediction spread / collapse diagnostics
            'pred_mean':       pred_mean,
            'pred_std':        pred_std,
            'actual_mean':     act_mean,
            'actual_std':      act_std,
            'variance_ratio':  float(variance_ratio),
            'is_collapsed':    is_collapsed,
            # Naive-mean benchmark
            'naive_mae':       naive_mae,
            'naive_dir_acc':   naive_dir,
            'mae_skill_score': mae_skill,
            # Raw arrays for plotting
            'predictions':         price_pred,
            'actuals':             y,
            'errors':              errors,
            'direction_pred_prob': direction_pred_raw,
            'direction_pred':      direction_pred_binary,
            'direction_actual':    y_direction_int,
        }

    def analyze_training_dynamics(self) -> Dict:
        """Analyse training history to diagnose convergence and overfitting."""
        if self.history is None:
            return {}

        h = self.history.history
        epochs_trained = len(h['loss'])
        best_epoch = int(np.argmin(h['val_price_output_loss']))
        best_val_loss = float(h['val_price_output_loss'][best_epoch])
        train_loss_at_best = float(h['price_output_loss'][best_epoch])
        overfit_gap   = best_val_loss - train_loss_at_best
        overfit_ratio = best_val_loss / train_loss_at_best if train_loss_at_best > 0 else None

        final_train_dir_acc = float(h['direction_output_accuracy'][-1]) * 100
        final_val_dir_acc   = float(h['val_direction_output_accuracy'][-1]) * 100
        best_val_dir_acc    = float(max(h['val_direction_output_accuracy'])) * 100

        final_train_mae = float(h['price_output_mae'][-1])
        final_val_mae   = float(h['val_price_output_mae'][-1])
        best_val_mae    = float(min(h['val_price_output_mae']))

        last_10_val = h['val_price_output_loss'][-10:]
        convergence_std = float(np.std(last_10_val))

        lr_history = h.get('lr', None)
        final_lr   = float(lr_history[-1]) if lr_history else None

        return {
            'epochs_trained':           epochs_trained,
            'max_epochs':               self.config.epochs,
            'early_stopped':            epochs_trained < self.config.epochs,
            'best_epoch':               best_epoch + 1,
            'best_val_price_loss':      best_val_loss,
            'train_price_loss_at_best': train_loss_at_best,
            'overfit_gap':              float(overfit_gap),
            'overfit_ratio':            float(overfit_ratio) if overfit_ratio is not None else None,
            'final_train_dir_accuracy': final_train_dir_acc,
            'final_val_dir_accuracy':   final_val_dir_acc,
            'best_val_dir_accuracy':    best_val_dir_acc,
            'final_train_mae':          final_train_mae,
            'final_val_mae':            final_val_mae,
            'best_val_mae':             best_val_mae,
            'convergence_std_last10':   convergence_std,
            'final_lr':                 final_lr,
            'training_time_sec':        self.training_time,
        }

    def plot_error_analysis(self, results: Dict, save_path: str = 'plots/'):
        """4-panel plot: residual histogram, Q-Q plot, residuals over time,
        and absolute error vs absolute actual return."""
        os.makedirs(save_path, exist_ok=True)
        errors     = results['errors']
        actuals    = results['actuals']
        split_name = results['split_name']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Error Analysis — {split_name.upper()} Split',
                     fontsize=14, fontweight='bold')

        # 1. Residual histogram with fitted normal
        ax = axes[0, 0]
        ax.hist(errors, bins=60, color='steelblue', alpha=0.7,
                edgecolor='white', density=True)
        mu, sigma = errors.mean(), errors.std()
        x_norm = np.linspace(errors.min(), errors.max(), 300)
        ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', lw=2,
                label=f'Normal fit\nμ={mu:.5f}\nσ={sigma:.5f}')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, lw=1)
        ax.set_title('Residual Distribution')
        ax.set_xlabel('Prediction Error (pred − actual)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Q-Q plot
        ax = axes[0, 1]
        (osm, osr), (slope, intercept, r) = stats.probplot(errors, dist='norm')
        ax.scatter(osm, osr, s=5, alpha=0.5, color='steelblue', label='Quantiles')
        ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2,
                label=f'Reference  (R={r:.4f})')
        ax.set_title(f'Normal Q-Q Plot  (R={r:.4f})')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Residuals over time
        ax = axes[1, 0]
        ax.plot(errors, color='steelblue', linewidth=0.7, alpha=0.8)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
        ax.fill_between(range(len(errors)), errors, 0,
                        where=(errors > 0), color='green', alpha=0.25,
                        label='Over-predicted')
        ax.fill_between(range(len(errors)), errors, 0,
                        where=(errors < 0), color='red', alpha=0.25,
                        label='Under-predicted')
        ax.set_title('Residuals Over Time')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Error (pred − actual)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. |Error| vs |Actual return|
        ax = axes[1, 1]
        abs_act = np.abs(actuals)
        abs_err = np.abs(errors)
        ax.scatter(abs_act, abs_err, alpha=0.25, s=6, color='steelblue')
        z = np.polyfit(abs_act, abs_err, 1)
        xline = np.linspace(abs_act.min(), abs_act.max(), 200)
        ax.plot(xline, np.polyval(z, xline), 'r-', lw=2,
                label=f'Trend  (slope={z[0]:.3f})')
        ax.set_title('|Error| vs |Actual Return|')
        ax.set_xlabel('|Actual Return|')
        ax.set_ylabel('|Prediction Error|')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = f'{save_path}error_analysis_{split_name}_v3.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Error analysis plot saved to {filepath}")

    def plot_trading_simulation(self, results: Dict,
                                save_path: str = 'plots/') -> Dict:
        """Simulate a long/short strategy driven by direction predictions.
        Returns a dict with Sharpe, total return, and max drawdown."""
        os.makedirs(save_path, exist_ok=True)
        actuals          = results['actuals']
        direction_pred   = results['direction_pred']
        direction_actual = results['direction_actual']
        split_name       = results['split_name']

        strategy_returns = np.where(direction_pred == 1, actuals, -actuals)
        cum_strategy = np.cumprod(1 + strategy_returns) - 1
        cum_bh       = np.cumprod(1 + actuals) - 1

        def _sharpe(ret, periods=252):
            return float(ret.mean() / ret.std() * np.sqrt(periods)) \
                if ret.std() != 0 else 0.0

        def _mdd(cum_ret):
            wealth = 1 + cum_ret
            peak   = np.maximum.accumulate(wealth)
            return float(((wealth - peak) / peak).min())

        strat_sharpe = _sharpe(strategy_returns)
        bh_sharpe    = _sharpe(actuals)
        strat_mdd    = _mdd(cum_strategy)
        bh_mdd       = _mdd(cum_bh)
        strat_total  = float(cum_strategy[-1] * 100)
        bh_total     = float(cum_bh[-1] * 100)

        window = min(30, max(5, len(actuals) // 10))
        rolling_acc = (pd.Series((direction_pred == direction_actual).astype(float))
                       .rolling(window, min_periods=1).mean() * 100)
        overall_acc = float(np.mean(direction_pred == direction_actual) * 100)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Trading Simulation — {split_name.upper()} Split',
                     fontsize=14, fontweight='bold')

        ax = axes[0]
        ax.plot(cum_strategy * 100, color='seagreen', lw=2,
                label=(f'Model Strategy  ({strat_total:+.1f}%  |  '
                       f'Sharpe={strat_sharpe:.2f}  |  MDD={strat_mdd*100:.1f}%)'))
        ax.plot(cum_bh * 100, color='royalblue', lw=2, linestyle='--',
                label=(f'Buy & Hold  ({bh_total:+.1f}%  |  '
                       f'Sharpe={bh_sharpe:.2f}  |  MDD={bh_mdd*100:.1f}%)'))
        ax.axhline(0, color='black', lw=0.8, alpha=0.5)
        ax.set_title('Cumulative Return (%)')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Return (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(rolling_acc, color='purple', lw=1.5,
                label=f'{window}-day rolling accuracy')
        ax.axhline(50, color='red', linestyle='--', lw=1.5, alpha=0.7,
                   label='Chance (50%)')
        ax.axhline(overall_acc, color='seagreen', linestyle='--', lw=1.5,
                   label=f'Overall ({overall_acc:.1f}%)')
        ax.set_title(f'{window}-Day Rolling Direction Accuracy')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = f'{save_path}trading_simulation_{split_name}_v3.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Trading simulation plot saved to {filepath}")

        return {
            'strategy_total_return_pct':  strat_total,
            'bh_total_return_pct':        bh_total,
            'strategy_sharpe':            strat_sharpe,
            'bh_sharpe':                  bh_sharpe,
            'strategy_max_drawdown_pct':  strat_mdd * 100,
            'bh_max_drawdown_pct':        bh_mdd * 100,
        }

    def plot_confusion_matrix_chart(self, results: Dict,
                                    save_path: str = 'plots/'):
        """Side-by-side raw and row-normalised confusion matrices."""
        os.makedirs(save_path, exist_ok=True)
        direction_actual = results['direction_actual']
        direction_pred   = results['direction_pred']
        split_name       = results['split_name']

        cm = confusion_matrix(direction_actual, direction_pred, labels=[0, 1])
        labels = ['Down', 'Up']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Direction Classification — {split_name.upper()} Split',
                     fontsize=13, fontweight='bold')

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=[f'Pred {l}' for l in labels],
                    yticklabels=[f'Actual {l}' for l in labels],
                    linewidths=0.5)
        axes[0].set_title('Confusion Matrix (counts)')

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', ax=axes[1],
                    xticklabels=[f'Pred {l}' for l in labels],
                    yticklabels=[f'Actual {l}' for l in labels],
                    linewidths=0.5)
        axes[1].set_title('Confusion Matrix (row-normalised)')

        plt.tight_layout()
        filepath = f'{save_path}confusion_matrix_{split_name}_v3.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {filepath}")

    def plot_split_comparison(self, split_results: List[Dict],
                              save_path: str = 'plots/'):
        """Bar chart comparing key metrics across train / val / test splits."""
        os.makedirs(save_path, exist_ok=True)
        metrics_to_plot = [
            ('MAE',               'MAE'),
            ('RMSE',              'RMSE'),
            ('R2',                'R²'),
            ('Direction_Accuracy', 'Direction Accuracy (%)'),
        ]
        colors = ['steelblue', 'darkorange', 'seagreen']
        splits = [r['split_name'].upper() for r in split_results]

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle('Performance Comparison Across Data Splits',
                     fontsize=13, fontweight='bold')

        for ax, (metric, label) in zip(axes, metrics_to_plot):
            values = [r[metric] for r in split_results]
            bars = ax.bar(splits, values, color=colors[:len(splits)],
                          alpha=0.85, edgecolor='white', width=0.5)
            ax.set_title(label)
            ax.set_ylabel(label)
            max_v = max(values) if max(values) != 0 else 1
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max_v * 0.02,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = f'{save_path}split_comparison_v3.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Split comparison plot saved to {filepath}")

    def save(self, save_path: str = 'models/'):
        """Save model and configuration"""
        os.makedirs(save_path, exist_ok=True)

        # Save model
        self.model.save(f'{save_path}hybrid_model_v3.keras')

        # Save config
        self.config.save(f'{save_path}config_v3.json')

        # Save feature importance (convert numpy types to Python floats)
        if self.feature_importance:
            importance_serializable = {k: float(v) for k, v in self.feature_importance.items()}
            with open(f'{save_path}feature_importance_v3.json', 'w') as f:
                json.dump(importance_serializable, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, model_path: str, config_path: str) -> 'HybridModelV3':
        """Load saved model"""
        config = ModelConfigV3.load(config_path)
        instance = cls(config)

        # Custom objects for loading
        custom_objects = {
            'PositionalEncoding': PositionalEncoding,
            'TransformerBlock': TransformerBlock,
            'FeatureAttention': FeatureAttention
        }

        instance.model = load_model(model_path, custom_objects=custom_objects)
        logger.info(f"Model loaded from {model_path}")

        return instance


def generate_full_report(model: 'HybridModelV3', data: Dict,
                         save_path: str = 'results/') -> str:
    """
    Generate a comprehensive post-training analysis report.

    Evaluates the model on training, validation, and test splits; computes
    detailed regression and classification metrics; runs a long/short trading
    simulation; and writes a human-readable report to disk alongside a
    machine-readable JSON summary.

    All plots are saved to plots/.
    Returns the report as a plain-text string.
    """
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('plots/', exist_ok=True)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines: List[str] = []

    def hdr(title: str, char: str = '=', width: int = 70):
        lines.extend(['', char * width, f'  {title}', char * width])

    def sub(title: str):
        lines.extend(['', f'  {title}', '  ' + '\u2500' * (len(title) + 2)])

    def row(label: str, value: str):
        lines.append(f'    {label:<38} {value}')

    # ── Header ────────────────────────────────────────────────────────────────
    lines.extend([
        '=' * 70,
        '  POST-TRAINING ANALYSIS REPORT \u2014 HYBRID LSTM+TRANSFORMER V3.0',
        f'  Generated: {now}',
        '=' * 70,
    ])

    # ── Section 1: Training dynamics ──────────────────────────────────────────
    hdr('SECTION 1: TRAINING DYNAMICS')
    dynamics = model.analyze_training_dynamics()

    if dynamics:
        early = 'YES' if dynamics['early_stopped'] else 'NO'
        row('Epochs trained', f"{dynamics['epochs_trained']} / {dynamics['max_epochs']}")
        row('Early stopping triggered', early)
        row('Best epoch (lowest val loss)', str(dynamics['best_epoch']))
        row('Best val price loss',           f"{dynamics['best_val_price_loss']:.6f}")
        row('Train price loss @ best epoch', f"{dynamics['train_price_loss_at_best']:.6f}")
        row('Overfitting gap (val \u2212 train)', f"{dynamics['overfit_gap']:+.6f}")
        if dynamics['overfit_ratio'] is not None:
            row('Overfitting ratio (val / train)', f"{dynamics['overfit_ratio']:.3f}")
        lines.append('')
        row('Final train direction accuracy', f"{dynamics['final_train_dir_accuracy']:.2f}%")
        row('Final val direction accuracy',   f"{dynamics['final_val_dir_accuracy']:.2f}%")
        row('Best val direction accuracy',    f"{dynamics['best_val_dir_accuracy']:.2f}%")
        lines.append('')
        row('Final train MAE', f"{dynamics['final_train_mae']:.6f}")
        row('Final val MAE',   f"{dynamics['final_val_mae']:.6f}")
        row('Best val MAE',    f"{dynamics['best_val_mae']:.6f}")
        lines.append('')
        row('Val loss std (last 10 epochs)', f"{dynamics['convergence_std_last10']:.2e}")
        if dynamics['final_lr'] is not None:
            row('Final learning rate', f"{dynamics['final_lr']:.2e}")
        if dynamics['training_time_sec'] is not None:
            t = dynamics['training_time_sec']
            row('Training time', f"{t:.0f}s  ({t / 60:.1f} min)")

        lines.append('')
        lines.append('  Interpretation:')
        or_ = dynamics.get('overfit_ratio')
        if or_ is not None:
            if or_ < 1.10:
                lines.append('    Generalisation  : GOOD     (val/train ratio < 1.10)')
            elif or_ < 1.50:
                lines.append('    Generalisation  : MODERATE (val/train ratio 1.10\u20131.50)')
            else:
                lines.append('    Generalisation  : POOR     (val/train ratio > 1.50 \u2014 likely overfitting)')
        cv = dynamics['convergence_std_last10']
        if cv < 1e-5:
            lines.append('    Convergence     : STABLE   (val loss std < 1e-5 over last 10 epochs)')
        elif cv < 1e-4:
            lines.append('    Convergence     : MODERATE (some oscillation in final epochs)')
        else:
            lines.append('    Convergence     : UNSTABLE (high variance in final val loss)')
    else:
        lines.append('    No training history available.')

    # ── Pre-evaluate test split for health summary before sections 2-4 ─────────
    # (we evaluate all splits below, but run test first for the health banner)
    _pre_test = model.evaluate_split(data['X_test'], data['y_test'], 'test')
    hdr('*** MODEL HEALTH SUMMARY ***', char='*')
    lines.append('')
    if _pre_test['is_collapsed']:
        lines.extend([
            '  !!! WARNING: MODEL OUTPUT COLLAPSE DETECTED !!!',
            '',
            f'  The model is predicting a near-constant value for all inputs.',
            f'  Predicted std = {_pre_test["pred_std"]:.2e}  vs  Actual std = {_pre_test["actual_std"]:.2e}',
            f'  Variance ratio = {_pre_test["variance_ratio"]:.4f}  (threshold for collapse < 0.05)',
            '',
            '  What this means:',
            '    - The prediction line will appear flat in plots',
            '    - Direction accuracy merely reflects the market\'s upward drift',
            f'      (naive always-up predictor scores {_pre_test["naive_dir_acc"]:.1f}%)',
            '    - MAE / R² describe a degenerate constant predictor, not real skill',
            f'    - MAE skill vs naive mean: {_pre_test["mae_skill_score"]:+.2f}%',
            '',
            '  Likely causes:',
            '    1. Asymmetric loss (under_prediction_weight=2.0) rewards small positive',
            '       constants that never under-predict large up-moves',
            '    2. Target scale mismatch: returns are tiny (~1e-3) while features are',
            '       MinMaxScaler [0,1] — the output head struggles to escape near-zero',
            '    3. Model may need: target standardisation, reduced under-prediction',
            '       weight, or longer training with lower learning rate',
        ])
    else:
        lines.extend([
            f'  Variance ratio (pred std / actual std) = {_pre_test["variance_ratio"]:.4f}',
            f'  Model is producing varied predictions. No collapse detected.',
            f'  MAE skill vs naive mean: {_pre_test["mae_skill_score"]:+.2f}%',
        ])
    lines.append('')

    # ── Sections 2\u20134: Per-split performance ────────────────────────────────────
    split_configs = [
        ('train', 'X_train', 'y_train', 'SECTION 2: TRAINING SPLIT PERFORMANCE'),
        ('val',   'X_val',   'y_val',   'SECTION 3: VALIDATION SPLIT PERFORMANCE'),
        ('test',  'X_test',  'y_test',  'SECTION 4: TEST SET (ACTUAL DATA) PERFORMANCE'),
    ]

    split_results: List[Dict] = []
    print('\nEvaluating splits for report...')
    for split_name, X_key, y_key, sec_title in split_configs:
        print(f'  {split_name}...', end=' ', flush=True)
        res = model.evaluate_split(data[X_key], data[y_key], split_name)
        split_results.append(res)
        print('done')

        hdr(sec_title)
        row('Samples', str(res['n_samples']))

        sub('Price / Return Regression')
        row('MSE',  f"{res['MSE']:.8f}")
        row('MAE',  f"{res['MAE']:.6f}")
        row('RMSE', f"{res['RMSE']:.6f}")
        row('R\u00b2',   f"{res['R2']:.4f}")
        row('MAPE', f"{res['MAPE']:.2f}%")

        sub('Error Distribution')
        row('Mean error (systematic bias)', f"{res['error_mean']:+.6f}")
        row('Std of errors',               f"{res['error_std']:.6f}")
        row('Skewness',                    f"{res['error_skew']:+.4f}")
        row('Excess kurtosis',             f"{res['error_kurt']:+.4f}")
        row('Predictions within \u00b10.5%',   f"{res['within_05pct']:.1f}%")
        row('Predictions within \u00b11.0%',   f"{res['within_1pct']:.1f}%")
        row('Predictions within \u00b12.0%',   f"{res['within_2pct']:.1f}%")

        sub('Direction Classification')
        row('Overall accuracy', f"{res['Direction_Accuracy']:.2f}%")
        row('Up precision',     f"{res['Up_Precision']:.2f}%")
        row('Up recall',        f"{res['Up_Recall']:.2f}%")
        row('Down precision',   f"{res['Down_Precision']:.2f}%")
        row('Down recall',      f"{res['Down_Recall']:.2f}%")
        row('TP / TN / FP / FN',
            f"{res['TP']} / {res['TN']} / {res['FP']} / {res['FN']}")

        sub('Prediction Health / Collapse Diagnostics')
        row('Actual mean return',     f"{res['actual_mean']:+.6f}")
        row('Actual std  (spread)',   f"{res['actual_std']:.6f}")
        row('Predicted mean',         f"{res['pred_mean']:+.6f}")
        row('Predicted std  (spread)',f"{res['pred_std']:.6f}")
        row('Variance ratio (p/a)',   f"{res['variance_ratio']:.4f}  "
            f"({'COLLAPSED — model predicts near-constant' if res['is_collapsed'] else 'OK'})")
        lines.append('')
        row('Naive-mean MAE (benchmark)', f"{res['naive_mae']:.6f}")
        row('Naive-mean dir. accuracy',   f"{res['naive_dir_acc']:.2f}%")
        row('MAE skill score vs naive',   f"{res['mae_skill_score']:+.2f}%  "
            f"({'better' if res['mae_skill_score'] > 0 else 'WORSE than naive'})")

    # ── Section 5: Train vs Test comparison ───────────────────────────────────
    hdr('SECTION 5: TRAIN vs TEST GENERALISATION SUMMARY')
    tr, _, te = split_results
    lines.append('')
    lines.append(f'    {"Metric":<28} {"Train":>10} {"Test":>10} {"\u0394 (Test\u2212Train)":>16}')
    lines.append('    ' + '\u2500' * 66)
    for metric, fmt in [('MAE', '.6f'), ('RMSE', '.6f'), ('R2', '.4f'),
                        ('Direction_Accuracy', '.2f')]:
        tr_val = tr[metric]
        te_val = te[metric]
        delta  = te_val - tr_val
        lines.append(
            f'    {metric:<28} {format(tr_val, fmt):>10} '
            f'{format(te_val, fmt):>10} {format(delta, "+.4f"):>16}'
        )

    # ── Section 6: Trading simulation (test set) ──────────────────────────────
    hdr('SECTION 6: TRADING SIMULATION (TEST SET \u2014 LONG / SHORT STRATEGY)')
    test_res    = split_results[2]
    trade_stats = model.plot_trading_simulation(test_res, 'plots/')

    lines.append('')
    lines.append(f'    {"Metric":<35} {"Strategy":>12} {"Buy & Hold":>12}')
    lines.append('    ' + '\u2500' * 62)

    def trow(lbl: str, s_val: str, b_val: str):
        lines.append(f'    {lbl:<35} {s_val:>12} {b_val:>12}')

    trow('Total Return (%)',
         f'{trade_stats["strategy_total_return_pct"]:+.2f}%',
         f'{trade_stats["bh_total_return_pct"]:+.2f}%')
    trow('Annualised Sharpe Ratio',
         f'{trade_stats["strategy_sharpe"]:.3f}',
         f'{trade_stats["bh_sharpe"]:.3f}')
    trow('Max Drawdown (%)',
         f'{trade_stats["strategy_max_drawdown_pct"]:.2f}%',
         f'{trade_stats["bh_max_drawdown_pct"]:.2f}%')
    lines.extend([
        '',
        '  Note: Strategy = long when model predicts up, short when down.',
        '        No transaction costs or slippage applied.',
    ])

    # ── Section 7: Feature importance ─────────────────────────────────────────
    hdr('SECTION 7: FEATURE IMPORTANCE (ATTENTION WEIGHTS \u2014 TOP 20)')
    if model.feature_importance:
        sorted_fi = sorted(model.feature_importance.items(),
                           key=lambda x: x[1], reverse=True)
        lines.append('')
        lines.append(f'    {"Rank":<6} {"Feature":<40} {"Attention Weight":>16}')
        lines.append('    ' + '\u2500' * 64)
        for rank, (name, weight) in enumerate(sorted_fi[:20], 1):
            lines.append(f'    {rank:<6} {name:<40} {weight:>16.4f}')
        if len(sorted_fi) > 20:
            lines.append(f'    ... and {len(sorted_fi) - 20} more features')
    else:
        lines.append('    Feature importance not available (feature attention disabled).')

    # ── Section 8: Generated files ────────────────────────────────────────────
    hdr('SECTION 8: GENERATED OUTPUT FILES')
    lines.append('')
    lines.append('  Plots:')
    for sn in ['train', 'val', 'test']:
        lines.append(f'    plots/error_analysis_{sn}_v3.png')
        lines.append(f'    plots/trading_simulation_{sn}_v3.png')
        lines.append(f'    plots/confusion_matrix_{sn}_v3.png')
    lines.append( '    plots/split_comparison_v3.png')
    lines.append( '    plots/training_history_v3.png')
    lines.append( '    plots/predictions_v3.png')
    lines.append('')
    lines.append('  Data:')
    lines.append(f'    {save_path}report_v3.txt')
    lines.append(f'    {save_path}detailed_metrics_v3.json')

    lines.extend(['', '=' * 70, '  END OF REPORT', '=' * 70, ''])

    report_text = '\n'.join(lines)

    # Save report text
    report_path = os.path.join(save_path, 'report_v3.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"Report saved to {report_path}")

    # Save detailed metrics JSON (numpy-safe serialisation)
    def _to_py(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    detailed_json = {
        'generated':          now,
        'training_dynamics':  dynamics,
        'trading_simulation': trade_stats,
        'splits': [
            {k: _to_py(v) for k, v in r.items() if not isinstance(v, np.ndarray)}
            for r in split_results
        ],
    }
    json_path = os.path.join(save_path, 'detailed_metrics_v3.json')
    with open(json_path, 'w') as f:
        json.dump(detailed_json, f, indent=2, default=_to_py)
    logger.info(f"Detailed metrics saved to {json_path}")

    # Generate remaining plots
    print('Generating analysis plots...')
    for res in split_results:
        model.plot_error_analysis(res, 'plots/')
        # trading simulation for train/val (test was already plotted above)
        if res['split_name'] != 'test':
            model.plot_trading_simulation(res, 'plots/')
        model.plot_confusion_matrix_chart(res, 'plots/')

    model.plot_split_comparison(split_results, 'plots/')
    model.plot_training_history('plots/')
    model.plot_predictions(test_res['predictions'], test_res['actuals'], 'plots/')

    print(report_text)
    return report_text


def load_prepared_data(data_path: str = 'data/processed/') -> Dict:
    """Load data prepared by SnP500DataPrepV3.py"""
    logger.info(f"Loading prepared data from {data_path}")

    data = {
        'X_train': np.load(f'{data_path}X_train.npy'),
        'y_train': np.load(f'{data_path}y_train.npy'),
        'X_val': np.load(f'{data_path}X_val.npy'),
        'y_val': np.load(f'{data_path}y_val.npy'),
        'X_test': np.load(f'{data_path}X_test.npy'),
        'y_test': np.load(f'{data_path}y_test.npy')
    }

    with open(f'{data_path}metadata.pkl', 'rb') as f:
        data['metadata'] = pickle.load(f)

    logger.info(f"Loaded data shapes:")
    logger.info(f"  X_train: {data['X_train'].shape}")
    logger.info(f"  X_val: {data['X_val'].shape}")
    logger.info(f"  X_test: {data['X_test'].shape}")
    logger.info(f"  Features: {data['metadata']['n_features']}")

    return data


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("HYBRID LSTM+TRANSFORMER MODEL V3.0 FOR S&P 500 PREDICTION")
    print("=" * 70 + "\n")

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Load prepared data
    try:
        data = load_prepared_data('data/processed/')
    except FileNotFoundError:
        print("ERROR: Prepared data not found!")
        print("Please run SnP500DataPrepV3.py first to prepare the data.")
        print("\nExample:")
        print("  python SnP500DataPrepV3.py")
        return None

    # Load configuration from file
    config_path = 'config/lstm_transformer_config.json'
    if os.path.exists(config_path):
        config = ModelConfigV3.from_json(config_path)
        # Override with actual data dimensions from metadata
        config.sequence_length = data['metadata']['sequence_length']
        config.n_features = data['metadata']['n_features']
        logger.info(f"Loaded model config from {config_path}")
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = ModelConfigV3(
            sequence_length=data['metadata']['sequence_length'],
            n_features=data['metadata']['n_features']
        )

    # Initialize model
    model = HybridModelV3(config)

    # Build and train
    model.build_model()

    feature_names = data['metadata'].get('feature_names', None)

    model.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        feature_names=feature_names
    )

    # Generate comprehensive post-training analysis report
    generate_full_report(model, data, save_path='results/')

    if model.feature_importance:
        model.plot_feature_importance('plots/')

    # Save model
    model.save('models/')

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - models/hybrid_model_v3.keras")
    print("  - models/config_v3.json")
    print("  - results/report_v3.txt           (full text report)")
    print("  - results/detailed_metrics_v3.json (machine-readable metrics)")
    print("  - plots/training_history_v3.png")
    print("  - plots/predictions_v3.png")
    print("  - plots/error_analysis_[train/val/test]_v3.png")
    print("  - plots/trading_simulation_[train/val/test]_v3.png")
    print("  - plots/confusion_matrix_[train/val/test]_v3.png")
    print("  - plots/split_comparison_v3.png")
    if model.feature_importance:
        print("  - plots/feature_importance_v3.png")
    print("=" * 70 + "\n")

    return model


if __name__ == "__main__":
    main()
