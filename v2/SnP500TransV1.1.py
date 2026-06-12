"""
Transformer Model for S&P 500 Prediction (V1.1)
================================================
This script implements a Transformer neural network with V1.1 enhancements:

V1.1 Enhancements (from LSTM V1.1):
- Dual outputs (price prediction + direction classification)
- Custom biased loss function (penalizes under-predictions 2x more)
- Direction labels derived from returns (y > 0 = up, y <= 0 = down)
- Simple growth bias applied to predictions
- Dataclass-based configuration

Features:
- Multi-head self-attention mechanism
- Positional encoding for time series
- Configurable architecture
- Proper training with validation monitoring
- Comprehensive evaluation metrics
- Visualization of predictions

Author: Ronald
Version: 1.1
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
import logging

# TensorFlow/Keras imports
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfigV1_1:
    """Configuration for Transformer V1.1"""
    sequence_length: int
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 0.0001
    patience: int = 30
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 3
    dff: int = 256
    dense_units: List[int] = None
    dropout_rate: float = 0.2
    growth_bias: float = 0.005  # 0.5% optimistic bias

    def __post_init__(self):
        if self.dense_units is None:
            self.dense_units = [64, 32]

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")


class PositionalEncoding(keras.layers.Layer):
    """
    Positional encoding layer for Transformer
    Adds positional information to input sequences
    """
    
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, sequence_length, d_model):
        angle_rads = self.get_angles(
            np.arange(sequence_length)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerBlock(keras.layers.Layer):
    """
    Single Transformer encoder block
    Contains multi-head attention and feed-forward network
    """
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2


class TransformerPredictorV1_1:
    """
    Transformer V1.1 with Dual Outputs (Price + Direction)
    Compatible with V1 data preparation workflow
    """

    def __init__(self, config: TransformerConfigV1_1, n_features: int):
        self.config = config
        self.n_features = n_features
        self.model = None
        self.history = None
        self.training_time = None

        logger.info(f"Transformer V1.1 initialized")
        logger.info(f"Architecture: {config.num_layers} layers, {config.num_heads} heads, d_model={config.d_model}")
        logger.info(f"Features: {n_features}, Sequence length: {config.sequence_length}")
        logger.info(f"Growth bias: {config.growth_bias:.3%}")
    
    def build_model(self):
        """
        Build Transformer model with dual outputs (V1.1)
        """
        print("\n" + "="*60)
        print("BUILDING TRANSFORMER MODEL (V1.1)")
        print("="*60)

        # Input layer
        inputs = Input(shape=(self.config.sequence_length, self.n_features))

        # Project input features to d_model dimension
        x = Dense(self.config.d_model, name='input_projection')(inputs)

        # Add positional encoding
        x = PositionalEncoding(self.config.sequence_length, self.config.d_model)(x)

        # Stack Transformer encoder blocks
        for i in range(self.config.num_layers):
            x = TransformerBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                dff=self.config.dff,
                dropout_rate=self.config.dropout_rate
            )(x, training=True)

        # Global average pooling
        x = GlobalAveragePooling1D(name='global_pooling')(x)

        # Dense layers
        for i, units in enumerate(self.config.dense_units):
            x = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = Dropout(self.config.dropout_rate)(x)

        # Dual outputs (V1.1)
        price_output = Dense(1, activation='linear', name='price_output')(x)
        direction_output = Dense(1, activation='sigmoid', name='direction_output')(x)

        self.model = Model(inputs=inputs, outputs=[price_output, direction_output])

        # Custom biased loss (penalizes under-predictions 2x more)
        def custom_biased_mse(y_true, y_pred):
            error = y_true - y_pred
            under_prediction_weight = 2.0
            weights = tf.where(error > 0, under_prediction_weight, 1.0)
            return tf.reduce_mean(weights * tf.square(error))

        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss={
                'price_output': custom_biased_mse,
                'direction_output': 'binary_crossentropy'
            },
            loss_weights={'price_output': 1.0, 'direction_output': 0.3},
            metrics={
                'price_output': ['mae', 'mse'],
                'direction_output': ['accuracy']
            }
        )

        print(f"\nModel architecture:")
        print(f"  - Transformer layers: {self.config.num_layers}")
        print(f"  - Attention heads: {self.config.num_heads}")
        print(f"  - d_model: {self.config.d_model}")
        print(f"  - Feed-forward dim: {self.config.dff}")
        print(f"  - Dropout rate: {self.config.dropout_rate}")
        print(f"  - Price output: Linear (custom biased MSE loss)")
        print(f"  - Direction output: Sigmoid (binary crossentropy)")
        print(f"  - Total parameters: {self.model.count_params():,}")
        print("="*60 + "\n")

        return self.model
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """
        Train the model (V1.1)

        Args:
            train_data: (X_train, y_train_price, y_train_direction)
            val_data: (X_val, y_val_price, y_val_direction)
        """
        X_train, y_train_price, y_train_direction = train_data
        X_val, y_val_price, y_val_direction = val_data

        if self.model is None:
            self.build_model()

        print("\n" + "="*60)
        print("TRAINING TRANSFORMER V1.1 MODEL")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Early stopping patience: {self.config.patience}")
        print("="*60 + "\n")

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_price_output_loss',
            patience=self.config.patience,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_price_output_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            mode='min',
            verbose=1
        )

        start_time = datetime.now()

        self.history = self.model.fit(
            X_train,
            {'price_output': y_train_price, 'direction_output': y_train_direction},
            validation_data=(X_val, {'price_output': y_val_price, 'direction_output': y_val_direction}),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
            shuffle=False
        )

        end_time = datetime.now()
        self.training_time = (end_time - start_time).total_seconds()

        print("\n" + "="*60)
        print(f"Training completed!")
        print(f"Training time: {self.training_time:.2f} seconds ({self.training_time/60:.2f} minutes)")
        print(f"Epochs trained: {len(self.history.history['loss'])}")
        print(f"Best val price loss: {min(self.history.history['val_price_output_loss']):.6f}")
        print("="*60 + "\n")

        return self.history

    def apply_growth_bias(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply simple growth bias to predictions
        """
        bias_factor = 1 + self.config.growth_bias
        predictions_biased = predictions * bias_factor
        logger.info(f"Applied growth bias of {self.config.growth_bias:.3%}")
        return predictions_biased
    
    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 dataset_name: str = 'Test') -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate the model (works in scaled space like V1)
        """
        X_test, y_test_price, y_test_direction = test_data

        if self.model is None:
            raise ValueError("Model not trained")

        # Predict both outputs
        price_predictions, direction_predictions = self.model.predict(X_test, verbose=0)
        price_predictions = price_predictions.flatten()
        direction_predictions = (direction_predictions.flatten() > 0.5).astype(int)

        # Apply growth bias
        price_predictions = self.apply_growth_bias(price_predictions)

        # Calculate metrics (in scaled space)
        mse = mean_squared_error(y_test_price, price_predictions)
        mae = mean_absolute_error(y_test_price, price_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_price, price_predictions)

        # MAPE
        mape = np.mean(np.abs((y_test_price - price_predictions) / (y_test_price + 1e-8))) * 100

        # Directional accuracy
        directional_accuracy = np.mean(y_test_direction == direction_predictions) * 100

        # Bias effect
        bias_effect = ((price_predictions.mean() - (price_predictions.mean() / (1 + self.config.growth_bias))) / (price_predictions.mean() / (1 + self.config.growth_bias))) * 100

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Bias_Effect_%': bias_effect
        }

        # Print results
        print("\n" + "="*60)
        print(f"TRANSFORMER V1.1 EVALUATION RESULTS - {dataset_name.upper()}")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*60 + "\n")

        return metrics, price_predictions, direction_predictions
    
    def plot_training_history(self, save_path='plots/'):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        os.makedirs(save_path, exist_ok=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Price loss
        ax1.plot(self.history.history['price_output_loss'], label='Training Price Loss')
        ax1.plot(self.history.history['val_price_output_loss'], label='Validation Price Loss')
        ax1.set_title('Price Prediction Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Direction loss
        ax2.plot(self.history.history['direction_output_loss'], label='Training Direction Loss')
        ax2.plot(self.history.history['val_direction_output_loss'], label='Validation Direction Loss')
        ax2.set_title('Direction Classification Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        # Direction accuracy
        ax3.plot(self.history.history['direction_output_accuracy'], label='Training Direction Accuracy')
        ax3.plot(self.history.history['val_direction_output_accuracy'], label='Validation Direction Accuracy')
        ax3.set_title('Direction Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_path}transformer_v1.1_training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}transformer_v1.1_training_history.png")
        plt.close()
    
    def plot_predictions(self, y_true, y_pred, dataset_name='Test', save_path='plots/'):
        """Plot predictions vs actual"""
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(14, 6))
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(f'Transformer V1.1 - {dataset_name} Set Predictions')
        plt.xlabel('Sample')
        plt.ylabel('Scaled Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}transformer_v1.1_predictions_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to: {save_path}transformer_v1.1_predictions_{dataset_name.lower()}.png")
        plt.close()
    
    def save_model(self, save_path='models/'):
        """Save model and configuration"""
        os.makedirs(save_path, exist_ok=True)

        self.model.save(f'{save_path}transformer_v1.1_model.keras')
        self.config.save(f'{save_path}transformer_v1.1_config.json')

        print(f"Model saved to: {save_path}transformer_v1.1_model.keras")
        print(f"Config saved to: {save_path}transformer_v1.1_config.json")


def create_default_config(sequence_length: int = 60) -> TransformerConfigV1_1:
    """Create default configuration"""
    return TransformerConfigV1_1(
        sequence_length=sequence_length,
        batch_size=64,
        epochs=50,
        learning_rate=0.0002,
        patience=5,
        d_model=258,
        num_heads=8,
        num_layers=7,
        dff=256,
        dense_units=[128, 64, 32],
        dropout_rate=0.2,
        growth_bias=0.0
    )


def main():
    """
    Main execution function - V1 workflow with V1.1 enhancements
    """
    print("\n" + "="*80)
    print("TRANSFORMER MODEL V1.1 FOR S&P 500 PREDICTION")
    print("="*80 + "\n")

    # ========================================================================
    # STEP 1: Load V1 prepared data
    # ========================================================================
    print("Loading V1 prepared data...")
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
    print(f"Sequence length: {metadata['sequence_length']}\n")

    # ========================================================================
    # STEP 2: Derive direction labels from returns (V1.1 enhancement)
    # ========================================================================
    print("Deriving direction labels from returns...")
    # Direction: 1 if return is positive (up), 0 if negative/zero (down)
    y_train_direction = (y_train > 0).astype(int)
    y_val_direction = (y_val > 0).astype(int)
    y_test_direction = (y_test > 0).astype(int)

    print(f"Direction distribution (train): {np.mean(y_train_direction)*100:.1f}% up, {(1-np.mean(y_train_direction))*100:.1f}% down")
    print(f"Direction distribution (val): {np.mean(y_val_direction)*100:.1f}% up, {(1-np.mean(y_val_direction))*100:.1f}% down")
    print(f"Direction distribution (test): {np.mean(y_test_direction)*100:.1f}% up, {(1-np.mean(y_test_direction))*100:.1f}% down\n")

    # ========================================================================
    # STEP 3: Create configuration and model
    # ========================================================================
    config = create_default_config(sequence_length=metadata['sequence_length'])
    config.save('models/transformer_v1.1_config.json')

    logger.info("=" * 70)
    logger.info("TRANSFORMER PREDICTOR V1.1")
    logger.info("=" * 70)
    logger.info(f"Growth Bias: {config.growth_bias:.3%}")
    logger.info(f"Sequence Length: {metadata['sequence_length']} steps")
    logger.info(f"Architecture: {config.num_layers} layers, {config.num_heads} heads, d_model={config.d_model}")
    logger.info(f"Epochs: {config.epochs}, Patience: {config.patience}")
    logger.info("=" * 70)

    transformer = TransformerPredictorV1_1(config, n_features=metadata['n_features'])
    transformer.build_model()

    # ========================================================================
    # STEP 4: Train model
    # ========================================================================
    transformer.train(
        (X_train, y_train, y_train_direction),
        (X_val, y_val, y_val_direction)
    )

    # ========================================================================
    # STEP 5: Evaluate on all datasets
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    train_metrics, train_pred, _ = transformer.evaluate(
        (X_train, y_train, y_train_direction),
        'Train'
    )

    val_metrics, val_pred, _ = transformer.evaluate(
        (X_val, y_val, y_val_direction),
        'Validation'
    )

    test_metrics, test_pred, test_dir_pred = transformer.evaluate(
        (X_test, y_test, y_test_direction),
        'Test'
    )

    # ========================================================================
    # STEP 6: Save results
    # ========================================================================
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'config': config.__dict__,
        'training_time_seconds': transformer.training_time
    }

    os.makedirs('results', exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = convert_to_serializable(results)

    with open('results/transformer_v1.1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: results/transformer_v1.1_results.json")

    # ========================================================================
    # STEP 7: Generate visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    transformer.plot_training_history()
    transformer.plot_predictions(y_test, test_pred, 'Test')

    # ========================================================================
    # STEP 8: Save model
    # ========================================================================
    os.makedirs('models', exist_ok=True)
    transformer.model.save('models/transformer_v1.1_model.keras')
    print(f"\nModel saved to: models/transformer_v1.1_model.keras")

    print("\n" + "="*80)
    print("TRANSFORMER V1.1 PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - models/transformer_v1.1_model.keras")
    print("  - models/transformer_v1.1_config.json")
    print("  - results/transformer_v1.1_results.json")
    print("  - plots/transformer_v1.1_training_history.png")
    print("  - plots/transformer_v1.1_predictions_test.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Run main pipeline
    main()