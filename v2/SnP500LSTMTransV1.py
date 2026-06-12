"""
Hybrid LSTM + Transformer Model for S&P 500 Prediction
========================================================
This script implements a hybrid architecture that combines
LSTM and Transformer layers for predicting S&P 500 returns.

Architecture:
- LSTM layers: Capture local temporal patterns and sequential dependencies
- Transformer layers: Capture long-range dependencies and attention patterns
- Combination: Best of both worlds

Features:
- Sequential hybrid architecture (LSTM → Transformer)
- Configurable layer sizes
- Proper training with validation monitoring
- Comprehensive evaluation metrics
- Visualization and model saving

Author: Ronald
Date: 2024
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

# TensorFlow/Keras imports
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization, LSTM, Bidirectional,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.losses import Huber

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class PositionalEncoding(keras.layers.Layer):
    """
    Positional encoding layer for Transformer component
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
    Transformer encoder block for hybrid model
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
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class HybridPredictor:
    """
    Hybrid LSTM + Transformer model for time series prediction of S&P 500
    
    Architecture Flow:
    Input → LSTM layers → Transformer layers → Dense layers → Output
    
    Rationale:
    - LSTM: Captures local patterns and sequential dependencies
    - Transformer: Captures long-range dependencies and attention patterns
    - Combination: Leverages strengths of both architectures
    """
    
    def __init__(self,
                 sequence_length=60,
                 n_features=None,
                 lstm_units=[64],
                 d_model=64,
                 num_heads=4,
                 num_transformer_layers=2,
                 dff=256,
                 dense_units=[32],
                 dropout_rate=0.15,
                 learning_rate=0.0005,
                 bidirectional_lstm=False):
        """
        Initialize Hybrid LSTM+Transformer model architecture
        
        Parameters:
        -----------
        sequence_length : int
            Number of time steps in input sequences
        n_features : int
            Number of features per time step
        lstm_units : list
            List of LSTM layer sizes [64] = one LSTM layer
        d_model : int
            Dimension of Transformer model (embedding dimension)
        num_heads : int
            Number of attention heads in Transformer
        num_transformer_layers : int
            Number of Transformer encoder blocks
        dff : int
            Dimension of feed-forward network in Transformer
        dense_units : list
            List of Dense layer sizes after Transformer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for Adam optimizer
        bidirectional_lstm : bool
            Use bidirectional LSTM layers
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.dff = dff
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.bidirectional_lstm = bidirectional_lstm
        
        self.model = None
        self.history = None
        self.training_time = None
    
    def build_model(self):
        """
        Build Hybrid LSTM+Transformer model architecture
        """
        print("\n" + "="*60)
        print("BUILDING HYBRID LSTM+TRANSFORMER MODEL")
        print("="*60)
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # ============================================
        # LSTM COMPONENT (Local Pattern Capture)
        # ============================================
        x = inputs
        
        for i, units in enumerate(self.lstm_units):
            # Always return sequences for Transformer to process
            return_seq = True
            
            if self.bidirectional_lstm:
                x = Bidirectional(
                    LSTM(units, return_sequences=return_seq),
                    name=f'bidirectional_lstm_{i+1}'
                )(x)
            else:
                x = LSTM(units, return_sequences=return_seq, name=f'lstm_{i+1}')(x)
            
            x = Dropout(self.dropout_rate, name=f'lstm_dropout_{i+1}')(x)
        
        # ============================================
        # PROJECTION TO TRANSFORMER DIMENSION
        # ============================================
        # Project LSTM output to d_model dimension if needed
        if self.bidirectional_lstm:
            lstm_output_dim = self.lstm_units[-1] * 2  # Bidirectional doubles dimension
        else:
            lstm_output_dim = self.lstm_units[-1]
        
        if lstm_output_dim != self.d_model:
            x = Dense(self.d_model, name='projection_to_transformer')(x)
        
        # ============================================
        # TRANSFORMER COMPONENT (Long-range Dependencies)
        # ============================================
        
        # Add positional encoding
        x = PositionalEncoding(self.sequence_length, self.d_model)(x)
        
        # Stack Transformer encoder blocks
        for i in range(self.num_transformer_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate
            )(x, training=True)
        
        # ============================================
        # AGGREGATION AND OUTPUT
        # ============================================
        
        # Global average pooling to aggregate sequence
        x = GlobalAveragePooling1D(name='global_pooling')(x)
        
        # Dense layers for final processing
        for i, units in enumerate(self.dense_units):
            x = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = Dropout(self.dropout_rate, name=f'dense_dropout_{i+1}')(x)
        
        # Output layer
        outputs = Dense(1, name='output')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='Hybrid_LSTM_Transformer')
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=Huber(),
            metrics=['mae', 'mse']
        )
        
        # Print model summary
        print("\nModel Architecture:")
        print("-" * 60)
        self.model.summary()
        print("-" * 60)
        
        # Count parameters
        total_params = self.model.count_params()
        print(f"\nTotal trainable parameters: {total_params:,}")
        print(f"\nArchitecture Components:")
        print(f"  LSTM layers: {len(self.lstm_units)} layers")
        print(f"  LSTM units: {self.lstm_units}")
        print(f"  Bidirectional LSTM: {self.bidirectional_lstm}")
        print(f"  Transformer d_model: {self.d_model}")
        print(f"  Transformer heads: {self.num_heads}")
        print(f"  Transformer layers: {self.num_transformer_layers}")
        print(f"  Feed-forward dim: {self.dff}")
        print("="*60 + "\n")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val,
              epochs=100,
              batch_size=32,
              patience=15,
              min_delta=0.0001,
              warmup_epochs=10,
              verbose=1):
        """
        Train the Hybrid model with learning rate warmup
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("\n" + "="*60)
        print("TRAINING HYBRID LSTM+TRANSFORMER MODEL")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Early stopping patience: {patience}")
        print(f"Learning rate warmup: {warmup_epochs} epochs")
        print("="*60 + "\n")
        
        # Learning rate warmup schedule
        def lr_schedule(epoch):
            if epoch < warmup_epochs:
                return self.learning_rate * (epoch + 1) / warmup_epochs
            else:
                return self.learning_rate
        
        lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_hybrid_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        # Train model
        start_time = datetime.now()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, model_checkpoint, lr_scheduler],
            verbose=verbose
        )
        
        end_time = datetime.now()
        self.training_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Training time: {self.training_time:.2f} seconds ({self.training_time/60:.2f} minutes)")
        print(f"Epochs trained: {len(self.history.history['loss'])}")
        print(f"Best validation loss: {min(self.history.history['val_loss']):.6f}")
        print("="*60 + "\n")
        
        return self.history
    
    def evaluate(self, X, y, dataset_name='Test'):
        """
        Evaluate model on a dataset
        
        Returns:
        --------
        Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Make predictions
        y_pred = self.model.predict(X, verbose=0).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        
        # Directional Accuracy (for returns prediction)
        direction_actual = np.sign(y)
        direction_pred = np.sign(y_pred)
        directional_accuracy = np.mean(direction_actual == direction_pred) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'predictions': y_pred,
            'actuals': y
        }
        
        print(f"\n{dataset_name} Set Evaluation:")
        print("-" * 60)
        print(f"RMSE:                  {rmse:.6f}")
        print(f"MAE:                   {mae:.6f}")
        print(f"R² Score:              {r2:.6f}")
        print(f"MAPE:                  {mape:.2f}%")
        print(f"Directional Accuracy:  {directional_accuracy:.2f}%")
        print("-" * 60)
        
        return metrics
    
    def plot_training_history(self, save_path='plots/'):
        """
        Plot training and validation loss curves
        """
        if self.history is None:
            print("No training history available.")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Mean Absolute Error During Training', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}hybrid_training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}hybrid_training_history.png")
        plt.close()
    
    def plot_predictions(self, y_actual, y_pred, dates=None, dataset_name='Test', save_path='plots/'):
        """
        Plot actual vs predicted values
        """
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series plot
        if dates is not None:
            x_axis = pd.to_datetime(dates)
        else:
            x_axis = np.arange(len(y_actual))
        
        axes[0].plot(x_axis, y_actual, label='Actual', alpha=0.7, linewidth=1.5)
        axes[0].plot(x_axis, y_pred, label='Predicted', alpha=0.7, linewidth=1.5)
        axes[0].set_xlabel('Date' if dates is not None else 'Sample', fontsize=12)
        axes[0].set_ylabel('Return', fontsize=12)
        axes[0].set_title(f'Hybrid LSTM+Transformer Predictions vs Actual - {dataset_name} Set', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_actual, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Prediction')
        
        axes[1].set_xlabel('Actual Return', fontsize=12)
        axes[1].set_ylabel('Predicted Return', fontsize=12)
        axes[1].set_title('Actual vs Predicted Scatter Plot', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Add R² score to scatter plot
        r2 = r2_score(y_actual, y_pred)
        axes[1].text(0.05, 0.95, f'R² = {r2:.4f}', 
                    transform=axes[1].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}hybrid_predictions_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to: {save_path}hybrid_predictions_{dataset_name.lower()}.png")
        plt.close()
    
    def save_model(self, save_path='models/'):
        """
        Save model and configuration
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save Keras model
        self.model.save(f'{save_path}hybrid_model.h5')
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_transformer_layers': self.num_transformer_layers,
            'dff': self.dff,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'bidirectional_lstm': self.bidirectional_lstm,
            'training_time': self.training_time
        }
        
        with open(f'{save_path}hybrid_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Model saved to: {save_path}hybrid_model.h5")
        print(f"Config saved to: {save_path}hybrid_config.json")
    
    @classmethod
    def load_model(cls, model_path='models/hybrid_model.h5', 
                   config_path='models/hybrid_config.json'):
        """
        Load saved model and configuration
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        predictor = cls(
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            lstm_units=config['lstm_units'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_transformer_layers=config['num_transformer_layers'],
            dff=config['dff'],
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            bidirectional_lstm=config['bidirectional_lstm']
        )
        
        predictor.model = load_model(model_path, custom_objects={
            'PositionalEncoding': PositionalEncoding,
            'TransformerBlock': TransformerBlock
        })
        predictor.training_time = config.get('training_time')
        
        print(f"Model loaded from: {model_path}")
        return predictor


# ============================================================================
# USAGE EXAMPLE AND MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - complete Hybrid training pipeline
    """
    print("\n" + "="*80)
    print("HYBRID LSTM+TRANSFORMER MODEL FOR S&P 500 PREDICTION")
    print("="*80 + "\n")
    
    # Load prepared data
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
    
    # Initialize Hybrid model
    hybrid = HybridPredictor(
        sequence_length=metadata['sequence_length'],
        n_features=metadata['n_features'],
        lstm_units=[64],              # One LSTM layer with 64 units
        d_model=64,                   # Transformer embedding dimension
        num_heads=4,                  # Number of attention heads
        num_transformer_layers=2,     # Two Transformer blocks
        dff=256,                      # Feed-forward dimension
        dense_units=[32],             # Dense layer after Transformer
        dropout_rate=0.15,            # Moderate dropout (between LSTM and Transformer)
        learning_rate=0.0005,         # Middle ground between LSTM and Transformer
        bidirectional_lstm=False      # Standard LSTM
    )
    
    # Build model
    hybrid.build_model()
    
    # Train model
    hybrid.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
        patience=15,
        warmup_epochs=10,
        verbose=1
    )
    
    # Evaluate on all datasets
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    train_metrics = hybrid.evaluate(X_train, y_train, dataset_name='Training')
    val_metrics = hybrid.evaluate(X_val, y_val, dataset_name='Validation')
    test_metrics = hybrid.evaluate(X_test, y_test, dataset_name='Test')
    
    # Save results
    results = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'config': {
            'lstm_units': hybrid.lstm_units,
            'd_model': hybrid.d_model,
            'num_heads': hybrid.num_heads,
            'num_transformer_layers': hybrid.num_transformer_layers,
            'dff': hybrid.dff,
            'dropout_rate': hybrid.dropout_rate,
            'learning_rate': hybrid.learning_rate,
            'bidirectional_lstm': hybrid.bidirectional_lstm,
            'training_time': hybrid.training_time
        }
    }
    
    # Remove predictions/actuals before saving (too large)
    for dataset in ['train', 'val', 'test']:
        results[dataset] = {k: v for k, v in results[dataset].items() 
                           if k not in ['predictions', 'actuals']}
    
    with open('results/hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to: results/hybrid_results.json")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    hybrid.plot_training_history(save_path='plots/')
    hybrid.plot_predictions(test_metrics['actuals'], test_metrics['predictions'], 
                           dataset_name='Test', save_path='plots/')
    
    # Save model
    hybrid.save_model(save_path='models/')
    
    print("\n" + "="*80)
    print("HYBRID MODEL PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - models/hybrid_model.h5")
    print("  - models/hybrid_config.json")
    print("  - results/hybrid_results.json")
    print("  - plots/hybrid_training_history.png")
    print("  - plots/hybrid_predictions_test.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Run main pipeline
    main()