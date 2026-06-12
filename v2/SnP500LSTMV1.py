"""
LSTM Model for S&P 500 Prediction
==================================
This script implements a Long Short-Term Memory (LSTM) neural network
for predicting S&P 500 returns.

Features:
- Configurable LSTM architecture
- Proper training with validation monitoring
- Early stopping and learning rate reduction
- Comprehensive evaluation metrics
- Visualization of predictions and training history
- Model saving and loading

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
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import Huber

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class LSTMPredictor:
    """
    LSTM model for time series prediction of S&P 500
    """
    
    def __init__(self,
                 sequence_length=60,
                 n_features=None,
                 lstm_units=[128, 64],
                 dense_units=[32],
                 dropout_rate=0.2,
                 learning_rate=0.001,
                 loss='huber',
                 bidirectional=False):
        """
        Initialize LSTM model architecture
        
        Parameters:
        -----------
        sequence_length : int
            Number of time steps in input sequences
        n_features : int
            Number of features per time step
        lstm_units : list
            List of LSTM layer sizes [128, 64] = two LSTM layers
        dense_units : list
            List of Dense layer sizes after LSTM
        dropout_rate : float
            Dropout rate for regularization (0.0 to 0.5)
        learning_rate : float
            Learning rate for Adam optimizer
        loss : str
            Loss function ('mse', 'mae', 'huber')
        bidirectional : bool
            Use bidirectional LSTM layers
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss = loss
        self.bidirectional = bidirectional
        
        self.model = None
        self.history = None
        self.training_time = None
        
    def build_model(self):
        """
        Build LSTM model architecture
        """
        print("\n" + "="*60)
        print("BUILDING LSTM MODEL")
        print("="*60)
        
        model = Sequential()
        
        # First LSTM layer
        if self.bidirectional:
            model.add(Bidirectional(
                LSTM(self.lstm_units[0], 
                     return_sequences=len(self.lstm_units) > 1,
                     input_shape=(self.sequence_length, self.n_features)),
                name='bidirectional_lstm_1'
            ))
        else:
            model.add(LSTM(self.lstm_units[0],
                          return_sequences=len(self.lstm_units) > 1,
                          input_shape=(self.sequence_length, self.n_features),
                          name='lstm_1'))
        
        model.add(Dropout(self.dropout_rate, name='dropout_1'))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)  # Last LSTM layer returns sequences=False
            
            if self.bidirectional:
                model.add(Bidirectional(
                    LSTM(units, return_sequences=return_seq),
                    name=f'bidirectional_lstm_{i}'
                ))
            else:
                model.add(LSTM(units, return_sequences=return_seq, name=f'lstm_{i}'))
            
            model.add(Dropout(self.dropout_rate, name=f'dropout_{i}'))
        
        # Dense layers
        for i, units in enumerate(self.dense_units, start=1):
            model.add(Dense(units, activation='relu', name=f'dense_{i}'))
            model.add(Dropout(self.dropout_rate, name=f'dropout_dense_{i}'))
        
        # Output layer
        model.add(Dense(1, name='output'))
        
        # Compile model
        if self.loss == 'huber':
            loss_fn = Huber()
        else:
            loss_fn = self.loss
        
        optimizer = Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        # Print model summary
        print("\nModel Architecture:")
        print("-" * 60)
        self.model.summary()
        print("-" * 60)
        
        # Count parameters
        total_params = self.model.count_params()
        print(f"\nTotal trainable parameters: {total_params:,}")
        print("="*60 + "\n")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val,
              epochs=100,
              batch_size=32,
              patience=15,
              min_delta=0.0001,
              verbose=1):
        """
        Train the LSTM model
        
        Parameters:
        -----------
        X_train : numpy array
            Training sequences (samples, sequence_length, n_features)
        y_train : numpy array
            Training targets (samples,)
        X_val : numpy array
            Validation sequences
        y_val : numpy array
            Validation targets
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Early stopping patience
        min_delta : float
            Minimum change to qualify as improvement
        verbose : int
            Verbosity level (0, 1, or 2)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Early stopping patience: {patience}")
        print("="*60 + "\n")
        
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
            'best_lstm_model.h5',
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
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
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
        plt.savefig(f'{save_path}lstm_training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}lstm_training_history.png")
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
        axes[0].set_title(f'LSTM Predictions vs Actual - {dataset_name} Set', 
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
        plt.savefig(f'{save_path}lstm_predictions_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to: {save_path}lstm_predictions_{dataset_name.lower()}.png")
        plt.close()
    
    def save_model(self, save_path='models/'):
        """
        Save model and configuration
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save Keras model
        self.model.save(f'{save_path}lstm_model.h5')
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'bidirectional': self.bidirectional,
            'training_time': self.training_time
        }
        
        with open(f'{save_path}lstm_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Model saved to: {save_path}lstm_model.h5")
        print(f"Config saved to: {save_path}lstm_config.json")
    
    @classmethod
    def load_model(cls, model_path='models/lstm_model.h5', config_path='models/lstm_config.json'):
        """
        Load saved model and configuration
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        predictor = cls(
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            lstm_units=config['lstm_units'],
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            loss=config['loss'],
            bidirectional=config['bidirectional']
        )
        
        predictor.model = load_model(model_path)
        predictor.training_time = config.get('training_time')
        
        print(f"Model loaded from: {model_path}")
        return predictor


# ============================================================================
# USAGE EXAMPLE AND MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - complete LSTM training pipeline
    """
    print("\n" + "="*80)
    print("LSTM MODEL FOR S&P 500 PREDICTION")
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
    
    # Initialize LSTM model
    lstm = LSTMPredictor(
        sequence_length=metadata['sequence_length'],
        n_features=metadata['n_features'],
        lstm_units=[128, 64],         # Two LSTM layers: 128 -> 64 units
        dense_units=[32],             # One dense layer with 32 units
        dropout_rate=0.2,             # 20% dropout for regularization
        learning_rate=0.001,          # Adam learning rate
        loss='huber',                 # Huber loss (robust to outliers)
        bidirectional=False           # Standard LSTM (not bidirectional)
    )
    
    # Build model
    lstm.build_model()
    
    # Train model
    lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
        patience=15,
        verbose=1
    )
    
    # Evaluate on all datasets
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    train_metrics = lstm.evaluate(X_train, y_train, dataset_name='Training')
    val_metrics = lstm.evaluate(X_val, y_val, dataset_name='Validation')
    test_metrics = lstm.evaluate(X_test, y_test, dataset_name='Test')
    
    # Save results
    results = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'config': {
            'lstm_units': lstm.lstm_units,
            'dense_units': lstm.dense_units,
            'dropout_rate': lstm.dropout_rate,
            'learning_rate': lstm.learning_rate,
            'training_time': lstm.training_time
        }
    }
    
    # Remove predictions/actuals before saving (too large)
    for dataset in ['train', 'val', 'test']:
        results[dataset] = {k: v for k, v in results[dataset].items() 
                           if k not in ['predictions', 'actuals']}
    
    with open('results/lstm_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to: results/lstm_results.json")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    lstm.plot_training_history(save_path='plots/')
    lstm.plot_predictions(test_metrics['actuals'], test_metrics['predictions'], 
                         dataset_name='Test', save_path='plots/')
    
    # Save model
    lstm.save_model(save_path='models/')
    
    print("\n" + "="*80)
    print("LSTM PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - models/lstm_model.h5")
    print("  - models/lstm_config.json")
    print("  - results/lstm_results.json")
    print("  - plots/lstm_training_history.png")
    print("  - plots/lstm_predictions_test.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Run main pipeline
    main()