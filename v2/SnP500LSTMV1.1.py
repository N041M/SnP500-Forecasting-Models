"""
Bidirectional LSTM Model for S&P 500 Prediction (V1.1)
=======================================================
This script merges V1's clean workflow with v1.1.2's accuracy improvements:

V1 Workflow (Preserved):
- Load pre-prepared numpy arrays from data/ folder
- Work in scaled space (no inverse transforms)
- Clean, simple training pipeline

V1.1.2 Enhancements (Added):
- Bidirectional LSTM architecture for capturing patterns in both directions
- Dual outputs (price prediction + direction classification)
- Custom biased loss function (penalizes under-predictions 2x more)
- Direction labels derived from returns (y > 0 = up, y <= 0 = down)
- Simple growth bias applied to predictions

Author: Ronald
Version: 1.1 (Fixed)
Date: 2026-01-25
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
import logging

# TensorFlow/Keras imports
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LSTMConfigV1_1:
    """Configuration for Bidirectional LSTM V1.1"""
    sequence_length: int
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 0.001
    patience: int = 30
    lstm_units: List[int] = None
    dropout_rate: float = 0.25
    growth_bias: float = 0.005  # 0.5% optimistic bias

    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")


class LSTMPredictorV1_1:
    """
    Bidirectional LSTM V1.1 with Dual Outputs (Price + Direction)
    Compatible with V1 data preparation workflow
    """

    def __init__(self, config: LSTMConfigV1_1, n_features: int):
        self.config = config
        self.n_features = n_features
        self.model = None
        self.history = None
        self.training_time = None

        logger.info(f"Bidirectional LSTM V1.1 initialized")
        logger.info(f"Architecture: Bidirectional LSTM {config.lstm_units}")
        logger.info(f"Features: {n_features}, Sequence length: {config.sequence_length}")
        logger.info(f"Growth bias: {config.growth_bias:.3%}")

    def build_model(self):
        """
        Build Bidirectional LSTM with dual outputs
        """
        print("\n" + "="*60)
        print("BUILDING BIDIRECTIONAL LSTM MODEL (V1.1)")
        print("="*60)

        inputs = Input(shape=(self.config.sequence_length, self.n_features))

        # Bidirectional LSTM layers
        x = inputs
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = (i < len(self.config.lstm_units) - 1)
            x = Bidirectional(
                LSTM(units, return_sequences=return_sequences),
                name=f'bidirectional_lstm_{i+1}'
            )(x)
            x = Dropout(self.config.dropout_rate)(x)

        # Dual outputs
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
        print(f"  - Bidirectional LSTM layers: {self.config.lstm_units}")
        print(f"  - Dropout rate: {self.config.dropout_rate}")
        print(f"  - Price output: Linear (custom biased MSE loss)")
        print(f"  - Direction output: Sigmoid (binary crossentropy)")
        print(f"  - Total parameters: {self.model.count_params():,}")
        print("="*60 + "\n")

        return self.model

    def train(self, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """
        Train the model

        Args:
            train_data: (X_train, y_train_price, y_train_direction)
            val_data: (X_val, y_val_price, y_val_direction)
        """
        X_train, y_train_price, y_train_direction = train_data
        X_val, y_val_price, y_val_direction = val_data

        if self.model is None:
            self.build_model()

        print("\n" + "="*60)
        print("TRAINING V1.1 LSTM MODEL")
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
        print(f"LSTM V1.1 EVALUATION RESULTS - {dataset_name.upper()}")
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

        import os
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
        plt.savefig(f'{save_path}lstm_v1.1_training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}lstm_v1.1_training_history.png")
        plt.close()

    def plot_predictions(self, y_true, y_pred, dataset_name='Test', save_path='plots/'):
        """Plot predictions vs actual"""
        import os
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(14, 6))
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(f'LSTM V1.1 - {dataset_name} Set Predictions')
        plt.xlabel('Sample')
        plt.ylabel('Scaled Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}lstm_v1.1_predictions_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to: {save_path}lstm_v1.1_predictions_{dataset_name.lower()}.png")
        plt.close()


def create_default_config(sequence_length: int = 60) -> LSTMConfigV1_1:
    """Create default configuration"""
    return LSTMConfigV1_1(
        sequence_length=sequence_length,
        batch_size=32,
        epochs=200,
        learning_rate=0.001,
        patience=30,
        lstm_units=[128, 64, 32],
        dropout_rate=0.25,
        growth_bias=0.005
    )


def main():
    """
    Main execution function - V1 workflow with V1.1 enhancements
    """
    print("\n" + "="*80)
    print("BIDIRECTIONAL LSTM MODEL V1.1 FOR S&P 500 PREDICTION")
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
    config.save('models/lstm_v1.1_config.json')

    logger.info("=" * 70)
    logger.info("📊 LSTM V1.1 PREDICTOR V1.1")
    logger.info("=" * 70)
    logger.info(f"📈 Growth Bias: {config.growth_bias:.3%}")
    logger.info(f"🔢 Sequence Length: {metadata['sequence_length']} steps")
    logger.info(f"🧠 Architecture: Bidirectional LSTM {config.lstm_units}")
    logger.info(f"🎯 Epochs: {config.epochs}, Patience: {config.patience}")
    logger.info("=" * 70)

    lstm = LSTMPredictorV1_1(config, n_features=metadata['n_features'])
    lstm.build_model()

    # ========================================================================
    # STEP 4: Train model
    # ========================================================================
    lstm.train(
        (X_train, y_train, y_train_direction),
        (X_val, y_val, y_val_direction)
    )

    # ========================================================================
    # STEP 5: Evaluate on all datasets
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    train_metrics, train_pred, _ = lstm.evaluate(
        (X_train, y_train, y_train_direction),
        'Train'
    )

    val_metrics, val_pred, _ = lstm.evaluate(
        (X_val, y_val, y_val_direction),
        'Validation'
    )

    test_metrics, test_pred, test_dir_pred = lstm.evaluate(
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
        'training_time_seconds': lstm.training_time
    }

    import os
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

    with open('results/lstm_v1.1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: results/lstm_v1.1_results.json")

    # ========================================================================
    # STEP 7: Generate visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    lstm.plot_training_history()
    lstm.plot_predictions(y_test, test_pred, 'Test')

    # ========================================================================
    # STEP 8: Save model
    # ========================================================================
    os.makedirs('models', exist_ok=True)
    lstm.model.save('models/lstm_v1.1_model.keras')
    print(f"\nModel saved to: models/lstm_v1.1_model.keras")

    print("\n" + "="*80)
    print("LSTM V1.1 V1.1 PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - models/lstm_v1.1_model.keras")
    print("  - models/lstm_v1.1_config.json")
    print("  - results/lstm_v1.1_results.json")
    print("  - plots/lstm_v1.1_training_history.png")
    print("  - plots/lstm_v1.1_predictions_test.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
