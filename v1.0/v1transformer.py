import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import json
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for alternative models."""
    
    csv_files: List[str]
    sequence_length: int
    train_split: float
    validation_split: float
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    
    # Model-specific parameters
    model_type: str = 'transformer'  # 'transformer', 'xgboost', 'ensemble'
    
    # Transformer parameters
    num_heads: int = 8
    ff_dim: int = 256
    num_transformer_blocks: int = 4
    transformer_dropout: float = 0.1
    
    # XGBoost parameters
    xgb_n_estimators: int = 1000
    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.01
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # Bias parameters (same as LSTM)
    growth_bias: float = 0.005
    top_performer_weight: float = 1.8
    low_rep_boost: float = 1.5
    top_performer_percentile: float = 0.8
    
    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        config_dict = self.__dict__.copy()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

class TransformerBlock(tf.keras.layers.Layer):
    """Custom Transformer block for time series."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class AlternativeSP500Predictor:
    """Alternative models for S&P 500 prediction."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.xgb_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.data = None
        self.features = None
        self.is_aggregate = False
        
        logger.info(f"Initialized {config.model_type.upper()} predictor")
        logger.info(f"TensorFlow version: {tf.__version__}")
    
    def load_csv_data(self) -> pd.DataFrame:
        """Load and process CSV data (same as LSTM version)."""
        logger.info(f"Loading data from {len(self.config.csv_files)} CSV files...")
        try:
            raw_dataframes = []
            for i, filepath in enumerate(self.config.csv_files):
                if not os.path.exists(filepath):
                    logger.error(f"File not found: {filepath}")
                    raise FileNotFoundError(f"File not found: {filepath}")
                logger.info(f"Loading file {i+1}: {filepath}")
                df = pd.read_csv(filepath)
                raw_dataframes.append(df)
                logger.info(f"  Shape: {df.shape}")
            
            price_df = None
            company_info_df = None
            
            for df in raw_dataframes:
                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    price_df = df.copy()
                    logger.info("  -> Identified as price data")
                elif 'Symbol' in df.columns and 'Company_Name' in df.columns:
                    company_info_df = df.copy()
                    logger.info("  -> Identified as company info")
            
            if price_df is None:
                raise ValueError("No price data found. Need a file with Open/High/Low/Close columns.")
            
            # Set date index
            if not isinstance(price_df.index, pd.DatetimeIndex):
                date_cols = [col for col in price_df.columns if 'date' in col.lower()]
                if date_cols:
                    price_df[date_cols[0]] = pd.to_datetime(price_df[date_cols[0]], errors='coerce')
                    price_df = price_df.dropna(subset=[date_cols[0]])
                    price_df.set_index(date_cols[0], inplace=True)
                else:
                    price_df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(price_df), freq='Q')
            
            # Use Close and Volume
            if 'Close' in price_df.columns:
                merged_data = price_df[['Close', 'Volume']].copy() if 'Volume' in price_df.columns else price_df[['Close']].copy()
                self.is_aggregate = True
            else:
                raise ValueError("No Close price column found")
            
            merged_data = merged_data.dropna()
            self.data = merged_data
            logger.info(f"Final data shape: {merged_data.shape}")
            logger.info(f"Date range: {merged_data.index[0]} to {merged_data.index[-1]}")
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error loading CSV files: {e}")
            raise
    
    def prepare_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features with technical indicators."""
        logger.info("Preparing enhanced features...")
        
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['price'] = data['Close']
        
        # Volume features
        if 'Volume' in data.columns:
            features['volume'] = data['Volume']
            features['volume_ma'] = features['volume'].rolling(20, min_periods=1).mean()
            features['volume_ratio'] = features['volume'] / (features['volume_ma'] + 1e-10)
        
        # Returns and volatility
        features['returns'] = features['price'].pct_change()
        features['log_returns'] = np.log(features['price'] / features['price'].shift(1))
        features['volatility'] = features['returns'].rolling(10, min_periods=1).std()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'ma_{period}'] = features['price'].rolling(period, min_periods=1).mean()
            features[f'price_to_ma{period}'] = features['price'] / features[f'ma_{period}']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = features['price'].rolling(bb_period, min_periods=1).std()
        features['bb_upper'] = features['ma_20'] + 2 * bb_std
        features['bb_lower'] = features['ma_20'] - 2 * bb_std
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (features['price'] - features['bb_lower']) / (features['bb_width'] + 1e-10)
        
        # RSI
        delta = features['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features['price'].ewm(span=12, adjust=False).mean()
        exp2 = features['price'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Stochastic Oscillator
        low_14 = features['price'].rolling(14, min_periods=1).min()
        high_14 = features['price'].rolling(14, min_periods=1).max()
        features['stoch_k'] = 100 * (features['price'] - low_14) / (high_14 - low_14 + 1e-10)
        features['stoch_d'] = features['stoch_k'].rolling(3, min_periods=1).mean()
        
        # ATR (Average True Range)
        high = features['price'].rolling(1).max()
        low = features['price'].rolling(1).min()
        close_prev = features['price'].shift(1)
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14, min_periods=1).mean()
        
        # Price momentum
        for period in [3, 7, 14, 30]:
            features[f'momentum_{period}'] = features['price'].pct_change(period)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Store original price for later use
        self.original_prices = features['price'].copy()
        
        # Normalize features (except price which we'll handle separately)
        feature_cols = [col for col in features.columns if col != 'price']
        features[feature_cols] = self.feature_scaler.fit_transform(features[feature_cols])
        
        logger.info(f"Features prepared: {features.shape} with {len(features.columns)} indicators")
        self.features = features
        return features
    
    def prepare_data_transformer(self, features: pd.DataFrame) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """Prepare data for Transformer model."""
        logger.info("Preparing data for Transformer model...")
        
        # Scale price separately
        price_scaled = self.scaler.fit_transform(features[['price']].values)
        
        # Create sequences for all features
        feature_cols = [col for col in features.columns if col != 'price']
        X, y = [], []
        
        for i in range(self.config.sequence_length, len(features)):
            # Include all features in sequence
            feature_sequence = features[feature_cols].iloc[i-self.config.sequence_length:i].values
            price_sequence = price_scaled[i-self.config.sequence_length:i]
            
            # Combine price and features
            combined_sequence = np.concatenate([price_sequence, feature_sequence], axis=1)
            X.append(combined_sequence)
            y.append(price_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_size = int(len(X) * self.config.train_split)
        val_size = int(len(X) * self.config.validation_split)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Feature dimension: {X.shape[2]}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def prepare_data_xgboost(self, features: pd.DataFrame) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """Prepare data for XGBoost model."""
        logger.info("Preparing data for XGBoost model...")
        
        # Scale price
        price_scaled = self.scaler.fit_transform(features[['price']].values)
        
        # Create lagged features for XGBoost
        X, y = [], []
        
        for i in range(self.config.sequence_length, len(features)):
            # Flatten the sequence into features
            row_features = []
            
            # Add lagged prices
            for lag in range(self.config.sequence_length):
                row_features.append(price_scaled[i - self.config.sequence_length + lag, 0])
            
            # Add current technical indicators
            for col in features.columns:
                if col != 'price':
                    row_features.append(features[col].iloc[i-1])
            
            # Add statistical features of the sequence
            price_seq = price_scaled[i-self.config.sequence_length:i, 0]
            row_features.extend([
                np.mean(price_seq),
                np.std(price_seq),
                np.min(price_seq),
                np.max(price_seq),
                price_seq[-1] - price_seq[0],  # trend
                np.percentile(price_seq, 25),
                np.percentile(price_seq, 75)
            ])
            
            X.append(row_features)
            y.append(price_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_size = int(len(X) * self.config.train_split)
        val_size = int(len(X) * self.config.validation_split)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
        
        logger.info(f"XGBoost data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Number of features: {X.shape[1]}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def build_transformer_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build Transformer model for time series prediction."""
        logger.info("Building Transformer model...")
        
        inputs = tf.keras.Input(shape=input_shape)
        
        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=input_shape[0], 
            output_dim=input_shape[1]
        )(positions)
        
        x = inputs + position_embedding
        
        # Stack Transformer blocks
        for _ in range(self.config.num_transformer_blocks):
            x = TransformerBlock(
                input_shape[1],
                self.config.num_heads,
                self.config.ff_dim,
                self.config.transformer_dropout
            )(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Final layers
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.config.transformer_dropout)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.config.transformer_dropout)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Custom loss with growth bias
        @tf.function
        def biased_loss(y_true, y_pred):
            mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
            diff = y_pred - y_true
            # Penalize under-predictions more
            bias_penalty = tf.where(
                diff < 0,
                tf.abs(diff) * (1 + self.config.growth_bias * 2),
                tf.abs(diff)
            )
            return mse + tf.reduce_mean(bias_penalty) * 0.1
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=biased_loss,
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info(f"Transformer model built with {model.count_params():,} parameters")
        return model
    
    def train_transformer(self, train_data: Tuple, val_data: Tuple) -> tf.keras.callbacks.History:
        """Train Transformer model."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        if self.model is None:
            self.build_transformer_model(X_train.shape[1:])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        logger.info("Training Transformer model...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Transformer training completed!")
        return self.history
    
    def train_xgboost(self, train_data: Tuple, val_data: Tuple):
        """Train XGBoost model."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        logger.info("Training XGBoost model...")
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.config.xgb_max_depth,
            'learning_rate': self.config.xgb_learning_rate,
            'n_estimators': self.config.xgb_n_estimators,
            'subsample': self.config.xgb_subsample,
            'colsample_bytree': self.config.xgb_colsample_bytree,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }
        
        self.xgb_model = xgb.XGBRegressor(**params)
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        logger.info("XGBoost training completed!")
        
        # Feature importance
        if hasattr(self.xgb_model, 'feature_importances_'):
            importance = self.xgb_model.feature_importances_
            indices = np.argsort(importance)[-10:]  # Top 10 features
            logger.info("Top 10 most important features:")
            for i in indices:
                logger.info(f"  Feature {i}: {importance[i]:.4f}")
    
    def predict_transformer(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Transformer model."""
        if self.model is None:
            raise ValueError("Transformer model not trained")
        
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
        
        # Apply growth bias
        predictions = self.apply_growth_bias(predictions.flatten())
        
        return predictions
    
    def predict_xgboost(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost model."""
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained")
        
        predictions_scaled = self.xgb_model.predict(X)
        predictions = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
        
        # Apply growth bias
        predictions = self.apply_growth_bias(predictions.flatten())
        
        return predictions
    
    def apply_growth_bias(self, predictions: np.ndarray) -> np.ndarray:
        """Apply growth bias to predictions."""
        bias_factor = 1 + self.config.growth_bias
        
        for i in range(len(predictions)):
            time_factor = 1 - np.exp(-i / 10)
            predictions[i] *= (1 + self.config.growth_bias * time_factor * bias_factor)
        
        return predictions
    
    def evaluate(self, test_data: Tuple, model_type: str = None) -> Dict[str, float]:
        """Evaluate model performance."""
        if model_type is None:
            model_type = self.config.model_type
        
        X_test, y_test = test_data
        
        if model_type == 'transformer':
            predictions = self.predict_transformer(X_test)
        elif model_type == 'xgboost':
            predictions = self.predict_xgboost(X_test)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, predictions)
        
        # MAPE
        non_zero_mask = y_test_actual != 0
        mape = np.mean(np.abs((y_test_actual[non_zero_mask] - predictions[non_zero_mask]) / 
                             y_test_actual[non_zero_mask])) * 100
        
        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(y_test_actual) > 0
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
        else:
            directional_accuracy = 0.0
        
        metrics = {
            'Model': model_type.upper(),
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2),
            'MAPE': float(mape),
            'Directional_Accuracy': float(directional_accuracy)
        }
        
        logger.info("\n" + "="*60)
        logger.info(f"{model_type.upper()} MODEL EVALUATION RESULTS")
        logger.info("="*60)
        for metric, value in metrics.items():
            if metric != 'Model':
                logger.info(f"{metric}: {value:.4f}")
        logger.info("="*60)
        
        return metrics, predictions, y_test_actual
    
    def plot_results(self, predictions: np.ndarray, actual: np.ndarray, 
                    model_name: str = "", n_points: int = 200):
        """Plot prediction results."""
        plt.figure(figsize=(15, 8))
        
        n_points = min(n_points, len(predictions))
        
        plt.subplot(2, 1, 1)
        plt.plot(actual[-n_points:], label='Actual Price', color='blue', linewidth=2)
        plt.plot(predictions[-n_points:], label=f'Predicted ({model_name})', 
                color='red', linewidth=2, alpha=0.8)
        plt.title(f'{model_name} Model - S&P 500 Predictions')
        plt.xlabel('Time Steps')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        errors = predictions[-n_points:] - actual[-n_points:]
        plt.plot(errors, label='Prediction Error', color='green', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Prediction Errors')
        plt.xlabel('Time Steps')
        plt.ylabel('Error ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """Plot training history for neural network models."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

class EnsemblePredictor:
    """Ensemble model combining Transformer and XGBoost."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.transformer_predictor = AlternativeSP500Predictor(config)
        self.xgboost_predictor = AlternativeSP500Predictor(config)
        self.weights = {'transformer': 0.6, 'xgboost': 0.4}
        
    def train(self, data: pd.DataFrame):
        """Train both models."""
        logger.info("Training Ensemble Model...")
        
        # Prepare features
        features = self.transformer_predictor.prepare_enhanced_features(data)
        
        # Prepare data for each model
        transformer_data = self.transformer_predictor.prepare_data_transformer(features)
        xgboost_data = self.xgboost_predictor.prepare_data_xgboost(features)
        
        # Train Transformer
        logger.info("Training Transformer component...")
        self.transformer_predictor.train_transformer(
            transformer_data[0], transformer_data[1]
        )
        
        # Train XGBoost
        logger.info("Training XGBoost component...")
        self.xgboost_predictor.xgb_model = None  # Reset
        self.xgboost_predictor.scaler = self.transformer_predictor.scaler  # Share scaler
        self.xgboost_predictor.train_xgboost(
            xgboost_data[0], xgboost_data[1]
        )
        
        return transformer_data[2], xgboost_data[2]
    
    def predict(self, transformer_test: Tuple, xgboost_test: Tuple) -> np.ndarray:
        """Make ensemble predictions."""
        X_trans, _ = transformer_test
        X_xgb, _ = xgboost_test
        
        trans_pred = self.transformer_predictor.predict_transformer(X_trans)
        xgb_pred = self.xgboost_predictor.predict_xgboost(X_xgb)
        
        # Weighted average
        ensemble_pred = (
            self.weights['transformer'] * trans_pred +
            self.weights['xgboost'] * xgb_pred
        )
        
        return ensemble_pred
    
    def evaluate(self, transformer_test: Tuple, xgboost_test: Tuple) -> Dict[str, float]:
        """Evaluate ensemble model."""
        _, y_test = transformer_test
        predictions = self.predict(transformer_test, xgboost_test)
        
        y_test_actual = self.transformer_predictor.scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, predictions)
        
        non_zero_mask = y_test_actual != 0
        mape = np.mean(np.abs((y_test_actual[non_zero_mask] - predictions[non_zero_mask]) / 
                             y_test_actual[non_zero_mask])) * 100
        
        metrics = {
            'Model': 'ENSEMBLE',
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2),
            'MAPE': float(mape)
        }
        
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE MODEL EVALUATION RESULTS")
        logger.info("="*60)
        for metric, value in metrics.items():
            if metric != 'Model':
                logger.info(f"{metric}: {value:.4f}")
        logger.info("="*60)
        
        return metrics, predictions, y_test_actual

def main():
    """Main execution function."""
    
    # Define your CSV files here
    csv_files = [
        'csv/sp500_analysis_20250830_094857.csv',
        'csv/sp500_quarterly_20250830_103017_prices.csv',
        'csv/sp500_quarterly_20250830_103017_summary.csv'
    ]
    
    # Check if files exist
    if not all(os.path.exists(f) for f in csv_files):
        missing = [f for f in csv_files if not os.path.exists(f)]
        logger.error(f"Missing files: {missing}")
        logger.info("\nPlease update the csv_files list with your actual file paths")
        return
    
    # Configuration
    config = ModelConfig(
        csv_files=csv_files,
        sequence_length=20,
        train_split=0.8,
        validation_split=0.1,
        batch_size=32,
        epochs=100,
        learning_rate=0.001,
        patience=20,
        model_type='transformer',  # Change to 'xgboost' or 'ensemble' as needed
        growth_bias=0.005,
        top_performer_weight=1.8,
        low_rep_boost=1.5,
        top_performer_percentile=0.8
    )
    
    logger.info("=" * 70)
    logger.info(f"🚀 ALTERNATIVE S&P 500 PREDICTOR - {config.model_type.upper()}")
    logger.info("=" * 70)
    
    results = {}
    
    # Test Transformer Model
    logger.info("\n" + "="*70)
    logger.info("TESTING TRANSFORMER MODEL")
    logger.info("="*70)
    
    transformer_predictor = AlternativeSP500Predictor(config)
    data = transformer_predictor.load_csv_data()
    features = transformer_predictor.prepare_enhanced_features(data)
    train_data, val_data, test_data = transformer_predictor.prepare_data_transformer(features)
    
    transformer_predictor.train_transformer(train_data, val_data)
    trans_metrics, trans_pred, trans_actual = transformer_predictor.evaluate(test_data, 'transformer')
    transformer_predictor.plot_training_history()
    transformer_predictor.plot_results(trans_pred, trans_actual, "Transformer")
    results['Transformer'] = trans_metrics
    
    # Test XGBoost Model
    logger.info("\n" + "="*70)
    logger.info("TESTING XGBOOST MODEL")
    logger.info("="*70)
    
    xgboost_predictor = AlternativeSP500Predictor(config)
    xgboost_predictor.data = data
    xgboost_predictor.features = features
    xgb_train_data, xgb_val_data, xgb_test_data = xgboost_predictor.prepare_data_xgboost(features)
    
    xgboost_predictor.train_xgboost(xgb_train_data, xgb_val_data)
    xgb_metrics, xgb_pred, xgb_actual = xgboost_predictor.evaluate(xgb_test_data, 'xgboost')
    xgboost_predictor.plot_results(xgb_pred, xgb_actual, "XGBoost")
    results['XGBoost'] = xgb_metrics
    
    # Test Ensemble Model
    logger.info("\n" + "="*70)
    logger.info("TESTING ENSEMBLE MODEL")
    logger.info("="*70)
    
    ensemble = EnsemblePredictor(config)
    ensemble.transformer_predictor.data = data
    ensemble.xgboost_predictor.data = data
    transformer_test, xgboost_test = ensemble.train(data)
    
    ensemble_metrics, ensemble_pred, ensemble_actual = ensemble.evaluate(
        (transformer_test[0], transformer_test[1]),
        (xgboost_test[0], xgboost_test[1])
    )
    results['Ensemble'] = ensemble_metrics
    
    # Compare all models
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*70)
    
    comparison_df = pd.DataFrame(results).T
    logger.info("\n" + comparison_df.to_string())
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    n_points = min(200, len(trans_pred))
    
    plt.subplot(2, 1, 1)
    plt.plot(trans_actual[-n_points:], label='Actual', color='black', linewidth=2)
    plt.plot(trans_pred[-n_points:], label='Transformer', alpha=0.7)
    plt.plot(xgb_pred[-n_points:], label='XGBoost', alpha=0.7)
    plt.plot(ensemble_pred[-n_points:], label='Ensemble', alpha=0.7, linewidth=2)
    plt.title('Model Comparison - S&P 500 Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    metrics_to_plot = ['MAE', 'RMSE', 'R²', 'MAPE']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, model in enumerate(['Transformer', 'XGBoost', 'Ensemble']):
        values = [results[model][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=model)
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, metrics_to_plot)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results, ensemble

if __name__ == "__main__":
    """
    Alternative S&P 500 Prediction Models
    =====================================
    
    This script provides three alternative models to LSTM:
    
    1. **Transformer Model**: Uses self-attention mechanism instead of recurrent connections.
       - Better at capturing long-range dependencies
       - Parallel processing capability
       - Often outperforms LSTMs on sequence modeling tasks
    
    2. **XGBoost Model**: Gradient boosting approach (non-neural network).
       - Tree-based ensemble method
       - Handles non-linear relationships well
       - Fast training and inference
       - Provides feature importance
    
    3. **Ensemble Model**: Combines Transformer and XGBoost predictions.
       - Weighted average of both models
       - Often more robust than individual models
    
    All models use the same:
    - Data loading and preprocessing pipeline
    - Feature engineering (technical indicators)
    - Growth bias and performance weighting
    - Evaluation metrics
    
    Key differences from LSTM:
    - Transformer: Attention-based vs. recurrent processing
    - XGBoost: Tree-based vs. neural network
    - Different ways of handling sequential data
    - Generally faster training than LSTM
    
    The script will automatically:
    1. Train all three models
    2. Evaluate performance
    3. Display comparison charts
    4. Show which model performs best
    """
    
    results, ensemble = main()