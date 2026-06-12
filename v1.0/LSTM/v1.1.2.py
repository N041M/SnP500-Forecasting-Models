import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
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
class EnhancedModelConfig:
    """Enhanced configuration with market bias parameters."""
    
    csv_files: List[str]
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
    def load(cls, filepath: Union[str, Path]) -> 'EnhancedModelConfig':
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls(**config_dict)

class EnhancedSP500Predictor:
    
    def __init__(self, config: EnhancedModelConfig):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.data = None
        self.company_weights = {}
        self.company_representations = {}
        self.negative_streak_counter = 0
        self.is_aggregate = False
        self.momentum_adjustment = 1.0
        self.company_info = None
        self.sectors = None
        
        logger.info("Enhanced Predictor initialized with custom biases")
        logger.info(f"Growth bias: {self.config.growth_bias}")
        logger.info(f"Top performer weight: {self.config.top_performer_weight}")
        logger.info(f"Low representation boost: {self.config.low_rep_boost}")
        logger.info(f"TensorFlow version: {tf.__version__}")
    
    def load_csv_data(self) -> pd.DataFrame:
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
                logger.info(f"  Shape: {df.shape}, Columns: {list(df.columns)[:5]}...")
                logger.info(f"  First 5 rows:\n{df.head().to_string()}")
            
            price_df = None
            company_info_df = None
            summary_df = None
            
            for df in raw_dataframes:
                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    price_df = df.copy()
                    logger.info("  -> Identified as price data")
                elif 'Symbol' in df.columns and 'Company_Name' in df.columns:
                    company_info_df = df.copy()
                    logger.info("  -> Identified as company info")
                elif 'Symbol' in df.columns and 'Quarters_Available' in df.columns:
                    summary_df = df.copy()
                    logger.info("  -> Identified as summary data")
            
            if price_df is None:
                logger.error("No price data found in CSV files!")
                raise ValueError("No price data found. Need a file with Open/High/Low/Close columns.")
            
            logger.info(f"Price DataFrame shape: {price_df.shape}")
            logger.info(f"Price DataFrame columns: {list(price_df.columns)}")
            logger.info(f"Price DataFrame head:\n{price_df.head().to_string()}")
            
            if not isinstance(price_df.index, pd.DatetimeIndex):
                date_cols = [col for col in price_df.columns if 'date' in col.lower()]
                if date_cols:
                    price_df[date_cols[0]] = pd.to_datetime(price_df[date_cols[0]], errors='coerce', utc=True)
                    if price_df[date_cols[0]].isna().any():
                        logger.warning("Some dates could not be parsed, dropping invalid rows")
                        price_df = price_df.dropna(subset=[date_cols[0]])
                    price_df.set_index(date_cols[0], inplace=True)
                    logger.info(f"Set index to {date_cols[0]}")
                else:
                    logger.warning("No date column found, creating date index assuming quarterly data")
                    price_df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(price_df), freq='Q')
            
            if 'Symbol' in price_df.columns and company_info_df is not None and 'Market_Cap_Billions' in company_info_df.columns:
                logger.info("Aggregating price data by market-cap weighting...")
                price_df['Symbol'] = price_df['Symbol'].str.upper().str.strip()
                company_info_df['Symbol'] = company_info_df['Symbol'].str.upper().str.strip()
                merged_data = pd.merge(price_df, company_info_df[['Symbol', 'Market_Cap_Billions']],
                                     on='Symbol', how='left')
                logger.info(f"Merged data shape after adding market caps: {merged_data.shape}")
                
                if merged_data['Market_Cap_Billions'].isna().any():
                    logger.warning("Some symbols have missing market caps, filling with median")
                    merged_data['Market_Cap_Billions'] = merged_data['Market_Cap_Billions'].fillna(
                        merged_data['Market_Cap_Billions'].median())
                
                merged_data['Weighted_Close'] = merged_data['Close'] * merged_data['Market_Cap_Billions']
                aggregated_data = merged_data.groupby(merged_data.index)['Weighted_Close'].sum() / \
                                merged_data.groupby(merged_data.index)['Market_Cap_Billions'].sum()
                merged_data = pd.DataFrame({
                    'Close': aggregated_data,
                    'Volume': merged_data.groupby(merged_data.index)['Volume'].sum()
                })
                self.is_aggregate = True
                logger.info("Created market-cap-weighted S&P 500 index")
            else:
                logger.warning("No Symbol or Market_Cap_Billions found, using raw Close prices")
                merged_data = price_df[['Close', 'Volume']].copy()
                self.is_aggregate = True
            
            if merged_data.index.duplicated().any():
                logger.warning("Duplicate index values found, keeping first occurrence")
                merged_data = merged_data[~merged_data.index.duplicated(keep='first')]
            
            if company_info_df is not None:
                self.company_info = company_info_df
                logger.info(f"Loaded company info for {len(company_info_df)} companies")
                if 'Sector' in company_info_df.columns:
                    self.sectors = company_info_df['Sector'].unique()
                    logger.info(f"Identified {len(self.sectors)} sectors: {', '.join(self.sectors[:5])}...")
            else:
                self.company_info = None
                logger.info("No company metadata found - using price data only")
            
            merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
            merged_data = merged_data.dropna()
            logger.info(f"Merged data shape after cleaning: {merged_data.shape}")
            
            if merged_data.empty:
                logger.error("Merged data is empty after cleaning!")
                raise ValueError("Merged data is empty after removing NaN values.")
            
            self._identify_companies(merged_data)
            self.data = merged_data
            logger.info(f"Final data shape: {merged_data.shape}")
            logger.info(f"Date range: {merged_data.index[0]} to {merged_data.index[-1]}")
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error loading CSV files: {e}")
            logger.info("Please ensure your CSV files have compatible structure")
            raise
    
    def _identify_companies(self, data: pd.DataFrame):
        price_keywords = ['close', 'price', 'adj close', 'adjusted']
        company_cols = {}
        
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in price_keywords):
                company_name = col.split('_')[0] if '_' in col else col.replace('Close', '').replace('Price', '').strip()
                if company_name:
                    company_cols[company_name] = col
        
        if not company_cols:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            company_cols = {f"Asset_{i+1}": col for i, col in enumerate(numeric_cols)}
        
        self.company_columns = company_cols
        logger.info(f"Identified {len(company_cols)} companies/assets")
        
        for company in company_cols:
            self.company_weights[company] = 1.0
            self.company_representations[company] = 1.0 / len(company_cols)
    
    def _apply_sector_rotation(self, features: pd.DataFrame):
        if not hasattr(self, 'company_info') or self.company_info is None:
            return
        
        if 'Market_Cap_Billions' in self.company_info.columns and 'Sector' in self.company_info.columns:
            sector_caps = self.company_info.groupby('Sector')['Market_Cap_Billions'].sum()
            sector_weights = sector_caps / sector_caps.sum()
            
            if 'bull_market' in features.columns and features['bull_market'].iloc[-1] == 1:
                growth_sectors = ['Technology', 'Communication Services', 'Consumer Discretionary']
                for sector in growth_sectors:
                    if sector in sector_weights.index:
                        sector_weights[sector] *= self.config.top_performer_weight
            elif features['volatility'].iloc[-1] > features['volatility'].quantile(0.7):
                defensive_sectors = ['Utilities', 'Consumer Staples', 'Health Care']
                for sector in defensive_sectors:
                    if sector in sector_weights.index:
                        sector_weights[sector] *= self.config.low_rep_boost
            
            sector_weights = sector_weights / sector_weights.sum()
            features['sector_weight_adjustment'] = sector_weights.mean()
    
    def calculate_performance_weights(self, data: pd.DataFrame, lookback: int = 20):
        if self.is_aggregate:
            logger.info("Calculating momentum-based adjustments...")
            if 'Close' in data.columns:
                recent_data = data['Close'].tail(lookback)
                returns = recent_data.pct_change().dropna()
                if len(returns) > 0:
                    momentum = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1 if len(recent_data) > 0 else 0
                    if momentum > 0.05:
                        self.momentum_adjustment = self.config.top_performer_weight
                        logger.info(f"Strong momentum detected: {momentum:.2%}, applying weight {self.momentum_adjustment}")
                    else:
                        self.momentum_adjustment = 1.0
        else:
            performances = {}
            for company, col in self.company_columns.items():
                if col in data.columns:
                    recent_data = data[col].tail(lookback)
                    returns = recent_data.pct_change().dropna()
                    avg_return = returns.mean()
                    sharpe = returns.mean() / (returns.std() + 1e-8)
                    cumulative_return = (1 + returns).prod() - 1
                    performances[company] = avg_return * 0.4 + sharpe * 0.3 + cumulative_return * 0.3
            
            if performances:
                threshold = np.percentile(list(performances.values()), 
                                         self.config.top_performer_percentile * 100)
                for company, perf in performances.items():
                    if perf >= threshold:
                        self.company_weights[company] = self.config.top_performer_weight
                        logger.info(f"  Top performer: {company} (weight: {self.config.top_performer_weight})")
                    else:
                        self.company_weights[company] = 1.0
    
    def adjust_for_negative_streaks(self, returns: pd.DataFrame):
        if isinstance(returns, pd.Series):
            recent_returns = returns.tail(self.config.negative_streak_threshold)
        else:
            recent_returns = returns.tail(self.config.negative_streak_threshold)
        
        if len(recent_returns) >= self.config.negative_streak_threshold:
            negative_days = (recent_returns < 0).sum()
            if negative_days >= self.config.negative_streak_threshold:
                logger.info(f"Negative streak detected ({negative_days} days)")
                if self.is_aggregate:
                    logger.info("Applying diversification boost for risk mitigation")
                    self.momentum_adjustment *= self.config.low_rep_boost
                else:
                    sorted_companies = sorted(self.company_representations.items(), key=lambda x: x[1])
                    n_boost = max(1, len(sorted_companies) // 3)
                    for company, rep in sorted_companies[:n_boost]:
                        self.company_weights[company] *= self.config.low_rep_boost
                        logger.info(f"  Boosted {company}: weight now {self.company_weights[company]:.2f}")
    
    def prepare_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preparing enhanced features...")
        if data.empty:
            logger.error("Input data is empty!")
            raise ValueError("Input data is empty")
        
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {list(data.columns)}")
        logger.info(f"Input data head:\n{data.head().to_string()}")
        
        features = pd.DataFrame(index=data.index)
        
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
        
        logger.info(f"Features shape after adding price: {features.shape}")
        
        if 'Volume' in data.columns:
            features['volume'] = data['Volume']
            features['volume_ma'] = features['volume'].rolling(20, min_periods=1).mean()
            features['volume_ratio'] = features['volume'] / features['volume_ma']
            logger.info("Added volume indicators")
        
        logger.info("Calculating technical indicators...")
        features['returns'] = features['price'].pct_change()
        features['ma_5'] = features['price'].rolling(5, min_periods=1).mean()
        features['ma_20'] = features['price'].rolling(10, min_periods=1).mean()
        features['ma_50'] = features['price'].rolling(20, min_periods=1).mean()
        features['volatility'] = features['returns'].rolling(10, min_periods=1).std()
        
        features['bollinger_upper'] = features['ma_20'] + 2 * features['price'].rolling(20, min_periods=1).std()
        features['bollinger_lower'] = features['ma_20'] - 2 * features['price'].rolling(20, min_periods=1).std()
        low_14 = features['price'].rolling(14, min_periods=1).min()
        high_14 = features['price'].rolling(14, min_periods=1).max()
        features['stochastic_k'] = 100 * (features['price'] - low_14) / (high_14 - low_14 + 1e-10)
        
        logger.info(f"Features shape after technical indicators: {features.shape}")
        
        features['price_to_ma5'] = features['price'] / features['ma_5']
        features['price_to_ma20'] = features['price'] / features['ma_20']
        
        delta = features['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = features['price'].ewm(span=6, adjust=False).mean()
        exp2 = features['price'].ewm(span=13, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=4, adjust=False).mean()
        
        if len(features) >= 10:
            features['bull_market'] = (features['ma_50'] > features['ma_50'].shift(10, fill_value=features['ma_50'].iloc[0])).astype(int)
            logger.info("Added bull_market indicator")
        else:
            features['bull_market'] = 0
            logger.warning("Insufficient data for bull_market calculation, setting to 0")
        
        features['performance_rank'] = features['returns'].rolling(10, min_periods=1).mean().rank(pct=True)
        
        features['momentum_bias'] = np.where(
            features['returns'].rolling(5, min_periods=1).mean() > 0,
            1 + self.config.growth_bias,
            1.0
        )
        
        logger.info("Detecting negative streaks...")
        features['negative_streak'] = 0.0
        streak = 0
        for i in range(1, len(features)):
            if not pd.isna(features['returns'].iloc[i]) and features['returns'].iloc[i] < 0:
                streak += 1
            else:
                streak = 0
            features.loc[features.index[i], 'negative_streak'] = float(streak)
        
        features['diversification_factor'] = np.where(
            features['negative_streak'] >= self.config.negative_streak_threshold,
            self.config.low_rep_boost,
            1.0
        )
        
        if hasattr(self, 'company_info') and self.company_info is not None:
            self._apply_sector_rotation(features)
        
        features['weighted_price'] = features['price'] * features['momentum_bias']
        
        logger.info(f"Features shape before dropping NaNs: {features.shape}")
        features = features.ffill().bfill()
        logger.info(f"Features shape after filling NaNs: {features.shape}")
        
        for col in features.columns:
            if col != 'price' and features[col].dtype in [np.float64, np.float32]:
                features[col] = MinMaxScaler().fit_transform(features[[col]])
        
        if len(features) < self.config.sequence_length + 10:
            logger.warning(f"Insufficient data after feature engineering. Need at least {self.config.sequence_length + 10} rows, got {len(features)}. Proceeding for testing.")
        
        logger.info(f"Features prepared: {features.shape} with {len(features.columns)} indicators")
        logger.info(f"Available features: {list(features.columns)}")
        return features
    
    def apply_growth_bias(self, predictions: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        bias_factor = 1 + self.config.growth_bias
        if 'bull_market' in features.columns and len(features) > 0:
            if features['bull_market'].iloc[-1] == 1:
                bias_factor *= 1.5
                logger.info("Applying enhanced growth bias due to bull market detection")
            else:
                logger.info("No bull market detected, applying standard growth bias")
        else:
            logger.warning("bull_market feature not found, applying standard growth bias")
        
        for i in range(len(predictions)):
            time_factor = 1 - np.exp(-i / 10)
            predictions[i] *= (1 + self.config.growth_bias * time_factor * bias_factor)
        return predictions
    
    def prepare_data(self, features: pd.DataFrame) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
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
        
        dataset_scaled = self.scaler.fit_transform(dataset)
        
        X, y_price, y_direction = [], [], []
        for i in range(self.config.sequence_length, len(dataset_scaled)):
            X.append(dataset_scaled[i-self.config.sequence_length:i, 0])
            y_price.append(dataset_scaled[i, 0])
            y_direction.append(1 if dataset_scaled[i, 0] > dataset_scaled[i-1, 0] else 0)
        
        X, y_price, y_direction = np.array(X), np.array(y_price), np.array(y_direction)
        
        train_size = int(len(X) * self.config.train_split)
        val_size = int(len(X) * self.config.validation_split)
        
        X_train = X[:train_size]
        y_train_price = y_price[:train_size]
        y_train_direction = y_direction[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val_price = y_price[train_size:train_size + val_size]
        y_val_direction = y_direction[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test_price = y_price[train_size + val_size:]
        y_test_direction = y_direction[train_size + val_size:]
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, y_train_price, y_train_direction), (X_val, y_val_price, y_val_direction), (X_test, y_test_price, y_test_direction)
    
    def build_model(self, input_shape: Tuple[int,]) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(input_shape[0], 1))
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.config.lstm_units[0], return_sequences=len(self.config.lstm_units) > 1)
        )(inputs)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        for i, units in enumerate(self.config.lstm_units[1:], 1):
            return_seq = i < len(self.config.lstm_units) - 1
            x = tf.keras.layers.LSTM(units, return_sequences=return_seq)(x)
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        x = tf.keras.layers.Dense(50, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate / 2)(x)
        x = tf.keras.layers.Dense(25, activation='relu')(x)
        price_output = tf.keras.layers.Dense(1, name='price_output')(x)
        direction_output = tf.keras.layers.Dense(1, activation='sigmoid', name='direction_output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=[price_output, direction_output])
        
        @tf.function
        def biased_mse(y_true, y_pred):
            mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
            diff = y_pred - y_true
            bias_penalty = tf.where(diff < 0, 
                                   tf.abs(diff) * (1 + self.config.growth_bias),
                                   tf.abs(diff))
            return mse + tf.reduce_mean(bias_penalty) * 0.1
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss={'price_output': biased_mse, 'direction_output': 'binary_crossentropy'},
            loss_weights={'price_output': 1.0, 'direction_output': 0.5},
            metrics={'price_output': 'mae', 'direction_output': 'accuracy'}
        )
        
        self.model = model
        logger.info(f"Enhanced model built with {model.count_params():,} parameters")
        return model
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray], 
              val_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> tf.keras.callbacks.History:
        X_train, y_train_price, y_train_direction = train_data
        X_val, y_val_price, y_val_direction = val_data
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        
        if self.model is None:
            self.build_model(X_train.shape[1:])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_price_output_loss',
                mode='min',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-5
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_price_output_loss',
                mode='min',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                monitor='val_price_output_loss',
                mode='min',
                filepath='best_model.keras',
                save_best_only=True,
                verbose=0
            )
        ]
        
        logger.info("Starting enhanced training with biases...")
        
        self.history = self.model.fit(
            X_train, {'price_output': y_train_price, 'direction_output': y_train_direction},
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_val, {'price_output': y_val_price, 'direction_output': y_val_direction}),
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        logger.info("Training completed!")
        return self.history
    
    def predict(self, X: np.ndarray, features: pd.DataFrame, apply_bias: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        price_predictions_scaled, direction_predictions = self.model.predict(X_reshaped, verbose=0)
        
        price_predictions = self.scaler.inverse_transform(price_predictions_scaled)
        price_predictions = price_predictions.flatten()
        
        if apply_bias:
            price_predictions = self.apply_growth_bias(price_predictions, features)
        
        return price_predictions, direction_predictions
    
    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        X_test, y_test_price, y_test_direction = test_data
        
        price_predictions, direction_predictions = self.predict(X_test, self.data, apply_bias=True)
        
        y_test_actual = self.scaler.inverse_transform(y_test_price.reshape(-1, 1)).flatten()
        
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
        logger.info("ENHANCED MODEL EVALUATION RESULTS")
        logger.info("="*60)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("="*60)
        
        return metrics, price_predictions, y_test_actual
    
    def plot_results(self, predictions: np.ndarray, actual: np.ndarray, 
                    title: str = "Enhanced S&P 500 Predictions", n_points: int = 200):
        plt.figure(figsize=(15, 8))
        n_points = min(n_points, len(predictions))
        plt.plot(actual[-n_points:], label='Actual Price', color='blue', linewidth=2)
        plt.plot(predictions[-n_points:], label='Predicted (with bias)', 
                color='red', linewidth=2, alpha=0.8)
        x = np.arange(n_points)
        trend = actual[-n_points] * (1 + self.config.growth_bias) ** (x / 252)
        plt.plot(trend, '--', label='Growth Bias Trend', color='green', alpha=0.5)
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
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        ax1.plot(self.history.history['price_output_loss'], label='Training Price Loss')
        ax1.plot(self.history.history['val_price_output_loss'], label='Validation Price Loss')
        ax1.set_title('Price Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history.history['price_output_mae'], label='Training Price MAE')
        ax2.plot(self.history.history['val_price_output_mae'], label='Validation Price MAE')
        ax2.set_title('Price MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(self.history.history['direction_output_accuracy'], label='Training Direction Accuracy')
        ax3.plot(self.history.history['val_direction_output_accuracy'], label='Validation Direction Accuracy')
        ax3.set_title('Direction Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
        
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

def create_default_config(csv_files: List[str]) -> EnhancedModelConfig:
    return EnhancedModelConfig(
        csv_files=csv_files,
        sequence_length=20,
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
    csv_files = [
        'csv/sp500_analysis_20250830_094857.csv',
        'csv/sp500_quarterly_20250830_103017_prices.csv',
        'csv/sp500_quarterly_20250830_103017_summary.csv'
    ]
    
    if not all(os.path.exists(f) for f in csv_files):
        missing = [f for f in csv_files if not os.path.exists(f)]
        logger.error(f"Missing files: {missing}")
        raise FileNotFoundError(f"Missing files: {missing}")
    
    config = create_default_config(csv_files)
    config.sequence_length = 20
    config.growth_bias = 0.005
    config.top_performer_weight = 1.8
    config.low_rep_boost = 1.5
    
    config.save('enhanced_config.json')
    
    logger.info("=" * 70)
    logger.info("📊 ENHANCED S&P 500 PREDICTOR CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"📁 CSV Files: {config.csv_files}")
    logger.info(f"📈 Growth Bias: {config.growth_bias:.3%}")
    logger.info(f"⭐ Top Performer Weight: {config.top_performer_weight}x")
    logger.info(f"📉 Low Rep Boost: {config.low_rep_boost}x")
    logger.info(f"🔢 Sequence Length: {config.sequence_length} days")
    logger.info(f"🧠 LSTM Architecture: {config.lstm_units}")
    logger.info(f"🎯 Max Epochs: {config.epochs}")
    logger.info("=" * 70)
    
    predictor = EnhancedSP500Predictor(config)
    
    try:
        data = predictor.load_csv_data()
        features = predictor.prepare_enhanced_features(data)
        (X_train, y_train_price, y_train_direction), (X_val, y_val_price, y_val_direction), (X_test, y_test_price, y_test_direction) = predictor.prepare_data(features)
        predictor.train((X_train, y_train_price, y_train_direction), (X_val, y_val_price, y_val_direction))
        metrics, price_predictions, y_test_actual = predictor.evaluate((X_test, y_test_price, y_test_direction))
        predictor.plot_training_history()
        predictor.plot_results(price_predictions, y_test_actual)
        predictor.save_model('enhanced_sp500_model.keras')
        
        if predictor.company_weights:
            logger.info("\n" + "="*60)
            logger.info("FINAL COMPANY WEIGHTS")
            logger.info("="*60)
            for company, weight in sorted(predictor.company_weights.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"{company}: {weight:.2f}")
        else:
            logger.info("\n" + "="*60)
            logger.info("Using aggregate S&P 500 data - no individual company weights")
            logger.info(f"Momentum adjustment factor: {predictor.momentum_adjustment:.2f}")
            logger.info("="*60)
        
        return predictor, metrics
        
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        logger.info("\nPlease update the csv_files list with your actual file paths:")
        logger.info("  csv_files = [")
        logger.info("      '/path/to/your/first.csv',")
        logger.info("      '/path/to/your/second.csv',")
        logger.info("      '/path/to/your/third.csv'")
        logger.info("  ]")
        raise
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.info("\nExpected CSV file structure:")
        logger.info("1. Price data file: Should contain columns like Open, High, Low, Close, Volume")
        logger.info("2. Company info file: Should contain Symbol, Company_Name, Market_Cap, Sector, etc.")
        logger.info("3. Summary file: Can contain any additional metadata")
        logger.info("\nThe script will automatically identify and use the appropriate data.")
        raise

if __name__ == "__main__":
    """
    Enhanced S&P 500 LSTM Predictor with Custom Biases (v1.1.6)
    ====================================================
    
    This script implements your requirements:
    1. Growth bias: Assumes market will grow (default 0.5%)
    2. Top performer weighting: Gives more weight to strong momentum periods
    3. Low representation boost: After negative streaks, increases diversification
    
    The script automatically:
    - Identifies which CSV contains price data
    - Uses aggregate S&P 500 data or individual company data
    - Applies technical indicators and market regime detection
    - Implements biased loss function for optimistic predictions
    
    Based on your files:
    - sp500_analysis_*.csv: Company metadata (sectors, market cap)
    - sp500_quarterly_*_prices.csv: S&P 500 price data (OHLCV)
    - sp500_quarterly_*_summary.csv: Additional summary data
    
    Changes in v1.1.6:
    - Fixed KeyError: 'bull_market' by ensuring robust feature creation
    - Added fallback in apply_growth_bias for missing bull_market
    - Retained mode='min' and learning rate fixes from v1.1.5
    """
    predictor, metrics = main()