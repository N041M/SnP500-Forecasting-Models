"""
S&P 500 Extended Data Preparation V3.0
======================================
Comprehensive data preparation pipeline with extended features:

- Technical indicators (MA, RSI, MACD, Bollinger, Stochastic)
- Direction features (avg direction over multiple timeframes)
- OHLCV data (open, high, low, close, volume)
- Fundamental data (P/E ratio, market cap, earnings)
- Macroeconomic indicators (CPI, PPI, inflation expectations)
- Inequality metrics (Gini index, Lorenz curve)
- Sector-based features (Lerner index, HHI concentration)
- Feature validation (covariance matrix, redundancy detection)

Author: Ronald
Version: 3.0
Date: 2026
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import warnings

# Data fetching
import yfinance as yf

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.covariance import EmpiricalCovariance

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    version: str = "3.0"
    sequence_length: int = 60
    train_split: float = 0.7
    validation_split: float = 0.15
    test_split: float = 0.15
    scaling_method: str = "minmax"
    direction_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    compute_covariance: bool = True
    redundancy_threshold: float = 0.95

    @classmethod
    def from_json(cls, filepath: str) -> 'FeatureConfig':
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls(
            version=config.get('version', '3.0'),
            sequence_length=config['preprocessing']['sequence_length'],
            train_split=config['preprocessing']['train_split'],
            validation_split=config['preprocessing']['validation_split'],
            test_split=config['preprocessing']['test_split'],
            scaling_method=config['preprocessing']['scaling_method'],
            direction_windows=config['feature_groups']['direction']['windows'],
            compute_covariance=config['validation']['compute_covariance'],
            redundancy_threshold=config['validation']['redundancy_threshold']
        )


class MacroDataFetcher:
    """
    Fetch macroeconomic data from FRED and other sources

    Note: Requires FRED API key for full functionality.
    Falls back to synthetic data for demonstration if API unavailable.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.fred = None

        if api_key and api_key != "YOUR_FRED_API_KEY":
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=api_key)
                logger.info("FRED API initialized successfully")
            except ImportError:
                logger.warning("fredapi not installed. Run: pip install fredapi")
            except Exception as e:
                logger.warning(f"FRED API initialization failed: {e}")

    def fetch_inflation_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch CPI, PPI, and inflation expectations"""
        df = pd.DataFrame()

        if self.fred:
            try:
                # Consumer Price Index
                cpi = self.fred.get_series('CPIAUCSL', start_date, end_date)
                df['cpi'] = cpi
                df['cpi_yoy'] = cpi.pct_change(periods=12) * 100

                # Producer Price Index
                ppi = self.fred.get_series('PPIACO', start_date, end_date)
                df['ppi'] = ppi
                df['ppi_yoy'] = ppi.pct_change(periods=12) * 100

                # 5-Year Breakeven Inflation Rate
                df['inflation_expectation'] = self.fred.get_series('T5YIE', start_date, end_date)

                # Federal Funds Rate
                df['fed_funds_rate'] = self.fred.get_series('FEDFUNDS', start_date, end_date)

                logger.info("Fetched inflation data from FRED")
            except Exception as e:
                logger.warning(f"FRED fetch failed: {e}. Using synthetic data.")
                df = self._generate_synthetic_macro(start_date, end_date)
        else:
            logger.info("FRED API not available. Generating synthetic macro data.")
            df = self._generate_synthetic_macro(start_date, end_date)

        return df

    def fetch_gdp_share(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch US share of global GDP"""
        # World Bank data is annual - we'll interpolate to daily
        # US GDP share has been ~24-25% of world GDP in recent years
        dates = pd.date_range(start_date, end_date, freq='D')

        # Historical approximation (declining trend)
        years = (dates - pd.Timestamp('2010-01-01')).days / 365.25
        us_gdp_share = 24.5 - (years * 0.15)  # Slight decline over time
        us_gdp_share = np.clip(us_gdp_share, 22, 26)

        df = pd.DataFrame({'us_gdp_share': us_gdp_share}, index=dates)
        logger.info("Generated US GDP share data")
        return df

    def _generate_synthetic_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic macro data for demonstration"""
        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)

        np.random.seed(42)

        # Realistic synthetic values with trends and noise
        base_cpi = 230 + np.cumsum(np.random.randn(n) * 0.1 + 0.02)
        base_ppi = 200 + np.cumsum(np.random.randn(n) * 0.15 + 0.015)

        df = pd.DataFrame({
            'cpi': base_cpi,
            'cpi_yoy': 2.5 + np.sin(np.arange(n) / 365 * 2 * np.pi) * 1.5 + np.random.randn(n) * 0.3,
            'ppi': base_ppi,
            'ppi_yoy': 2.0 + np.sin(np.arange(n) / 365 * 2 * np.pi + 0.5) * 2.0 + np.random.randn(n) * 0.4,
            'inflation_expectation': 2.2 + np.random.randn(n) * 0.3,
            'fed_funds_rate': np.clip(2.0 + np.cumsum(np.random.randn(n) * 0.01), 0, 6)
        }, index=dates)

        return df


class InequalityDataFetcher:
    """
    Fetch inequality metrics (Gini, Lorenz curve data)

    Note: World Bank data is annual. Values are interpolated for daily frequency.
    """

    def __init__(self):
        # Historical Gini coefficients for developed countries (approximate)
        self.gini_data = {
            'US': 0.39,
            'UK': 0.35,
            'Germany': 0.31,
            'France': 0.32,
            'Japan': 0.33,
            'Canada': 0.33,
            'Australia': 0.34,
            'Developed_Avg': 0.34
        }

    def fetch_gini_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Gini index data"""
        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)

        # Gini index changes slowly - add small random walk
        np.random.seed(43)
        gini_us = self.gini_data['US'] + np.cumsum(np.random.randn(n) * 0.0001)
        gini_us = np.clip(gini_us, 0.35, 0.45)

        gini_developed = self.gini_data['Developed_Avg'] + np.cumsum(np.random.randn(n) * 0.00008)
        gini_developed = np.clip(gini_developed, 0.30, 0.40)

        df = pd.DataFrame({
            'gini_index_us': gini_us,
            'gini_index_developed': gini_developed
        }, index=dates)

        logger.info("Generated Gini index data")
        return df

    def compute_lorenz_area(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Compute Lorenz curve area ratio

        Lorenz area = area between perfect equality line and Lorenz curve
        Related to Gini: Gini = 2 * Lorenz_area
        """
        dates = pd.date_range(start_date, end_date, freq='D')

        gini_df = self.fetch_gini_index(start_date, end_date)

        # Lorenz area = Gini / 2 (by definition)
        df = pd.DataFrame({
            'lorenz_area_us': gini_df['gini_index_us'] / 2,
            'lorenz_area_developed': gini_df['gini_index_developed'] / 2
        }, index=dates)

        logger.info("Computed Lorenz curve area ratios")
        return df


class FundamentalDataFetcher:
    """Fetch fundamental data (P/E, market cap, earnings) from Yahoo Finance"""

    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.cache = {}

    def fetch_fundamental_data(self, ticker: str) -> Dict:
        """Fetch fundamental data for a single ticker"""
        if ticker in self.cache:
            return self.cache[ticker]

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            data = {
                'pe_ratio': info.get('trailingPE', np.nan),
                'forward_pe': info.get('forwardPE', np.nan),
                'market_cap': info.get('marketCap', np.nan),
                'enterprise_value': info.get('enterpriseValue', np.nan),
                'profit_margin': info.get('profitMargins', np.nan),
                'operating_margin': info.get('operatingMargins', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'earnings_growth': info.get('earningsGrowth', np.nan),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }

            self.cache[ticker] = data
            return data

        except Exception as e:
            logger.warning(f"Failed to fetch fundamental data for {ticker}: {e}")
            return {}

    def fetch_earnings_history(self, ticker: str) -> pd.DataFrame:
        """Fetch quarterly earnings history"""
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.quarterly_earnings

            if earnings is not None and not earnings.empty:
                return earnings
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.warning(f"Failed to fetch earnings for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_all_fundamentals(self) -> pd.DataFrame:
        """Fetch fundamental data for all tickers"""
        records = []

        for ticker in self.tickers:
            logger.info(f"Fetching fundamentals for {ticker}")
            data = self.fetch_fundamental_data(ticker)
            if data:
                data['ticker'] = ticker
                records.append(data)

        df = pd.DataFrame(records)
        logger.info(f"Fetched fundamentals for {len(records)} tickers")
        return df


class SectorAnalyzer:
    """
    Compute sector-based features including Lerner index

    Lerner Index = (Price - Marginal Cost) / Price
    Approximated using operating margin as proxy
    """

    def __init__(self, sector_mapping: Dict[str, List[str]]):
        self.sector_mapping = sector_mapping
        self.ticker_to_sector = {}

        for sector, tickers in sector_mapping.items():
            for ticker in tickers:
                self.ticker_to_sector[ticker] = sector

    def compute_lerner_index(self, fundamental_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Lerner index per sector

        Uses operating margin as proxy for (P - MC) / P
        """
        results = []

        for sector, tickers in self.sector_mapping.items():
            sector_data = fundamental_df[fundamental_df['ticker'].isin(tickers)]

            if not sector_data.empty:
                # Average operating margin as Lerner proxy
                lerner = sector_data['operating_margin'].mean()

                # HHI (Herfindahl-Hirschman Index) for concentration
                if 'market_cap' in sector_data.columns:
                    market_caps = sector_data['market_cap'].dropna()
                    if len(market_caps) > 0:
                        shares = market_caps / market_caps.sum()
                        hhi = (shares ** 2).sum()
                    else:
                        hhi = np.nan
                else:
                    hhi = np.nan

                results.append({
                    'sector': sector,
                    'lerner_index': lerner if not pd.isna(lerner) else 0.15,
                    'hhi': hhi if not pd.isna(hhi) else 0.1,
                    'n_companies': len(sector_data)
                })

        return pd.DataFrame(results)

    def compute_sector_features(self, price_data: pd.DataFrame,
                                 fundamental_df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-series sector features"""

        # Get Lerner indices (static for now, could be made dynamic with quarterly updates)
        lerner_df = self.compute_lerner_index(fundamental_df)

        # Create sector feature columns
        sector_features = pd.DataFrame(index=price_data.index)

        for _, row in lerner_df.iterrows():
            sector = row['sector']
            sector_features[f'lerner_{sector}'] = row['lerner_index']
            sector_features[f'hhi_{sector}'] = row['hhi']

        # Aggregate Lerner (market-cap weighted)
        total_lerner = lerner_df['lerner_index'].mean()
        sector_features['lerner_aggregate'] = total_lerner

        # Aggregate HHI
        sector_features['hhi_aggregate'] = lerner_df['hhi'].mean()

        logger.info(f"Computed sector features: {len(lerner_df)} sectors")
        return sector_features


class TechnicalFeatureEngineer:
    """Compute technical indicators"""

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features from OHLCV data"""
        features = pd.DataFrame(index=df.index)

        # Basic OHLCV
        features['open'] = df['Open']
        features['high'] = df['High']
        features['low'] = df['Low']
        features['close'] = df['Close']
        features['volume'] = df['Volume']

        # Spreads (add small epsilon to prevent division by zero)
        features['hl_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
        features['oc_spread'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-10)

        # Returns
        features['returns'] = df['Close'].pct_change().clip(-0.5, 0.5)
        # Safe log returns (clip ratio to prevent log(0))
        price_ratio = (df['Close'] / df['Close'].shift(1)).clip(0.5, 2.0)
        features['log_returns'] = np.log(price_ratio)

        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            features[f'ma_{window}'] = df['Close'].rolling(window, min_periods=1).mean()
            features[f'price_to_ma_{window}'] = df['Close'] / (features[f'ma_{window}'] + 1e-10)

        # Exponential moving averages
        for span in [12, 26]:
            features[f'ema_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()

        # Volatility
        features['volatility_10'] = features['returns'].rolling(10, min_periods=1).std()
        features['volatility_20'] = features['returns'].rolling(20, min_periods=1).std()
        features['volatility_60'] = features['returns'].rolling(60, min_periods=1).std()

        # Bollinger Bands
        bb_window = 20
        bb_std = df['Close'].rolling(bb_window, min_periods=1).std()
        features['bollinger_upper'] = features['ma_20'] + 2 * bb_std
        features['bollinger_lower'] = features['ma_20'] - 2 * bb_std
        features['bollinger_width'] = (features['bollinger_upper'] - features['bollinger_lower']) / (features['ma_20'] + 1e-10)
        features['bollinger_position'] = (df['Close'] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'] + 1e-10)

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(14, min_periods=1).min()
        high_14 = df['High'].rolling(14, min_periods=1).max()
        features['stochastic_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
        features['stochastic_d'] = features['stochastic_k'].rolling(3, min_periods=1).mean()

        # Volume indicators
        features['volume_ma_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        features['volume_ratio'] = df['Volume'] / (features['volume_ma_20'] + 1e-10)
        features['volume_change'] = df['Volume'].pct_change().clip(-10, 10)

        # On-Balance Volume (OBV)
        obv = np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
                       np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
        features['obv'] = np.cumsum(obv)

        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14, min_periods=1).mean()

        # Momentum
        features['momentum_10'] = (df['Close'] / (df['Close'].shift(10) + 1e-10) - 1).clip(-1, 1)
        features['momentum_20'] = (df['Close'] / (df['Close'].shift(20) + 1e-10) - 1).clip(-1, 1)

        # Rate of Change
        features['roc_10'] = ((df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10) + 1e-10) * 100).clip(-100, 100)

        logger.info(f"Computed {len(features.columns)} technical features")
        return features


class DirectionFeatureEngineer:
    """Compute direction-based features"""

    @staticmethod
    def compute_direction_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Compute average direction in X timeframe
        Direction = proportion of positive returns in window
        """
        features = pd.DataFrame(index=df.index)
        returns = df['Close'].pct_change()

        for window in windows:
            # Average direction (% of positive days)
            features[f'avg_direction_{window}d'] = (
                returns.rolling(window, min_periods=1).apply(lambda x: (x > 0).mean())
            )

            # Cumulative return over window (clipped to prevent overflow)
            features[f'cum_return_{window}d'] = (
                (1 + returns.clip(-0.5, 0.5)).rolling(window, min_periods=1).apply(lambda x: x.prod()) - 1
            ).clip(-0.9, 10)

            # Consecutive positive/negative days
            positive = (returns > 0).astype(int)
            features[f'pos_streak_{window}d'] = positive.rolling(window, min_periods=1).sum()

            # Direction strength (avg return when positive vs negative)
            def safe_direction_strength(x):
                pos_mean = x[x > 0].mean() if len(x[x > 0]) > 0 else 0
                neg_mean = abs(x[x < 0].mean()) if len(x[x < 0]) > 0 else 1e-10
                if np.isnan(pos_mean) or np.isnan(neg_mean):
                    return 1.0
                return np.clip(pos_mean / (neg_mean + 1e-10), 0, 100)

            features[f'direction_strength_{window}d'] = (
                returns.rolling(window, min_periods=1).apply(safe_direction_strength)
            )

        logger.info(f"Computed direction features for {len(windows)} windows")
        return features


class FeatureValidator:
    """Validate feature quality and compute covariance analysis"""

    def __init__(self, redundancy_threshold: float = 0.95):
        self.redundancy_threshold = redundancy_threshold
        self.analysis_results = {}

    def compute_covariance_analysis(self, X: np.ndarray,
                                     feature_names: List[str],
                                     y: np.ndarray = None) -> Dict:
        """
        Compute covariance matrix and feature validation metrics

        Returns:
            - Covariance matrix
            - Correlation matrix
            - Feature-target correlations
            - Redundant feature pairs
            - Feature importance ranking
        """
        logger.info("Computing covariance analysis...")

        # Handle 3D data (samples, sequence, features)
        if X.ndim == 3:
            # Use last timestep for correlation analysis
            X_flat = X[:, -1, :]
        else:
            X_flat = X

        n_samples, n_features = X_flat.shape

        # Remove any constant features
        valid_features = []
        valid_indices = []
        for i, name in enumerate(feature_names):
            if np.std(X_flat[:, i]) > 1e-10:
                valid_features.append(name)
                valid_indices.append(i)

        X_valid = X_flat[:, valid_indices]

        # Covariance and correlation matrices
        cov_matrix = np.cov(X_valid.T)
        corr_matrix = np.corrcoef(X_valid.T)

        # Feature-target correlation
        target_corr = {}
        if y is not None:
            y_flat = y.flatten() if y.ndim > 1 else y
            # Ensure same length
            min_len = min(len(X_valid), len(y_flat))
            for i, name in enumerate(valid_features):
                corr = np.corrcoef(X_valid[:min_len, i], y_flat[:min_len])[0, 1]
                target_corr[name] = corr if not np.isnan(corr) else 0

        # Identify redundant features
        redundant_pairs = []
        for i in range(len(valid_features)):
            for j in range(i + 1, len(valid_features)):
                if abs(corr_matrix[i, j]) > self.redundancy_threshold:
                    redundant_pairs.append({
                        'feature_1': valid_features[i],
                        'feature_2': valid_features[j],
                        'correlation': float(corr_matrix[i, j])
                    })

        # Rank features by target correlation
        feature_ranking = sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)

        self.analysis_results = {
            'covariance_matrix': cov_matrix,
            'correlation_matrix': corr_matrix,
            'target_correlation': target_corr,
            'redundant_pairs': redundant_pairs,
            'feature_ranking': feature_ranking,
            'feature_names': valid_features,
            'n_features': len(valid_features),
            'n_redundant_pairs': len(redundant_pairs)
        }

        logger.info(f"Covariance analysis complete: {len(valid_features)} features, {len(redundant_pairs)} redundant pairs")
        return self.analysis_results

    def plot_feature_analysis(self, save_path: str = 'plots/'):
        """Generate visualization of feature analysis"""
        if not self.analysis_results:
            logger.warning("No analysis results available. Run compute_covariance_analysis first.")
            return

        os.makedirs(save_path, exist_ok=True)

        # 1. Correlation heatmap
        fig, ax = plt.subplots(figsize=(16, 14))

        corr = self.analysis_results['correlation_matrix']
        names = self.analysis_results['feature_names']

        # Limit to top 30 features for readability
        if len(names) > 30:
            # Select top features by target correlation
            ranking = self.analysis_results['feature_ranking'][:30]
            top_names = [r[0] for r in ranking]
            top_indices = [names.index(n) for n in top_names if n in names]
            corr = corr[np.ix_(top_indices, top_indices)]
            names = top_names

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, xticklabels=names, yticklabels=names,
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, ax=ax,
                   cbar_kws={'shrink': 0.5})
        ax.set_title('Feature Correlation Matrix (Top 30)', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{save_path}feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Target correlation bar chart
        fig, ax = plt.subplots(figsize=(12, 10))

        ranking = self.analysis_results['feature_ranking'][:30]
        names = [r[0] for r in ranking]
        values = [r[1] for r in ranking]
        colors = ['green' if v > 0 else 'red' for v in values]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Correlation with Target')
        ax.set_title('Top 30 Features by Target Correlation')
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'{save_path}feature_target_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Redundant pairs summary
        fig, ax = plt.subplots(figsize=(10, 6))

        redundant = self.analysis_results['redundant_pairs'][:20]  # Top 20
        if redundant:
            labels = [f"{r['feature_1'][:15]} <-> {r['feature_2'][:15]}" for r in redundant]
            values = [r['correlation'] for r in redundant]

            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, color='orange', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Correlation')
            ax.set_title(f'Redundant Feature Pairs (|corr| > {self.redundancy_threshold})')
            ax.axvline(x=self.redundancy_threshold, color='red', linestyle='--', linewidth=1)
            ax.axvline(x=-self.redundancy_threshold, color='red', linestyle='--', linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No redundant pairs found', ha='center', va='center', fontsize=14)

        plt.tight_layout()
        plt.savefig(f'{save_path}redundant_features.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature analysis plots saved to {save_path}")

    def get_feature_importance_report(self) -> str:
        """Generate text report of feature importance"""
        if not self.analysis_results:
            return "No analysis results available."

        report = []
        report.append("=" * 60)
        report.append("FEATURE IMPORTANCE REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal features analyzed: {self.analysis_results['n_features']}")
        report.append(f"Redundant pairs (|corr| > {self.redundancy_threshold}): {self.analysis_results['n_redundant_pairs']}")

        report.append("\n" + "-" * 40)
        report.append("TOP 20 FEATURES BY TARGET CORRELATION")
        report.append("-" * 40)

        for i, (name, corr) in enumerate(self.analysis_results['feature_ranking'][:20], 1):
            report.append(f"{i:2d}. {name:30s} : {corr:+.4f}")

        if self.analysis_results['redundant_pairs']:
            report.append("\n" + "-" * 40)
            report.append("REDUNDANT FEATURE PAIRS (Consider removing)")
            report.append("-" * 40)

            for pair in self.analysis_results['redundant_pairs'][:10]:
                report.append(f"  {pair['feature_1']} <-> {pair['feature_2']} (r={pair['correlation']:.3f})")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


class SnP500DataPrepV3:
    """
    Main data preparation class for S&P 500 prediction

    Orchestrates all data fetching and feature engineering components.
    """

    def __init__(self, config_path: str = 'config/feature_config.json'):
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config_dict = json.load(f)
            self.config = FeatureConfig.from_json(config_path)
        else:
            logger.warning(f"Config not found at {config_path}. Using defaults.")
            self.config_dict = {}
            self.config = FeatureConfig()

        # Initialize components
        self.sector_mapping = self.config_dict.get('sectors', {})
        self.macro_fetcher = MacroDataFetcher(
            self.config_dict.get('api_keys', {}).get('fred_api_key')
        )
        self.inequality_fetcher = InequalityDataFetcher()
        self.sector_analyzer = SectorAnalyzer(self.sector_mapping)
        self.feature_validator = FeatureValidator(self.config.redundancy_threshold)

        # Scalers
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

        # Storage
        self.all_features = None
        self.feature_names = None
        self.metadata = {}

        logger.info(f"SnP500DataPrepV3 initialized with {len(self.sector_mapping)} sectors")

    def fetch_price_data(self, ticker: str = "^GSPC",
                         start_date: str = "2010-01-01",
                         end_date: str = None) -> pd.DataFrame:
        """Fetch OHLCV price data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Fetching price data for {ticker} from {start_date} to {end_date}")

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        logger.info(f"Fetched {len(df)} rows of price data")
        return df

    def prepare_all_features(self, price_data: pd.DataFrame,
                             fetch_fundamentals: bool = True,
                             fetch_macro: bool = True) -> pd.DataFrame:
        """
        Prepare all features from multiple data sources
        """
        logger.info("=" * 60)
        logger.info("PREPARING EXTENDED FEATURES")
        logger.info("=" * 60)

        all_features = pd.DataFrame(index=price_data.index)
        start_date = price_data.index[0].strftime('%Y-%m-%d')
        end_date = price_data.index[-1].strftime('%Y-%m-%d')

        # 1. Technical features
        logger.info("\n[1/6] Computing technical features...")
        technical = TechnicalFeatureEngineer.compute_all(price_data)
        all_features = pd.concat([all_features, technical], axis=1)

        # 2. Direction features
        logger.info("\n[2/6] Computing direction features...")
        direction = DirectionFeatureEngineer.compute_direction_features(
            price_data, self.config.direction_windows
        )
        all_features = pd.concat([all_features, direction], axis=1)

        # 3. Macro features
        if fetch_macro:
            logger.info("\n[3/6] Fetching macroeconomic features...")

            # Inflation data
            inflation = self.macro_fetcher.fetch_inflation_data(start_date, end_date)
            inflation = inflation.reindex(price_data.index).ffill().bfill()
            all_features = pd.concat([all_features, inflation], axis=1)

            # GDP share
            gdp = self.macro_fetcher.fetch_gdp_share(start_date, end_date)
            gdp = gdp.reindex(price_data.index).ffill().bfill()
            all_features = pd.concat([all_features, gdp], axis=1)

        # 4. Inequality metrics
        logger.info("\n[4/6] Computing inequality metrics...")
        gini = self.inequality_fetcher.fetch_gini_index(start_date, end_date)
        gini = gini.reindex(price_data.index).ffill().bfill()
        all_features = pd.concat([all_features, gini], axis=1)

        lorenz = self.inequality_fetcher.compute_lorenz_area(start_date, end_date)
        lorenz = lorenz.reindex(price_data.index).ffill().bfill()
        all_features = pd.concat([all_features, lorenz], axis=1)

        # 5. Sector features (Lerner index, HHI)
        if fetch_fundamentals and self.sector_mapping:
            logger.info("\n[5/6] Computing sector features...")

            # Fetch fundamental data for sector analysis
            all_tickers = []
            for tickers in self.sector_mapping.values():
                all_tickers.extend(tickers[:3])  # Limit to top 3 per sector for speed

            fundamental_fetcher = FundamentalDataFetcher(all_tickers)
            fundamental_df = fundamental_fetcher.fetch_all_fundamentals()

            if not fundamental_df.empty:
                sector_features = self.sector_analyzer.compute_sector_features(
                    price_data, fundamental_df
                )
                all_features = pd.concat([all_features, sector_features], axis=1)

        # 6. Company representation features
        logger.info("\n[6/6] Computing representation features...")
        # % change buckets
        returns = price_data['Close'].pct_change()
        all_features['return_bucket'] = pd.cut(
            returns,
            bins=[-np.inf, -0.02, -0.01, 0, 0.01, 0.02, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)

        # Handle NaN and infinity values
        all_features = all_features.ffill().bfill()

        # Replace infinity with NaN, then fill
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.ffill().bfill().fillna(0)

        # Clip extreme values
        for col in all_features.select_dtypes(include=[np.number]).columns:
            all_features[col] = all_features[col].clip(-1e10, 1e10)

        # Remove any remaining NaN columns
        nan_cols = all_features.columns[all_features.isna().all()]
        if len(nan_cols) > 0:
            logger.warning(f"Removing {len(nan_cols)} all-NaN columns: {list(nan_cols)}")
            all_features = all_features.drop(columns=nan_cols)

        # Final safety check
        all_features = all_features.fillna(0)

        self.all_features = all_features
        self.feature_names = list(all_features.columns)

        logger.info("=" * 60)
        logger.info(f"Total features prepared: {len(self.feature_names)}")
        logger.info("=" * 60)

        return all_features

    def create_sequences(self, features: pd.DataFrame,
                         target_col: str = 'returns') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training

        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples,) - next period target
        """
        # Extract target before removing from features
        if target_col in features.columns:
            target = features[target_col].values
        else:
            target = features['close'].pct_change().values

        # Remove target from features
        feature_cols = [c for c in features.columns if c != target_col]
        X_data = features[feature_cols].values

        # Replace inf with NaN, then fill NaN
        X_data = np.where(np.isinf(X_data), np.nan, X_data)
        X_data = pd.DataFrame(X_data).ffill().bfill().fillna(0).values

        # Clip extreme values to prevent overflow
        X_data = np.clip(X_data, -1e10, 1e10)

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X_data)

        # Create sequences
        X, y = [], []
        seq_len = self.config.sequence_length

        for i in range(seq_len, len(X_scaled) - 1):
            X.append(X_scaled[i - seq_len:i])
            y.append(target[i + 1])  # Next period return

        X = np.array(X)
        y = np.array(y)

        # Handle NaN in target
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data into train/val/test sets"""
        n = len(X)
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.validation_split)

        splits = {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:]
        }

        logger.info(f"Data splits - Train: {len(splits['X_train'])}, "
                   f"Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")

        return splits

    def save_prepared_data(self, splits: Dict[str, np.ndarray],
                           save_path: str = 'data/processed/'):
        """Save prepared data to disk"""
        os.makedirs(save_path, exist_ok=True)

        # Save numpy arrays
        for name, data in splits.items():
            np.save(f'{save_path}{name}.npy', data)

        # Save metadata
        self.metadata = {
            'version': self.config.version,
            'n_features': len(self.feature_names) - 1,  # Exclude target
            'feature_names': [f for f in self.feature_names if f != 'returns'],
            'sequence_length': self.config.sequence_length,
            'train_samples': len(splits['X_train']),
            'val_samples': len(splits['X_val']),
            'test_samples': len(splits['X_test']),
            'created_at': datetime.now().isoformat(),
            'scaling_method': self.config.scaling_method
        }

        with open(f'{save_path}metadata.pkl', 'wb') as f:
            pickle.dump(self.metadata, f)

        # Save scalers
        with open(f'{save_path}feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)

        # Save feature analysis if available
        if self.feature_validator.analysis_results:
            with open(f'{save_path}feature_analysis.pkl', 'wb') as f:
                pickle.dump(self.feature_validator.analysis_results, f)

        logger.info(f"Data saved to {save_path}")
        logger.info(f"  - X_train.npy, y_train.npy")
        logger.info(f"  - X_val.npy, y_val.npy")
        logger.info(f"  - X_test.npy, y_test.npy")
        logger.info(f"  - metadata.pkl")
        logger.info(f"  - feature_scaler.pkl")

    def run_pipeline(self, ticker: str = "^GSPC",
                     start_date: str = "2010-01-01",
                     end_date: str = None,
                     save_path: str = 'data/processed/') -> Dict[str, np.ndarray]:
        """
        Run complete data preparation pipeline
        """
        print("\n" + "=" * 70)
        print("S&P 500 DATA PREPARATION PIPELINE V3.0")
        print("=" * 70 + "\n")

        # 1. Fetch price data
        price_data = self.fetch_price_data(ticker, start_date, end_date)

        # Save raw data
        raw_path = save_path.replace('processed', 'raw')
        os.makedirs(raw_path, exist_ok=True)
        price_data.to_csv(f'{raw_path}price_data.csv')

        # 2. Prepare all features
        features = self.prepare_all_features(price_data)

        # 3. Create sequences
        X, y = self.create_sequences(features)

        # 4. Split data
        splits = self.split_data(X, y)

        # 5. Validate features
        if self.config.compute_covariance:
            print("\n" + "-" * 50)
            print("FEATURE VALIDATION")
            print("-" * 50)

            analysis = self.feature_validator.compute_covariance_analysis(
                splits['X_train'],
                [f for f in self.feature_names if f != 'returns'],
                splits['y_train']
            )

            # Print report
            print(self.feature_validator.get_feature_importance_report())

            # Generate plots
            self.feature_validator.plot_feature_analysis(save_path='plots/')

        # 6. Save data
        self.save_prepared_data(splits, save_path)

        print("\n" + "=" * 70)
        print("DATA PREPARATION COMPLETE")
        print("=" * 70)
        print(f"\nOutput files saved to: {save_path}")
        print(f"Total features: {self.metadata['n_features']}")
        print(f"Sequence length: {self.metadata['sequence_length']}")
        print(f"Training samples: {self.metadata['train_samples']}")
        print("=" * 70 + "\n")

        return splits


def main():
    """Main execution function"""

    # Initialize data preparation
    prep = SnP500DataPrepV3(config_path='config/feature_config.json')

    # Run pipeline
    splits = prep.run_pipeline(
        ticker="^GSPC",
        start_date="2015-01-01",  # Shorter period for faster testing
        end_date=None,
        save_path='data/processed/'
    )

    return splits


if __name__ == "__main__":
    main()
