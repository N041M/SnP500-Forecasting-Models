import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SP500DataPreprocessor:
    """Preprocessor for S&P 500 individual stock data to create either:
    1. A market-cap weighted index
    2. Clean individual stock data
    3. Sector indices
    """
    
    def __init__(self, 
                 analysis_file: str,
                 prices_file: str, 
                 summary_file: str):
        self.analysis_file = analysis_file
        self.prices_file = prices_file
        self.summary_file = summary_file
        
        self.analysis_df = None
        self.prices_df = None
        self.summary_df = None
        self.sp500_index = None
        self.clean_stocks = {}
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three CSV files and perform initial inspection."""
        
        logger.info("="*70)
        logger.info("LOADING S&P 500 DATA FILES")
        logger.info("="*70)
        
        # Load analysis file (company info with market caps)
        self.analysis_df = pd.read_csv(self.analysis_file)
        logger.info(f"Analysis file: {self.analysis_df.shape[0]} companies")
        logger.info(f"Columns: {', '.join(self.analysis_df.columns[:10])}")
        
        # Load prices file (individual stock prices)
        self.prices_df = pd.read_csv(self.prices_file)
        self.prices_df['Date'] = pd.to_datetime(self.prices_df['Date'])
        logger.info(f"\nPrices file: {self.prices_df.shape[0]} rows")
        logger.info(f"Date range: {self.prices_df['Date'].min()} to {self.prices_df['Date'].max()}")
        
        # Check unique symbols
        unique_symbols = self.prices_df['Symbol'].nunique()
        logger.info(f"Unique symbols in prices: {unique_symbols}")
        
        # Load summary file
        self.summary_df = pd.read_csv(self.summary_file)
        logger.info(f"\nSummary file: {self.summary_df.shape[0]} companies")
        
        return self.analysis_df, self.prices_df, self.summary_df
    
    def diagnose_price_data(self):
        """Diagnose issues in the price data."""
        
        logger.info("\n" + "="*70)
        logger.info("PRICE DATA DIAGNOSIS")
        logger.info("="*70)
        
        # Check price ranges by symbol
        price_stats = self.prices_df.groupby('Symbol')['Close'].agg([
            'count', 'min', 'max', 'mean', 'std'
        ]).round(2)
        
        # Find outliers
        outlier_stocks = price_stats[price_stats['max'] > 1000].sort_values('max', ascending=False)
        
        if len(outlier_stocks) > 0:
            logger.info("\nStocks with prices > $1000:")
            for symbol in outlier_stocks.head(10).index:
                max_price = outlier_stocks.loc[symbol, 'max']
                company_name = self.analysis_df[
                    self.analysis_df['Symbol'] == symbol
                ]['Company_Name'].values
                company_name = company_name[0] if len(company_name) > 0 else "Unknown"
                logger.info(f"  {symbol}: ${max_price:.2f} ({company_name})")
        
        # Check for missing data
        missing_by_symbol = self.prices_df.groupby('Symbol').apply(
            lambda x: x[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum().sum()
        )
        
        if missing_by_symbol.sum() > 0:
            logger.warning(f"\nTotal missing values: {missing_by_symbol.sum()}")
            logger.warning(f"Symbols with missing data: {(missing_by_symbol > 0).sum()}")
        
        # Check for suspicious values
        zero_prices = (self.prices_df['Close'] <= 0).sum()
        if zero_prices > 0:
            logger.error(f"Found {zero_prices} rows with Close <= 0")
        
        return price_stats
    
    def create_market_cap_weighted_index(self) -> pd.DataFrame:
        """Create a market-cap weighted S&P 500 index from individual stocks."""
        
        logger.info("\n" + "="*70)
        logger.info("CREATING MARKET-CAP WEIGHTED S&P 500 INDEX")
        logger.info("="*70)
        
        # Merge price data with market cap data
        prices_with_cap = pd.merge(
            self.prices_df,
            self.analysis_df[['Symbol', 'Market_Cap_Billions']],
            on='Symbol',
            how='left'
        )
        
        # Remove rows with missing market caps
        missing_caps = prices_with_cap['Market_Cap_Billions'].isna().sum()
        if missing_caps > 0:
            logger.warning(f"Removing {missing_caps} rows with missing market caps")
            prices_with_cap = prices_with_cap.dropna(subset=['Market_Cap_Billions'])
        
        # Group by date and calculate weighted average
        logger.info("Calculating weighted averages by date...")
        
        def weighted_avg(group):
            weights = group['Market_Cap_Billions']
            total_weight = weights.sum()
            
            if total_weight == 0:
                return pd.Series({
                    'Open': np.nan,
                    'High': np.nan,
                    'Low': np.nan,
                    'Close': np.nan,
                    'Volume': 0
                })
            
            return pd.Series({
                'Open': (group['Open'] * weights).sum() / total_weight,
                'High': (group['High'] * weights).sum() / total_weight,
                'Low': (group['Low'] * weights).sum() / total_weight,
                'Close': (group['Close'] * weights).sum() / total_weight,
                'Volume': group['Volume'].sum(),
                'Market_Cap_Total': total_weight,
                'Stocks_Included': len(group)
            })
        
        sp500_index = prices_with_cap.groupby('Date').apply(weighted_avg).reset_index()
        
        # Drop any remaining NaN rows
        sp500_index = sp500_index.dropna(subset=['Close'])
        
        # Sort by date
        sp500_index = sp500_index.sort_values('Date')
        sp500_index = sp500_index.set_index('Date')
        
        # Scale to realistic S&P 500 levels (multiply by adjustment factor)
        # The actual S&P 500 is around 4000-5000, so we'll scale appropriately
        current_avg = sp500_index['Close'].mean()
        target_avg = 4000  # Approximate S&P 500 level
        scale_factor = target_avg / current_avg if current_avg > 0 else 1
        
        for col in ['Open', 'High', 'Low', 'Close']:
            sp500_index[col] *= scale_factor
        
        # Determine if data is quarterly or daily
        date_diffs = pd.Series(sp500_index.index).diff().dt.days.dropna()
        avg_days = date_diffs.mean()
        
        if avg_days > 60:  # Quarterly data
            freq_type = "quarterly"
            logger.info(f"\nCreated S&P 500 index with {len(sp500_index)} quarterly records")
        else:  # Daily data
            freq_type = "daily"
            logger.info(f"\nCreated S&P 500 index with {len(sp500_index)} daily records")
        
        logger.info(f"Date range: {sp500_index.index[0]} to {sp500_index.index[-1]}")
        logger.info(f"Price range: ${sp500_index['Close'].min():.2f} to ${sp500_index['Close'].max():.2f}")
        logger.info(f"Average stocks included per period: {sp500_index['Stocks_Included'].mean():.0f}")
        
        self.sp500_index = sp500_index
        return sp500_index
    
    def clean_individual_stock(self, symbol: str) -> Optional[pd.DataFrame]:
        """Clean and prepare individual stock data."""
        
        stock_data = self.prices_df[self.prices_df['Symbol'] == symbol].copy()
        
        if len(stock_data) == 0:
            logger.warning(f"No data found for symbol {symbol}")
            return None
        
        # Handle timezone-aware dates
        if hasattr(stock_data['Date'].iloc[0], 'tz'):
            # If already datetime with timezone, convert to UTC then remove tz
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.tz_localize(None)
        else:
            # Otherwise, just convert to datetime
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        stock_data = stock_data.set_index('Date')
        
        # Sort by date
        stock_data = stock_data.sort_index()
        
        # Remove outliers (prices that change by more than 50% in one day)
        returns = stock_data['Close'].pct_change()
        outlier_mask = np.abs(returns) > 0.5
        
        if outlier_mask.sum() > 0:
            logger.warning(f"Removing {outlier_mask.sum()} outlier days for {symbol}")
            stock_data = stock_data[~outlier_mask]
        
        # Forward fill missing values (maximum 5 days)
        stock_data = stock_data.ffill(limit=5)
        
        # Drop remaining NaNs
        stock_data = stock_data.dropna()
        
        # Keep relevant columns
        columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        stock_data = stock_data[columns_to_keep]
        
        return stock_data
    
    def prepare_top_stocks_by_market_cap(self, n: int = 10) -> Dict[str, pd.DataFrame]:
        """Prepare data for the top N stocks by market cap."""
        
        logger.info(f"\n" + "="*70)
        logger.info(f"PREPARING TOP {n} STOCKS BY MARKET CAP")
        logger.info("="*70)
        
        # Get top stocks by market cap
        top_stocks = self.analysis_df.nlargest(n, 'Market_Cap_Billions')
        
        clean_stocks = {}
        for _, row in top_stocks.iterrows():
            symbol = row['Symbol']
            company_name = row['Company_Name']
            market_cap = row['Market_Cap_Billions']
            
            logger.info(f"\nProcessing {symbol} ({company_name}): ${market_cap:.1f}B market cap")
            
            stock_data = self.clean_individual_stock(symbol)
            if stock_data is not None and len(stock_data) > 100:
                clean_stocks[symbol] = stock_data
                logger.info(f"  ✓ {len(stock_data)} clean records")
                logger.info(f"  Price range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")
            else:
                logger.warning(f"  ✗ Insufficient data for {symbol}")
        
        self.clean_stocks = clean_stocks
        return clean_stocks
    
    def create_sector_indices(self) -> Dict[str, pd.DataFrame]:
        """Create sector-based indices."""
        
        logger.info("\n" + "="*70)
        logger.info("CREATING SECTOR INDICES")
        logger.info("="*70)
        
        # Get unique sectors
        sectors = self.analysis_df['Sector'].unique()
        sector_indices = {}
        
        for sector in sectors:
            if pd.isna(sector):
                continue
                
            logger.info(f"\nProcessing {sector} sector...")
            
            # Get companies in this sector
            sector_companies = self.analysis_df[
                self.analysis_df['Sector'] == sector
            ]['Symbol'].tolist()
            
            # Filter price data for these companies
            sector_prices = self.prices_df[
                self.prices_df['Symbol'].isin(sector_companies)
            ].copy()
            
            if len(sector_prices) == 0:
                logger.warning(f"  No price data for {sector}")
                continue
            
            # Merge with market caps
            sector_prices = pd.merge(
                sector_prices,
                self.analysis_df[['Symbol', 'Market_Cap_Billions']],
                on='Symbol',
                how='left'
            )
            
            # Calculate weighted average for the sector
            def sector_weighted_avg(group):
                weights = group['Market_Cap_Billions'].fillna(1)
                total_weight = weights.sum()
                
                if total_weight == 0:
                    return pd.Series({'Close': np.nan, 'Volume': 0})
                
                return pd.Series({
                    'Close': (group['Close'] * weights).sum() / total_weight,
                    'Volume': group['Volume'].sum(),
                    'Companies': len(group)
                })
            
            sector_index = sector_prices.groupby('Date').apply(sector_weighted_avg)
            sector_index = sector_index.dropna(subset=['Close'])
            
            if len(sector_index) > 100:
                sector_indices[sector] = sector_index
                logger.info(f"  ✓ Created index with {len(sector_index)} records")
                logger.info(f"  Average companies: {sector_index['Companies'].mean():.0f}")
        
        return sector_indices
    
    def save_clean_data(self, output_dir: str = "./clean_data"):
        """Save all cleaned data to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"\n" + "="*70)
        logger.info(f"SAVING CLEAN DATA TO {output_dir}")
        logger.info("="*70)
        
        # Save S&P 500 index
        if self.sp500_index is not None:
            filepath = output_path / "sp500_index.csv"
            self.sp500_index.to_csv(filepath)
            logger.info(f"✓ Saved S&P 500 index to {filepath}")
        
        # Save individual stocks
        for symbol, data in self.clean_stocks.items():
            filepath = output_path / f"stock_{symbol}.csv"
            data.to_csv(filepath)
            logger.info(f"✓ Saved {symbol} data to {filepath}")
    
    def get_recommendation(self) -> str:
        """Provide recommendation on which data to use for prediction."""
        
        recommendations = []
        
        if self.sp500_index is not None:
            recommendations.append(
                "1. **S&P 500 Index**: Best for predicting overall market movement. "
                f"Has {len(self.sp500_index)} records with stable price range."
            )
        
        if self.clean_stocks:
            top_stock = list(self.clean_stocks.keys())[0] if self.clean_stocks else None
            recommendations.append(
                f"2. **Individual Stocks**: Good for company-specific predictions. "
                f"Example: {top_stock} has clean, consistent data."
            )
        
        recommendations.append(
            "3. **Data Quality**: Removed outliers and handled missing values. "
            "Ready for LSTM training."
        )
        
        return "\n".join(recommendations)


def main():
    """Main preprocessing pipeline."""
    
    # Initialize preprocessor with full paths
    preprocessor = SP500DataPreprocessor(
        analysis_file='csv/sp500_analysis_20250830_094857.csv',
        prices_file='csv/sp500_quarterly_20250830_103017_prices.csv',
        summary_file='csv/sp500_quarterly_20250830_103017_summary.csv'
    )
    
    # Load all data
    analysis_df, prices_df, summary_df = preprocessor.load_all_data()
    
    # Diagnose issues
    price_stats = preprocessor.diagnose_price_data()
    
    # Create market-cap weighted S&P 500 index
    sp500_index = preprocessor.create_market_cap_weighted_index()
    
    # Prepare top 10 stocks
    top_stocks = preprocessor.prepare_top_stocks_by_market_cap(n=10)
    
    # Create sector indices
    sector_indices = preprocessor.create_sector_indices()
    
    # Save clean data
    preprocessor.save_clean_data()
    
    # Print recommendations
    logger.info("\n" + "="*70)
    logger.info("PREPROCESSING COMPLETE - RECOMMENDATIONS")
    logger.info("="*70)
    print(preprocessor.get_recommendation())
    
    return preprocessor, sp500_index, top_stocks


if __name__ == "__main__":
    preprocessor, sp500_index, top_stocks = main()