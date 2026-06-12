#!/usr/bin/env python3
"""
Consolidated S&P 500 Analysis Tool
Combines visualization capabilities with dynamic company fetching and predictions
Fixed: Handles duplicate share classes (GOOG/GOOGL, BRK.A/BRK.B, etc.)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ConsolidatedSP500Analyzer:
    def __init__(self):
        self.companies = {}
        self.historical_data = {}
        self.metrics_df = None
        self.predictions = {}
        self.all_company_data = {}  # Store data for ALL companies, not just top 10
        self.market_caps = {}  # Store all market cap data
        
    def fetch_top_companies(self, method='dynamic', custom_list=None):
        """
        Fetch top S&P 500 companies using different methods
        method: 'dynamic' (fetch current), 'custom' (user provided)
        """
        if method == 'dynamic':
            return self._fetch_dynamic_top10()
        elif method == 'custom' and custom_list:
            return {symbol: 1.0 for symbol in custom_list}  # Equal weights
        
    def _fetch_dynamic_top10(self):
        """Fetch actual top 10 S&P 500 companies by market cap"""
        print("Fetching current S&P 500 top companies...")
        
        # Try to get S&P 500 list from Wikipedia
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            
            dfs = pd.read_html(response.text)
            sp500_df = dfs[0]
            symbols = sp500_df['Symbol'].tolist()  # Get ALL symbols, not just top 50
            
        except Exception as e:
            print(f"Wikipedia fetch failed: {e}")
            symbols = self._get_fallback_symbols()
        
        # Remove duplicate companies (same underlying business, different share classes)
        symbols = self._deduplicate_symbols(symbols)
        
        print(f"Will check {len(symbols)} companies for market cap data...")
        print("⚠ Note: This may take 5-10 minutes due to API rate limits")
        
        # Get market caps and determine top 10
        market_caps = {}
        failed_symbols = []
        
        print(f"Fetching market cap data for all {len(symbols)} companies...")
        
        for i, symbol in enumerate(symbols):
            print(f"  Progress: {i+1}/{len(symbols)} - Checking {symbol}...", end="")
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Try multiple market cap fields
                market_cap = (info.get('marketCap', 0) or 
                             info.get('enterpriseValue', 0) or 
                             info.get('totalCash', 0))
                
                if market_cap and market_cap > 1e9:  # At least $1B market cap
                    market_caps[symbol] = market_cap
                    print(f" ${market_cap/1e12:.2f}T")
                    
                    # Store ALL company data for later saving
                    self.all_company_data[symbol] = {
                        'market_cap': market_cap,
                        'company_name': info.get('longName', symbol),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'employees': info.get('fullTimeEmployees', None),
                        'country': info.get('country', 'Unknown'),
                        'website': info.get('website', None),
                        'pe_ratio': info.get('trailingPE', None),
                        'forward_pe': info.get('forwardPE', None),
                        'price_to_book': info.get('priceToBook', None),
                        'dividend_yield': info.get('dividendYield', None),
                        'beta': info.get('beta', None),
                        '52_week_high': info.get('fiftyTwoWeekHigh', None),
                        '52_week_low': info.get('fiftyTwoWeekLow', None)
                    }
                else:
                    failed_symbols.append(f"{symbol} (no market cap data)")
                    print(f" No data")
                    
            except Exception as e:
                failed_symbols.append(f"{symbol} ({str(e)[:30]})")
                print(f" Error: {str(e)[:30]}")
                continue
            
            # Add small delay to avoid rate limiting
            if i % 10 == 0 and i > 0:
                import time
                time.sleep(1)  # 1 second delay every 10 requests
        
        # Store market caps for later use
        self.market_caps = market_caps
        
        # Calculate weights and show results
        if not market_caps:
            print("❌ No market cap data retrieved. Check internet connection or try again later.")
            return None
            
        total_cap = sum(market_caps.values())
        weights = {k: (v / total_cap) * 100 for k, v in market_caps.items()}
        
        # Return top 10
        top_10 = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
        
        print(f"\n✓ Successfully analyzed {len(market_caps)} companies")
        print(f"📊 TOP 10 COMPANIES BY MARKET CAP:")
        print("-" * 60)
        for i, (symbol, weight) in enumerate(top_10.items(), 1):
            market_cap = market_caps[symbol]
            print(f"  {i:2}. {symbol:5} | Weight: {weight:5.2f}% | Market Cap: ${market_cap/1e12:.2f}T")
        
        # Check if NVDA made it
        if 'NVDA' in top_10:
            print(f"\n✓ NVIDIA (NVDA) is included with {top_10['NVDA']:.2f}% weight")
        else:
            print(f"\n⚠ NVIDIA (NVDA) not in top 10. Checking if it was fetched...")
            if 'NVDA' in market_caps:
                nvda_weight = weights['NVDA']
                nvda_rank = sorted(weights.items(), key=lambda x: x[1], reverse=True).index(('NVDA', nvda_weight)) + 1
                print(f"   NVDA found but ranked #{nvda_rank} with {nvda_weight:.2f}% weight")
            else:
                print(f"   NVDA market cap data was not successfully retrieved")
        
        if failed_symbols:
            print(f"\n⚠ Failed to fetch data for {len(failed_symbols)} companies")
            if len(failed_symbols) <= 10:
                print(f"   Failed: {', '.join(failed_symbols)}")
            else:
                print(f"   First 10 failures: {', '.join(failed_symbols[:10])}")
                print(f"   ... and {len(failed_symbols)-10} others")
        
        return top_10
    
    def _deduplicate_symbols(self, symbols):
        """Remove duplicate companies with different share classes"""
        # Dictionary of companies with multiple share classes
        # Format: {preferred_symbol: [alternative_symbols_to_remove]}
        duplicates = {
            'GOOGL': ['GOOG'],  # Prefer Class A (voting) over Class C
            'BRK.A': ['BRK.B'],  # This might be handled by market cap anyway
            # Add other known duplicates as needed
        }
        
        # Create clean list
        clean_symbols = []
        symbols_to_skip = set()
        
        # First, identify symbols to skip
        for preferred, alternatives in duplicates.items():
            if preferred in symbols:
                for alt in alternatives:
                    if alt in symbols:
                        symbols_to_skip.add(alt)
                        print(f"  Removing {alt} (keeping {preferred} instead)")
        
        # Build clean list
        for symbol in symbols:
            if symbol not in symbols_to_skip:
                clean_symbols.append(symbol)
        
        return clean_symbols
    
    def _get_fallback_symbols(self):
        """Fallback list of major S&P 500 companies (no duplicates) - prioritizing largest companies"""
        return ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK.B', 
                'LLY', 'V', 'JPM', 'UNH', 'XOM', 'JNJ', 'MA', 'WMT', 'PG', 'AVGO', 
                'HD', 'CVX', 'MRK', 'ABBV', 'ORCL', 'COST', 'PEP', 'BAC', 'ADBE', 'KO']
    

    
    def fetch_quarterly_data(self, symbols_list=None, years_back=10):
        """
        Fetch quarterly data for companies over specified years
        WARNING: This is VERY time-intensive and may take several hours
        """
        if symbols_list is None:
            symbols_list = list(self.all_company_data.keys())[:50]  # Limit to 50 for testing
        
        print("="*70)
        print(f"QUARTERLY DATA COLLECTION ({years_back} YEARS)")
        print("="*70)
        print(f"WARNING: This will make ~{len(symbols_list) * years_back * 4} API calls")
        print(f"Estimated time: {len(symbols_list) * years_back * 4 * 2 / 3600:.1f} hours")
        print("This includes rate limiting delays to avoid API blocks")
        
        confirm = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Quarterly data collection cancelled.")
            return None
        
        quarterly_data = {}
        from datetime import datetime, timedelta
        import time
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        for i, symbol in enumerate(symbols_list):
            print(f"\nProgress: {i+1}/{len(symbols_list)} - Fetching quarterly data for {symbol}...")
            
            try:
                ticker = yf.Ticker(symbol)
                
                # Get quarterly financials
                quarterly_financials = ticker.quarterly_financials
                quarterly_balance_sheet = ticker.quarterly_balance_sheet
                quarterly_cashflow = ticker.quarterly_cashflow
                
                # Get historical price data
                hist_data = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                         end=end_date.strftime('%Y-%m-%d'))
                
                if not hist_data.empty:
                    # Resample to quarterly
                    quarterly_prices = hist_data.resample('Q').agg({
                        'Open': 'first',
                        'High': 'max', 
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'mean'
                    })
                    
                    quarterly_data[symbol] = {
                        'prices': quarterly_prices,
                        'financials': quarterly_financials,
                        'balance_sheet': quarterly_balance_sheet,
                        'cashflow': quarterly_cashflow,
                        'company_info': self.all_company_data.get(symbol, {})
                    }
                    
                    print(f"   ✓ {symbol}: {len(quarterly_prices)} quarters of data")
                else:
                    print(f"   ⚠ {symbol}: No historical data available")
                    
            except Exception as e:
                print(f"   ✗ {symbol}: Error - {str(e)[:50]}")
                continue
            
            # Rate limiting: 2-3 seconds between requests
            time.sleep(2.5)
            
            # Every 25 companies, take a longer break
            if (i + 1) % 25 == 0:
                print(f"   Taking 30-second break to avoid rate limits...")
                time.sleep(30)
        
        print(f"\n✓ Collected quarterly data for {len(quarterly_data)} companies")
        return quarterly_data
    
    def save_quarterly_data(self, quarterly_data, filename=None):
        """Save quarterly data to multiple CSV files"""
        if filename is None:
            base_filename = f"sp500_quarterly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            base_filename = filename.replace('.csv', '')
        
        # Save price data
        all_price_data = []
        for symbol, data in quarterly_data.items():
            if 'prices' in data and not data['prices'].empty:
                price_df = data['prices'].copy()
                price_df['Symbol'] = symbol
                price_df['Quarter'] = price_df.index.strftime('%Y-Q%q')
                all_price_data.append(price_df)
        
        if all_price_data:
            combined_prices = pd.concat(all_price_data, ignore_index=False)
            price_filename = f"{base_filename}_prices.csv"
            combined_prices.to_csv(price_filename)
            print(f"✓ Price data saved to {price_filename}")
        else:
            price_filename = None
        
        # Save summary
        summary_data = []
        for symbol, data in quarterly_data.items():
            row = {
                'Symbol': symbol,
                'Quarters_Available': len(data.get('prices', [])),
                'Financials_Available': not data.get('financials', pd.DataFrame()).empty,
                'Balance_Sheet_Available': not data.get('balance_sheet', pd.DataFrame()).empty,
                'Cashflow_Available': not data.get('cashflow', pd.DataFrame()).empty,
            }
            if 'company_info' in data:
                row.update(data['company_info'])
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{base_filename}_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"✓ Summary saved to {summary_filename}")
        
        if price_filename:
            return [price_filename, summary_filename]
        else:
            return [summary_filename]
    
    def fetch_historical_data(self, period="2y"):
        """Fetch historical data for selected companies"""
        print(f"Fetching {period} of historical data...")
        
        symbols = list(self.companies.keys()) + ['^GSPC']  # Include S&P 500 index
        
        try:
            # Batch download for efficiency
            data = yf.download(symbols, start=datetime.now().replace(year=datetime.now().year-2))
            
            if 'Adj Close' in data.columns.levels[0]:
                price_data = data['Adj Close']
            else:
                price_data = data['Close']
            
            # Store individual company data
            for symbol in symbols:
                if symbol in price_data.columns:
                    self.historical_data[symbol] = price_data[symbol].dropna()
            
            print(f"✓ Retrieved data for {len(self.historical_data)} symbols")
            
        except Exception as e:
            print(f"Batch download failed: {e}. Trying individual downloads...")
            # Fallback to individual downloads
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        self.historical_data[symbol] = hist['Close']
                        print(f"  ✓ {symbol}: {len(hist)} days")
                except Exception as ex:
                    print(f"  ✗ {symbol}: {str(ex)[:50]}")
                    continue
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        print("Calculating performance metrics...")
        
        metrics = {}
        
        for symbol, weight in self.companies.items():
            if symbol in self.historical_data:
                prices = self.historical_data[symbol]
                
                if len(prices) < 20:
                    continue
                
                # Calculate returns
                total_return = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
                
                # Daily returns for volatility and other metrics
                returns = prices.pct_change().dropna()
                
                # Annualized volatility
                volatility = returns.std() * np.sqrt(252) * 100
                
                # Sharpe ratio (assuming 3% risk-free rate)
                annual_return = ((prices.iloc[-1] / prices.iloc[0]) ** (252 / len(prices)) - 1) * 100
                sharpe = (annual_return - 3) / volatility if volatility > 0 else 0
                
                # Beta vs S&P 500
                if '^GSPC' in self.historical_data:
                    sp500_returns = self.historical_data['^GSPC'].pct_change().dropna()
                    
                    # Align dates
                    common_dates = returns.index.intersection(sp500_returns.index)
                    if len(common_dates) > 50:
                        stock_aligned = returns.loc[common_dates]
                        sp500_aligned = sp500_returns.loc[common_dates]
                        beta = stock_aligned.cov(sp500_aligned) / sp500_aligned.var()
                    else:
                        beta = 1.0
                else:
                    beta = 1.0
                
                metrics[symbol] = {
                    'Weight': weight,
                    'Total_Return': total_return,
                    'Volatility': volatility,
                    'Sharpe_Ratio': sharpe,
                    'Beta': beta,
                    'Current_Price': prices.iloc[-1]
                }
        
        self.metrics_df = pd.DataFrame(metrics).T
        print(f"✓ Calculated metrics for {len(self.metrics_df)} companies")
        return self.metrics_df
    
    def generate_predictions(self):
        """Generate simple predictions based on momentum and mean reversion"""
        print("Generating 6-month predictions...")
        
        for symbol in self.companies.keys():
            if symbol in self.historical_data:
                prices = self.historical_data[symbol]
                
                if len(prices) < 60:
                    continue
                
                # Momentum indicators
                short_momentum = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]
                medium_momentum = (prices.iloc[-1] - prices.iloc[-60]) / prices.iloc[-60]
                
                # Simple prediction model
                expected_return = (short_momentum * 0.6 + medium_momentum * 0.4) * 300  # 6 months
                
                # Generate signal
                if expected_return > 10:
                    signal = "STRONG BUY"
                elif expected_return > 5:
                    signal = "BUY"
                elif expected_return > -5:
                    signal = "HOLD"
                elif expected_return > -10:
                    signal = "SELL"
                else:
                    signal = "STRONG SELL"
                
                self.predictions[symbol] = {
                    'Expected_Return_6M': expected_return,
                    'Signal': signal
                }
        
        print(f"✓ Generated predictions for {len(self.predictions)} companies")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("No metrics data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Normalized price performance
        normalized_data = {}
        for symbol in self.companies.keys():
            if symbol in self.historical_data:
                prices = self.historical_data[symbol]
                normalized_data[symbol] = (prices / prices.iloc[0]) * 100
        
        if normalized_data:
            norm_df = pd.DataFrame(normalized_data)
            norm_df.plot(ax=axes[0, 0], legend=True, linewidth=2)
            axes[0, 0].set_title("Normalized Price Performance (Base = 100)", fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel("Index Value")
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total return bar chart
        returns_sorted = self.metrics_df['Total_Return'].sort_values()
        colors = ['red' if x < 0 else 'green' for x in returns_sorted.values]
        returns_sorted.plot(kind='bar', ax=axes[0, 1], color=colors, alpha=0.7)
        axes[0, 1].set_title("Total Return Comparison", fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel("Return (%)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Risk vs Reward scatter
        x = self.metrics_df['Volatility']
        y = self.metrics_df['Total_Return']
        sizes = self.metrics_df['Weight'] * 20  # Scale bubble sizes
        
        scatter = axes[0, 2].scatter(x, y, s=sizes, alpha=0.6, c=y, cmap='RdYlGn')
        
        for symbol in self.metrics_df.index:
            axes[0, 2].annotate(symbol, 
                              (self.metrics_df.loc[symbol, 'Volatility'], 
                               self.metrics_df.loc[symbol, 'Total_Return']),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[0, 2].set_title("Risk vs Reward (Bubble size = Weight)", fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel("Volatility (%)")
        axes[0, 2].set_ylabel("Total Return (%)")
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Weight distribution pie chart
        weights = self.metrics_df['Weight']
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        axes[1, 0].pie(weights.values, labels=weights.index, autopct='%1.1f%%', 
                       colors=colors_pie, startangle=90)
        axes[1, 0].set_title("Portfolio Weight Distribution", fontsize=14, fontweight='bold')
        
        # 5. Sharpe ratio comparison
        sharpe_data = self.metrics_df['Sharpe_Ratio'].sort_values()
        colors_sharpe = ['red' if x < 0 else 'green' for x in sharpe_data.values]
        sharpe_data.plot(kind='bar', ax=axes[1, 1], color=colors_sharpe, alpha=0.7)
        axes[1, 1].set_title("Sharpe Ratio Comparison", fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel("Sharpe Ratio")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=1, color='blue', linestyle=':', alpha=0.5, label='Good (>1.0)')
        axes[1, 1].legend()
        
        # 6. Prediction signals (if available)
        if self.predictions:
            signals_df = pd.DataFrame(self.predictions).T
            signal_counts = signals_df['Signal'].value_counts()
            signal_colors = {'STRONG BUY': 'darkgreen', 'BUY': 'green', 'HOLD': 'orange', 
                           'SELL': 'red', 'STRONG SELL': 'darkred'}
            colors_signals = [signal_colors.get(x, 'gray') for x in signal_counts.index]
            
            signal_counts.plot(kind='bar', ax=axes[1, 2], color=colors_signals, alpha=0.7)
            axes[1, 2].set_title("Prediction Signal Distribution", fontsize=14, fontweight='bold')
            axes[1, 2].set_ylabel("Number of Stocks")
            axes[1, 2].tick_params(axis='x', rotation=45)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Predictions\nGenerated', 
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 2].set_title("Predictions", fontsize=14, fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.show()
        
        # Create correlation heatmap separately
        self._create_correlation_heatmap()
    
    def _create_correlation_heatmap(self):
        """Create a separate correlation heatmap"""
        if len(self.historical_data) > 1:
            returns_data = {}
            for symbol in self.companies.keys():
                if symbol in self.historical_data:
                    returns_data[symbol] = self.historical_data[symbol].pct_change().dropna()
            
            if len(returns_data) > 1:
                returns_df = pd.DataFrame(returns_data).dropna()
                
                plt.figure(figsize=(12, 10))
                correlation_matrix = returns_df.corr()
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                           center=0, linewidths=0.5, fmt='.2f', mask=mask,
                           square=True, cbar_kws={"shrink": .8})
                plt.title("Daily Returns Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.show()
    
    def generate_report(self):
        """Generate a comprehensive text report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE S&P 500 ANALYSIS REPORT")
        print("="*80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Companies Analyzed: {len(self.companies)}")
        print(f"Analysis Period: {len(next(iter(self.historical_data.values())))} trading days" if self.historical_data else "N/A")
        
        if self.metrics_df is not None and len(self.metrics_df) > 0:
            print("\n" + "="*60)
            print("TOP PERFORMERS (by Total Return)")
            print("="*60)
            top_performers = self.metrics_df.nlargest(5, 'Total_Return')
            for i, symbol in enumerate(top_performers.index, 1):
                row = top_performers.loc[symbol]
                print(f"{i}. {symbol:5} | Return: {row['Total_Return']:+7.2f}% | "
                      f"Weight: {row['Weight']:5.2f}% | "
                      f"Volatility: {row['Volatility']:5.2f}% | "
                      f"Sharpe: {row['Sharpe_Ratio']:+5.2f}")
            
            print("\n" + "="*60)
            print("WORST PERFORMERS (by Total Return)")
            print("="*60)
            worst_performers = self.metrics_df.nsmallest(3, 'Total_Return')
            for i, symbol in enumerate(worst_performers.index, 1):
                row = worst_performers.loc[symbol]
                print(f"{i}. {symbol:5} | Return: {row['Total_Return']:+7.2f}% | "
                      f"Weight: {row['Weight']:5.2f}% | "
                      f"Volatility: {row['Volatility']:5.2f}% | "
                      f"Sharpe: {row['Sharpe_Ratio']:+5.2f}")
        
        if self.predictions:
            print("\n" + "="*60)
            print("6-MONTH PREDICTIONS & SIGNALS")
            print("="*60)
            
            # Group by signal
            signals = {}
            for symbol, pred in self.predictions.items():
                signal = pred['Signal']
                if signal not in signals:
                    signals[signal] = []
                signals[signal].append((symbol, pred['Expected_Return_6M']))
            
            signal_order = ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL']
            
            for signal in signal_order:
                if signal in signals:
                    print(f"\n{signal}:")
                    for symbol, expected_return in sorted(signals[signal], key=lambda x: x[1], reverse=True):
                        weight = self.companies.get(symbol, 0)
                        print(f"  {symbol:5} | Expected Return: {expected_return:+6.2f}% | Weight: {weight:5.2f}%")
        
        # Portfolio summary
        if self.metrics_df is not None and len(self.metrics_df) > 0:
            total_weight = self.metrics_df['Weight'].sum()
            weighted_return = (self.metrics_df['Total_Return'] * self.metrics_df['Weight']).sum() / total_weight
            weighted_volatility = (self.metrics_df['Volatility'] * self.metrics_df['Weight']).sum() / total_weight
            
            print(f"\n" + "="*60)
            print("PORTFOLIO SUMMARY")
            print("="*60)
            print(f"Total Portfolio Weight: {total_weight:.1f}%")
            print(f"Weighted Average Return: {weighted_return:+.2f}%")
            print(f"Weighted Average Volatility: {weighted_volatility:.2f}%")
            
            best_performer = self.metrics_df['Total_Return'].idxmax()
            worst_performer = self.metrics_df['Total_Return'].idxmin()
            
            print(f"Best Performer: {best_performer} ({self.metrics_df.loc[best_performer, 'Total_Return']:+.2f}%)")
            print(f"Worst Performer: {worst_performer} ({self.metrics_df.loc[worst_performer, 'Total_Return']:+.2f}%)")
            
            # Risk assessment
            high_risk_stocks = self.metrics_df[self.metrics_df['Volatility'] > 30]
            if len(high_risk_stocks) > 0:
                print(f"\nHigh Risk Stocks (>30% volatility): {', '.join(high_risk_stocks.index)}")
            
            # Quality assessment
            quality_stocks = self.metrics_df[
                (self.metrics_df['Sharpe_Ratio'] > 0.5) & (self.metrics_df['Total_Return'] > 0)
            ]
            if len(quality_stocks) > 0:
                print(f"Quality Stocks (Sharpe > 0.5 & positive returns): {', '.join(quality_stocks.index)}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
    
    def save_results(self, filename=None, save_all_companies=True):
        """Save results to CSV file"""
        if filename is None:
            filename = f"sp500_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if save_all_companies and self.all_company_data:
            print(f"Saving data for {len(self.all_company_data)} companies...")
            
            results = []
            for symbol, data in self.all_company_data.items():
                row = {
                    'Symbol': symbol,
                    'Company_Name': data.get('company_name', symbol),
                    'Market_Cap_Billions': data.get('market_cap', 0) / 1e9,
                    'Sector': data.get('sector', 'Unknown'),
                    'Industry': data.get('industry', 'Unknown'),
                    'Employees': data.get('employees', None),
                    'Country': data.get('country', 'Unknown'),
                    'Website': data.get('website', None),
                    'PE_Ratio': data.get('pe_ratio', None),
                    'Forward_PE': data.get('forward_pe', None),
                    'Price_to_Book': data.get('price_to_book', None),
                    'Dividend_Yield': data.get('dividend_yield', None),
                    'Beta': data.get('beta', None),
                    '52_Week_High': data.get('52_week_high', None),
                    '52_Week_Low': data.get('52_week_low', None)
                }
                
                # Add weight if it's in top 10
                if symbol in self.companies:
                    row['Top10_Weight'] = self.companies[symbol]
                    row['In_Top10'] = True
                else:
                    row['Top10_Weight'] = 0
                    row['In_Top10'] = False
                
                # Add metrics if available
                if self.metrics_df is not None and symbol in self.metrics_df.index:
                    metrics = self.metrics_df.loc[symbol]
                    row.update({
                        'Total_Return': metrics.get('Total_Return', None),
                        'Volatility': metrics.get('Volatility', None),
                        'Sharpe_Ratio': metrics.get('Sharpe_Ratio', None),
                        'Beta_vs_SP500': metrics.get('Beta', None),
                        'Current_Price': metrics.get('Current_Price', None)
                    })
                
                # Add predictions if available
                if symbol in self.predictions:
                    row.update({
                        'Expected_Return_6M': self.predictions[symbol].get('Expected_Return_6M', None),
                        'Signal': self.predictions[symbol].get('Signal', None)
                    })
                
                results.append(row)
                
        else:
            # Fallback to top 10 only
            results = []
            for symbol, weight in self.companies.items():
                row = {'Symbol': symbol, 'Weight': weight}
                
                # Add metrics if available
                if self.metrics_df is not None and symbol in self.metrics_df.index:
                    metrics = self.metrics_df.loc[symbol]
                    row.update({
                        'Total_Return': metrics['Total_Return'],
                        'Volatility': metrics['Volatility'],
                        'Sharpe_Ratio': metrics['Sharpe_Ratio'],
                        'Beta': metrics['Beta'],
                        'Current_Price': metrics['Current_Price']
                    })
                
                # Add predictions if available
                if symbol in self.predictions:
                    row.update(self.predictions[symbol])
                
                results.append(row)
        
        df = pd.DataFrame(results)
        
        # Sort by market cap (largest first)
        if 'Market_Cap_Billions' in df.columns:
            df = df.sort_values('Market_Cap_Billions', ascending=False)
        
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        print(f"Saved {len(results)} companies")
        return filename
    
    def run_complete_analysis(self, method='dynamic', include_predictions=True, 
                            create_visuals=True, custom_symbols=None):
        """Run the complete analysis pipeline"""
        print("Starting Consolidated S&P 500 Analysis...")
        print("-" * 50)
        
        try:
            # Step 1: Get companies
            print("Step 1: Fetching companies...")
            self.companies = self.fetch_top_companies(method, custom_symbols)
            if not self.companies:
                print("❌ Failed to fetch companies")
                return None
            print(f"✓ Selected {len(self.companies)} companies")
            
            # Step 2: Fetch data
            print("\nStep 2: Fetching historical data...")
            self.fetch_historical_data()
            if not self.historical_data:
                print("❌ Failed to fetch historical data")
                return None
            
            # Step 3: Calculate metrics
            print("\nStep 3: Calculating metrics...")
            self.calculate_metrics()
            if self.metrics_df is None or len(self.metrics_df) == 0:
                print("❌ Failed to calculate metrics")
                return None
            
            # Step 4: Generate predictions (optional)
            if include_predictions:
                print("\nStep 4: Generating predictions...")
                self.generate_predictions()
            
            # Step 5: Create visualizations (optional)
            if create_visuals:
                print("\nStep 5: Creating visualizations...")
                self.create_visualizations()
            
            # Step 6: Generate report
            print("\nStep 6: Generating report...")
            self.generate_report()
            
            return {
                'companies': self.companies,
                'metrics': self.metrics_df,
                'predictions': self.predictions if include_predictions else None,
                'historical_data': self.historical_data
            }
            
        except Exception as e:
            print(f"❌ Analysis failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# Example usage and main execution
def main():
    """Main execution function with user interaction"""
    print("="*70)
    print("CONSOLIDATED S&P 500 ANALYSIS TOOL")
    print("="*70)
    print("Required packages: yfinance, pandas, numpy, matplotlib, seaborn, requests, beautifulsoup4")
    print()
    
    analyzer = ConsolidatedSP500Analyzer()
    
    try:
        # User can choose analysis method
        print("Choose analysis method:")
        print("1. Dynamic - Fetch current top 10 S&P 500 companies by market cap")
        print("2. Custom - Enter your own list of symbols")
        
        choice = input("\nEnter choice (1-2, or Enter for default=1): ").strip()
        
        if choice == '2':
            method = 'custom'
            symbols_input = input("Enter symbols separated by commas (e.g., AAPL,MSFT,GOOGL): ").strip()
            custom_symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            if not custom_symbols:
                print("No valid symbols entered. Using dynamic method.")
                method = 'dynamic'
                custom_symbols = None
        else:
            method = 'dynamic'
            custom_symbols = None
        
        # Run analysis
        results = analyzer.run_complete_analysis(
            method=method, 
            custom_symbols=custom_symbols,
            include_predictions=True,
            create_visuals=True
        )
        
        if results:
            # Option to save results
            save_choice = input("\nSave results to CSV? (y/n, Enter=n): ").strip().lower()
            if save_choice == 'y':
                save_all = input("Save ALL companies or just top 10? (all/top10, Enter=all): ").strip().lower()
                save_all_companies = save_all != 'top10'
                filename = analyzer.save_results(save_all_companies=save_all_companies)
            
            # Option for quarterly data collection
            quarterly_choice = input("\nCollect quarterly data for past 10 years? (y/n, Enter=n): ").strip().lower()
            if quarterly_choice == 'y':
                symbols_input = input("How many companies? (Enter number or 'all', default=50): ").strip()
                if symbols_input.lower() == 'all':
                    symbols_list = list(analyzer.all_company_data.keys())
                elif symbols_input.isdigit():
                    num_companies = int(symbols_input)
                    symbols_list = list(analyzer.all_company_data.keys())[:num_companies]
                else:
                    symbols_list = list(analyzer.all_company_data.keys())[:50]
                
                quarterly_data = analyzer.fetch_quarterly_data(symbols_list)
                if quarterly_data:
                    analyzer.save_quarterly_data(quarterly_data)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# Alternative quick run examples:
def quick_examples():
    """Quick example runs"""
    analyzer = ConsolidatedSP500Analyzer()
    
    print("\n" + "="*50)
    print("QUICK EXAMPLE 1: Dynamic Analysis")
    print("="*50)
    analyzer.run_complete_analysis(method='dynamic', create_visuals=False)
    
    print("\n" + "="*50)
    print("QUICK EXAMPLE 2: Custom Analysis")
    print("="*50)
    custom_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    analyzer.run_complete_analysis(method='custom', custom_symbols=custom_list, create_visuals=False)

# Uncomment to run quick examples:
# quick_examples()