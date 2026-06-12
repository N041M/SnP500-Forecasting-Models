#!/usr/bin/env python3
"""
S&P 500 Top 10 Companies Analysis (Fixed with Multiple Fallbacks)
- Multiple data source options for fetching S&P 500 companies
- Robust error handling
- Free-float-adjusted market cap for weighting
- Historical tracking and predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import warnings
warnings.filterwarnings('ignore')


class SP500Analyzer:
    def __init__(self):
        self.top_companies = []
        self.historical_data = {}
        self.predictions = {}
        self.all_symbols = []

    # ----------------------------
    # 1. Fetch S&P 500 companies with multiple methods
    # ----------------------------
    def fetch_sp500_companies(self):
        """Try multiple methods to fetch S&P 500 companies"""
        
        # Method 1: Try Wikipedia with better headers
        symbols = self._fetch_from_wikipedia()
        if symbols:
            self.all_symbols = symbols
            return True
            
        # Method 2: Try DataHub.io
        symbols = self._fetch_from_datahub()
        if symbols:
            self.all_symbols = symbols
            return True
            
        # Method 3: Try S&P 500 ETF holdings
        symbols = self._fetch_from_spy_etf()
        if symbols:
            self.all_symbols = symbols
            return True
            
        # Method 4: Use hardcoded list
        print("\nUsing hardcoded fallback list...")
        symbols = self._get_fallback_list()
        self.all_symbols = symbols
        return True

    def _fetch_from_wikipedia(self):
        """Fetch from Wikipedia with improved parsing"""
        try:
            print("Attempting to fetch from Wikipedia...")
            
            # Use headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Try pandas read_html directly first
            try:
                dfs = pd.read_html(response.text)
                if dfs and len(dfs[0]) > 400:  # S&P 500 should have 500+ rows
                    df = dfs[0]
                    # Try different column name variations
                    for col in df.columns:
                        if 'symbol' in col.lower() or 'ticker' in col.lower():
                            symbols = df[col].tolist()
                            # Clean symbols
                            symbols = [str(s).strip() for s in symbols if pd.notna(s)]
                            if len(symbols) > 400:
                                print(f"✓ Successfully fetched {len(symbols)} symbols from Wikipedia")
                                return symbols
            except:
                pass
                
            # Fallback to BeautifulSoup parsing
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Try multiple table selection methods
            table = None
            
            # Method 1: By ID
            table = soup.find("table", {"id": "constituents"})
            
            # Method 2: By class and content
            if not table:
                tables = soup.find_all("table", {"class": "wikitable"})
                for t in tables:
                    # Check if this looks like the S&P 500 table
                    if t.find("th", string=lambda x: x and "symbol" in x.lower()):
                        table = t
                        break
            
            if table:
                # Parse with pandas
                df = pd.read_html(str(table))[0]
                
                # Find symbol column
                symbol_col = None
                for col in df.columns:
                    if 'symbol' in str(col).lower() or 'ticker' in str(col).lower():
                        symbol_col = col
                        break
                
                if symbol_col:
                    symbols = df[symbol_col].tolist()
                    symbols = [str(s).strip() for s in symbols if pd.notna(s)]
                    if len(symbols) > 400:
                        print(f"✓ Successfully fetched {len(symbols)} symbols from Wikipedia")
                        return symbols
                        
        except Exception as e:
            print(f"  Wikipedia fetch failed: {str(e)[:100]}")
            
        return None

    def _fetch_from_datahub(self):
        """Fetch from DataHub.io S&P 500 dataset"""
        try:
            print("Attempting to fetch from DataHub.io...")
            url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            symbols = [item['Symbol'] for item in data if 'Symbol' in item]
            
            if len(symbols) > 400:
                print(f"✓ Successfully fetched {len(symbols)} symbols from DataHub")
                return symbols
                
        except Exception as e:
            print(f"  DataHub fetch failed: {str(e)[:100]}")
            
        return None

    def _fetch_from_spy_etf(self):
        """Fetch top holdings from SPY ETF"""
        try:
            print("Attempting to fetch SPY ETF holdings...")
            spy = yf.Ticker("SPY")
            
            # Try to get holdings info
            holdings_info = spy.info.get('holdings', [])
            
            if not holdings_info:
                # Alternative: fetch SPY's top holdings from Yahoo Finance
                url = "https://finance.yahoo.com/quote/SPY/holdings"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Parse holdings table
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    if len(rows) > 5:  # Likely a holdings table
                        symbols = []
                        for row in rows[1:]:  # Skip header
                            cells = row.find_all('td')
                            if cells and len(cells) > 0:
                                symbol = cells[0].text.strip()
                                if symbol and len(symbol) < 6:  # Valid ticker length
                                    symbols.append(symbol)
                        
                        if symbols:
                            # SPY might only show top holdings, so we'll combine with known list
                            print(f"  Found {len(symbols)} top holdings from SPY")
                            return None  # Continue to fallback for full list
                            
        except Exception as e:
            print(f"  SPY ETF fetch failed: {str(e)[:100]}")
            
        return None

    def _get_fallback_list(self):
        """Return hardcoded list of S&P 500 companies (top 100 by market cap)"""
        # This is a representative sample as of 2024
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'BRK.A',
            'LLY', 'V', 'JPM', 'UNH', 'XOM', 'JNJ', 'MA', 'WMT', 'PG', 'AVGO',
            'HD', 'CVX', 'MRK', 'ABBV', 'ORCL', 'COST', 'PEP', 'BAC', 'ADBE', 'KO',
            'CRM', 'TMO', 'CSCO', 'ACN', 'MCD', 'NFLX', 'ABT', 'AMD', 'WFC', 'PFE',
            'DIS', 'DHR', 'NKE', 'TMUS', 'VZ', 'INTC', 'TXN', 'INTU', 'PM', 'AMGN',
            'CMCSA', 'COP', 'UNP', 'NEE', 'IBM', 'QCOM', 'HON', 'RTX', 'UPS', 'BA',
            'SPGI', 'LOW', 'BMY', 'CAT', 'GE', 'SYK', 'ELV', 'BLK', 'GS', 'NOW',
            'AMAT', 'SBUX', 'PLD', 'ISRG', 'TJX', 'MDLZ', 'DE', 'GILD', 'MMC', 'ADP',
            'ADI', 'LMT', 'CVS', 'AMT', 'VRTX', 'REGN', 'CI', 'SLB', 'ZTS', 'C',
            'BDX', 'EOG', 'SO', 'SCHW', 'LRCX', 'FI', 'BSX', 'EQIX', 'ITW', 'CME'
        ]

    # ----------------------------------------
    # 2. Calculate weights using market cap
    # ----------------------------------------
    def calculate_weights_for_top10(self):
        """Calculate weights for top 10 companies by market cap"""
        print("\nCalculating weights for top companies...")
        
        market_caps = {}
        company_info = {}
        
        # Try to get market cap for all symbols (or at least top ones)
        symbols_to_check = self.all_symbols[:50] if len(self.all_symbols) > 50 else self.all_symbols
        
        print(f"Fetching market cap data for {len(symbols_to_check)} companies...")
        
        for i, symbol in enumerate(symbols_to_check):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(symbols_to_check)}")
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                market_cap = info.get("marketCap", 0)
                
                # Try alternative fields if marketCap is not available
                if not market_cap:
                    market_cap = info.get("enterpriseValue", 0)
                
                if market_cap > 0:
                    # Store additional info
                    company_info[symbol] = {
                        'name': info.get('longName', symbol),
                        'market_cap': market_cap,
                        'sector': info.get('sector', 'Unknown'),
                        'float_shares': info.get('floatShares', None),
                        'shares_outstanding': info.get('sharesOutstanding', None)
                    }
                    
                    # Calculate free-float adjusted market cap if possible
                    if info.get('floatShares') and info.get('sharesOutstanding'):
                        ff_ratio = info['floatShares'] / info['sharesOutstanding']
                        ff_market_cap = market_cap * ff_ratio
                    else:
                        ff_market_cap = market_cap
                    
                    market_caps[symbol] = ff_market_cap
                    
            except Exception as e:
                continue
        
        if not market_caps:
            # Use fallback with estimated weights
            print("\n⚠ Could not fetch live market caps. Using estimated weights...")
            self.top_companies = [
                ('AAPL', 7.5), ('MSFT', 7.2), ('NVDA', 5.0), ('GOOGL', 3.8),
                ('AMZN', 3.5), ('META', 2.8), ('TSLA', 1.9), ('BRK.B', 1.7),
                ('LLY', 1.5), ('V', 1.3)
            ]
        else:
            # Calculate weights
            total_cap = sum(market_caps.values())
            weights = {k: (v / total_cap) * 100 for k, v in market_caps.items()}
            
            # Get top 10
            top_10 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
            self.top_companies = [(symbol, weight) for symbol, weight in top_10]
        
        # Display results
        print("\n" + "="*70)
        print("TOP 10 S&P 500 COMPANIES BY WEIGHT")
        print("="*70)
        
        for i, (symbol, weight) in enumerate(self.top_companies, 1):
            if symbol in company_info:
                info = company_info[symbol]
                print(f"{i:2}. {symbol:5} - {info['name'][:30]:30} | {weight:5.2f}% | {info['sector']}")
            else:
                print(f"{i:2}. {symbol:5} | Weight: {weight:5.2f}%")

    # -----------------------------------------
    # 3. Fetch historical data
    # -----------------------------------------
    def fetch_historical_data(self, period="2y"):
        print(f"\n{'='*70}")
        print(f"FETCHING HISTORICAL DATA ({period})")
        print("="*70)

        for symbol, weight in self.top_companies:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    self.historical_data[symbol] = hist
                    print(f"✓ {symbol}: Retrieved {len(hist)} days of data")
                else:
                    # Try shorter period
                    hist = ticker.history(period="1y")
                    if not hist.empty:
                        self.historical_data[symbol] = hist
                        print(f"⚠ {symbol}: Only 1y data available ({len(hist)} days)")
                    else:
                        print(f"✗ {symbol}: No data available")
                        
            except Exception as e:
                print(f"✗ {symbol}: Error - {str(e)[:50]}")

    # --------------------------------
    # 4. Calculate performance metrics
    # --------------------------------
    def calculate_metrics(self):
        print(f"\n{'='*70}")
        print("PERFORMANCE METRICS")
        print("="*70)
        
        metrics = {}
        
        for symbol, weight in self.top_companies:
            if symbol in self.historical_data:
                data = self.historical_data[symbol]
                
                if 'Close' in data.columns and len(data) > 20:
                    current_price = data['Close'].iloc[-1]
                    start_price = data['Close'].iloc[0]
                    
                    # Calculate various metrics
                    total_return = ((current_price - start_price) / start_price) * 100
                    
                    # 1-year return if available
                    # Handle timezone-aware datetime index from yfinance
                    try:
                        # Option 1: Use integer indexing for 1 year of data (approximately 252 trading days)
                        if len(data) >= 252:
                            year_ago_price = data['Close'].iloc[-252]
                            year_return = ((current_price - year_ago_price) / year_ago_price) * 100
                        else:
                            year_return = total_return
                    except:
                        # Fallback: use the total return
                        year_return = total_return
                    
                    # Volatility
                    daily_returns = data['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252) * 100
                    
                    # Sharpe ratio (assuming 3% risk-free rate)
                    annual_return = ((current_price / start_price) ** (252 / len(data)) - 1) * 100
                    sharpe = (annual_return - 3) / volatility if volatility > 0 else 0
                    
                    metrics[symbol] = {
                        'Weight': weight,
                        'Price': current_price,
                        'Total Return': total_return,
                        '1Y Return': year_return,
                        'Volatility': volatility,
                        'Sharpe': sharpe
                    }
                    
                    print(f"{symbol:5} | Wt: {weight:5.2f}% | 1Y: {year_return:+7.2f}% | "
                          f"Vol: {volatility:5.2f}% | Sharpe: {sharpe:+.2f}")
        
        return metrics

    # --------------------------------
    # 5. Generate predictions
    # --------------------------------
    def predict_future_performance(self):
        print(f"\n{'='*70}")
        print("6-MONTH PREDICTIONS")
        print("="*70)
        
        for symbol, weight in self.top_companies:
            if symbol in self.historical_data:
                data = self.historical_data[symbol]
                
                if 'Close' in data.columns and len(data) > 60:
                    prices = data['Close'].values
                    
                    # Calculate momentum indicators
                    recent_momentum = (prices[-1] - prices[-20]) / prices[-20]  # 20-day momentum
                    medium_momentum = (prices[-1] - prices[-60]) / prices[-60]  # 60-day momentum
                    
                    # Mean reversion factor
                    ma_200 = data['Close'].rolling(window=min(200, len(data)-1)).mean().iloc[-1]
                    deviation_from_ma = (prices[-1] - ma_200) / ma_200
                    
                    # Volatility
                    daily_returns = data['Close'].pct_change().dropna()
                    volatility = daily_returns.std()
                    
                    # Simple prediction model
                    base_prediction = (recent_momentum * 0.6 + medium_momentum * 0.4) * 3  # Weight recent more
                    
                    # Apply mean reversion
                    if abs(deviation_from_ma) > 0.2:  # If >20% from MA200
                        base_prediction *= 0.7  # Expect some reversion
                    
                    # Convert to percentage
                    expected_return = base_prediction * 100
                    
                    # Calculate confidence interval
                    std_6m = volatility * np.sqrt(126)  # 6 months ≈ 126 trading days
                    upper_bound = expected_return + (std_6m * 100 * 1.96)  # 95% CI
                    lower_bound = expected_return - (std_6m * 100 * 1.96)
                    
                    # Signal generation
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
                        'Expected Return': expected_return,
                        'Lower Bound': lower_bound,
                        'Upper Bound': upper_bound,
                        'Signal': signal
                    }
                    
                    print(f"\n{symbol} (Weight: {weight:.2f}%)")
                    print(f"  Expected Return: {expected_return:+.2f}%")
                    print(f"  95% CI: [{lower_bound:+.2f}%, {upper_bound:+.2f}%]")
                    print(f"  Signal: {signal}")

    # -------------------------------
    # 6. Main analysis pipeline
    # -------------------------------
    def run_analysis(self):
        print("="*70)
        print("S&P 500 TOP 10 ANALYSIS - ENHANCED VERSION")
        print("="*70)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Step 1: Fetch S&P 500 companies
        if not self.fetch_sp500_companies():
            print("ERROR: Could not fetch S&P 500 companies from any source")
            return

        print(f"\n✓ Total companies loaded: {len(self.all_symbols)}")

        # Step 2: Calculate weights for top 10
        self.calculate_weights_for_top10()

        # Step 3: Fetch historical data
        self.fetch_historical_data()

        # Step 4: Calculate metrics
        metrics = self.calculate_metrics()

        # Step 5: Generate predictions
        self.predict_future_performance()

        # Step 6: Portfolio summary
        self._generate_portfolio_summary()

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)

    def _generate_portfolio_summary(self):
        """Generate overall portfolio recommendation"""
        print(f"\n{'='*70}")
        print("PORTFOLIO SUMMARY & RECOMMENDATION")
        print("="*70)
        
        if self.predictions:
            # Calculate weighted expected return
            total_weight = sum([w for _, w in self.top_companies])
            weighted_return = 0
            
            signals_count = {'STRONG BUY': 0, 'BUY': 0, 'HOLD': 0, 'SELL': 0, 'STRONG SELL': 0}
            
            for symbol, weight in self.top_companies:
                if symbol in self.predictions:
                    pred = self.predictions[symbol]
                    weighted_return += (pred['Expected Return'] * weight / total_weight)
                    signals_count[pred['Signal']] = signals_count.get(pred['Signal'], 0) + 1
            
            print(f"\nWeighted Portfolio Expected Return (6M): {weighted_return:+.2f}%")
            
            print("\nSignal Distribution:")
            for signal, count in signals_count.items():
                if count > 0:
                    print(f"  {signal}: {count} companies")
            
            # Overall recommendation
            print("\n" + "-"*50)
            if weighted_return > 8:
                print("📈 OVERALL OUTLOOK: STRONGLY BULLISH")
                print("   Recommendation: Consider increasing S&P 500 exposure")
            elif weighted_return > 3:
                print("📊 OVERALL OUTLOOK: MODERATELY BULLISH")
                print("   Recommendation: Maintain or slightly increase position")
            elif weighted_return > -3:
                print("➡️  OVERALL OUTLOOK: NEUTRAL")
                print("   Recommendation: Hold current allocation")
            elif weighted_return > -8:
                print("📉 OVERALL OUTLOOK: MODERATELY BEARISH")
                print("   Recommendation: Consider reducing exposure")
            else:
                print("⚠️  OVERALL OUTLOOK: STRONGLY BEARISH")
                print("   Recommendation: Defensive positioning recommended")


def main():
    """Main execution function"""
    analyzer = SP500Analyzer()
    
    try:
        analyzer.run_analysis()
        
        # Save results option
        save = input("\nSave results to CSV? (y/n): ").strip().lower()
        if save == 'y':
            results = []
            for symbol, weight in analyzer.top_companies:
                row = {'Symbol': symbol, 'Weight': weight}
                if symbol in analyzer.predictions:
                    row.update(analyzer.predictions[symbol])
                results.append(row)
            
            df = pd.DataFrame(results)
            filename = f"sp500_top10_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
            
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("S&P 500 Top 10 Analysis Tool")
    print("-" * 30)
    print("Required packages: yfinance, pandas, numpy, requests, beautifulsoup4, lxml")
    print("Install: pip install yfinance pandas numpy requests beautifulsoup4 lxml\n")
    
    main()