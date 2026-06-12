import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
tickers = ["^GSPC", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM"]
top10 = tickers[1:]  # exclude S&P 500 itself
start = "2022-01-01"

# ------------------------------------------------------------
# DATA FETCH
# ------------------------------------------------------------
print("🚀 Starting S&P 500 vs Top 10 Companies Analysis...")
print(f"Analyzing top 10 S&P 500 companies: {', '.join(top10)}")

print("Fetching market data...")
df = yf.download(tickers, start=start)
print("  Data shape:", df.shape)
print("  Column levels:", df.columns.nlevels)

if "Adj Close" in df.columns.levels[0]:
    df = df["Adj Close"]
else:
    df = df["Close"]

print("Successfully loaded data for", len(df.columns), "tickers with", len(df), "trading days")

# ------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------
def calculate_metrics(df):
    returns = df.pct_change().dropna()
    metrics = {}

    for col in df.columns:
        total_return = (df[col].iloc[-1] / df[col].iloc[0]) - 1
        annual_vol = returns[col].std() * np.sqrt(252)
        sharpe = (returns[col].mean() / returns[col].std()) * np.sqrt(252)
        beta = returns[col].cov(returns["^GSPC"]) / returns["^GSPC"].var() if col != "^GSPC" else 1.0

        metrics[col] = {
            "Total Return": total_return,
            "Volatility": annual_vol,
            "Sharpe Ratio": sharpe,
            "Beta vs SP500": beta,
        }

    return pd.DataFrame(metrics).T, returns

metrics_df, returns = calculate_metrics(df)

print("\nCalculated Performance Metrics:")
print(metrics_df.round(3))

# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------
def plot_performance_comparison(df, metrics_df, returns):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Normalized price trends
    (df / df.iloc[0] * 100).plot(ax=axes[0,0])
    axes[0,0].set_title("Normalized Price Performance")
    axes[0,0].set_ylabel("Index (100 = start)")

    # 2. Total return bar chart
    metrics_df["Total Return"].sort_values().plot(kind="bar", ax=axes[0,1], color="skyblue")
    axes[0,1].set_title("Total Return since 2022")
    axes[0,1].set_ylabel("Return")

    # 3. Risk vs reward scatter
    axes[1,0].scatter(metrics_df["Volatility"], metrics_df["Total Return"], s=100)
    for i, txt in enumerate(metrics_df.index):
        axes[1,0].annotate(txt, (metrics_df["Volatility"].iloc[i], metrics_df["Total Return"].iloc[i]))
    axes[1,0].set_title("Risk vs Reward")
    axes[1,0].set_xlabel("Volatility")
    axes[1,0].set_ylabel("Total Return")

    # 4. Heatmap of metrics
    heatmap_metrics = ["Total Return", "Volatility", "Sharpe Ratio", "Beta vs SP500"]
    available_metrics = [m for m in heatmap_metrics if m in metrics_df.columns]
    if available_metrics:
        heatmap_data = metrics_df[available_metrics].T
        normalized_heatmap = pd.DataFrame(index=heatmap_data.index, columns=heatmap_data.columns)

        for metric in available_metrics:
            row = heatmap_data.loc[metric]
            if metric == "Beta vs SP500":
                normalized_heatmap.loc[metric] = row
            else:
                if row.max() != row.min():
                    normalized_heatmap.loc[metric] = (row - row.min()) / (row.max() - row.min())
                else:
                    normalized_heatmap.loc[metric] = 0.5

        sns.heatmap(normalized_heatmap.astype(float),
                    annot=heatmap_data.astype(float), fmt=".2f",
                    cmap="RdYlGn", center=0.5, ax=axes[1,1],
                    cbar_kws={"label": "Normalized Score"})
        axes[1,1].set_title("Performance Metrics Heatmap")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 5. Correlation Heatmap (separate figure)
    # --------------------------------------------------------
    corr = returns.corr()
    print("\nCorrelation Matrix (daily returns):")
    print(corr.round(2))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation of Daily Returns")
    plt.show()

plot_performance_comparison(df, metrics_df, returns)
