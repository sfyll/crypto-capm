import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import gmean
import matplotlib.pyplot as plt

def plot_dual_rolling_beta(beta_vs_sp500, beta_vs_crypto, price_data, token_symbol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Comparative Rolling Beta Analysis for {token_symbol}', fontsize=18, y=0.96)

    ax1.plot(beta_vs_sp500.index, beta_vs_sp500, color='tab:red', label='Beta vs S&P 500')
    avg_beta_sp500 = beta_vs_sp500.mean()
    ax1.axhline(y=avg_beta_sp500, color='r', linestyle='--', label=f'Avg Beta vs S&P 500 ({avg_beta_sp500:.2f})')
    ax1.set_ylabel('Rolling Beta vs S&P 500', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(price_data.index, price_data, color='tab:green', alpha=0.4, label=f'{token_symbol} Price')
    ax1_twin.set_ylabel(f'{token_symbol} Price (USD) [Log Scale]', color='tab:green')
    ax1_twin.tick_params(axis='y', labelcolor='tab:green')
    ax1_twin.set_yscale('log')
    ax1_twin.legend(loc='upper right')
    
    ax2.plot(beta_vs_crypto.index, beta_vs_crypto, color='tab:blue', label='Beta vs Crypto Index (BITW)')
    avg_beta_crypto = beta_vs_crypto.mean()
    ax2.axhline(y=avg_beta_crypto, color='b', linestyle='--', label=f'Avg Beta vs BITW ({avg_beta_crypto:.2f})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rolling Beta vs BITW', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(price_data.index, price_data, color='tab:green', alpha=0.4, label=f'{token_symbol} Price')
    ax2_twin.set_ylabel(f'{token_symbol} Price (USD) [Log Scale]', color='tab:green')
    ax2_twin.tick_params(axis='y', labelcolor='tab:green')
    ax2_twin.set_yscale('log')
    ax2_twin.legend(loc='upper right')

    filename = f"dual_beta_analysis_{token_symbol.replace('-USD', '')}.png"
    plt.savefig(filename)
    print(f"Dual-beta chart saved to {filename}")
    plt.close(fig)

def calculate_final_model(token_symbol, data_df, market_index_symbol, crypto_index_symbol, implied_erp):
    print(f"\n{'='*60}")
    print(f"Starting Final Model Analysis for: {token_symbol}")
    print(f"{'='*60}")

    returns = data_df.pct_change()
    
    daily_risk_free_rate = (data_df['^TNX'] / 100 / 252).ffill()
    returns['daily_rf'] = daily_risk_free_rate
    returns.dropna(inplace=True)

    rolling_window = 252
    
    rolling_cov_sp500 = returns[token_symbol].rolling(window=rolling_window).cov(returns[market_index_symbol])
    rolling_var_sp500 = returns[market_index_symbol].rolling(window=rolling_window).var()
    rolling_beta_sp500 = (rolling_cov_sp500 / rolling_var_sp500).dropna()
    avg_beta_sp500 = rolling_beta_sp500.mean()
    print(f"Average Rolling Beta (β) vs S&P 500 ({rolling_window}-Day Window): {avg_beta_sp500:.2f}")

    rolling_cov_crypto = returns[token_symbol].rolling(window=rolling_window).cov(returns[crypto_index_symbol])
    rolling_var_crypto = returns[crypto_index_symbol].rolling(window=rolling_window).var()
    rolling_beta_crypto = (rolling_cov_crypto / rolling_var_crypto).dropna()
    avg_beta_crypto = rolling_beta_crypto.mean()
    print(f"Average Rolling Beta (β) vs Crypto (BITW) ({rolling_window}-Day Window): {avg_beta_crypto:.2f}")

    current_risk_free_rate = returns['daily_rf'].iloc[-1] * 252
    base_capm_rate = current_risk_free_rate + (avg_beta_sp500 * implied_erp)
    
    volatility_premium = 0.0
    
    if token_symbol != "COIN":
        volatility_sp500 = returns[market_index_symbol].std() * np.sqrt(252)
        volatility_crypto = returns[crypto_index_symbol].std() * np.sqrt(252)
        relative_volatility = volatility_crypto / volatility_sp500
        print(f"Crypto Index Volatility is {relative_volatility:.2f}x that of S&P 500.")
        volatility_premium = (relative_volatility - 1) * implied_erp
        volatility_premium = max(0, volatility_premium)
    
    final_cost_of_equity = base_capm_rate + volatility_premium
    
    print(f"\n--- Final Cost of Equity for {token_symbol} ---")
    print(f"Base CAPM Rate: {base_capm_rate:.2%}")
    if token_symbol != "COIN":
        print(f"+ Volatility Premium: {volatility_premium:.2%}")
    print(f"-------------------------------------------------")
    print(f"Final Cost of Equity (Ke): {final_cost_of_equity:.2%}")
    print(f"{'='*60}\n")
    
    plot_dual_rolling_beta(rolling_beta_sp500, rolling_beta_crypto, data_df[token_symbol], token_symbol)
    
    return {
        "token": token_symbol,
        "cost_of_equity": final_cost_of_equity,
        "beta_vs_sp500": avg_beta_sp500,
        "beta_vs_crypto": avg_beta_crypto,
        "base_capm_rate": base_capm_rate,
        "volatility_premium": volatility_premium,
    }

if __name__ == "__main__":
    START_DATE = "2020-01-01"
    IMPLIED_EQUITY_RISK_PREMIUM = 0.04  # 4.00% based on Damodaran data for Jan 2025
    
    TOKENS_TO_ANALYZE = ["AAVE-USD", "UNI7083-USD", "COIN"]
    MARKET_INDEX_SYMBOL = "^GSPC"
    CRYPTO_INDEX_SYMBOL = "BITW"
    
    all_tickers_to_fetch = list(set(TOKENS_TO_ANALYZE + [MARKET_INDEX_SYMBOL, CRYPTO_INDEX_SYMBOL, '^TNX']))
    
    print("Fetching all required historical data...")
    raw_data = yf.download(all_tickers_to_fetch, start=START_DATE, progress=False, auto_adjust=True)['Close']

    print(f"Aligning all data to {MARKET_INDEX_SYMBOL} trading calendar...")
    aligned_data = raw_data.dropna(subset=[MARKET_INDEX_SYMBOL])

    valid_start_dates = [aligned_data[ticker].first_valid_index() for ticker in TOKENS_TO_ANALYZE if ticker in aligned_data and pd.notna(aligned_data[ticker].first_valid_index())]
    common_start_date = max(valid_start_dates)
    print(f"Synchronizing all data to common start date: {common_start_date.date()}")
    analysis_data = aligned_data[common_start_date:]

    results = []
    
    for token in TOKENS_TO_ANALYZE:
        if token in analysis_data.columns and not analysis_data[token].isnull().all():
            result = calculate_final_model(token, analysis_data.copy(), MARKET_INDEX_SYMBOL, CRYPTO_INDEX_SYMBOL, IMPLIED_EQUITY_RISK_PREMIUM)
            if result:
                results.append(result)
        else:
            print(f"Skipping {token} as it has no data in the specified time window.")
            
    print("\n\n--- Final Summary ---")
    if results:
        summary_df = pd.DataFrame(results)
        
        for col in ['cost_of_equity', 'base_capm_rate', 'volatility_premium']:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2%}")

        summary_df['beta_vs_sp500'] = summary_df['beta_vs_sp500'].apply(lambda x: f"{x:.2f}")
        summary_df['beta_vs_crypto'] = summary_df['beta_vs_crypto'].apply(lambda x: f"{x:.2f}")
        
        summary_df = summary_df[['token', 'cost_of_equity', 'base_capm_rate', 'volatility_premium', 'beta_vs_sp500', 'beta_vs_crypto']]
        
        print(summary_df.to_string(index=False))
    else:
        print("No results to display.")

