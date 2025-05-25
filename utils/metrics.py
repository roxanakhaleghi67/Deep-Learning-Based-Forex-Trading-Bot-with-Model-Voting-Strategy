import numpy as np

def calculate_winrate_and_profit(strategy_series):
    trades = strategy_series.dropna()
    wins = (trades > 0).sum()
    total = len(trades)
    winrate = wins / total if total > 0 else 0
    profit = trades.sum()
    return round(winrate, 3), round(profit, 2)
