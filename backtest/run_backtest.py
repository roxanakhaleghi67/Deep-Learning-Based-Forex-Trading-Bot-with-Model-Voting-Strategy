def run_backtest(prediction_df, returns_col="returns"):
    prediction_df["strategy"] = prediction_df["position"].shift(1) * prediction_df[returns_col]
    prediction_df["strategy_cumsum"] = prediction_df["strategy"].cumsum()
    return prediction_df[["strategy", "strategy_cumsum"]]
