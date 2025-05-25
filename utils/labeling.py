import numpy as np

def generate_labels(df, future_return_column="returns", threshold=0.0015):
    """
    Labels:
        1 = Buy (future return > threshold)
        -1 = Sell (future return < -threshold)
        0 = Neutral (in between)
    """
    df["dummy"] = 0
    df.loc[df[future_return_column] > threshold, "dummy"] = 1
    df.loc[df[future_return_column] < -threshold, "dummy"] = -1
    return df
