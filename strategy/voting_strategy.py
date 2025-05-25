import numpy as np

def generate_voting_position(pred_df, model_names):
    signals = np.where(pred_df[model_names] == 0, 0, np.sign(pred_df[model_names] - 0.5))
    pred_df["Increase"] = (signals == 1).sum(axis=1)
    pred_df["Decrease"] = (signals == -1).sum(axis=1)

    pred_df["position"] = np.nan
    pred_df.loc[pred_df["Increase"] > pred_df["Decrease"], "position"] = 1
    pred_df.loc[pred_df["Increase"] < pred_df["Decrease"], "position"] = -1
    return pred_df
