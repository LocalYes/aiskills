import numpy as np

def top_k_masked_mean_pooling(k, arrays):
    S_masked = np.full_like(arrays, np.nan)

    for i, row in enumerate(arrays):
        top_k = row.argsort()[-k:]
        S_masked[i, top_k] = row[top_k]

    scores = np.nanmean(S_masked, axis=0)
    return np.nan_to_num(scores, nan=0.0)
