"""
Compute segmentation metrics (F1, Precision, Recall).
"""

import numpy as np
import pandas as pd
from stardist.matching import matching
from tqdm import tqdm

input_file = "/home/qbx911/datadir/cellpose_results.pkl"
df = pd.read_pickle(input_file)

f1_scores = [] # harmonic mean of precision and recall
precisions = [] # how many of the predicted positives are actually correct
recalls = [] # how many of the actual positives were correctly predicted.
mean_true_scores = [] # average IoU of all correctly matched ground‑truth objects

for row in tqdm(df.itertuples(index=False), total=len(df)):
    pred_mask = row.pred_mask.astype(np.int32)
    gt_mask   = row.gt_mask.astype(np.int32)

    match = matching(
        gt_mask,
        pred_mask,
        thresh=0.5, #IoU threshold for a match is 50%
        report_matches=False
    )

    f1_scores.append(match.f1)
    precisions.append(match.precision)
    recalls.append(match.recall)
    mean_true_scores.append(match.mean_true_score)

df["f1_score"] = f1_scores
df["precision"] = precisions
df["recall"] = recalls
df["mean_true_score"] = mean_true_scores
``
