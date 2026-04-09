"""
Compute segmentation metrics (F1, Precision, Recall).
"""


# Plotting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stardist.matching import matching
from tqdm import tqdm

input_file = "/home/qbx911/datadir/cellpose_results.pkl"
df = pd.read_pickle(input_file)

# Initializing lists
f1_scores = []
precisions = []
recalls = []
mean_true_scores = []

# Looping over dataframe
for row in tqdm(df.itertuples(index=False), total=len(df)):
    pred_mask = row.pred_mask.astype(np.int32)
    gt_mask   = row.gt_mask.astype(np.int32)

    # Compute matching metrics (StarDist)
    match = matching(
        gt_mask,
        pred_mask,
        thresh=0.5,
    )

    # Metrics update
    f1_scores.append(match.f1)
    precisions.append(match.precision)
    recalls.append(match.recall)
    mean_true_scores.append(match.mean_true_score)

# DataFram-ing metrics
df["f1_score"] = f1_scores
df["precision"] = precisions
df["recall"] = recalls
df["mean_true_score"] = mean_true_scores

print("Performance evaluation done")

# Aggregating per model
model_summary = (
    df.groupby("model")[["f1_score", "precision", "recall", "mean_true_score"]]
    .mean()
    .reset_index()
)

# Sort by F1 score
model_summary = model_summary.sort_values("f1_score", ascending=False)
print("\nModel performance summary:")
print(model_summary)

# Top 5 models

# Get top 5
top5_models = model_summary.head(5)

# Remove 5th model (index 4)
top5_models = top5_models.drop(index=4)

# Get 13th model (Cyto3)
cyto3_model = model_summary.iloc[12]

# Add Cyto3
top5_models = pd.concat([top5_models, cyto3_model.to_frame().T], ignore_index=True)

print("\nModified Top Models (with Cyto3 instead of 5th):")
print(top5_models)
print("\nTop 5 Models:")
print(top5_models)

# Save Top 5
top5_models.to_csv("/home/qbx911/datadir/top5_models.csv", index=False)
print("Top 5 models saved.")

# Plot
import seaborn as sns
sns.set(style="whitegrid")

# Filter df to only top 5 models
top5_model_names = top5_models["model"].tolist()
df_top5 = df[df["model"].isin(top5_model_names)]

plt.figure(figsize=(10,6))

# Stripplot
sns.stripplot(
    x="model",
    y="f1_score",
    data=df_top5,
    hue="image_name",
    jitter=True,
    size=8
)
plt.ylim(0,1)
plt.xlabel("Top 5 Models")
plt.ylabel("F1 Score")
plt.title("Top 5 Models – Per-Image F1 Scores")
plt.xticks(rotation=45)
plt.legend(title="Image", bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.show()

# Boxplot
sns.boxplot(
    x="model",
    y="f1_score",
    data=df_top5,
    showfliers=False
)
sns.stripplot(
    x="model",
    y="f1_score",
    data=df_top5,
    color="black",
    jitter=True,
    size=6,
    alpha=0.7
)
plt.ylim(0,1)
plt.xlabel("Main 5 Models")
plt.ylabel("F1 Score")
plt.title("Main 5 Models – F1 Score Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
