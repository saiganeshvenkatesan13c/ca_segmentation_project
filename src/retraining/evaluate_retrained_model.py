"""
Evaluate pretrained vs retrained model performance.
"""

-----------------------------------------------------------------------------------------

# Performance comparison after retraining

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from cellpose import models
from stardist.matching import matching
import torch

VAL_IMG_DIR = "/home/qbx911/datadir/retrain/validation/Images"
VAL_MASK_DIR = "/home/qbx911/datadir/retrain/validation/Masks"

MODELS = {
    "pretrained": "/home/qbx911/datadir/cellpose_models/cytotorch_1",
    "retrained": "/maps/projects/dan1/people/qbx911/retrain/models/cytotorch_1_retrain_9"
}

use_gpu = torch.cuda.is_available()
results = []

image_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith("_image.tiff")])

for model_label, model_path in MODELS.items():

    model = models.CellposeModel(
        pretrained_model=model_path,
        gpu=use_gpu
    )

    for fname in tqdm(image_files, desc=f"Evaluating {model_label}"):

        img = imread(os.path.join(VAL_IMG_DIR, fname))
        gt  = imread(os.path.join(
            VAL_MASK_DIR, fname.replace("_image", "_mask"))
        )

        out = model.eval(img, diameter=None)

        if len(out) == 4:
            masks, _, _, _ = out
        else:
            masks, _, _ = out

        
        
        match = matching(
            gt.astype(np.int32),
            masks.astype(np.int32),
            thresh=0.5,
            report_matches=False
        )

        results.append({
            "model": model_label,
            "image": fname,
            "f1": match.f1,
            "precision": match.precision,
            "recall": match.recall,
            "mean_true_score": match.mean_true_score
        })

df = pd.DataFrame(results)
df.to_csv("results_fig3_pretrained_vs_retrained.csv", index=False)

print(df.groupby("model")[["f1", "precision", "recall", "mean_true_score"]].mean())

----------------------------------------------------------------------------------------

# Plotting linked slope graph

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_fig3_pretrained_vs_retrained.csv")

metrics = ["f1", "precision", "recall", "mean_true_score"]
titles = ["F1", "Precision", "Recall", "Mean true score"]

fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)

for ax, metric, title in zip(axes, metrics, titles):

    pivot = df.pivot(index="image", columns="model", values=metric)

    for _, row in pivot.iterrows():
        ax.plot(
            ["pretrained", "retrained"],
            [row["pretrained"], row["retrained"]],
            marker="o",
            color="black",
            alpha=0.8
        )

    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)

    import os
    print(os.getcwd())
    
plt.suptitle("Pretrained vs Retrained Model Performance (Per Validation Image)")
plt.tight_layout()
plt.savefig("figure_3_slope_all_metrics.png", dpi=300)

--------------------------------------------------------------------------------

# Representative image demonstration (qualitative)

import numpy as np
from skimage.io import imsave

STACKS = [
    "/home/qbx911/datadir/representative_retraining/G03.tiff",
    "/home/qbx911/datadir/representative_retraining/G04.tiff"
]

OUT_DIR = "/home/qbx911/datadir/representative_retraining/output_masks"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "pretrained": "/home/qbx911/datadir/cellpose_models/cytotorch_1",
    "retrained": "/home/qbx911/datadir/cellpose_models/cytotorch_1_retrain_9"
}

for label, model_path in MODELS.items():

    model = models.CellposeModel(
        pretrained_model=model_path,
        gpu=use_gpu
    )

    for stack_path in STACKS:

        stack_name = os.path.basename(stack_path).replace(".tiff", "")
        stack = imread(stack_path)

        sliced = stack[1:]
        proj = np.max(sliced, axis=0)

        masks, _, _ = model.eval(proj, diameter=None)

        out_path = os.path.join(
            OUT_DIR,
            f"{stack_name}_{label}_mask.tiff"
        )

        imsave(out_path, masks.astype("uint16"), check_contrast=False)

-----------------------------------------------------------------------------------------

# Effect of training size

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("results_fig6_training_size.csv")

train_sizes = sorted(df["train_size"].unique())
x = np.arange(len(train_sizes))

plt.figure(figsize=(6, 5))

# Mean bars
means = df.groupby("train_size")["f1"].mean()
plt.bar(x, means.values)

# Individual datapoints
for i, size in enumerate(train_sizes):
    vals = df[df["train_size"] == size]["f1"]
    plt.scatter(
        np.full(len(vals), x[i]),
        vals,
        color="black",
        zorder=10
    )

plt.xticks(x, train_sizes)
plt.xlabel("Training set size (images)")
plt.ylabel("F1 score")
plt.title("Effect of Training Set Size on Segmentation Performance")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("figure_6_training_size.png", dpi=300)
plt.show()

-----------------------------------------------------------------------------------------

# Training & validation loss curves

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

loss_file = "/home/qbx911/datadir/cellpose_models/cytotorch_1_retrain_9_losses.pkl"

with open(loss_file, "rb") as f:
    train_losses, val_losses = pickle.load(f)

epochs = np.arange(len(train_losses))

peak_idx = np.where(val_losses > 0)[0] # Find the indices where validation has non-zero peaks

val_interp = np.interp(epochs, peak_idx, val_losses[peak_idx]) # Interpolate linearly between the peaks

plt.plot(epochs, train_losses, label="Training loss")
plt.plot(epochs, val_interp, label="Validation loss (connected peaks)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss (cytotorch_1 retraining)")
plt.legend()
plt.tight_layout()
plt.savefig("figure_8_loss_curves.png")

-----------------------------------------------------------------------------------------
