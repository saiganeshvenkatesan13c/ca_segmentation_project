"""
Evaluate pretrained vs retrained model performance.
Supports Results 3.4.
"""

# Figure 3: Performance comparison after retraining

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
    "retrained": "/home/qbx911/datadir/cellpose_models/cytotorch_1_retrain_9"
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
        gt  = imread(
            os.path.join(
                VAL_MASK_DIR,
                fname.replace("_image", "_mask")
            )
        )

        masks, _, _ = model.eval(img, diameter=None)

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
print(df.groupby("model")[["f1", "precision", "recall"]].mean())



# Figure 4: Representative image demonstration (qualitative)

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


# Figure 6: Training set size experiment

random.seed(911)

DATA_DIR = "/home/qbx911/datadir/retrain/train_and_test"
images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_img.tiff")])

test_set = random.sample(images, 3)
remaining = [img for img in images if img not in test_set]

train_sets = {
    3: remaining[:3],
    6: remaining[:6],
    9: remaining[:9]
}

results = []

for train_size in train_sets.keys():

    model_path = f"/home/qbx911/datadir/cellpose_models/cytotorch_1_retrain_{train_size}"

    model = models.CellposeModel(
        pretrained_model=model_path,
        gpu=use_gpu
    )

    for fname in sorted(os.listdir(VAL_IMG_DIR)):
        img = imread(os.path.join(VAL_IMG_DIR, fname))
        gt  = imread(
            os.path.join(
                VAL_MASK_DIR,
                fname.replace("_image", "_mask")
            )
        )

        masks, _, _ = model.eval(img, diameter=None)
        match = matching(gt, masks, thresh=0.5, report_matches=False)

        results.append({
            "train_size": train_size,
            "image": fname,
            "f1": match.f1
        })

df = pd.DataFrame(results)
df.to_csv("results_fig6_training_size.csv", index=False)


# Figure 7: Augmentation experiment

MODELS = {
    "no_aug": "/home/qbx911/datadir/cellpose_models/cytotorch_1_retrain_9_noaug",
    "aug": "/home/qbx911/datadir/cellpose_models/cytotorch_1_retrain_9_aug"
}

results = []

for label, model_path in MODELS.items():

    model = models.CellposeModel(
        pretrained_model=model_path,
        gpu=use_gpu
    )

    for fname in tqdm(os.listdir(VAL_IMG_DIR), desc=label):
        if not fname.endswith("_image.tiff"):
            continue

        img = imread(os.path.join(VAL_IMG_DIR, fname))
        gt  = imread(
            os.path.join(
                VAL_MASK_DIR,
                fname.replace("_image", "_mask")
            )
        )

        masks, _, _ = model.eval(img, diameter=None)
        match = matching(gt, masks, thresh=0.5, report_matches=False)

        results.append({
            "augmentation": label,
            "f1": match.f1
        })

df = pd.DataFrame(results)
df.to_csv("results_fig7_augmentation.csv", index=False)


# Figure 8: Training and validation loss curves

import matplotlib.pyplot as plt
import pickle

with open(
    "/home/qbx911/datadir/cellpose_models/cytotorch_1_retrain_9_losses.pkl",
    "rb"
) as f:
    train_losses, val_losses = pickle.load(f)

plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("figure_8_loss_curves.png")
