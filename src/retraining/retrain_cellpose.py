"""
Retrain the selected Cellpose model using train_seg.
"""
# retrain_cellpose.py

import os
import random
import numpy as np
from cellpose import models, train, io
from skimage.io import imread
import torch
import pickle

DATA_DIR = "/home/qbx911/datadir/retrain/train_and_test"
MODEL_OUT_DIR = "/home/qbx911/datadir/cellpose_models"
PRETRAINED_MODEL = "/home/qbx911/datadir/cellpose_models/cytotorch_1"

use_gpu = torch.cuda.is_available()
random.seed(911)
np.random.seed(911)

io.logger_setup()

# Image loading
all_images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_img.tiff")])

# Fixed-random test set
test_imgs = random.sample(all_images, 3)
remaining = [img for img in all_images if img not in test_imgs]


def load_data(img_list):
    images = []
    masks = []
    for fname in img_list:
        img = imread(os.path.join(DATA_DIR, fname))
        mask = imread(os.path.join(DATA_DIR, fname.replace("_img", "_mask")))
        images.append(img[..., None])
        masks.append(mask)
    return images, masks

test_images, test_masks = load_data(test_imgs)

# Training set sizes
train_sizes = [3, 6, 9]

for train_size in train_sizes:

    train_imgs = remaining[:train_size]
    train_images, train_masks = load_data(train_imgs)

    model_name = f"cytotorch_1_retrain_{train_size}"
    print(f"\nTraining model: {model_name}")

    model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=PRETRAINED_MODEL
    )

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images,
        test_labels=test_masks,
        channels=[0, 0],
        channel_axis=-1,
        learning_rate=1e-5,
        weight_decay=0.1,
        n_epochs=200,
        model_name=model_name
    )

    with open(f"{MODEL_OUT_DIR}/{model_name}_losses.pkl", "wb") as f:
        pickle.dump((train_losses, test_losses), f)
