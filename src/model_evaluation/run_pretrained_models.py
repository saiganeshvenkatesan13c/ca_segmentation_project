"""
Run segmentation using pretrained Cellpose models.
Supports Methods 2.2 and Results 3.2.
"""

import os
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imsave
from cellpose import models
import torch
import numpy as np

TEST_IMG_DIR = "/home/qbx911/datadir/Image_mask_pairs"
GT_MASK_DIR  = "/home/qbx911/datadir/Image_mask_pairs"
CELLPOSE_MODELS_DIR = "/home/qbx911/datadir/cellpose_models"
OUTPUT_DIR = "/home/qbx911/datadir/cellpose_results_masks"

images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith("_image.tiff")]

CELLPOSE_MODELS = sorted([
    os.path.join(CELLPOSE_MODELS_DIR, f)
    for f in os.listdir(CELLPOSE_MODELS_DIR)
    if os.path.isdir(os.path.join(CELLPOSE_MODELS_DIR, f)) or os.path.isfile(os.path.join(CELLPOSE_MODELS_DIR, f))
]) #only takes all files and folders

use_gpu = torch.cuda.is_available()
print(f"GPU: {use_gpu}")

def iou_metric(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
    return intersection / union if union > 0 else 0
    # calculation of intersection over union value between predicted mask and ground truth

results = []

for model_path in CELLPOSE_MODELS:
    
    model_name = os.path.basename(model_path)
    print(f"\nCellpose model: {model_name}")

    # credits to Thomas Hamelryck for making me use try//except
    try:
        model = models.CellposeModel(pretrained_model=model_path, gpu=use_gpu)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        continue

    model_output_dir = os.path.join(OUTPUT_DIR, model_name)
    
    for img_name in tqdm(images, desc=f"Segmenting with {model_name}"):
        mask_name = img_name.replace("_image", "_mask")
        img_path  = os.path.join(TEST_IMG_DIR, img_name)
        mask_path = os.path.join(GT_MASK_DIR, mask_name)

        try:
            img     = imread(img_path)
            gt_mask = imread(mask_path)
        except Exception as e:
            print(f"Error reading {img_name} or {mask_name}: {e}")
            continue

        try:
            res = model.eval(img, diameter=None) # cellpose segmentation model
            if len(res) == 4: # number of outputs depend on cellpose model version
                masks, flows, styles, diams = res
            else:
                masks, flows, styles = res
                diams = None
        except Exception as e:
            print(f"Error segmenting {img_name} with {model_name}: {e}")
            continue

        output_mask_path = os.path.join(
            model_output_dir,
            img_name.replace("_image.tiff", "_pred_mask.tiff")
        )

        imsave(
            output_mask_path,
            masks.astype(np.uint16),
            check_contrast=False
        )

        iou = iou_metric(masks, gt_mask)
        
        results.append({
            "image_name": img_name,
            "model": model_name,
            "iou": iou,
            "pred_mask": masks,
            "gt_mask": gt_mask
        })

df = pd.DataFrame(results)
df.to_pickle("/home/qbx911/datadir/cellpose_results.pkl")
