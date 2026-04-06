"""
Evaluate stack projection and slicing strategies.
Supports Methods 2.1 and Results 3.1 in the report.
"""

import numpy as np
from napari.layers import Image

layer_names = [
    "LST XAM11-3 ARC-AM26 14-08-2025_20250814_105533_D08",
    "LST XAM11-3 ARC-AM32B 14-08-2025_20250814_124628_C07",
    "XAM13.2 AM42 tanycytes_20260115_120254_C04",
    "XAM13.2 AM42 tanycytes_20260115_120254_D09",
    "XAM13.2 AM43 tanycytes_20260115_141811_C08",
    "XAM13.2 AM43 tanycytes_20260115_141811_D10",
]

for name in layer_names:
    layer = viewer.layers[name]
    data = layer.data

    sliced = data[1:]

    nr_name = f"{name}_nr" # nuclear frame removed
    nr_layer = viewer.add_image(sliced, name=nr_name)
    nr_layer.contrast_limits = (1, 200) # change contrast

    viewer.layers.remove(layer) # remove original layer with nuclear frame

    viewer.add_image(
        np.mean(sliced, axis=0),
        name=f"{nr_name}_mean"
    ) # mean projection

    viewer.add_image(
        np.max(sliced, axis=0),
        name=f"{nr_name}_max"
    ) # max projection

for layer in viewer.layers:
    if isinstance(layer, Image):
        layer.contrast_limits = (1, 200) # change contrast for all loaded layers
