from src.trackformer.util.misc import NestedTensor
from typing import Any, Dict, List, Optional, Union, Tuple

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import torch
import numpy as np


def normalize_numpy_image(img_: np.ndarray,
                          min_val: float = None,
                          max_val: float = None) -> np.ndarray:
    if min_val is None:
        min_val = np.min(img_)
    if max_val is None:
        max_val = np.max(img_)

    img_ -= min_val
    img_ /= (max_val - min_val)

    return img_


def torch_tensor_to_numpy_image_0_1(img_torch: torch.Tensor) -> np.ndarray:
    img_numpy = np.transpose(img_torch.detach().cpu().numpy(), [1, 2, 0])
    return normalize_numpy_image(img_numpy)


def visualize_batch(samples: NestedTensor,
                    targets: Tuple[dict],
                    fig_id: int = None,):
    imgs, mask = samples.decompose()
    imgs_np = np.transpose(imgs.numpy(), [0, 2, 3, 1])
    imgs_np = normalize_numpy_image(imgs_np)

    batch_size = len(targets)
    prev_imgs = [torch_tensor_to_numpy_image_0_1(target['prev_image']) for target in targets]

    if 'prev_prev_image' in targets[0]:
        prev_prev_imgs = [torch_tensor_to_numpy_image_0_1(target['prev_image']) for target in targets]
    else:
        prev_prev_imgs = None

    for ii in range(len(targets)):
        imgs = []
        targets_now = []
        if 'prev_prev_image' in targets[0]:
            imgs.append(prev_prev_imgs[ii])
            targets_now.append(targets[ii]['prev_prev_target'])

        imgs.extend([prev_imgs[ii], imgs_np[ii]])
        targets_now.extend([targets[ii]['prev_target'], targets[ii]])
        if fig_id is None or fig_id <= 0:
            plt.figure()
        else:
            plt.figure(fig_id + ii * 2)

        all_id = []
        for kk in range(len(targets_now)):
            all_id = all_id + targets_now[kk]['track_ids'].numpy().tolist()
        all_id = list(set(all_id))
        all_id.sort()
        num_tracks = len(all_id)
        colors = plt.cm.get_cmap('tab20c', num_tracks)
        col_dict = {}
        for kk, id__ in enumerate(all_id):
            col_dict[id__] = colors(kk)

        for kk in range(len(imgs)):
            img_ = imgs[kk]
            targ_ = targets_now[kk]
            plt.subplot(1, len(imgs), kk + 1)
            plt.imshow(img_)
            plt.title(targ_['image_id'][0])
            img_sz = targ_['size'].numpy()
            img_sz__img_sz_x_y = np.concatenate((img_sz[::-1], img_sz[::-1]))
            for ii_box in range(targ_['boxes'].shape[0]):
                box_now_nor = targ_['boxes'][ii_box].numpy()
                box_now = box_now_nor * img_sz__img_sz_x_y
                track_id = int(targ_['track_ids'][ii_box].numpy())
                x_c, y_c, dx, dy = box_now
                x = x_c - dx / 2
                y = y_c - dy / 2
                col_now = col_dict[track_id]
                plt.gca().add_patch(plt.Rectangle((x, y), dx, dy,
                                                  fill=False,
                                                  edgecolor=col_now,  # Contour color (black)
                                                  linewidth=2,  # Thickness of contour
                                                  label=f"Rect {track_id}"))
                plt.text(x, y + 15, f'{track_id}', fontsize=12)
        plt.show()

    return
