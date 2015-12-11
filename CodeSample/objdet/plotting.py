from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_prediction(image_scaled, prediction, patches_coordinates, predicted_patch_prob, overlap_threshold):
    fig, ax = plt.subplots()
    ax.imshow(image_scaled.T, cmap=matplotlib.cm.gray)
    for index, probability in enumerate(prediction[:, 1]):
        patch = patches_coordinates[index]
        if probability > overlap_threshold:
            print(probability)
            #plot patches with a probability above the threshold in blue
            #plot_patch(ax, patch, color="y", linewidth=1, alpha=0.5)
            
    if debug:
        print('predicted_patch_prob=')
        print(predicted_patch_prob)
    # plot the predicted patch in red
    plot_patch(ax, predicted_patch_prob, color="r", linewidth=1.5, alpha=1)


def plot_patch(ax, patch, color="b", linewidth=1.0, alpha=1.0):
    # copy first element of patch to the last place to draw the complete box
    patch_closed = np.concatenate([patch, patch[0, :][np.newaxis, :]])
    ax.plot(patch_closed[:, 0], patch_closed[:, 1], color=color, linewidth=linewidth, alpha=alpha)
    
def plot_nms_prediction(image_scaled, predicted_patch_nms):
    fig, ax = plt.subplots()
    ax.imshow(image_scaled.T, cmap=matplotlib.cm.gray)
    plot_patch(ax, predicted_patch_nms, color="g", linewidth=3, alpha=1)
