from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np
import skimage.draw


def overlap_measure(A, B, resolution=250):
    # implemented according to ginneken 2002 equation (10), performance = TP/(TP+FP+FN)
    # performance = (intersection of shape 1 and 2) / (union of shape 1 and 2)
    # resolution determines the grid resolution (resolution * resolution) that is used to calculate overlap
    A = np.asanyarray(A)
    B = np.asanyarray(B)

    A_x_min, A_y_min = A.min(axis=0)
    A_x_max, A_y_max = A.max(axis=0)

    B_x_min, B_y_min = B.min(axis=0)
    B_x_max, B_y_max = B.max(axis=0)

    AB_min = np.array([np.min([A_x_min, B_x_min]), np.min([A_y_min, B_y_min])])
    AB_max = np.array([np.max([A_x_max, B_x_max]), np.max([A_y_max, B_y_max])])

    AB_size = AB_max - AB_min

    A_unit = (A - AB_min) / AB_size
    B_unit = (B - AB_min) / AB_size

    A_unit *= resolution
    B_unit *= resolution

    A_polygon = np.zeros((resolution, resolution))
    B_polygon = np.zeros((resolution, resolution))

    A_rr, A_cc = skimage.draw.polygon(A_unit[:, 1], A_unit[:, 0])
    B_rr, B_cc = skimage.draw.polygon(B_unit[:, 1], B_unit[:, 0])

    A_polygon[A_rr, A_cc] = 1
    B_polygon[B_rr, B_cc] = 1

    AB_union = np.logical_or(A_polygon, B_polygon)
    AB_intersection = np.logical_and(A_polygon, B_polygon)

    return np.count_nonzero(AB_intersection) / np.count_nonzero(AB_union)
