from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np
import skimage
import skimage.feature
import sklearn.ensemble
import matplotlib
import matplotlib.pyplot as plt
from . import contours, plotting
from sklearn.base import BaseEstimator
from skimage import transform


class ObjectDetectorHOG(BaseEstimator):
    """
    Object detection using histogram of oriented gradients, based on a blog post by Adrian Rosebrock on November 10,
    2014. Available at: http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
    """

    def __init__(self, patch_size=(96, 96), step_size=10, scale_fraction=0.0625, overlap_threshold=0.4,
                 orientations=9, pixels_per_cell=(32, 32), cells_per_block=(3, 3)):
        self.patch_size = patch_size
        self.step_size = step_size
        self.scale_fraction = scale_fraction
        self.overlap_threshold = overlap_threshold
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y, debug=False):
        """
        Train/fit a random forest classifier on HOG features calculated for patches of a list of images.

        Parameters
        ----------
        X : list
            List of images for which the HOG patches will be computed.
        y : list
            List of correct bounding boxes of the object that needs to be detected.
        """
        # validation
        assert isinstance(self.patch_size, tuple), "patch_size should be a tuple"
        assert isinstance(self.step_size, int), "step_size should be an int"
        assert 0 <= self.scale_fraction <= 1, "scale_fraction should be between 0 and 1"
        assert 0 <= self.overlap_threshold <= 1, "overlap_threshold should be between 0 and 1"
        assert isinstance(self.orientations, int), "orientations should be an int"
        assert isinstance(self.pixels_per_cell, tuple), "pixels_per_cell should be a tuple"
        assert isinstance(self.cells_per_block, tuple), "cells_per_block should be a tuple"

        # initialize
        train_features = []
        labels = []

        # iterate over images and bounding boxes to obtain HOG features and overlap fractions for patches in the image.
        for image, bounding_box in zip(X, y):
            # scale image for faster computation time.
            image_scaled = self._rescale(image, self.scale_fraction)
            scaled_box = bounding_box * self.scale_fraction
            feature_vector, _, patch_overlaps = self._sliding_window_hog_patches(image_scaled, bounding_box=scaled_box,
                                                                                 debug=debug)
            train_features += feature_vector
            labels += patch_overlaps

        train_features = np.array(train_features)
        labels = np.array(labels)

        # create classification labels for patches based on the amount of overlap
        labels[labels > self.overlap_threshold] = 1
        labels[labels < 1] = 0
        # train random forest classifier
        self.classifier_ = sklearn.ensemble.RandomForestClassifier()
        self.classifier_.fit(train_features, labels)

    def predict(self, X, debug=False):
        """
        Make a prediction of the bounding box of the object in the input image, by classifying HOG features of patches
        of the input images.

        Parameters
        ----------
        X_test : array or list
            Image or list of images that has an object that needs to be detected.

        Returns
        -------
        predicted_patches : array, [n_images, 4 points, 2 coordinates]
            Coordinates of predicted patches.
        """
        # if there is only one input image make it list
        if np.asanyarray(X).ndim == 2:
            X = [X]

        predicted_patches_prob = []

        # iterate over images to obtain a prediction for each image
        for image in X:
            image_scaled = self._rescale(image, self.scale_fraction)
            # calculate HOG features for patches created by moving a sliding window over the image
            feature_vector, patches_coordinates, _ = self._sliding_window_hog_patches(image_scaled)
            # predict the probability of the two classes for all patches
            prediction = self.classifier_.predict_proba(np.array(feature_vector))

            # The patch with maximum probability of being the second class is the predicted patch
            index_max = np.argmax(prediction[:, 1])
            predicted_patch_prob = patches_coordinates[index_max]
            predicted_patches_prob.append(predicted_patch_prob)
            
            if debug:
                plotting.plot_prediction(image_scaled, prediction, patches_coordinates, predicted_patch_prob,
                                         self.overlap_threshold)
            
            # predicted_patch_nms: patch predicted using nms 
            predicted_patch_nms = self._non_max_suppression_fast(patches_coordinates, overlapThresh=self.overlap_threshold)
            predicted_patches_nms.append(predicted_patch_nms)
            
            plotting.plot_nms_prediction(image_scaled, predicted_patch_nms) # USE DIFFERENT COLORS

            

        return np.array(predicted_patches_prob) / self.scale_fraction#, np.array(predicted_patches_nms) / self.scale_fraction

    def score(self, X, y):
        """
        Calculate overlap between predicted bounding box and correct(manually selected) bounding box

        Parameters
        ----------
        X_test : array or list
            Image or list of images that has an object that needs to be detected.
        y : array, [n_boxes, n_points, 2]
            x- and y-coordinates of correct(manually selected) polygon of bounding box consisting of n_points

        Returns
        -------
        score : float
            mean score
        """
        # if there is only one input make it list
        if np.asanyarray(X).ndim == 2:
            X = [X]
        if np.asanyarray(y).ndim == 2:
            y = [y]

        X = np.asanyarray(X)
        y = np.asanyarray(y)
        
        predictions = self.predict(X)
        scores = []
        
        for predicted_box, correct_box in zip(predictions, y):
            overlap = contours.overlap_measure(correct_box, predicted_box)
            scores.append(overlap)

        return np.mean(scores)

    def _rescale(self, image, scale_factor):
        return transform.rescale(image, scale_factor, preserve_range=True).astype(image.dtype)

    def _sliding_window_hog_patches(self, image, bounding_box=None, debug=False):
        """
        HOG features are calculated for patches by moving a sliding window over the image. If a bounding box is given
        for all patches the overlap between that bounding box and the patches is returned.

        Parameters
        ----------
        image : array
            Image for which the HOG features are calculated
        bounding_box : array, [n_points, 2], optional
            x- and y-coordinates of correct polygon of bounding box consisting of n_points. Default is None
        debug : bool, optional
            If True, a visual debugging matplotlib figure is generated along the way. Default is False.

        Returns
        -------
        score : float
            overlap fraction between input polygons
        """
        patches_coordinates = []
        overlap_fractions = []
        feature_vector = []

        image_width, image_height = image.shape
        patch_width, patch_height = self.patch_size

        if debug:
            fig, ax = plt.subplots()
            ax.imshow(image.T, cmap=matplotlib.cm.gray)

        # Move over x and y coordinates to create a sliding window and calculate the HOG features for each window
        for x in range(0, image_width - patch_width - 1, self.step_size):
            for y in range(0, image_height - patch_height - 1, self.step_size):
                # calculate patch coordinates of sliding window using x and y coordinates and the patch size
                patch_coordinates = np.array([[x, y], [x + patch_width, y], [x + patch_width, y + patch_height],
                                              [x, y + patch_height]])
                patches_coordinates.append(patch_coordinates)

                # if there is a shape, overlap is also calculated
                if bounding_box is not None:
                    # determine overlap between current patch and the correct bounding box
                    patch_overlap_fraction = contours.overlap_measure(patch_coordinates, bounding_box,
                                                                      resolution=max(self.patch_size))
                    overlap_fractions.append(patch_overlap_fraction)

                # calculate hog feature vector of image data of current patch
                hog_vector = skimage.feature.hog(image[x: x + patch_width, y: y + patch_height],
                                                 orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                                                 cells_per_block=self.cells_per_block, visualise=False, normalise=True)
                feature_vector.append(hog_vector)

                if debug:
                    plotting.plot_patch(ax, patch_coordinates, color="r", linewidth=1, alpha=0.1)
                    if bounding_box is not None:
                        if patch_overlap_fraction > 0.2:
                            plotting.plot_patch(ax, patch_coordinates, color="y", linewidth=1, alpha=0.5)

        if debug:
            if bounding_box is not None:
                plotting.plot_patch(ax, bounding_box, color="g", linewidth=2, alpha=1)

        return feature_vector, np.array(patches_coordinates), overlap_fractions
    
    # Malisiewicz et al.
    def _non_max_suppression_fast(self, patches_coordinates, overlapThresh):
        # if there are no boxes, return an empty list
        #boxes = np.concatenate([boxes_tup, boxes_tup[0, :][np.newaxis, :]])
        #if len(boxes) == 0:
            #return []
 
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        #if boxes.dtype.kind == "i":
            #boxes = boxes.astype("float")
 
        # initialize the list of picked indexes	
        pick = []
        x1=[]
        x2=[]
        y1=[]
        y2=[]
 
        # grab the coordinates of the bounding boxes
        #x1.append(boxes[:,0])
        x1 = patches_coordinates[:,0]
        y1 = patches_coordinates[:,1]
        x2 = patches_coordinates[:,2]
        y2 = patches_coordinates[:,3]
 
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
 
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
 
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
 
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
 
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
 
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")
    
    def overlapping_area(detection_1, detection_2):
        '''
        Function to calculate overlapping area'si
        `detection_1` and `detection_2` are 2 detections whose area
        of overlap needs to be found out.
        Each detection is list in the format ->
        [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
        The function returns a value between 0 and 1,
        which represents the area of overlap.
        0 is no overlap and 1 is complete overlap.
        Area calculated from ->
        http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
        '''
        # Calculate the x-y co-ordinates of the 
        # rectangles
    
        x1_tl = detection_1[0]
        x2_tl = detection_2[0]
        x1_br = detection_1[0] + detection_1[3]
        x2_br = detection_2[0] + detection_2[3]
        y1_tl = detection_1[1]
        y2_tl = detection_2[1]
        y1_br = detection_1[1] + detection_1[4]
        y2_br = detection_2[1] + detection_2[4]
        # Calculate the overlapping Area
        x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
        y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
        overlap_area = x_overlap * y_overlap
        area_1 = detection_1[3] * detection_2[4]
        area_2 = detection_2[3] * detection_2[4]
        total_area = area_1 + area_2 - overlap_area
        return overlap_area / float(total_area)

    def nms(detections, threshold=.5):
        '''
        This function performs Non-Maxima Suppression.
        `detections` consists of a list of detections.
        Each detection is in the format ->
        [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
        If the area of overlap is greater than the `threshold`,
        the area with the lower confidence score is removed.
        The output is a list of detections.
        '''
        if len(detections) == 0:
            return []
        # Sort the detections based on confidence score
        detections = sorted(detections, key=lambda detections: detections[2],
                reverse=True)
        # Unique detections will be appended to this list
        new_detections=[]
        # Append the first detection
        new_detections.append(detections[0])
        # Remove the detection from the original list
        del detections[0]
        # For each detection, calculate the overlapping area
        # and if area of overlap is less than the threshold set
        # for the detections in `new_detections`, append the 
        # detection to `new_detections`.
        # In either case, remove the detection from `detections` list.
        for index, detection in enumerate(detections):
            for new_detection in new_detections:
                if overlapping_area(detection, new_detection) > threshold:
                    del detections[index]
                    break
            else:
                new_detections.append(detection)
                del detections[index]
        return new_detections
    
    if __name__ == "__main__":
        # Example of how to use the NMS Module
        detections = [[31, 31, .9, 10, 10], [31, 31, .12, 10, 10], [100, 34, .8,10, 10]]
        print("Detections before NMS = {}".format(detections))
        print("Detections after NMS = {}".format(nms(detections)))
    
    