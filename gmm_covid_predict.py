#!/usr/bin/env python

import numpy as np
import SimpleITK as sitk
from sklearn import mixture

import sys, os
import joblib

import lungs_processing

"""
Usage: python ./gmm_covid_predict.py PRETRAINED_MODEL CT.nrrd
"""

# Read CT
print("Reading file...")
ct_fn = os.path.abspath(sys.argv[2])
ct_sitk, ct = lungs_processing.read_image(ct_fn)

# Create lung mask
print("Threshold masking...")
thr_img = lungs_processing.threshold_image(ct, -155)
print("Extracting only lungs islands...")
lungs_mask = lungs_processing.extract_only_lungs_islands(thr_img)
print("Closing mask...")
closed_lungs_mask = lungs_processing.close_lungs_mask(lungs_mask)

# Apply mask
ct[closed_lungs_mask==0]=-1000
ct_flatten = ct.flatten()

# Remove background
print("Removing non lung voxels...")
indexes_to_remove = np.argwhere(closed_lungs_mask.flatten()==0)
lungs = np.delete(ct_flatten, indexes_to_remove)

# Run GMM
print("Running GMM prediction...")
gmm = joblib.load(sys.argv[1])
gmm_labels = gmm.predict(lungs.reshape(-1,1)).reshape(lungs.shape)

# Make label values fixed
sorted_label = np.zeros_like(lungs, dtype=np.uint8)
sorted_gmm_means = np.argsort([i[0] for i in gmm.means_])

sorted_label[gmm_labels==[sorted_gmm_means[0]]]=1
sorted_label[gmm_labels==[sorted_gmm_means[1]]]=2
sorted_label[gmm_labels==[sorted_gmm_means[2]]]=3
sorted_label[gmm_labels==[sorted_gmm_means[3]]]=4
sorted_label[gmm_labels==[sorted_gmm_means[4]]]=5

# Restore background voxels
print("Restoring non lung voxels...")
indexes_to_leave = np.argwhere(closed_lungs_mask.flatten()==1)
indexes_to_leave_list = [i[0] for i in indexes_to_leave]

final_label = np.zeros_like(ct_flatten, dtype=np.uint8)

counter = 0
for i in indexes_to_leave_list:
    final_label[i] = sorted_label[counter]
    counter += 1

# Reshape array labels. From 1D to 3D
final_label = final_label.reshape(ct.shape)

# Write segmentation file
print("Writing file...")
final_label_fn = ct_fn.replace(".nrrd", "_GMM_LABELS.nii.gz")
final_label_sitk = sitk.GetImageFromArray(final_label)
final_label_sitk.CopyInformation(ct_sitk)
final_label_sitk = sitk.Cast(final_label_sitk, sitk.sitkUInt8)
sitk.WriteImage(final_label_sitk, final_label_fn, True)

print("Done!")

