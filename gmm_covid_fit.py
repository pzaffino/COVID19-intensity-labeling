#!/usr/bin/env python

import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from sklearn import mixture

import sys, os
import joblib

import lungs_processing

"""
Usage: python gmm_covid_fit.py cases.txt model.joblib
"""

subsample = 4
n_init = 6

# Read case text file (1 CT path per line)
with open(sys.argv[1], "r") as f:
    cases = [os.path.abspath(s.strip()) for s in f.readlines()]

for i, case in enumerate(cases):

    print("Processing case %s (%d/%d)" % (case, i+1, len(cases)))

    # Read CT
    print("Reading file...")
    ct_fn = os.path.abspath(case)
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
    ct = ct.flatten()

    # Remove voxels that are not lungs (not useful for classification)
    print("Removing non lung voxels...")
    only_lungs = np.delete(ct, np.argwhere(closed_lungs_mask.flatten()==0))

    # Create or concatenate vector for GMM
    if i == 0:
        vector_for_gmm = only_lungs
    else:
        vector_for_gmm = np.hstack((vector_for_gmm, only_lungs))

# Print information about GMM vector
print("Printing information about GMM vector...")
print("  GMM vector has %d elements" % (vector_for_gmm.shape[0]))

# Run GMM fit
print("Running GMM fitting...")
gmm = mixture.GaussianMixture(n_components=5, n_init=n_init)
gmm.fit(vector_for_gmm[::subsample].reshape(-1,1))

# Save model
joblib.dump(gmm, sys.argv[2])

print("Done!")

