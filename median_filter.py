#!/usr/bin/env python

import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import sys, os

print("Running median filter...")

# Read file
mask_fn= os.path.abspath(sys.argv[1])
mask_sitk = sitk.ReadImage(mask_fn)
mask = sitk.GetArrayFromImage(mask_sitk)

# Run median filter and round to the closest integer
filtered_mask = scipy.ndimage.median_filter(mask, 4)
rounded_filtered_mask = np.rint(filtered_mask).astype(np.uint8)

# Write filtered mask
final_mask_fn = mask_fn.replace(".nii.gz", "_MEDIAN.nii.gz")
final_mask_sitk = sitk.GetImageFromArray(rounded_filtered_mask)
final_mask_sitk.CopyInformation(mask_sitk)
final_mask_sitk = sitk.Cast(final_mask_sitk, sitk.sitkUInt8)
sitk.WriteImage(final_mask_sitk, final_mask_fn, True)

