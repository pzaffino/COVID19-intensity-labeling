#!/usr/bin/env python

import numpy as np
import SimpleITK as sitk
import scipy.ndimage

def extract_only_lungs_islands(thr_img):
    """
    Extract only lung islands from patient's binary image
    """

    # Create final mask
    final_mask = np.zeros_like(thr_img, dtype=np.uint8)

    # Compute islands
    label_im, nb_labels = scipy.ndimage.label(thr_img)
    sizes = scipy.ndimage.sum(thr_img, label_im, range(nb_labels + 1))

    # investigate each island
    for i in range(nb_labels):

        # discard small islands
        if sizes[i] < 5.0e5:
            continue

        # Check if island is background (bbox overlapping with image corner)
        img_coords = np.zeros_like(thr_img, dtype=np.uint8)
        img_coords[label_im==i]=1
        coords = bbox(img_coords, margin=0)

        if (coords[2] != 0 and coords[4]!=0 and
           coords[3] != thr_img.shape[1]-1 and coords[5] != thr_img.shape[2]-1): # non background, set as lung

            final_mask[img_coords==1]=1

    return final_mask

def bbox(img, margin=20):
    """
    Compute bounding box of a binary mask and add a maring (only in axial plane).
    """

    coords=[0,img.shape[0],0,img.shape[1],0,img.shape[2]]

    # i
    for i in range(img.shape[0]):
        if 1 in img[i,:,:]:
            coords[0]=i
            break
    for i in range(img.shape[0]-1,-1,-1):
        if 1 in img[i,:,:]:
            coords[1]=i
            break
    # j     
    for j in range(img.shape[1]):
        if 1 in img[:,j,:]:
            coords[2]=j - margin
            break
    for j in range(img.shape[1]-1,-1,-1):
        if 1 in img[:,j,:]:
            coords[3]=j + margin
            break
    # k
    for k in range(img.shape[2]):
        if 1 in img[:,:,k]:
            coords[4]=k - margin
            break
    for k in range(img.shape[2]-1,-1,-1):
        if 1 in img[:,:,k]:
            coords[5]=k + margin
            break

    assert coords[0] >= 0 and coords[2] >= 0 and coords[4] >= 0
    assert coords[1] <= img.shape[0]-1 and coords[3] <= img.shape[1]-1 and coords[5] <= img.shape[2]-1

    return coords


def binary_closing_sitk(img_np, radius_list):
    """
    SimpleITK much faster and less compute-intesive than skimage
    """

    img_sitk = sitk.GetImageFromArray(img_np)

    for radius in radius_list:
        print("  radius %d" % (radius))
        img_sitk = sitk.BinaryMorphologicalClosing(img_sitk, radius)

    return sitk.GetArrayFromImage(img_sitk).astype(np.uint8)

def read_image(image_fn):
    """
    Read CT image.
    """

    ct_sitk = sitk.ReadImage(image_fn)
    ct = sitk.GetArrayFromImage(ct_sitk)

    ct[ct<-1000]=-1000

    return ct_sitk, ct

def threshold_image(ct, intensity_thr=-155):
    """
    Execute a threshold based segmentation and fill holes
    """

    thr_img = np.zeros_like(ct, dtype=np.uint8)
    thr_img[ct>=intensity_thr]=1
    thr_img = 1 - thr_img
    thr_img = scipy.ndimage.binary_opening(thr_img, iterations=3)

    return thr_img

def close_lungs_mask(lungs_mask):
    """
    Close lungs binary mask.
    """

    # Do bounding box (to sepped up morph filters)
    coords = bbox(lungs_mask)
    bb_lungs_mask = lungs_mask[coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]]

    # Binary closing
    closed_bb_lung_mask = binary_closing_sitk(bb_lungs_mask, [30, 20])

    assert closed_bb_lung_mask.sum() > 1000

    # Undo bounding box
    closed_lung_mask = np.zeros_like(lungs_mask, dtype=np.uint8)
    closed_lung_mask[coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]] = closed_bb_lung_mask

    return closed_lung_mask

