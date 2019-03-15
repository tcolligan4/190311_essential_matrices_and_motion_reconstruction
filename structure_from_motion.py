from camera_utils import Camera, sfm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import piexif


def compute_sift_and_match(i1, i2):
    h, w, d = i1.shape
    sift = cv2.xfeatures2d.SIFT_create()

    kp1,des1 = sift.detectAndCompute(i1, None)
    kp2,des2 = sift.detectAndCompute(i2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
        
    u1 = []
    u2 = []

    for m in good:
        u1.append(kp1[m.queryIdx].pt)
        u2.append(kp2[m.trainIdx].pt)
        
    u1 = np.array(u1)
    u2 = np.array(u2)
    u1 = np.c_[u1,np.ones(u1.shape[0])]
    u2 = np.c_[u2,np.ones(u2.shape[0])]

    return u1, u2

def u_to_x(u, im):
    ''' Converts camera coordinates to
    generalized image coordinates. '''
    h, w, d = I_1.shape
    exif = piexif.load('pens_1.jpg')
    f = exif['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm]/36*w
    cu = w // 2
    cv = h // 2
    K_cam = 

