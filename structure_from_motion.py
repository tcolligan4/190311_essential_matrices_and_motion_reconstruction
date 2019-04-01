import numpy as np
import matplotlib.pyplot as plt
import cv2
import piexif
from camera_utils import sfm
from scipy.optimize import leastsq
from glob import glob


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

def u_to_x(u, im, f_im):
    ''' Converts camera coordinates to
    generalized image coordinates. '''
    h, w, d = im.shape
    exif = piexif.load(f_im)
    f = exif['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm]/36*w
    cu = w // 2
    cv = h // 2
    K_cam = np.array([[f,0,cu],[0,f,cv],[0,0,1]])
    K_inv = np.linalg.inv(K_cam)
    x = u @ K_inv.T
    return x, K_cam

def triangulate(P0,P1,x1,x2):
    # P0,P1: projection matrices for each of two cameras/images
    # x1,x1: corresponding points in each of two images (If using P that has been scaled by K, then use camera
    # coordinates, otherwise use generalized coordinates)
    A = np.array([[P0[2,0]*x1[0] - P0[0,0], P0[2,1]*x1[0] - P0[0,1], P0[2,2]*x1[0] - P0[0,2], P0[2,3]*x1[0] - P0[0,3]],
                  [P0[2,0]*x1[1] - P0[1,0], P0[2,1]*x1[1] - P0[1,1], P0[2,2]*x1[1] - P0[1,2], P0[2,3]*x1[1] - P0[1,3]],
                  [P1[2,0]*x2[0] - P1[0,0], P1[2,1]*x2[0] - P1[0,1], P1[2,2]*x2[0] - P1[0,2], P1[2,3]*x2[0] - P1[0,3]],
                  [P1[2,0]*x2[1] - P1[1,0], P1[2,1]*x2[1] - P1[1,1], P1[2,2]*x2[1] - P1[1,2], P1[2,3]*x2[1] - P1[1,3]]])
    u,s,vt = np.linalg.svd(A)
    return vt[-1]


class Camera(object):

    def __init__(self, cam_mat):
        self.rot_mat = cam_mat[:, :3]
        self.trans_vect = cam_mat[:, 3]
        self.trans_vect = np.expand_dims(self.trans_vect, 1)

    def predict(self, uv):
        return np.hstack((self.rot_mat, self.trans_vect)) @ uv.T

    def fit_func(self, x, trans_vect):
        self.trans_vect = trans_vect
        return self.predict(x)
    
    def err_func(self, trans_vect, uv, real_world_coords, fit_func, err):
        out = fit_func(uv, trans_vect)
        return real_world_coords - out

    def estimate_pose(self, X_gcp, u_gcp):
        err = np.ones(X_gcp.shape)
        out = leastsq(self.err_func, self.trans_vect, args=(X_gcp, u_gcp, self.fit_func, err), full_output=1)
        self.trans_vect = out[0]
        return out[0]
     

if __name__ == '__main__':

    pth = 'pictures/helmet/'
    out = []
    files = [f for f in glob(pth + "*.JPG")]
    files = sorted(files)
    i = 0
    first = True
    P_Nc = None
    f1 = files[i] 
    f2 = files[i+1] 
    f3 = files[i+2]
    i1 = cv2.imread(f1)
    i2 = cv2.imread(f2)
    i3 = cv2.imread(f3)
    # First two image
    u1, u2 = compute_sift_and_match(i1, i2)
    x1, k_cam1 = u_to_x(u1, i1, f1)
    x2, k_cam2 = u_to_x(u2, i2, f2)
    E, inliers = cv2.findEssentialMat(x1[:,:2],x2[:,:2],np.eye(3),method=cv2.RANSAC,threshold=1e-3)
    inliers = inliers.ravel().astype(bool)
    n_in, R, t, _ = cv2.recoverPose(E, x1[inliers,:2], x2[inliers,:2])
    P_1 = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0]])
    P_2 = np.hstack((R,t))
    P_1c = k_cam1 @ P_1
    P_2c = k_cam2 @ P_2
    # print(P_2c.shape)
    # c2 = Camera(P_2c)
    # x1_h = np.ones((x1.shape[0], x1.shape[1]+1))
    # print(x1.shape)
    # print(x1_h.shape)
    # x1_h[:, :x1.shape[1]] = x1
    # c2.predict(x1_h[inliers])
    # for xx1, xx2 in zip(u1[inliers, :2], u2[inliers, :2]):
    #     a = triangulate(P_1c, P_2c, xx1, xx2)
    #     out.append(a)

    # Last two images
    u2_p, u3 = compute_sift_and_match(i2, i3)
    x2_p, k_cam2_p = u_to_x(u2_p, i2, f2)
    x3, k_cam3 = u_to_x(u3, i3, f3)
    E, inliers = cv2.findEssentialMat(x2_p[:,:2],x3[:,:2],np.eye(3),method=cv2.RANSAC,threshold=1e-3)
    inliers = inliers.ravel().astype(bool)
    n_in, R, t, _ = cv2.recoverPose(E, x2_p[inliers,:2], x3[inliers,:2])
    P_2_p = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0]])
    P_3 = np.hstack((R,t))
    P_2_pc = k_cam2_p @ P_2_p
    P_3c = k_cam3 @ P_3

    print(P_2c.shape, P_3c.shape)

    initial_pose_guess = P_2c @ P_3c.T
    print(initial_pose_guess.shape)
