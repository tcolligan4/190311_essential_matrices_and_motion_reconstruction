from scipy.optimize import leastsq
import numpy as np

class Camera(object):
    '''Reduce the misfit between the projection of the GCPs and their identified location in the image.
       Adjust the pose s.t. the squared difference between the projection of the GCP and its location
       in the image is minimized.
       Project the GCPs onto the sensor.
       Recall: Guess at pose vector.
       Then, reduce the misfit between the GCP coordinates that we predicted
       using our guessed-at coordinates, and the true coordinates.'''
    
    def __init__(self, focal_length=None, sensor_x=None, sensor_y=None, pose=None):
        self.p = pose # Pose: x, y, z, phi, theta, psi
        self.focal_length = focal_length                   # Focal Length in Pixels
        self.sensor_x = sensor_x
        self.sensor_y = sensor_y
        
    def projective_transform(self, X):
        """  
        This function performs the projective transform on generalized coordinates in the camera reference frame.
        Expects x, y, z (non-generalized).
        """
        x = X[:, 0]/X[:, 2]
        y = X[:, 1]/X[:, 2]
        u = self.focal_length*x + self.sensor_x / 2
        v = self.focal_length*y + self.sensor_y / 2 # the coordinates that input intensities map to
        u = np.hstack(u)
        v = np.hstack(v)
        return u, v
    
    def rotational_transform(self, X):
        '''Expects non-homogeneous coordinates.'''
        if len(X.shape) < 2:
            X = np.reshape(X, (1, X.shape[0]))
        s = np.sin
        c = np.cos
        X_h = np.zeros((X.shape[0], X.shape[1]+1))
        X_h[:, :X.shape[1]] = X
        X_h[:, 3] = np.ones((X.shape[0]))
        X_cam = self.p
        phi = X_cam[3]
        theta = X_cam[4]
        p = X_cam[5]
        trans = np.mat(([1, 0, 0, -X_cam[0]], [0, 1, 0, -X_cam[1]], 
                          [0, 0, 1, -X_cam[2]], [0, 0, 0, 1]))
        r_yaw = np.mat(([c(phi), -s(phi), 0, 0], [s(phi), c(phi), 0, 0], [0, 0, 1, 0]))
        r_pitch = np.mat(([1, 0, 0], [0, c(theta), s(theta)], [0, -s(theta), c(theta)]))
        r_roll = np.mat(([c(p), 0, -s(p)], [0, 1, 0], [s(p), 0, c(p)]))
        r_axis = np.mat(([1, 0, 0], [0, 0, -1], [0, 1, 0]))
        C = r_axis @ r_roll @ r_pitch @ r_yaw @ trans
        Xt = C @ X_h.T
        return Xt.T
    
    def _func(self, p, x):
        self.p = p 
        X = self.rotational_transform(x)
        u, v = self.projective_transform(X)
        u = u.T
        v = v.T
        z = np.asarray(np.hstack((u, v)))
        return z.ravel()
    
    def _errfunc(self, p, x, y, func, err):
        xx = func(p, x)
        ss = y.ravel() - xx
        return ss
    
    def estimate_pose(self, X_gcp, u_gcp):
        """
        This function adjusts the pose vector such that the difference between the observed pixel coordinates u_gcp 
        and the projected pixels coordinates of X_gcp is minimized.
        """
        err = np.ones(X_gcp.shape)
        out = leastsq(self._errfunc, self.p, args=(X_gcp, u_gcp, self._func, err), full_output=1)
        self.p = out[0]
        return out[0]
    

def sfm(c1, c2, guess, u_gcp):
    '''Estimates world coordinates of a given point in an
       image given their image coordinates and a guess at
       the world coordinates. This requires two calibrated
       cameras. If we had more images from different positions, we'd 
       minimize the misfit between all of the predicted camera
       coordinates given a guess at the world coordinates. '''
    
    def func(x, c1, c2):
        '''Return u, v given guess at X, Y, Z'''
        xt_1 = c1.rotational_transform(x)
        xt_2 = c2.rotational_transform(x)
        u1, v1 = c1.projective_transform(xt_1)
        u2, v2 = c2.projective_transform(xt_2)
        u1 = u1.T
        u2 = u2.T
        v1 = v1.T
        v2 = v2.T
        z = np.asarray(np.hstack((u1, v1, u2, v2)))
        return z.ravel()

    def errfunc(p, c1, c2, y):
        xx = func(p, c1, c2)
        ss = y.ravel() - xx
        return ss
    
    out = leastsq(errfunc, guess, args=(c1, c2, u_gcp), full_output=1)
    
    return out[0]

if __name__ == '__main__':
    pass
