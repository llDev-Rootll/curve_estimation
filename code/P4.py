import numpy as np
from numpy import linalg as LA
from utils import SingleValueDecompositon
import cv2

def calcHomography(P1, P2):
    """Calculates the homography matrix between set of src and dst points P1, and P2

    Args:
        P1 (list or numpy array): Set of source Points
        P2 (list or numpy array): Set of destination Points

    Returns:
        _type_: Returns the homograph matrix of numpy array type
    """
    P1 = np.array(P1)
    P2 = np.array(P2)
    x = P1[:,0]
    y = P1[:,1]
    xp = P2[:,0]
    yp = P2[:,1]

    A=[]
    # Generate A matrix for Ax=0 formulation
    for i in range(4):
        A.append(np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]]))
        A.append(np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]]))
    A = np.array(A)
    # Compute the singular value decompostion of A
    svd = SingleValueDecompositon()
    U, S, V = svd.SVD(A)
    # Least square approximate of x corresponds to the last column of V
    # Reshape and divide each element by H33 to obtain the final matrix
    H = V[:,V.shape[1]-1].reshape(3,3)
    H = H/H[2,2]
    return H
if __name__ == "__main__":

    
    P1 = np.array([[5, 5], [150, 5], [150, 150], [5, 150]])
    P2 = np.array([[100, 100], [200, 80], [220, 80], [100, 200]])
    H = calcHomography(P1,P2)
    h, _ = cv2.findHomography(P1, P2)
    print("Homograph Matrix: ", H)
    if np.allclose(H, h):
        print("Verified by OpenCV findHomography function")
    else:
        print("Estimated Homography Matrix doesn't match exactly with OpenCV estiamtion")