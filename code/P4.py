import numpy as np
from numpy import linalg as LA
from utils import SingleValueDecompositon


def calcHomography(P1, P2):
    P1 = np.array(P1)
    P2 = np.array(P2)
    x = P1[:,0]
    y = P1[:,1]
    xp = P2[:,0]
    yp = P2[:,1]
    # print(x, y, xp, yp)
    A=[]
    for i in range(4):
        A.append(np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]]))
        A.append(np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]]))
    A = np.array(A)

    svd = SingleValueDecompositon()
    U, S, V = svd.SVD(A)
    H = V[:,V.shape[1]-1].reshape(3,3)
    H = H/H[2,2]
    return H
if __name__ == "__main__":

    # a = np.array([[3,2,2], [2,3,-2]])
    # print(a.shape)

    # svd = SingleValueDecompositon()

    # print(svd.SVD(a))
    P1 = [[5, 5], [150, 5], [150, 150], [5, 150]]
    P2 = [[100, 100], [200, 80], [220, 80], [100, 200]]
    print(calcHomography(P1,P2))
    
