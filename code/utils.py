import numpy as np
from numpy import linalg as LA

def covariance(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_diff = x - x_mean
    y_diff = y - y_mean
    cov_xy = np.sum(np.multiply(x_diff, y_diff))/(x.size-1)

    return cov_xy

class SingleValueDecompositon:
    
    def SVD(self, A):
        A = np.array(A)

        M1 = np.dot(A,A.T)

        M2 = np.dot(A.T,A)
        
        eig_val1, eig_vec1 = LA.eig(M1)
        sg_val = np.flip(np.sort(np.sqrt(eig_val1)))
        # U = eig_vec1[:,np.flip(np.argsort(eig_val1))]

        eig_val2, eig_vec2 = LA.eig(M2)
        # print(eig_vec2)
        V = eig_vec2[:,np.flip(np.argsort(eig_val2))]

        S = np.zeros(A.shape)
        for i in range(len(sg_val)):
            S[i][i] = sg_val[i]
        U=[]
        # print(np.dot(A,V[:,0].T))
        for idx, i in enumerate(sg_val):
            U.append(np.dot(A,V[:,idx].T)/i)
        U=np.array(U).T
        # print("U: ",U)
        # print("S: ",S)
        # print("V: ",V)
        # print("A:")
        A_r = np.dot(np.dot(U, S),V.T)

        # print(np.allclose(A,A_r))
        if np.allclose(A, A_r):
            return U, S, V

class TotalLeastSquare:
    def fit_line(self, x, y):
        x = np.array(x)
        y = np.array(y)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        U = np.vstack([x-x_mean, y-y_mean]).T
        A = np.dot(U.T, U)
        svd = SingleValueDecompositon()
        U, S, V = svd.SVD(A)
        a, b = V[:, V.shape[1]-1]
        d = a*x_mean + b*y_mean
        m = -a/b
        c = d/b

        return m, c

class LinearLeastSquare:

    def fit_line(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x_square_sum = np.sum(np.square(x))

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_sum = np.sum(x)
        y_sum = np.sum(y)
        xy_sum = np.sum(np.multiply(x,y))
        n = x.size
        ss_xx = x_square_sum - x_sum**2/n 
        ss_xy = xy_sum - x_sum*y_sum/n 

        m = ss_xy/ss_xx
        c = y_mean - m*x_mean

        return m,c
    def fit_parabola(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x_square = np.square(x)
        A = np.vstack([x_square,x,np.ones(x.size)]).T
        Y = y.reshape(-1,1)
        beta = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),Y)
        return beta


class RANSAC():

    def computerError(self, x, y, m, c):
        error = y - m*x - c
        return error**2
    def fit_line(self, x, y, threshold, p=0.95, e=0.5, optimal_fit = False):
        N = int(np.ceil(np.log(1-p)/np.log(1 - np.power(1 - e, 2))))
        n = 2
        N = np.max([N, 30])
        x = np.array(x)

        y = np.array(y)
        m_best = 0
        c_best = 0
        n_inliers_max=0
        inlier_mask_best = np.zeros(x.size)
        for i in range(N):

            idxs_r = np.random.choice(x.size, size = n)
            
            x_r = x[idxs_r]
            y_r = y[idxs_r]

            ls = LinearLeastSquare()
            m, c = ls.fit_line(x_r, y_r)
            
            error_list = self.computerError(x, y, m, c)
            inlier_mask = error_list<threshold
            # print(error_list, inlier_mask)
            n_inliers = np.count_nonzero(error_list<threshold)
            # print("n_in", n_inliers)
            # print("n_max", n_inliers_max)
            if n_inliers > n_inliers_max:
                n_inliers_max = n_inliers
                m_best = m 
                c_best = c
                inlier_mask_best = inlier_mask
                
                

            if n_inliers_max/x.size >= p:
                print("HERE")
                break
        # plt.scatter(x[np.invert(inlier_mask_best)], y[np.invert(inlier_mask_best)], color = 'tab:orange')
        # plt.scatter(x[inlier_mask_best], y[inlier_mask_best], color='b')
        # print(np.invert(inlier_mask_best))
        # print(m_best, c_best)
        # y1 = m_best*x + c_best
        # plt.plot(x, y1, color='r')
        # plt.show()
        if optimal_fit:
            m_best, c_best = ls.fit_line(x[inlier_mask_best], y[inlier_mask_best])
        return m_best, c_best, inlier_mask_best
        
        
