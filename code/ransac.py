import numpy as np
import os 
import pandas as pd
from utils import LinearLeastSquare
import matplotlib.pyplot as plt
class RANSAC():

    def computerError(self, x, y, m, c):
        error = y - m*x - c
        return error**2
    def fit_line(self, x, y, threshold, p=0.95, e=0.5, optimal_fit = True):
        N = int(np.ceil(np.log(1-p)/np.log(1 - np.power(1 - e, 2))))
        n = 2

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
                plt.scatter(x[np.invert(inlier_mask_best)], y[np.invert(inlier_mask_best)], color = 'tab:orange')
                plt.scatter(x[inlier_mask_best], y[inlier_mask_best], color='b')
                print(np.invert(inlier_mask_best))
                print(m_best, c_best)
                y1 = m_best*x + c_best
                plt.plot(x, y1, color='r')
                plt.show()
                        
                

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
        
        



        








if __name__ == "__main__":

    file =os.path.join("../data","insurance_data.csv")

    data = pd.read_csv(file)
    x = data['age'].to_numpy()
    y = data['charges'].to_numpy()

    x_n = (x - np.min(x))/(np.max(x) - np.min(x))
    y_n = (y - np.min(y))/(np.max(y) - np.min(y))
    x_n = x_n - np.mean(x_n)
    y_n = y_n - np.mean(y_n)

    ransac = RANSAC()
    ransac.fit_line(x_n, y_n, 0.0015, p=0.99)