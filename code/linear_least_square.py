import numpy as np

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

if __name__ == "__main__":
    lls = LinearLeastSqaure()
    # x = np.array([2,2,6,8,10])
    # y = np.array([0,1,2,3,3])
    
    x = np.array([1,2,3,4])
    y = np.array([6,11,18,27])
    print(np.polyfit(x,y,2))
    print(lls.fit_parabola(x, y))