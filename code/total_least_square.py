import numpy as np
from numpy import linalg as LA
from linear_least_square import LinearLeastSquare
import math
import pandas as pd
import os
from utils import SingleValueDecompositon
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



if __name__ == "__main__":
    file =os.path.join("../data","insurance_data.csv")
    data = pd.read_csv(file)
    x = data['age'].to_numpy()
    y = data['charges'].to_numpy()
    x_n = (x - np.min(x))/(np.max(x) - np.min(x))
    y_n = (y - np.min(y))/(np.max(y) - np.min(y))
    x = x_n - np.mean(x_n)
    y = y_n - np.mean(y_n)
    ls = LinearLeastSquare()
    tls = TotalLeastSquare()
    print(tls.fit_line(x ,y))
    print(ls.fit_line(x,y))
    # cov_mat = np.array([[variance(x), covariance(x, y)],[covariance(x, y), variance(y)]])
    
    # print(cov_mat)
    from scipy import odr 
    import random as r
    

    def target_function(p, x):
        m, c = p
        return m*x + c
    
    #  model fitting.
    odr_model = odr.Model(target_function)
    
    # Create a Data object using sample data created.
    data = odr.Data(x, y)
    
    # Set ODR with the model and data.
    ordinal_distance_reg = odr.ODR(data, odr_model,
                                beta0=[0, 1])
    
    # Run the regression.
    out = ordinal_distance_reg.run()
    
    # print the results
    out.pprint()