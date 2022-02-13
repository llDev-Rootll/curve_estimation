import numpy as np 
from numpy import linalg as LA
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from utils import covariance, LinearLeastSquare, TotalLeastSquare, RANSAC



np.set_printoptions(formatter={'all':lambda x: str(x)})
file =os.path.join("../data","insurance_data.csv")

data = pd.read_csv(file)
x = data['age'].to_numpy()
y = data['charges'].to_numpy()

x_n = (x - np.min(x))/(np.max(x) - np.min(x))
y_n = (y - np.min(y))/(np.max(y) - np.min(y))
# x_n = x_n - np.mean(x_n)
# y_n = y_n - np.mean(y_n)
# plt.figure(1)
# plt.scatter(x,y)
# plt.figure(2)
# plt.scatter(x_n, y_n)
# plt.show()

cov_mat = np.array([[covariance(x_n, x_n), covariance(x_n, y_n)], [covariance(x_n, y_n), covariance(y_n, y_n)]])

eig_val, eig_vec = LA.eig(cov_mat)

eig_vec1 = eig_vec[:,np.argmax(eig_val)]
eig_vec2 = eig_vec[:,np.argmin(eig_val)]
plt.figure(1)
plt.scatter(x, y)
plt.quiver(np.mean(x), np.mean(y), eig_vec1[0]*(np.max(x) - np.min(x)) + np.min(x) , eig_vec1[1]*(np.max(y) - np.min(y)) + np.min(y) , label = 'PC_1', color = 'r', units='xy', angles='xy', scale_units='xy', scale=4)
plt.quiver(np.mean(x), np.mean(y), eig_vec2[0]*(np.max(x) - np.min(x)) + np.min(x) , eig_vec2[1]*(np.max(y) - np.min(y)) + np.min(y), label = 'PC_2', units='xy', angles='xy', scale_units='xy', scale =4)
plt.legend()


plt.figure(2)
plt.scatter(x, y)
ls = LinearLeastSquare()
m, c = ls.fit_line(x, y)
y1 = m*x + c
plt.plot(x, y1, label='LS', color='r')

# plt.scatter(x, y)
tls = TotalLeastSquare()
m, c = tls.fit_line(x_n, y_n)

x2 = np.linspace(np.min(x_n), np.max(x_n), x_n.size) 
x_ps = np.linspace(np.min(x), np.max(x), x.size) 
y2 = m*x2 + c
y2 = y2*(np.max(y) - np.min(y)) + np.min(y)
plt.plot(x_ps, y2, label="TLS", color='g')
plt.legend()

plt.figure(3)
ransac = RANSAC()
m, c, mask = ransac.fit_line(x_n, y_n, 0.002, p=0.99)

plt.scatter(x[np.invert(mask)], y[np.invert(mask)], label='Outliers', color = 'tab:orange')
plt.scatter(x[mask], y[mask], label='Inliers', color='g')
y3 = m*x2 + c
y3 = y3*(np.max(y) - np.min(y)) + np.min(y)
plt.plot(x_ps, y3, color='r')
plt.legend()
plt.show()







from scipy import odr 
import random as r


def target_function(p, x):
    m, c = p
    return m*x + c

#  model fitting.
odr_model = odr.Model(target_function)

# Create a Data object using sample data created.
data = odr.Data(x_n, y_n)

# Set ODR with the model and data.
ordinal_distance_reg = odr.ODR(data, odr_model,
                            beta0=[0, 1])

# Run the regression.
out = ordinal_distance_reg.run()

# print the results
out.pprint()






plt.show()