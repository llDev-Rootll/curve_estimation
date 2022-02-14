import numpy as np 
from numpy import linalg as LA
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from utils import covariance, LinearLeastSquare, TotalLeastSquare, RANSAC



np.set_printoptions(formatter={'all':lambda x: str(x)})
file =os.path.join("../data","insurance_data.csv")
# Read cvs data into a dataframe
data = pd.read_csv(file)
x = data['age'].to_numpy()
y = data['charges'].to_numpy()
# Perform min-max normalization to bring data to the same scale
x_n = (x - np.min(x))/(np.max(x) - np.min(x))
y_n = (y - np.min(y))/(np.max(y) - np.min(y))

# Q1 _____________________________________________________________
# Compute Covariance matrix of original & normalized data
cov_mat = np.array([[covariance(x, x), covariance(x, y)], [covariance(x, y), covariance(y, y)]])
print("Covariance Matrix of original data: ")
print(cov_mat)

cov_mat = np.array([[covariance(x_n, x_n), covariance(x_n, y_n)], [covariance(x_n, y_n), covariance(y_n, y_n)]])
print("Covariance Matrix of normalized data: ")
print(cov_mat)
eig_val, eig_vec = LA.eig(cov_mat)

eig_vec1 = eig_vec[:,np.argmax(eig_val)] # Eigen vector corresponding to max eigen value
eig_vec2 = eig_vec[:,np.argmin(eig_val)] # Eigen vector corresponding to max eigen value

print("Dot product of eigen vectors is: ", np.dot(eig_vec1, eig_vec2))

plt.figure(1)
plt.scatter(x, y)
plt.title("Principal Components")
plt.xlabel('age')
plt.ylabel('insurance_charges')

# Rescale each eigen vector to fit the graph and plot them at the mean of the data
plt.quiver(np.mean(x), np.mean(y), eig_vec1[0]*(np.max(x) - np.min(x))*np.max(eig_val) , eig_vec1[1]*(np.max(y) - np.min(y))*np.max(eig_val) , label = 'PC_1', color = 'r', units='xy', angles='xy', scale_units='xy', scale=0.2)
plt.quiver(np.mean(x), np.mean(y), eig_vec2[0]*(np.max(x) - np.min(x))*np.min(eig_val) , eig_vec2[1]*(np.max(y) - np.min(y))*np.min(eig_val) , label = 'PC_2', units='xy', angles='xy', scale_units='xy', scale =0.2)

plt.legend()
# __________________________________________________________________


# Q2________________________________________________________________

plt.figure(2)
plt.scatter(x, y)
plt.title("Line Fitting using LS & TLS")
plt.xlabel('age')
plt.ylabel('insurance_charges')
ls = LinearLeastSquare()
# Fit original data using LinearLeastSquare as it is scale invariant
m, c = ls.fit_line(x, y)
y1 = m*x + c
plt.plot(x, y1, label='LS', color='r')

# Fit normalized data using TotalLeastSquare
tls = TotalLeastSquare()
m, c = tls.fit_line(x_n, y_n)

x2 = np.linspace(np.min(x_n), np.max(x_n), x_n.size) 
x_ps = np.linspace(np.min(x), np.max(x), x.size) 
print(m, c)
y2 = m*x2 + c
# Rescale data by performing inverse of min-max normalization
y2 = y2*(np.max(y) - np.min(y)) + np.min(y)
plt.plot(x_ps, y2, label="TLS", color='g')
plt.legend()

plt.figure(3)
# Fit normalized data using RANSAC
"""Optional : fit_line function of RANSAC class takes an optional boolean flag, "optimal_fit" while returns the slope and intercept after running Linear Least Square on only the inliers as a final step. 

Set to False as default for comparison between LS, TLS & RANSAC
"""
ransac = RANSAC()
m, c, mask = ransac.fit_line(x_n, y_n, 0.001, p=0.99, optimal_fit = False)
# Plot outliers
plt.scatter(x[np.invert(mask)], y[np.invert(mask)], label='Outliers', color = 'tab:orange')
# Plot inliers
plt.scatter(x[mask], y[mask], label='Inliers', color='g')
plt.title("Line Fitting using RANSAC")
plt.xlabel('age')
plt.ylabel('insurance_charges')
y3 = m*x2 + c
# Rescale data by performing inverse of min-max normalization
y3 = y3*(np.max(y) - np.min(y)) + np.min(y)
plt.plot(x_ps, y3, color='r')
plt.legend()


plt.show()