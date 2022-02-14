import numpy as np 
import matplotlib.pyplot as plt 
import json
from utils import LinearLeastSquare
import cv2
import os 
# CASE 1 : Ball without noise
with open('../data/ball_without_noise.json') as json_file:
    data = json.load(json_file)
# Load x and y data
x=np.array(data['x'])
y=data['y']
# Fit parabola using Linear Least Sqaure 
lls = LinearLeastSquare()
a, b, c = lls.fit_parabola(x,y)
y1=a*x**2+b*x+c
plt.figure(1)
# Plot data
plt.scatter(x,y)
plt.title("Parabola Fitting Without Noise")

# Plot Curve
plt.plot(x,y1, label="LS", color='r')
plt.legend()
plt.gca().invert_yaxis()


# CASE 2 : Ball with noise
with open('../data/ball_with_noise.json') as json_file:
    data = json.load(json_file)
# Load x and y data
x=np.array(data['x'])
y=data['y']
# Fit parabola using Linear Least Sqaure 
lls = LinearLeastSquare()
a, b, c = lls.fit_parabola(x,y)
y1=a*x**2+b*x+c
plt.figure(2)
# Plot data
plt.scatter(x,y)
plt.title("Parabola Fitting With Noise")

# Plot Curve
plt.plot(x,y1, label="LS", color='r')
plt.legend()
plt.gca().invert_yaxis()



plt.show()