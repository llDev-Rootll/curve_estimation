import numpy as np 
import matplotlib.pyplot as plt 
import json
from utils import LinearLeastSquare
import cv2
import os 

with open('../data/ball_without_noise.json') as json_file:
    data = json.load(json_file)

x=np.array(data['x'])
y=data['y']
lls = LinearLeastSquare()
a, b, c = lls.fit_parabola(x,y)
y1=a*x**2+b*x+c
plt.figure(1)
plt.scatter(x,y)
plt.plot(x,y1, color='r')
plt.gca().invert_yaxis()

with open('../data/ball_with_noise.json') as json_file:
    data = json.load(json_file)

x=np.array(data['x'])
y=data['y']
lls = LinearLeastSquare()
a, b, c = lls.fit_parabola(x,y)
y1=a*x**2+b*x+c
plt.figure(2)
plt.scatter(x,y)
plt.plot(x,y1, color='r')
plt.gca().invert_yaxis()



plt.show()