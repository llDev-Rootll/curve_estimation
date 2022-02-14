import cv2
import numpy as np
import json
import os
"""Script to generate datapoints from the ball videos.
  - Performs inverse binary thresholding on grayscale frames 
    to obtain bounding boxes for the ball in each case.
-  Points are then saved in corresponding JSON files for easy 
    access.

"""

def get_rect_patch(indices):
  """Function to return bbox co-ordinates from a blob

  """
  xmin, ymin = int(min(indices[1])), int(min(indices[0]))
  xmax, ymax = int(max(indices[1])), int(max(indices[0]))

  return [xmin, ymin, xmax, ymax]


files = ["ball_without_noise", "ball_with_noise"]
for file in files:
  # Read Video 
  cap = cv2.VideoCapture(os.path.join("../data",file + ".mp4"))
  
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")

  points={'x': [], 'y' : []}
  print("Processing : ", file)
  while(cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:
      # Convert to grayscale
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Perform inverse binary thresholding
      th, dst = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY_INV)
      # Extract region of the ball
      indices = np.where(dst>200)
      # Generate bbox co-ordinates
      bboxes = get_rect_patch(indices)
      bboxes[0] = int((bboxes[0] + bboxes[2]) / 2)
      bboxes[2] = bboxes[0]
      point1 = [bboxes[0], bboxes[1]]
      point2 = [bboxes[2], bboxes[3]]
      # Append points
      points['x'].append(point1[0])
      points['x'].append(point2[0])
      points['y'].append(point1[1])
      points['y'].append(point2[1])

      print("Points : ",point1, point2)
      frame = cv2.resize(frame,(640,480))
      cv2.imshow('Frame', frame)
      
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break


    else: 
      break
  # Save points to disk
  with open(os.path.join("../data",file+".json"), 'w') as f:
      json.dump(points, f)

  cap.release()
  cv2.destroyAllWindows()