import cv2
import numpy as np
import json
import os
files = ["ball_without_noise.mp4", "ball_with_noise.mp4"]
for file in files:
  cap = cv2.VideoCapture(os.path.join("../data",file))


  if (cap.isOpened()== False): 
    print("Error opening video stream or file")

  def get_rect_patch(indices):

      xmin, ymin = int(min(indices[1])), int(min(indices[0]))
      xmax, ymax = int(max(indices[1])), int(max(indices[0]))

      return [xmin, ymin, xmax, ymax]

  points={'x': [], 'y' : []}
  while(cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:
      # frame = cv2.resize(frame, (640, 480))
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      th, dst = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY_INV)
      indices = np.where(dst>200)
      bboxes = get_rect_patch(indices)
      bboxes[0] = int((bboxes[0] + bboxes[2]) / 2)
      bboxes[2] = bboxes[0]
      point1 = [bboxes[0], bboxes[1]]
      point2 = [bboxes[2], bboxes[3]]
      points['x'].append(point1[0])
      points['x'].append(point2[0])
      points['y'].append(point1[1])
      points['y'].append(point2[1])
      # cv2.circle(frame, (point1[0], point1[1]), 2, (0,0,255), 3)
      print("Points : ",point1, point2)
      frame = cv2.resize(frame,(640,480))
      cv2.imshow('Frame', frame)
      
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break


    else: 
      break

  with open(file+".json", 'w') as f:
      json.dump(points, f)

  cap.release()
  cv2.destroyAllWindows()