import imutils
import cv2
import os
import sys
import numpy as np

from scipy.spatial import distance as dist

import numpy as np
a=np.array([(1,1)])
b=np.array([(7,2),(1,1)])
min_ind=dist.cdist(a, b)


class optical_flow:
    def __init__(self,video_filename):
        self.video_filename=video_filename
        self.vs = cv2.VideoCapture(self.video_filename)
        _,frame=self.vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (23, 23), 0)
        self.firstFrame = gray
        self.i=0
        
    def get_location(self,frame):
        points=[]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (23, 23), 0)
        frameDelta = cv2.absdiff(self.firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=3)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
             (x, y, w, h) = cv2.boundingRect(c)
             if cv2.contourArea(c) > 2500:
                 continue
             #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

             M = cv2.moments(c)
             cX = int(M["m10"] / M["m00"])
             cY = int(M["m01"] / M["m00"])
             points.append((cX,cY))
             
        return np.array(points)
            
        
        
       