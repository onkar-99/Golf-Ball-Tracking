# Golf-Ball-Tracking
This is a script which can be used for tracking a golf ball and drawing curve along the swing swing.

First we found the golf head locations in the video using YOLOv5. The model was trained on custom data and the weights were saved in .pt format.
However, since the golf swing was very fast, the golf club was blurred in most of the frames and difficult to detect. 

To overcome this problem we've used 2 approaches:
## 1. Optical Flow
Here we used optical flow to get the thresholded binary image of the swing.  
#### **Optical flow or optic flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene.**
Then using the centroid distance approach we find the contour in the thresholded image nearest to the last known golf club location. 

## 2. Dlib Tracker
In this approach, for the frames where the golf head wasn't detected, we implemented a tracker to track the golf club based on the last known location and updated the location with the current predicted location by the tracker.

Although both these approaches were implemented correctly, they were not as promising as were expected to be.

# Areas of Improvement:
Since the golf club is blurred while downswing, the tracking and detection fails. If you think any approach could tackle this problem, you can connect with me  

# YOLOv5 Weights
Yolov5 model weights for golf head detection can be found [here](https://drive.google.com/file/d/1KMdxNBPA-HZhrm2ht4W_cDrBWic-j9vT/view?usp=sharing)

# Output
![img](golf_club_tracking.PNG)
