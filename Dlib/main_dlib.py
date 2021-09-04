import cv2
import sys
import os
from yolov5_detect_updated import get_detection
import dlib
from imutils.video import FPS
#from draw_curve import draw

video_filename='webtest/web1.mp4'
if not os.path.isfile(video_filename):
    print('Video not found.')
    sys.exit()
    
op_name=video_filename.split('/')[-1]

#choice=int(input('Select from 1 or 2:\n1. DLIB Tracker '))
output=os.path.join('Output/dlib',op_name.split('.')[0])
if not os.path.isdir(output):
    os.makedirs(os.path.join(output,'Frames'))
    os.makedirs(os.path.join(output,'Output_video'))
cap = cv2.VideoCapture(video_filename)
model=get_detection()
count=0
fps = FPS().start()
trackers = []
pts=[]
trackers=None
width  = cap.get(3)  # float `width`
height = cap.get(4)  # float `height`
downswing=False
color=[]
frames=[]
stop_swing=False
quadrant=False

def to_video(vid_dir,folder, filename):
    img_array = []
    lsorted = sorted(os.listdir(folder),key=lambda x: int(os.path.splitext(x)[0]))
    for file in lsorted:
        img = cv2.imread(os.path.join(folder,file))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter(os.path.join(vid_dir,filename),cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('Video saved to {}'.format(os.path.join(vid_dir,filename)))
    
while True:
    # get frame from the video
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    #converting frame form BGR to RGB for dlib 
    if not stop_swing:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        points,_=model.find_coordinates(rgb)
        if len(points)!=1:
            trackers = []
            x,y,w,h=points
            cx=x+((w-x)//2)
            cy=y+((h-y)//2)
            if (cx > width/2) and (cy > (height/2)) and downswing:
                quadrant=True
                
            if quadrant and (cx > (width/2)) and (cy < (height/4)):
                stop_swing=True
                continue
            centroid=(cx,cy)
            pts.append(centroid)
            model.initial=(cx,cy)
            frames.append(count)

            if len(pts) == 1:
                color.append((0,255,0))
                cv2.circle(frame,centroid,3,(0,255,0),-1)
            else:
                if not downswing:
                    #print(pts[-1][0],pts[-2][0])
    
                    if (pts[-1][0] > width/2) and (pts[-1][1] < height/2) and (pts[-2][0]-pts[-1][0]>=5):
                        downswing=True
                        color.append((0,0,255))
                    
                    else:
                        color.append((0,255,0))
                else:
                    color.append((0,0,255))
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], color[i], 4)
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x, y, w, h)
            tracker.start_track(rgb, rect)
            trackers.append(tracker)
            #cv2.rectangle(frame, (x, y), (w, h),(0, 0, 255), 2)
    
        else:
            if trackers!=None:
                for tracker in trackers:
                    tracker.update(rgb)
                    pos = tracker.get_position()
                    # unpack the position object
                    x = int(pos.left())
                    y = int(pos.top())
                    w = int(pos.right())
                    h = int(pos.bottom())
                    #print(x,y,w,h)
                    cx=x+((w-x)//2)
                    cy=y+((h-y)//2)
                    if (cx > width/2) and (cy > (height/2)) and downswing:
                        quadrant=True
                    if quadrant and (cx > (width/2)) and (cy < (height/4)):
                        stop_swing=True
                        continue
                    centroid=(cx,cy)
                    pts.append(centroid)
                    model.initial=centroid
                    frames.append(count)

                    if not downswing:
                        if (pts[-1][0] > width/2) and (pts[-1][1] < height/2) and (pts[-2][0]-pts[-1][0]>=5):
                            downswing=True
                            color.append((0,0,255))
                        
                        else:
                            color.append((0,255,0))
                    else:
                        color.append((0,0,255))
                        
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i - 1], pts[i], color[i], 4)
            
    cv2.imwrite(output+'/Frames/{}.png'.format(count),frame)
    count+=1
    cv2.imshow('winName', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()
fps.stop()

textfile = open(os.path.join(output,op_name+"_points.txt"), "w")
textfile.write('\t\t\t CENTROID LOCATIONS \n')
for i,element in enumerate(pts):
    textfile.write('Frame {}: {} \n'.format(frames[i],str(element)))
textfile.close()

to_video(os.path.join(output,'Output_video'),os.path.join(output,'Frames'),op_name.split('.')[0]+'_tracked.mp4')
cv2.destroyAllWindows()
cap.release()
    

  
    