import torch
import cv2
import math

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom
class get_detection:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default
        self.model.conf = 0.3
        self.initial=[]
        
    def find_coordinates(self,im):
        #im=cv2.imread('1.png')[..., ::-1]
        results = self.model(im)
        #results.show()
        results=results.pandas().xyxy[0]
        #print(results)
        if not results.empty:  
            for i in range(len(results)):
                l=results.loc[i, :].values.tolist()
                x,y,w,h=int(l[0]),int(l[1]),int(l[2]),int(l[3])
                cx=x+((w-x)//2)
                cy=y+((h-y)//2)
                if not len(self.initial):
                    return (x,y,w,h),0
                    #break
                if math.dist((cx,cy), self.initial) > 150:
                    continue
                
                return (x,y,w,h),math.dist((cx,cy), self.initial)
        
        return [0],None
        
