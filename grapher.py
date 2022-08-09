import cv2
import numpy as np
from math import sqrt

def dist(c1, c2):
    return sqrt(sum([(c1[i]-c2[i])**2 for i in range(3)]))

def movingAverage(centers, beta=0.9):
    pass

class GraphDrawer:
    def __init__(self):
        self.centers_queue=[]
    
    def _push(self, centers):
    
        self.centers_queue.append(centers)
    def getGraph(self):
        pass


def getGraph(centers, mask, threshold, show_labels=True):
    obj_ids=centers.keys()

    if show_labels:
        # graphImage=mask*30
        graphImage = cv2.threshold(mask*20, 5, 255, cv.THRESH_BINARY)
    else:
        graphImage=np.zeros_like(mask)



    max_distance=sqrt(mask.shape[1]**2+mask.shape[0]**2+400**2)
    
    for first_id in obj_ids:
        first_center=centers[first_id]

        for second_id in obj_ids:
            second_center=centers[second_id]

            if second_center<=first_center:
                continue

            distance=int(255*dist(first_center, second_center)/max_distance)

            if threshold<distance:
                cv2.line(graphImage, (first_center[1], first_center[0]), (second_center[1], second_center[0]), distance, 2)
                cv2.putText(graphImage, distance, ((first_center[1]-second_center[1])//2, (first_center[0]-second_center[0])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    graphImage=cv2.applyColorMap(graphImage, cv2.COLORMAP_OCEAN)
    

    for _id in obj_ids:
        center=centers[_id]
        cv2.circle(graphImage, (center[1], center[0]), 3, (255,255,255), -1)
        cv2.putText(graphImage, f"{_id}", (center[1] - 1, center[0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return graphImage