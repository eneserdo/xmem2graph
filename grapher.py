import cv2
import numpy as np
from math import sqrt

def dist(c1, c2):
    # return sqrt(sum([(c1[i]-c2[i])**2 for i in range(3)]))
    return sqrt((c1[0]-c2[0])**2+ (c1[1]-c2[1])**2+ (c1[2]-c2[2])**2)


class GraphDrawer:
    def __init__(self, num_obj, queue_length=5):
        self.centers_queue=[]
        self.queue_length=queue_length
        self.num_obj=num_obj
        self.avr_centers={}
        for i in range(1,1+num_obj):
            self.avr_centers[i]=np.zeros((3))


    def _impute_and_push(self, centers):
        
        if len(self.centers_queue)==self.queue_length:
            
            for key, value in centers.items():
                # z koordinatı 0 gelirse onun yerine ortalamayı koy
                if value[2]==0:
                    print("imputed: ", key)
                    centers[key][2]=self.avr_centers[key][2]
            
            self.centers_queue.append(centers) 
            self.centers_queue.pop(0)
        else:
            print("queue not yet filled")
            self.centers_queue.append(centers)


    def _average(self):
        counter=np.zeros((self.num_obj))
        for i in range(1,1+self.num_obj):
            self.avr_centers[i]=np.zeros((3))

        for t in self.centers_queue:
            for key, value in t.items():
                counter[int(key)-1]+=1
                self.avr_centers[key]+=value

        for i in range(1,1+self.num_obj):
            self.avr_centers[i]=(self.avr_centers[i]/counter[i-1]).astype(np.int64)

        # self.avr_centers=self.centers_queue[-1]

    def _movingAverage(self, beta=0.95):
        curr_center=self.centers_queue[0]
        
        for key, value in curr_center.items():
            self.avr_centers[key]=(self.avr_centers[key]*beta +(1-beta)*value).astype(np.int64)

    def getGraph(self, curr_centers, mask, threshold, show_labels=True):
        self._impute_and_push(curr_centers)
        self._average()
        # self._movingAverage()
        

        centers=self.avr_centers
        obj_ids=centers.keys()

        if show_labels:
            # graphImage=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # graphImage[(graphImage>0).all(-1)] = [204, 51, 51]

            graphImage=cv2.applyColorMap(mask*(240//self.num_obj)+15, cv2.COLORMAP_OCEAN)
        else:
            graphImage=np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

            
        # max_distance=sqrt(mask.shape[1]**2+mask.shape[0]**2+400**2)
        max_distance=1000
        
        for first_id in obj_ids:
            first_center=centers[first_id]

            for second_id in obj_ids:
                second_center=centers[second_id]

                if second_id<=first_id:
                    continue

                distance=int(255*dist(first_center, second_center)/max_distance)

                if threshold>distance:
                    try:
                        cv2.line(graphImage, (first_center[1], first_center[0]), (second_center[1], second_center[0]), (51,204,51), 2)
                        cv2.putText(graphImage, f"{distance}", ((first_center[1]+second_center[1])//2, (first_center[0]+second_center[0])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0 ,204), 1)
                    except:
                        print("Error1: ", first_center[0], first_center[1])
                        print("Error2: ", second_center[0], second_center[1])

        # graphImage=cv2.applyColorMap(graphImage, cv2.COLORMAP_OCEAN)
        

        for _id in obj_ids:
            center=centers[_id]
            cv2.circle(graphImage, (center[1], center[0]), 3, (0,51,255), -1)
            
            try:
                curr_center=curr_centers[_id]            
                cv2.circle(graphImage, (curr_center[1], curr_center[0]), 3, (20,255,10), -1)
            except IndexError:
                print("IndexError passed")
            cv2.putText(graphImage, f"{center}", (center[1] - 1, center[0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,204), 1)

        return graphImage