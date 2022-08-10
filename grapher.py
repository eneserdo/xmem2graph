import cv2
import numpy as np
from math import sqrt

def dist(c1, c2):
    # return sqrt(sum([(c1[i]-c2[i])**2 for i in range(3)]))
    return sqrt((c1[0]-c2[0])**2+ (c1[1]-c2[1])**2+ (c1[2]-c2[2])**2)


def getMinDist(points1, points2):
    minDist=np.inf
    p1_min=None
    p2_min=None

    for p1 in points1:
        for p2 in points2:
            dist=np.linalg.norm(p1-p2)
            if minDist>dist:
                minDist=dist
                p1_min=p1
                p2_min=p2

    return p1_min, p2_min, minDist


def mask2location(obj_mask, depth, every_nth_element=8):

    _, thresh = cv2.threshold((obj_mask*2).astype(np.uint8), 1, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours=np.vstack(contours)

    coords=contours[::every_nth_element, :, ::-1]
    coords=coords.reshape(-1,2)

    return coords

# def closestDistance(obj_mask1, center1, obj_mask2, center2, depth, graphImage):
    
#     points1=mask2location(obj_mask1, depth)
#     unit_vector1=(center2-center1)/np.linalg.norm(center1-center2)

#     max_val=-np.inf
#     closest1=None
#     for p in points1:
#         p_translated=(p-center1[:2])/np.linalg.norm(p-center1[:2])
#         curr_val=np.dot(p_translated,unit_vector1[:2])

#         if max_val<curr_val:
#             max_val=curr_val
#             closest1=p

    
#     points2=mask2location(obj_mask2, depth)
#     unit_vector2=-unit_vector1
    
#     max_val=-np.inf
#     closest2=None
#     for p in points2:
#         p_translated=(p-center2[:2])/np.linalg.norm(p-center2[:2])
#         curr_val=np.dot(p_translated,unit_vector2[:2])
#         if max_val<curr_val:
#             max_val=curr_val
#             closest2=p
    
#     for p in points1:
#         cv2.circle(graphImage, (p[1], p[0]), 2, (20,100,90), -1)

#     for p in points2:
#         cv2.circle(graphImage, (p[1], p[0]), 2, (20,100,90), -1)

#     # cv2.line(graphImage, (int(closest1[1]), int(closest1[0])), (int(closest2[1]), int(closest2[0])), (20,100,90), 1)


#     return closest1, closest2


def closestDistance(obj_mask1, center1, obj_mask2, center2, depth, graphImage):
    
    points1=mask2location(obj_mask1, depth)
    unit_vector1=(center2-center1)/np.linalg.norm(center1-center2)

    related_indicies1=[]
    for i,p in enumerate(points1):
        p_translated=(p-center1[:2])/np.linalg.norm(p-center1[:2])
        curr_val=np.dot(p_translated,unit_vector1[:2])

        if curr_val>np.cos(np.pi/3):
            related_indicies1.append(i)

    
    points1=points1[related_indicies1]

    points2=mask2location(obj_mask2, depth)
    unit_vector2=-unit_vector1

    
    related_indicies2=[]
    for i,p in enumerate(points2):
        p_translated=(p-center2[:2])/np.linalg.norm(p-center2[:2])
        curr_val=np.dot(p_translated,unit_vector2[:2])

        if curr_val>np.cos(np.pi/3):
            related_indicies2.append(i)
    
    points2=points2[related_indicies2]

    for p in points1:
        cv2.circle(graphImage, (p[1], p[0]), 2, (20,100,90), -1)

    for p in points2:
        cv2.circle(graphImage, (p[1], p[0]), 2, (20,100,90), -1)

    closest1, closest2, minDist = getMinDist(points1, points2)

    # cv2.line(graphImage, (int(closest1[1]), int(closest1[0])), (int(closest2[1]), int(closest2[0])), (20,100,90), 1)


    # return closest1, closest2
    
    return closest1, closest2, minDist





class GraphDrawer:
    """
    TODO: queue length 1 ise direkt moving avr olsun, ve beta verilmemişse hata versin
    """
    def __init__(self, num_obj, queue_length=1, beta=0, avr_on=False):
        
        if queue_length==1 and beta==0:
            raise ValueError("For queue length 1, moving avr is used, so specify the beta")

        self.centers_queue=[]
        self.queue_length=queue_length
        self.num_obj=num_obj
        self.beta=beta
        self.avr_centers={}
        self.avr_on = avr_on
        # for i in range(1,1+num_obj):
        #     self.avr_centers[i]=np.zeros((3))


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

    def _movingAverage(self):
        
        curr_center=self.centers_queue[0]

        # if it is first call
        if len(self.avr_centers)==0:
            self.avr_centers=curr_center
            return

        
        for key, value in curr_center.items():
            self.avr_centers[key]=(self.avr_centers[key]*self.beta +(1-self.beta)*value).astype(np.int64)

    def getGraph(self, curr_centers, mask, check_threshold, proximity, depth, show_labels=True):
        
        # self._impute_and_push(curr_centers)
        
        if self.avr_on:
            if self.queue_length==1:
                self._movingAverage()
            else:
                self._average()
            
            centers=self.avr_centers
        else:
            centers=curr_centers
        
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

                # if their center is closer than the threshold then they are candidate for "touch"
                if check_threshold>distance:
                    
                    cv2.line(graphImage, (first_center[1], first_center[0]), (second_center[1], second_center[0]), (51,204,51), 1)

                    # if their closest points are touching they are touched for sure
                    # if closestDistance((mask==int(first_id))*1, first_center, (mask==int(second_id))*1, second_center, depth)
                    closest_p1, closest_p2, minDist= closestDistance((mask==int(first_id))*1, first_center, (mask==int(second_id))*1, second_center, depth, graphImage,)

                    if proximity>minDist:
                        cv2.line(graphImage, (closest_p1[1], closest_p1[0]), (closest_p2[1], closest_p2[0]), (20,100,90), 3)
                        cv2.putText(graphImage, f"{distance}", ((first_center[1]+second_center[1])//2, (first_center[0]+second_center[0])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,50,150), 1)


        # graphImage=cv2.applyColorMap(graphImage, cv2.COLORMAP_OCEAN)
        

        for _id in obj_ids:
            center=centers[_id]
            cv2.circle(graphImage, (center[1], center[0]), 3, (0,51,255), -1)
            
            # try:
            #     curr_center=curr_centers[_id]            
            #     cv2.circle(graphImage, (curr_center[1], curr_center[0]), 3, (20,255,10), -1)
            # except KeyError:
            #     print("KeyError passed")
            cv2.putText(graphImage, f"{center}", (center[1] - 1, center[0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,204), 1)

        return graphImage