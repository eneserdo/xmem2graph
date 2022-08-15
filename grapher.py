import cv2
import numpy as np
from math import sqrt
from operator import itemgetter

from utils import convert_from_uvd, Coord, print_debug


EVERY_NTH_ELEMENT=10
SUBSAMPLING_DEGREE=np.pi/2

# from matplotlib.colors import to_rgb, CSS4_COLORS
def colorMap(name):
    raise NotImplementedError

    # colorName={"b1": "",
    #            "b2": "",
    #            "y1": "",}

    # try: 
    #     return (255*to_rgb(CSS4_COLORS[name])[::-1]).astype(np.uint8)
    # except:
    #     return (255*to_rgb(CSS4_COLORS[colorName[name]])[::-1]).astype(np.uint8)
        


# def realDistance(c1, c2, depth):
#     real_c1=convert_from_uvd(c1[0], c1[1], depth[c1[0], c1[1]][0])
#     real_c2=convert_from_uvd(c2[0], c2[1], depth[c2[0], c2[1]][0])

#     # return sqrt((c1[0]-c2[0])**2+ (c1[1]-c2[1])**2+ (c1[2]-c2[2])**2)
#     # return sqrt(sum([(c1[i]-c2[i])**2 for i in range(3)]))
#     return sqrt(np.sum((real_c1-real_c2)**2))



def getMinDistBtwPoints(points1, points2):
    minDist=np.inf
    p1_min=None
    p2_min=None

    for p1 in points1:
        for p2 in points2:
            dist=p1.realDistanceTo(p2)

            if minDist>dist:
                minDist=dist
                p1_min=p1
                p2_min=p2

    return p1_min, p2_min, minDist


def mask2BoundaryLocs(obj_mask:np.ndarray, depth:np.ndarray, every_nth_element:int=EVERY_NTH_ELEMENT):
    """
        Calculates the boundary points of the given mask (for single object), 
        then only choose every $(every_nth_element) th element
    """

    _, thresh = cv2.threshold((obj_mask*2).astype(np.uint8), 1, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours=np.vstack(contours)

    boundaryPoints=contours[::every_nth_element, :, ::-1]
    boundaryPoints=boundaryPoints.reshape(-1,2)

    bP_as_coord=[]
    for bP in boundaryPoints:
        bP_as_coord.append(Coord(bP[0], bP[1], depth[bP[0],bP[1]][0]))

    return bP_as_coord


def closestDistanceBtwMasks(obj_mask1, center1: Coord, obj_mask2, center2: Coord, depth, graphImage):
    """
        From masks, calculates the boundary points. 
        Then, sub-samples them. Finally measure the min distance between two point groups
    """

    ### First Mask ###
 
    points1=mask2BoundaryLocs(obj_mask1, depth)

    unit_vector1=center1.realUnitVectorTo(center2)

    related_indicies1=[]
    for i,p in enumerate(points1):
        
        from_center1_to_p=center1.realUnitVectorTo(p)

        curr_val=np.dot(from_center1_to_p, unit_vector1)

        if curr_val>np.cos(SUBSAMPLING_DEGREE):
            related_indicies1.append(i)

    

    # if len(related_indicies2)!=0:
    # points1=itemgetter(*related_indicies1)(points1)
    points1_subset=[points1[i] for i in related_indicies1]
    for p in points1_subset:
        cv2.circle(graphImage, (p.v, p.u), 2, (20,100,90), -1)

    ### Second Mask ###

    points2=mask2BoundaryLocs(obj_mask2, depth)
    unit_vector2 = -unit_vector1
    
    related_indicies2=[]
    for i,p in enumerate(points2):
        
        from_center2_to_p=center2.realUnitVectorTo(p)

        curr_val=np.dot(from_center2_to_p, unit_vector2)

        if curr_val>np.cos(SUBSAMPLING_DEGREE):
            related_indicies2.append(i)
    
    # points2=points2[related_indicies2]

    # if len(related_indicies2)!=0:
    # points2=itemgetter(*related_indicies2)(points2)
    points2_subset=[points2[i] for i in related_indicies2]

    for p in points2_subset:
        cv2.circle(graphImage, (p.v, p.u), 2, (20,100,90), -1)


    points1_subset.append(center1)
    points2_subset.append(center2)
    closest1, closest2, minDist = getMinDistBtwPoints(points1_subset, points2_subset)
    # cv2.line(graphImage, (int(closest1[1]), int(closest1[0])), (int(closest2[1]), int(closest2[0])), (20,100,90), 1)
    
    return closest1, closest2, minDist



class GraphDrawer:
    """
        Transforms semantic segmented image masks to graphs
    """
    def __init__(self, num_obj, queue_length=-1, beta=0.95, avr_on=False):
        
        # if queue_length==1 and beta==0:
            # raise ValueError("For queue length 1, moving avr is used, so specify the beta")

        if (queue_length!=-1) or (avr_on is True):
            raise NotImplementedError("Averaging queue is not implemented")
        # self.queue_length=queue_length
        # self.centers_queue=[]

        self.movingAverageCenters={} 
        self.num_obj=num_obj
        self.beta=beta
        
        # self.avr_centers={}
        # self.avr_on = avr_on
        # for i in range(1,1+num_obj):
        #     self.avr_centers[i]=np.zeros((3))

    # def _impute_and_push(self, centers):
        
    #     if len(self.centers_queue)==self.queue_length:
            
    #         for key, value in centers.items():
    #             # z koordinatı 0 gelirse onun yerine ortalamayı koy
    #             if value[2]==0:
    #                 print("imputed: ", key)
    #                 centers[key][2]=self.avr_centers[key][2]
            
    #         self.centers_queue.append(centers) 
    #         self.centers_queue.pop(0)
    #     else:
    #         print("queue not yet filled")
    #         self.centers_queue.append(centers)

    def _impute_and_push(self, curr_center:dict):
        """
            Only keeps track of d (as moving average). If it gets 0, imputes it and updates x,y,z. 
        """
        if len(self.movingAverageCenters)==0: 
            self.movingAverageCenters=curr_center
        
        else:
            for id_, center in curr_center.items():
                
                # for new comers (add them)
                if id_ in self.movingAverageCenters.keys():
                    if center.d==0:
                        curr_center[id_].update_d(self.movingAverageCenters[id_].d)
                    
                else:
                    self.movingAverageCenters[id_]=center

            # for unused keys (remove them)
            for k in (self.movingAverageCenters.keys()-curr_center.keys()):
                del self.movingAverageCenters[k]

                # if center.d==0:
                #     try:
                #         # for already available ones (update them)
                #         curr_center[id_].update_d(self.movingAverageCenters[id_].d)
                #     except:
                #         # for new comers (add them)
                #         self.movingAverageCenters[id_]=center
                #         # print("ERROR:")
                #         # print(f"{curr_center=}")
                #         # print(f"{self.movingAverageCenters=}")
                #         # raise KeyError
                    

    # def _average(self):
    #     counter=np.zeros((self.num_obj))
    #     for i in range(1,1+self.num_obj):
    #         self.avr_centers[i]=np.zeros((3))

    #     for t in self.centers_queue:
    #         for key, value in t.items():
    #             counter[int(key)-1]+=1
    #             self.avr_centers[key]+=value

    #     for i in range(1,1+self.num_obj):
    #         self.avr_centers[i]=(self.avr_centers[i]/counter[i-1]).astype(np.int64)

        # self.avr_centers=self.centers_queue[-1]

    def _movingAverage(self, curr_center):
        
        # if it is first call
        if len(self.movingAverageCenters)==0:
            if len(curr_center)==0:
                pass
            else:
                print("ERROR:")
                print(f"{curr_center=}")
                print(f"{self.movingAverageCenters=}")
                raise ValueError
        else:
            for id_, center in curr_center.items():
                self.movingAverageCenters[id_].d=(self.movingAverageCenters[id_].d*self.beta +(1-self.beta)*center.d).astype(np.int64)
          
            # for id_, center in curr_center.items():
            #     try:
            #         # for already available ones (update them)
            #         self.movingAverageCenters[id_].d=(self.movingAverageCenters[id_].d*self.beta +(1-self.beta)*center.d).astype(np.int64)
            #     except KeyError:
            #         # for new comers (add them)
            #         self.movingAverageCenters[id_]=center

            # # for unused keys (remove them)
            # for k in (self.movingAverageCenters.keys()-curr_center.keys()):
            #     del self.movingAverageCenters[k]





    def getGraph(self, curr_centers: dict, mask, check_threshold, touch_threshold, depth, show_labels=True):
        
        self._impute_and_push(curr_centers)
        self._movingAverage(curr_centers)
        

        centers=curr_centers

        
        obj_ids=centers.keys()

        if show_labels:
            # graphImage=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # graphImage[(graphImage>0).all(-1)] = [204, 51, 51]

            graphImage=cv2.applyColorMap(mask*(240//self.num_obj)+15, cv2.COLORMAP_OCEAN)
        else:
            graphImage=np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

            
        # max_distance=sqrt(mask.shape[1]**2+mask.shape[0]**2+400**2)
        # max_distance=10
        
        for first_id in obj_ids:
            first_center=centers[first_id]

            for second_id in obj_ids:
                second_center=centers[second_id]

                if second_id<=first_id:
                    continue

                # distance=int(255*dist(first_center, second_center)/max_distance)
                distance=round(first_center.realDistanceTo(second_center), 1)

                
                if touch_threshold>distance:
                    cv2.line(graphImage, (first_center.v, first_center.u), (second_center.v, second_center.u), (20,100,90), 3)
                    cv2.putText(graphImage, f"{distance}", ((first_center.v+second_center.v)//2, (first_center.u+second_center.u)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,50,150), 1)

                
                # if their center is closer than the threshold then they are candidate for "touch"
                elif check_threshold>distance:
                    cv2.putText(graphImage, f"{distance}", ((first_center.v+second_center.v)//2, (first_center.u+second_center.u)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,50,150), 1)

                    # candidate
                    cv2.line(graphImage, (first_center.v, first_center.u), (second_center.v, second_center.u), (51,204,51), 1)
                    
                    # if their closest points are touching they are touched for sure
                    closest_p1, closest_p2, minDist = closestDistanceBtwMasks((mask==int(first_id))*1, first_center, (mask==int(second_id))*1, second_center, depth, graphImage)
                    
                    if touch_threshold>minDist:
                        cv2.line(graphImage, (first_center.v, first_center.u), (second_center.v, second_center.u), (20,100,90), 3)
                        cv2.line(graphImage, (closest_p1.v, closest_p1.u), (closest_p2.v, closest_p2.u), (20,100,90), 3)
                        # cv2.putText(graphImage, f"{distance}", ((first_center[1]+second_center[1])//2, (first_center[0]+second_center[0])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,50,150), 1)

        

        for _id in obj_ids:
            center=centers[_id]
            cv2.circle(graphImage, (center.v, center.u), 3, (0,51,255), -1)
            
            cv2.putText(graphImage, f"{center}", (center.v - 25, center.u + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,204), 1)

        return graphImage



# class TableRelation:
#     def __init__(self, table_points):
        
#         # Construct a plane eqn by fitting to points
#         pass
#     def getTableRelations(mask, centers, graphImage):
#         num_obj=np.unique(mask)
        
#         for i in range(1, 1+num_obj):
#             obj_mask=(mask==i)*1

#             all_points=mask2BoundaryLocs(obj_mask)

#             # en alttaki n pointi seç
#             # plane denklemine koyunca thresholdun kaç üstünü veriyo?
#             if condition:
#                 cv2.putText(graphImage, f"on_table", (centers[i-1][1] - 5, center[i-1][0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,204), 1)

#         return

        