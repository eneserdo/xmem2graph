import cv2
import numpy as np
from math import sqrt
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from utils import convert_from_uvd, Coord, print_debug, arrayAndDepthToCoord, almost_equal


EVERY_NTH_ELEMENT = 10
SUBSAMPLING_DEGREE = np.pi/2
MIN_INSIDE=10
SUBSET_INSIDE=0.002



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

class TableRelation:
    def __init__(self, img, depth, num_obj):

        self._selectPoints(img, depth) # sets self.table_points

        assert self.table_points.shape[0]>=3

        # zeros = np.zeros((self.table_points.shape[0])).reshape(-1,1)
        
        # Since "A@[x, y, z, 1] = 0" has a trivial solution (p=0), instead I used "A@[x, y, 1] = z" 
        ones = np.ones((self.table_points.shape[0])).reshape(-1,1)
        A = np.concatenate([self.table_points[:,0:2], ones], axis=1) 
        z=self.table_points[:, 2]

        # z= a*x + b*y + c
        self.p, _, _, _ = lstsq(A, z)
        print("p: ", self.p)
        # self._distToPoint = lambda x, y, z: (x*p[0]+ y*p[1]+ z*p[2]+p[3])/(sqrt(p[0]**2+p[1]**2+ p[2]**2))
        self._distToPoint = lambda _x, _y, _z: abs(_x*self.p[0] + _y*self.p[1] + self.p[2] -1*_z)/(sqrt(self.p[0]**2 + self.p[1]**2 + 1))
        
        # self.relations=np.zeros((num_obj))
        self.min_dists=np.ones((num_obj))*np.inf
        self.num_obj=num_obj

    # def setRelationValue(self, obj_id, val):
    #     self.relations[obj_id]=val

    def setMinDistFromTable(self, points, obj_id):
        minDist=np.inf
        for p in points:
            dist=self.plane(p.x, p.y, p.z)
            if dist<minDist:
                minDist=self.plane(p)
        
        self.min_dists[obj_id]=minDist


    def _selectPoints(self, img_point_select, depth):
        ### inner func ###
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img_point_select, (x, y), 2, (0, 51, 255), -1)
                cv2.imshow("Choose points on table", img_point_select)
                x1, y1, z1=convert_from_uvd(int(x), int(y), depth.item((int(y), int(x))))
                print(f"({round(x1,1)}, {round(y1,1)}, {round(z1,1)}) is selected")
                self.table_points=np.append(self.table_points, [x1, y1, z1])
        ### inner func ###

        self.table_points=np.array([])
        # cv2.destroyAllWindows()
        cv2.imshow("Choose points on table", img_point_select)

        cv2.setMouseCallback('Choose points on table', click_event)

        cv2.waitKey(0)
        self.table_points=self.table_points.reshape(-1, 3)
        cv2.destroyWindow("Choose points on table")



def getMinDistBtwPoints(points1, points2):
    minDist = np.inf
    p1_min = None
    p2_min = None

    for p1 in points1:
        if almost_equal(p1.d, 0):
            continue
        
        for p2 in points2:
            if almost_equal(p2.d, 0):
                continue

            dist = p1.realDistanceTo(p2)

            if minDist > dist:
                minDist = dist
                p1_min = p1
                p2_min = p2

    return p1_min, p2_min, minDist


def getN_MinDistsBtwPoints(points1, points2, N=6):
    
    dist_array=np.ones((len(points1), len(points2)))*np.inf

    for i, p1 in enumerate(points1):
        if almost_equal(p1.d, 0):
            continue
        
        for j, p2 in enumerate(points2):
            if almost_equal(p2.d, 0):
                continue

            dist_array[i,j]=p1.realDistanceTo(p2)

    
    # return np.sort(dist_array.flatten())[:5]
    # plt.hist(dist_array.flatten(), bins=100, range=(0,100))
    # plt.show()
    return np.mean(np.sort(dist_array.flatten())[:N])


def mask2BoundaryLocs(obj_mask: np.ndarray, depth: np.ndarray, every_nth_element: int = EVERY_NTH_ELEMENT):
    """
        Calculates the boundary points of the given mask (for single object), 
        then only choose every $(every_nth_element) th element
    """

    _, thresh = cv2.threshold((obj_mask*3).astype(np.uint8), 1, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # If more htan one, combines them
    contours = np.vstack(contours)

    boundaryPoints = contours[::every_nth_element, :, ::-1]
    boundaryPoints = boundaryPoints.reshape(-1, 2)

    bP_as_coord = []
    for bP in boundaryPoints:
        # bP[0]=v, bP[1]=u
        bP_as_coord.append(Coord(bP.item(1), bP.item(0), depth.item((bP[0], bP[1]))))

    return bP_as_coord


def closestDistanceBtwMasks(obj_mask1, center1: Coord, first_id, obj_mask2, center2: Coord, second_id, depth, graphImage):
    """
        From masks, calculates the boundary points and inner points. 
        Then, sub-samples them. Finally measure the min distance between two point groups
    """

    # IMPORTANT!: Instead of boundary points I used random points on mask,
    # because edges of objects have very unreliable depth values

    ### First Mask ###

    # Get boundary points
    boundary_points1=mask2BoundaryLocs(obj_mask1, depth)

    # get subset
    # unit_vector1=center1.realUnitVectorTo(center2)
    # related_indicies1=[]
    # for i,p in enumerate(points1):

    #     from_center1_to_p=center1.realUnitVectorTo(p)

    #     curr_val=np.dot(from_center1_to_p, unit_vector1)

    #     if curr_val>np.cos(SUBSAMPLING_DEGREE):
    #         related_indicies1.append(i)

    # points1_subset=[points1[i] for i in related_indicies1]

    points1_v, points1_u = np.where(obj_mask1*5 > 1)
    points1 = np.concatenate([points1_v.reshape(-1, 1), points1_u.reshape(-1, 1)], axis=1)

    subset1 = np.random.choice(len(points1), max(int(len(points1)*SUBSET_INSIDE), MIN_INSIDE))
    points1_subset = points1[subset1]
    points1_subset = arrayAndDepthToCoord(points1_subset, depth)


    all_points1=boundary_points1+points1_subset
    for p in all_points1:
        cv2.circle(graphImage, (p.u, p.v), 2, (20, 100, 90), -1)

    ### Second Mask ###

    boundary_points2=mask2BoundaryLocs(obj_mask2, depth)
    # unit_vector2 = -unit_vector1

    # related_indicies2=[]
    # for i,p in enumerate(points2):

    #     from_center2_to_p=center2.realUnitVectorTo(p)

    #     curr_val=np.dot(from_center2_to_p, unit_vector2)

    #     if curr_val>np.cos(SUBSAMPLING_DEGREE):
    #         related_indicies2.append(i)

    # points2_subset=[points2[i] for i in related_indicies2]

    points2_v, points2_u = np.where(obj_mask2*5 > 1)
    points2 = np.concatenate([points2_v.reshape(-1, 1), points2_u.reshape(-1, 1)], axis=1)
    subset2 = np.random.choice(len(points2), max(int(len(points2)*SUBSET_INSIDE), MIN_INSIDE))
    points2_subset = points2[subset2]
    # points2_subset=points2
    points2_subset = arrayAndDepthToCoord(points2_subset, depth)

    if len(points2_subset) == 0:
        print(points2_subset)
        print(points2)
        raise ValueError

    all_points2=boundary_points2+points2_subset
    for p in all_points2:
        cv2.circle(graphImage, (p.u, p.v), 2, (20, 100, 90), -1)

    # points1_subset.append(center1)
    # points2_subset.append(center2)


    ### TABLE RELATION ###
    # table.setMinDistFromTable(points1, first_id)
    # if table.num_obj==second_id[0]+1:
    #     table.setMinDistFromTable(points2, second_id)

    # closest1, closest2, minDist = getMinDistBtwPoints(all_points1, all_points2)
    # return closest1, closest2, minDist
    
    minDist = getN_MinDistsBtwPoints(all_points1, all_points2)
    return None, None, minDist




class GraphDrawer:
    """
        Transforms semantic segmented image masks to graphs
    """

    def __init__(self, num_obj, queue_length=-1, beta=0.95, avr_on=False):

        # if queue_length==1 and beta==0:
        # raise ValueError("For queue length 1, moving avr is used, so specify the beta")

        if (queue_length != -1) or (avr_on is True):
            raise NotImplementedError("Averaging queue is not implemented")
        # self.queue_length=queue_length
        # self.centers_queue=[]

        self.movingAverageCenters = {}
        self.num_obj = num_obj
        self.beta = beta

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

    def _impute_and_push(self, curr_center: dict):
        """
            Only keeps track of d (as moving average). If it gets 0, imputes it and updates x,y,z. 
        """
        if len(self.movingAverageCenters) == 0:
            self.movingAverageCenters = curr_center

        else:
            for id_, center in curr_center.items():

                # for new comers (add them)
                if id_ in self.movingAverageCenters.keys():
                    if center.d == 0:
                        curr_center[id_].update_d(
                            self.movingAverageCenters[id_].d)

                else:
                    self.movingAverageCenters[id_] = center

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
        if len(self.movingAverageCenters) == 0:
            if len(curr_center) == 0:
                pass
            else:
                print("ERROR:")
                print(f"{curr_center=}")
                print(f"{self.movingAverageCenters=}")
                raise ValueError
        else:
            for id_, center in curr_center.items():
                self.movingAverageCenters[id_].d = self.movingAverageCenters[id_].d*self.beta + (1-self.beta)*center.d

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

    def getGraph(self, curr_centers: dict, mask, check_threshold, touch_threshold, table_threshold, depth, show_labels=True):

        self._impute_and_push(curr_centers)
        self._movingAverage(curr_centers)

        centers = curr_centers

        obj_ids = centers.keys()

        if show_labels:
            # graphImage=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # graphImage[(graphImage>0).all(-1)] = [204, 51, 51]

            graphImage = cv2.applyColorMap(mask*(240//self.num_obj)+15, cv2.COLORMAP_OCEAN)
        else:
            graphImage = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)


        for first_id in obj_ids:
            first_center = centers[first_id]

            for second_id in obj_ids:
                second_center = centers[second_id]

                if second_id <= first_id:
                    continue

                distance = round(first_center.realDistanceTo(second_center), 1)

                if touch_threshold > distance:
                    cv2.line(graphImage, (first_center.u, first_center.v),(second_center.u, second_center.v), (20, 100, 90), 3)
                    cv2.putText(graphImage, f"{distance}", ((first_center.u+second_center.u)//2, (first_center.v+second_center.v)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 50, 200), 1)

                # if their center is closer than the threshold then they are candidate for "touch"
                elif check_threshold > distance:
                    cv2.putText(graphImage, f"{distance}", ((first_center.u+second_center.u)//2, (first_center.v+second_center.v)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 50, 200), 1)

                    # candidate
                    cv2.line(graphImage, (first_center.u, first_center.v), (second_center.u, second_center.v), (51, 204, 51), 1)

                    # if their closest points are touching they are touched for sure
                    closest_p1, closest_p2, minDist = closestDistanceBtwMasks((mask == int(first_id))*1, first_center, first_id, (mask == int(second_id))*1, second_center, second_id, depth, graphImage)

                    if touch_threshold > minDist:
                        cv2.line(graphImage, (first_center.u, first_center.v), (second_center.u, second_center.v), (20, 100, 90), 3)
                        if closest_p1 is not None:
                            cv2.line(graphImage, (closest_p1.u, closest_p1.v), (closest_p2.u, closest_p2.v), (20, 100, 90), 1)
                        # cv2.putText(graphImage, f"{distance}", ((first_center[1]+second_center[1])//2, (first_center[0]+second_center[0])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,50,150), 1)

        for _id in obj_ids:
            center = centers[_id]
            cv2.circle(graphImage, (center.u, center.v), 3, (0, 51, 255), -1)

            cv2.putText(graphImage, f"{center}", (center.u - 25, center.v + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 204), 1)

        return graphImage

    # def setTablePlane(self, img, depth):
    #     # Call once
    #     self.table_relations=TableRelation(img, depth)
    
    # def getTableRelations(self):
    #     # Call to show on table relation
    #     return self.table_relations.relations
