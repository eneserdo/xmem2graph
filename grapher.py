import cv2
import numpy as np
import networkx as nx
from math import sqrt
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from utils import convert_from_uvd, Coord, print_debug, arrayAndDepthToCoord, almost_equal, load_config


cfg=load_config("./config.yml")


# from matplotlib.colors import to_rgb, CSS4_COLORS
def colorMap(name):

    # colorName={"b1": "",
    #            "b2": "",
    #            "y1": "",}

    # try:
    #     return (255*to_rgb(CSS4_COLORS[name])[::-1]).astype(np.uint8)
    # except:
    #     return (255*to_rgb(CSS4_COLORS[colorName[name]])[::-1]).astype(np.uint8)
    raise NotImplementedError

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
                d_=depth.item((int(y), int(x)))
                if not almost_equal(d_, 0):
                    cv2.circle(img_point_select, (x, y), 2, (0, 255, 51), -1)
                    cv2.imshow("Choose points on table", img_point_select)
                    x1, y1, z1=convert_from_uvd(int(x), int(y), d_)
                    print(f"({round(x1,1)}, {round(y1,1)}, {round(z1,1)}) is selected")
                    self.table_points=np.append(self.table_points, [x1, y1, z1])
                else:
                    cv2.circle(img_point_select, (x, y), 2, (0, 51, 255), -1)
                    cv2.imshow("Choose points on table", img_point_select)

        ### inner func ###

        self.table_points=np.array([])
        # cv2.destroyAllWindows()
        cv2.imshow("Choose points on table", img_point_select)

        cv2.setMouseCallback('Choose points on table', click_event)

        cv2.waitKey(0)
        self.table_points=self.table_points.reshape(-1, 3)
        cv2.destroyWindow("Choose points on table")
    
    def _test(self, table_thrshld, img, depth):
        img_test=img.copy()
        
        for i in range(img_test.shape[0]): 
            for j in range(img_test.shape[1]): 
                x1, y1, z1=convert_from_uvd(int(j), int(i), depth.item((int(i), int(j))))
                
                if table_thrshld > self._distToPoint(x1, y1, z1):
                    img_test[i, j]=[70, 20, 250]

        img_test = cv2.resize(img_test, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Table: img_test", img_test)



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


def getAvrMinDistBtwPoints(points1, points2, N):
    
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


def mask2BoundaryPoints(obj_mask: np.ndarray, depth: np.ndarray):
    """
        Calculates the boundary points of the given mask (for single object), 
        then only choose every $(every_nth_element) th element
    """

    _, thresh = cv2.threshold((obj_mask*3).astype(np.uint8), 1, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # If more htan one, combines them
    contours = np.vstack(contours)

    boundaryPoints = contours[::cfg.GRAPHER.EVERY_NTH_ELEMENT, :, ::-1]
    boundaryPoints = boundaryPoints.reshape(-1, 2)

    bP_as_coord = []
    for bP in boundaryPoints:
        # bP[0]=v, bP[1]=u
        # if not almost_equal(depth.item((bP[0], bP[1])), 0):
        bP_as_coord.append(Coord(bP.item(1), bP.item(0), depth.item((bP[0], bP[1]))))

    return bP_as_coord


def mask2InnerPoints(obj_mask: np.ndarray, depth: np.ndarray):
    points_v, points_u = np.where(obj_mask*5 > 1)
    points = np.concatenate([points_v.reshape(-1, 1), points_u.reshape(-1, 1)], axis=1)

    subset = np.random.choice(len(points), max(int(len(points)*cfg.GRAPHER.SUBSET_INSIDE), cfg.GRAPHER.MIN_INSIDE))
    points_subset = points[subset]
    points_subset = arrayAndDepthToCoord(points_subset, depth)
    return points_subset


# def disPointToPCL(point, listOfPoints):
#     return minDist


class GraphDrawer:
    """
        Transforms semantic segmented image masks to graphs on image
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
        self.relation_matrix=None

        self.simpleGraph=SimpleStableGraph(num_obj)

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
 
    def _drawSimpleGraph(self):
        self.G.clear()
        plt.clf()
        is_ok=False
        #self.G.add_node(0)
        for i in range(self.num_obj):
            for j in range(i, self.num_obj):
                if self.relation_matrix[i,j]==1:
                    self.G.add_edge(i+1, j+1)
                    is_ok=True

        if is_ok:
            random_pos = nx.random_layout(self.G, seed=42)
            pos = nx.spring_layout(self.G, pos=random_pos)
            nx.draw_networkx(self.G, pos, **self.options)
            plt.pause(0.00001)

    @staticmethod
    def _getPCL(graphImage, object_mask, depth_img, show_labels):
        allPoints=mask2InnerPoints(object_mask,depth_img) + mask2BoundaryPoints(object_mask,depth_img)

        for i in range(-1, -len(allPoints)-1):
            if almost_equal(allPoints[i].d,0):
                allPoints.pop(i)

        if show_labels:
            for p in allPoints:
                cv2.circle(graphImage, (p.u, p.v), 1, (20, 100, 90), -1)
        
        return allPoints

    def update_cfg(self):
        cfg=load_config("./config.yml")
        self.simpleGraph._update_names()

    def _testProximity(self, img, depth, id_, touch_threshold):
        img_test=img.copy()
        print(self.pcl_dict[1])
        for i in range(img_test.shape[0]): 
            for j in range(img_test.shape[1]): 
                _,_, dist=getMinDistBtwPoints([Coord(int(j), int(i), depth.item((int(i), int(j))))], self.pcl_dict[id_])
                
                if touch_threshold > dist:
                    img_test[i, j]=[70, 20, 250]

        img_test = cv2.resize(img_test, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("img_test", img_test)
        
    def getGraph(self, curr_centers: dict, mask, check_threshold, touch_threshold, table_threshold, depth, show_details=True):
        # Since detailed checking is costly, first check the strict condition; if they are not touching, select candidates
        # assert touch_threshold < check_threshold

        self._impute_and_push(curr_centers)
        self._movingAverage(curr_centers)

        centers = curr_centers

        obj_ids = centers.keys()

        if show_details:
            # graphImage=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # graphImage[(graphImage>0).all(-1)] = [204, 51, 51]

            graphImage = cv2.applyColorMap(mask*(240//self.num_obj)+15, cv2.COLORMAP_OCEAN)
        else:
            graphImage = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)


        self.relation_matrix=np.zeros((self.num_obj, self.num_obj))
        

        # Create pcl for all of them
        self.pcl_dict={}
        for id_ in obj_ids:
            # obj_mask = (mask == int(first_id))*1
            self.pcl_dict[id_]=self._getPCL(graphImage, (mask == int(id_))*1, depth, show_details)

        # In the future to implement multiprocessing, we need to modify following

        for first_id in obj_ids:
            first_center = centers[first_id]

            for second_id in obj_ids:
                second_center = centers[second_id]

                if second_id <= first_id:
                    continue

                center2centerDistance = round(first_center.realDistanceTo(second_center), 1)

                # touch_threshold < check_threshold
                # Since detailed checking is costly, first check the strict condition; if they are not touching, select candidates
                if touch_threshold > center2centerDistance:
                    self.relation_matrix[first_id-1, second_id-1]=1
                    self.relation_matrix[second_id-1, first_id-1]=1

                    cv2.line(graphImage, (first_center.u, first_center.v),(second_center.u, second_center.v), (20, 100, 90), 3)
                    cv2.putText(graphImage, f"{center2centerDistance}", ((first_center.u+second_center.u)//2, (first_center.v+second_center.v)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 50, 200), 1)
                
                # if their center is closer than the threshold then they are candidate for "touch"
                elif check_threshold > center2centerDistance:

                    # candidate
                    #cv2.line(graphImage, (first_center.u, first_center.v), (second_center.u, second_center.v), (51, 204, 51), 1)

                    # if their closest points are touching they are touched for sure
                    # closest_p1, closest_p2, minDist = closestDistanceBtwMasks((mask == int(first_id))*1, first_center, first_id, (mask == int(second_id))*1, second_center, second_id, depth, graphImage, show_labels)
                    minDistance=getAvrMinDistBtwPoints(self.pcl_dict[first_id], self.pcl_dict[second_id], N=10)

                    if touch_threshold > minDistance:
                        self.relation_matrix[first_id-1, second_id-1]=1
                        self.relation_matrix[second_id-1, first_id-1]=1

                        cv2.line(graphImage, (first_center.u, first_center.v), (second_center.u, second_center.v), (20, 100, 90), 3)
                        cv2.putText(graphImage, f"{round(minDistance)}", ((first_center.u+second_center.u)//2, (first_center.v+second_center.v)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 20), 1)

                        # if closest_p1 is not None:
                            # cv2.line(graphImage, (closest_p1.u, closest_p1.v), (closest_p2.u, closest_p2.v), (20, 100, 90), 1)
                        # cv2.putText(graphImage, f"{distance}", ((first_center[1]+second_center[1])//2, (first_center[0]+second_center[0])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,50,150), 1)
                    
                    # cv2.putText(graphImage, f"{round(minDistance)}", ((first_center.u+second_center.u)//2, (first_center.v+second_center.v)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 20), 1)

        # Emphasize object centers
        for _id in obj_ids:
            center = centers[_id]
            cv2.circle(graphImage, (center.u, center.v), 3, (0, 51, 255), -1)
            if show_details:
                cv2.putText(graphImage, f"{center}", (center.u - 25, center.v + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 204), 1)

        # self._drawSimpleGraph()
        self.simpleGraph.show(self.relation_matrix)

        return graphImage

    # def setTablePlane(self, img, depth):
    #     # Call once
    #     self.table_relations=TableRelation(img, depth)
    
    # def getTableRelations(self):
    #     # Call to show on table relation
    #     return self.table_relations.relations


class SimpleStableGraph:
    def __init__(self, num_obj) -> None:

        self.num_obj=num_obj
        self.options = {
            "font_size": 14,
            "node_size": 2100,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 2,
            "width": 2,
        }
        
        self.objNames=[]
        for i in range(1,1+self.num_obj):
            try:
                self.objNames.append(cfg.LABEL_MAP[str(i)])
            except:
                self.objNames.append(str(i))


        self.G=nx.empty_graph(self.objNames)

        self.pos = nx.circular_layout(self.G)
        self.listOfRelationMatrix=[]

    def _stable(self, relation_matrix):

        
        self.stabileRelationMatrix=np.zeros((self.num_obj, self.num_obj))
        
        # queue with lenght N
        self.listOfRelationMatrix.append(relation_matrix)
        if len(self.listOfRelationMatrix) > cfg.SIMPLEGRAPH.QUEUE_LENGTH:
            self.listOfRelationMatrix.pop(0)

        
        for i in range(self.num_obj):
            for j in range(i, self.num_obj):
                temp=0
                for t in range(len(self.listOfRelationMatrix)):
                    if self.listOfRelationMatrix[t][i,j]==1:
                        temp +=1
                    
                if temp >= cfg.SIMPLEGRAPH.MIN_VISIBILITY:
                    self.stabileRelationMatrix[i,j]=1
                    self.stabileRelationMatrix[j,i]=1

    def _update_names(self):
        cfg=load_config("./config.yml")
        self.objNames=[]
        for i in range(1,1+self.num_obj):
            try:
                self.objNames.append(cfg.LABEL_MAP[str(i)])
            except:
                self.objNames.append(str(i))
        self.G=nx.empty_graph(self.objNames)
        self.pos = nx.circular_layout(self.G)


    def show(self, relation_matrix):

        self._stable(relation_matrix)

        self.G.remove_edges_from(list(self.G.edges()))
        plt.clf()
        plt.axis("off")
        axis = plt.gca()
        axis.margins(0.20)

        for i in range(self.num_obj):
            for j in range(self.num_obj):
                if self.stabileRelationMatrix[i,j]==1:
                    self.G.add_edge(self.objNames[i], self.objNames[j])
                    
        nx.draw_networkx(self.G, self.pos, **self.options)
        plt.savefig("simpleStableGraph.png")
        cv2.imshow("Simple Stable Graph", cv2.imread("simpleStableGraph.png"))        