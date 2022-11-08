import json
from os import makedirs, mkdir, path
import cv2
import numpy as np
import networkx as nx
from math import sqrt
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


from utils import convert_from_uvd, Coord, almost_equal, load_config
from Mask import mask2BoundaryPoints,  mask2BoundaryPoints, getAvrMinDistBtwPoints, getMinDistBtwPoints
from bimacs_recording import writeForBimacs


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
    """
    Express the table as plane equation, and use it to calculate the distance to table
    """
    def __init__(self, img, depth, num_obj, use_existing):

        if use_existing:
            self.table_points=np.load("table_points.npy")
            print("Found table_points.npy")
        else:
            self._selectPoints(img, depth) # sets self.table_points

        assert self.table_points.shape[0]>=3
        
        # Since "A@[x, y, z, 1] = 0" has a trivial solution (p=0), instead, I used "A@[x, y, 1] = z" 
        ones = np.ones((self.table_points.shape[0])).reshape(-1,1)
        A = np.concatenate([self.table_points[:,0:2], ones], axis=1) 
        z=self.table_points[:, 2]

        # z= a*x + b*y + c
        self.p, _, _, _ = lstsq(A, z)
        print("p: ", self.p)
        self._distToPoint = lambda _x, _y, _z: abs(_x*self.p[0] + _y*self.p[1] + self.p[2] -1*_z)/(sqrt(self.p[0]**2 + self.p[1]**2 + 1))
        
        # self.relations=np.zeros((num_obj))
        self.min_dists=np.ones((num_obj))*np.inf
        self.num_obj=num_obj

    def getMinDistFromTable(self, points):
        minDist=np.inf
        for p in points:
            dist=self._distToPoint(p.x, p.y, p.z)
            if dist<minDist:
                minDist=dist
        
        return minDist


    def _selectPoints(self, img_point_select, depth):
        ### inner func: start ###
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

        ### inner func: end ###

        self.table_points=np.array([])
        cv2.imshow("Choose points on table", img_point_select)

        cv2.setMouseCallback('Choose points on table', click_event)

        cv2.waitKey(0)
        self.table_points=self.table_points.reshape(-1, 3)
        np.save("table_points.npy", self.table_points)
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


class GraphDrawer:
    """
        Transforms semantic segmented image masks to graphs on image
    """

    def __init__(self, num_obj, beta=0.95, tr=None):


        assert (beta > 0.8) and (beta <=1.0)

        self.movingAverageCenters = {}
        self.num_obj = num_obj
        self.beta = beta
        self.relation_matrix=None

        self.simpleGraph=SimpleStableGraph(num_obj+1) # +1 is table
        
        if tr is None:
            raise ValueError
        self.tableRelation=tr


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


    def _movingAverage(self, curr_center: dict):

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

 
    @staticmethod
    def _getPCL(graphImage: np.ndarray, object_mask: np.ndarray, depth_img: np.ndarray):
        allPoints=mask2BoundaryPoints(object_mask,depth_img, addCenterToBoundary=True) #+ mask2InnerPoints(object_mask,depth_img)

        filtered_allPoints = [point for point in allPoints if not almost_equal(point.d, 0)]

        if cfg.VISIBLE.PCL:
            for p in filtered_allPoints:
                cv2.circle(graphImage, (p.u, p.v), 1, (20, 100, 90), -1)
        
        return filtered_allPoints

    def update_cfg(self):
        cfg=load_config("./config.yml")
        self.simpleGraph._update_names()

    def _testProximity(self, img, depth, mask, touch_threshold):

        assert np.unique(mask).shape[0]==2 # Only one object must be selected

        pcl=self._getPCL(None, mask, depth)

        img_test=img.copy()
        print(pcl)
        for i in range(50, img_test.shape[0]-50, 3): 
            for j in range(50, img_test.shape[1]-50, 3): 
                _,_, dist=getMinDistBtwPoints([Coord(int(j), int(i), depth.item((i, j)) )], pcl)
                
                if touch_threshold > dist:
                    img_test[i, j]=[70, 20, 250]

        img_test = cv2.resize(img_test, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("img_test", img_test)
        
    def getGraph(self, curr_centers: dict, mask, check_threshold, touch_threshold, table_threshold, depth):
        # Since detailed checking is costly, first check the strict condition; if they are not touching, select candidates
        
        # assert touch_threshold < check_threshold
        
        self._impute_and_push(curr_centers)
        self._movingAverage(curr_centers)

        centers = curr_centers

        obj_ids = centers.keys()

        if cfg.VISIBLE.SEGMENTATION:
            # graphImage=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # graphImage[(graphImage>0).all(-1)] = [204, 51, 51]

            graphImage = cv2.applyColorMap(mask*(240//self.num_obj)+15, cv2.COLORMAP_VIRIDIS) # OCEAN , COLORMAP_VIRIDIS
        else:
            graphImage = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)


        # Number_of_object + table
        self.relation_matrix=np.zeros((self.num_obj+1, self.num_obj+1))
        
        # Record all relevant distances
        distance_dict={}

        # Create pcl for all of them, and form table-object relations
        self.pcl_dict={}
        avrNumPoints=0
        for id_ in obj_ids:
            # obj_mask = (mask == int(first_id))*1
            self.pcl_dict[id_]=self._getPCL(graphImage, (mask == int(id_))*1, depth)
            avrNumPoints+=len(self.pcl_dict[id_])

            # Also calculate the distance to table
            tempDist=self.tableRelation.getMinDistFromTable(self.pcl_dict[id_])
            
            if table_threshold>tempDist:
                self.relation_matrix[id_-1, -1]=1
                self.relation_matrix[-1, id_-1]=1
                distance_dict[cfg.LABEL_MAP[str(id_)]+" to table"]=tempDist
            else:
                distance_dict["*"+cfg.LABEL_MAP[str(id_)]+" to table"]=tempDist

        print("Average selected points: ", avrNumPoints/len(self.pcl_dict))

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
                    
                    if cfg.VISIBLE.MIN_DIST:
                        cv2.putText(graphImage, f"{center2centerDistance}", ((first_center.u+second_center.u)//2, (first_center.v+second_center.v)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 20), 1)
                    
                    distance_dict[cfg.LABEL_MAP[str(first_id)]+" to "+ cfg.LABEL_MAP[str(second_id)]]=center2centerDistance

                # if their center is closer than the threshold then they are candidate for "touch"
                elif check_threshold > center2centerDistance:
                    # candidate

                    # if their closest points are touching they are touched for sure
                    # Increase N if there are many false positive 
                    minDistance=getAvrMinDistBtwPoints(self.pcl_dict[first_id], self.pcl_dict[second_id], N=6)

                    if touch_threshold > minDistance:
                        self.relation_matrix[first_id-1, second_id-1]=1
                        self.relation_matrix[second_id-1, first_id-1]=1

                        cv2.line(graphImage, (first_center.u, first_center.v), (second_center.u, second_center.v), (20, 100, 90), 3)
                        if cfg.VISIBLE.MIN_DIST:
                            cv2.putText(graphImage, f"{round(minDistance)}", ((first_center.u+second_center.u)//2, (first_center.v+second_center.v)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 20), 1)

                        distance_dict[cfg.LABEL_MAP[str(first_id)]+" to "+ cfg.LABEL_MAP[str(second_id)]]=minDistance
                    
                    elif 2*touch_threshold > minDistance:
                        distance_dict["*"+cfg.LABEL_MAP[str(first_id)]+" to "+ cfg.LABEL_MAP[str(second_id)]]=minDistance


        # Emphasize object centers
        for _id in obj_ids:
            center = centers[_id]
            cv2.circle(graphImage, (center.u, center.v), 2, (0, 51, 255), -1)

            if cfg.VISIBLE.CENTER_COORDS:
                cv2.putText(graphImage, f"{center}", (center.u - 25, center.v + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 204), 1)

        ssg=self.simpleGraph.show(self.relation_matrix)

        return graphImage, distance_dict, ssg


class SimpleStableGraph:
    def __init__(self, num_obj) -> None:

        self.num_obj=num_obj
        self.options = {
            "font_size": 9,
            "node_size": 1500,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 1.4,
            "width": 1.4,
        }
        
        self.frame_num=0

        self.objNames=[]
        self.listOfRelationMatrix=[]
        self._getNames()

    def _getNames(self):
        global cfg
        for i in range(1,1+self.num_obj):
            try:
                self.objNames.append(cfg.LABEL_MAP[str(i)]+"_"+str(i))
            except:
                self.objNames.append(str(i))


        self.G=nx.empty_graph(self.objNames)
        self.pos = nx.circular_layout(self.G)


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
        global cfg
        cfg=load_config("./config.yml")
        self.objNames=[]
        # for i in range(1,1+self.num_obj):
        #     try:
        #         self.objNames.append(cfg.LABEL_MAP[str(i)])
        #     except:
        #         self.objNames.append(str(i))
        # self.G=nx.empty_graph(self.objNames)
        # self.pos = nx.circular_layout(self.G)
        self._getNames()


    def show(self, relation_matrix):

        self._stable(relation_matrix)

        self.G.remove_edges_from(list(self.G.edges()))
        fig = plt.figure()
        plt.clf()
        plt.axis("off")
        axis = fig.gca()
        axis.margins(0.20)

        print(self.stabileRelationMatrix)

        writeForBimacs(self.stabileRelationMatrix, "exp7_graphs", self.frame_num)
        

        for i in range(self.num_obj):
            for j in range(self.num_obj):
                if self.stabileRelationMatrix[i,j]==1:
                    self.G.add_edge(self.objNames[i], self.objNames[j])
                    
        nx.draw_networkx(self.G, self.pos, **self.options)
        # plt.savefig("simpleStableGraph.png")
        # cv2.imshow("Simple Stable Graph", cv2.imread("simpleStableGraph.png"))
        # cv2.imshow("Simple Stable Graph", ssg)
        fig.canvas.draw()
        ssg = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        ssg = ssg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        trim_coef=70
        ssg = cv2.resize(ssg[trim_coef:-trim_coef, trim_coef:-trim_coef], (ssg.shape[1], ssg.shape[0]))

        # makedirs(path.join("exp3_graphs", "graph_imgs"), exist_ok=True)
        # cv2.imwrite(path.join("exp3_graphs", "graph_imgs", f"graph_{self.frame_num}.png"), ssg)


        self.frame_num+=1
             
        return ssg


