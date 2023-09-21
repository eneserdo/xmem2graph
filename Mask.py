import cv2
import numpy as np
from PIL import Image

from utils import Coord, convert_from_uvd, print_debug, arrayAndDepthToCoord, almost_equal, load_config

cfg=load_config("./config.yml")


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

    # filter out points with depth close to 0
    mask_to_keep = depth[boundaryPoints[:, 0], boundaryPoints[:, 1]] > 1e-4
    boundaryPoints = boundaryPoints[mask_to_keep]

    boundaryPoints_xyz = convert_from_uvd(boundaryPoints[:, 0], boundaryPoints[:, 1], depth[boundaryPoints[:, 0], boundaryPoints[:, 1]], is_np=True)
    # bP_as_coord = []
    # if not addCenterToBoundary:
        # for bP in boundaryPoints:
        #     # bP[0]=v, bP[1]=u
        #     # if not almost_equal(depth.item((bP[0], bP[1])), 0):
        #     bP_as_coord.append(Coord(bP.item(1), bP.item(0), depth.item((bP[0], bP[1]))))
    # else:
    #     M = cv2.moments(thresh)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     center=np.array([cY, cX])

    #     ratios=[0.4, 0.75, 0.98]

    #     for i, bP in enumerate(boundaryPoints):
    #         for k, r in enumerate(ratios):
    #             """
    #             k=0: get every 3rd
    #             k=1: get every 2nd
    #             k=2: get everything
    #             """
    #             if i%(len(ratios)-k)==0:
    #                 tempP=(bP*r+(1-r)*center).astype(np.uint32)
    #                 bP_as_coord.append(Coord(tempP.item(1), tempP.item(0), depth.item((tempP[0], tempP[1]))))
                
    # return bP_as_coord
    return boundaryPoints_xyz


def mask2InnerPoints(obj_mask: np.ndarray, depth: np.ndarray):
    points_v, points_u = np.where(obj_mask*5 > 1)
    points = np.concatenate([points_v.reshape(-1, 1), points_u.reshape(-1, 1)], axis=1)

    # subset = np.random.choice(len(points), max(int(len(points)*cfg.GRAPHER.SUBSET_INSIDE), cfg.GRAPHER.MIN_INSIDE))
    subset = np.random.choice(len(points), cfg.GRAPHER.MIN_INSIDE)
    points_subset = points[subset]
    points_subset = arrayAndDepthToCoord(points_subset, depth)
    return points_subset


def _mask2SingleCenter(pred, depth):
    """
        If pred has SINGLE segments
    """
    _, thresh = cv2.threshold(pred*20, 5, 255, 0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])


    # return np.array([cY, cX, depth[cY, cX][0]])
    # return np.array([cY, cX])
    return Coord(cX, cY, depth.item((cY, cX)))
     

def mask2orderedCenters(pred, depth):

    ids = np.unique(pred)[1:]
    centers = {}

    for id in ids:
        # centers[id] = getSingleCenterFromMask((pred == id).astype(pred.dtype), depth)
        centers[id] = _mask2SingleCenter((pred == id).astype(pred.dtype), depth)
        # if display:
        #     cv2.circle(pred, (centers[id][1], centers[id][0]), 3, 255, -1)
        #     cv2.putText(pred, f"{centers[id]}", (centers[id][1] - 1, centers[id][0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    return centers


def adaptMask(mask_name):
    """
    XMem requires mask image to have unique values of [0, 1, 2 ... num_obj]
    """
    mask_image = np.array(Image.open(mask_name))
    uniques = np.unique(mask_image)
    # print(uniques)

    num_obj = len(uniques) - 1

    new_mask = np.zeros_like(mask_image).astype(np.int64)

    for i, m in enumerate(uniques):
        new_mask += (mask_image == m)*i

    mask_image = new_mask.astype(np.uint8).copy()

    # print(np.unique(mask_image))
    num_obj = len(np.unique(mask_image)) - 1
    print("Mask is ready")
    return num_obj, mask_image
