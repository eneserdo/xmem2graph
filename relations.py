# Functions to compute relations between objects based on bounding boxes (not point clouds) 
# I kinda imitated the bimanual's "feature-extraction-pipeline" repo (more specifically evaluate_relations.cpp)

import numpy as np
import cv2

# Static relations: contact, above, below, left of, right of, behind of, in front of, inside, surround
# Dynamic relations: moving together, halting together, fixed moving together, getting close, moving apart, stable

RELATION_LIST = ['contact', 'above', 'below', 'left of', 'right of', 'behind of', 'in front of',
             'inside', 'surround', 'moving together', 'halting together', 'fixed moving together',
             'getting close', 'moving apart', 'stable']

RELATION = lambda a : RELATION_LIST.index(a)

def IS_CONTACTING(bbox_a, bbox_b): 
    return bbox_a[0] <= bbox_b[1] and bbox_a[1] >= bbox_b[0] and bbox_a[2] <= bbox_b[3] and bbox_a[3] >= bbox_b[2] and bbox_a[4] <= bbox_b[5] and bbox_a[5] >= bbox_b[4]


def distance_btw_bb(bbox_a, bbox_b):
    # return np.sqrt((bbox_a[0]-bbox_b[0])**2 + (bbox_a[1]-bbox_b[1])**2 + (bbox_a[2]-bbox_b[2])**2)
    cax = (bbox_a[1] + bbox_a[0]) / 2
    cay = (bbox_a[3] + bbox_a[2]) / 2
    caz = (bbox_a[5] + bbox_a[4]) / 2
    cbx = (bbox_b[1] + bbox_b[0]) / 2
    cby = (bbox_b[3] + bbox_b[2]) / 2
    cbz = (bbox_b[5] + bbox_b[4]) / 2

    radicand = (cax - cbx)**2 + (cay - cby)**2 + (caz - cbz)**2
    distance = np.sqrt(radicand)

    return distance



def get_pointcloud_list(mask_img, depth_img):
    # Mask of the objects and depth image to point cloud
    ...


def get_single_bbox_from_pcl(coords_list):
    min_x, min_y, min_z = np.inf, np.inf, np.inf
    max_x, max_y, max_z = -np.inf, -np.inf, -np.inf

    for c in coords_list:
        if c.x < min_x:
            min_x = c.x
        if c.x > max_x:
            max_x = c.x


def get_pcl_to_bbox(pointcloud_dict, N):
    # Get the bounding box of each object
    # pointcloud_dict is a dictionary of list of Coords
    
    bbox=np.zeros((N, 6)) # [x0, x1, y0, y1, z0, z1]

    for idx, pcl in pointcloud_dict.items():
        bbox[idx] = 

def get_relation_matrix(bbox_list, past_bbox_list, ignore_mask, distance_equality_threshold):
    # bbox_list: N x 6 : [x0, x1, y0, y1, z0, z1]
    # ignore_mask: N x 1 : 1 if the object is not visible, 0 otherwise

    # FIXME Assumes that the objects are ordered in the same way in both bbox_list and past_bbox_list and they are the same objects

    relation_matrix = np.zeros((len(bbox_list), len(bbox_list), len(RELATION_LIST)))
    N=len(bbox_list)
    
    # Static relations
    for subject_idx in range(N):
        for object_idx in range(subject_idx+1, N):
            
            subject_bb = bbox_list[subject_idx]
            object_bb = bbox_list[object_idx]
            
            # contact:
            if IS_CONTACTING(subject_bb, object_bb):
                relation_matrix[subject_idx, object_idx, RELATION('contact')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('contact')] = 1
            
            # left of and right of:
            if subject_bb[1] < object_bb[0]:
                relation_matrix[subject_idx, object_idx, RELATION('left of')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('right of')] = 1

            elif subject_bb[0] > object_bb[1]:
                relation_matrix[subject_idx, object_idx, RELATION('right of')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('left of')] = 1

            # above and below:
            if subject_bb[3] < object_bb[2]:
                relation_matrix[subject_idx, object_idx, RELATION('below')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('above')] = 1

            elif subject_bb[2] > object_bb[3]:
                relation_matrix[subject_idx, object_idx, RELATION('above')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('below')] = 1
            
            # behind of and in front of:
            if subject_bb[5] < object_bb[4]:
                relation_matrix[subject_idx, object_idx, RELATION('behind of')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('in front of')] = 1
            
            elif subject_bb[4] > object_bb[5]:
                relation_matrix[subject_idx, object_idx, RELATION('in front of')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('behind of')] = 1
            
            # inside and surround:x
            if subject_bb[0] >= object_bb[0] and subject_bb[1] <= object_bb[1] and \
               subject_bb[2] >= object_bb[2] and subject_bb[3] <= object_bb[3] and \
               subject_bb[4] >= object_bb[4] and subject_bb[5] <= object_bb[5]:
        
                relation_matrix[subject_idx, object_idx, RELATION('inside')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('surround')] = 1
            
            elif object_bb[0] >= subject_bb[0] and object_bb[1] <= subject_bb[1] and \
                 object_bb[2] >= subject_bb[2] and object_bb[3] <= subject_bb[3] and \
                 object_bb[4] >= subject_bb[4] and object_bb[5] <= subject_bb[5]:
                
                relation_matrix[subject_idx, object_idx, RELATION('surround')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('inside')] = 1


            # Dynamic relations
            subject_past_bb = past_bbox_list[subject_idx]
            object_past_bb = past_bbox_list[object_idx]
            delta_so=distance_btw_bb(subject_bb, object_bb)
            delta_so_past=distance_btw_bb(subject_past_bb, object_past_bb)


            p1 = IS_CONTACTING(subject_bb, object_bb) and IS_CONTACTING(subject_past_bb, object_past_bb)

            p2 = not IS_CONTACTING(subject_bb, object_bb) and not IS_CONTACTING(subject_past_bb, object_past_bb)

            if p1:

                p3 = distance_btw_bb(object_bb, object_past_bb) < (distance_equality_threshold / 2)
                p4 = distance_btw_bb(subject_bb, subject_past_bb) < (distance_equality_threshold / 2)

                if p3 and p4:
                    relation_matrix[subject_idx, object_idx, RELATION('moving together')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('moving together')] = 1
                
                elif not p3 and not p4:
                    relation_matrix[subject_idx, object_idx, RELATION('halting together')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('halting together')] = 1
                
                elif p3 ^ p4:
                    relation_matrix[subject_idx, object_idx, RELATION('fixed moving together')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('fixed moving together')] = 1
            
            elif p2:
                if delta_so - delta_so_past < -distance_equality_threshold:
                    relation_matrix[subject_idx, object_idx, RELATION('getting close')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('getting close')] = 1
                
                elif delta_so - delta_so_past > distance_equality_threshold:
                    relation_matrix[subject_idx, object_idx, RELATION('moving apart')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('moving apart')] = 1
                
                else:
                    relation_matrix[subject_idx, object_idx, RELATION('stable')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('stable')] = 1



    return relation_matrix