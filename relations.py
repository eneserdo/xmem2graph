# Functions to compute relations between objects based on bounding boxes (not point clouds) 
# I kinda imitated the bimanual's "feature-extraction-pipeline" repo (more specifically evaluate_relations.cpp)

import numpy as np
import cv2

# Static relations: contact, above, below, left of, right of, behind of, in front of, inside, surround
# Dynamic relations: moving together, halting together, fixed moving together, getting close, moving apart, stable

RELATION_LIST = ['contact', 'above', 'below', 'left of', 'right of', 'behind of', 'in front of',
                 'inside', 'surround', 'moving together', 'halting together', 'fixed moving together',
                 'getting close', 'moving apart', 'stable']

RELATION_LIST_SHORT = ['C', 'A', 'B', 'L', 'R', 'BE', 'FR',
                       'IN', 'SUR', 'MOV', 'HLT', 'FMOV',
                       'CLO', 'MOVA', 'S']


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


def get_single_bbox_from_pcl(coords_list):
    min_x, min_y, min_z = np.inf, np.inf, np.inf
    max_x, max_y, max_z = -np.inf, -np.inf, -np.inf

    for c in coords_list:
        if c.x < min_x:
            min_x = c.x
        if c.x > max_x:
            max_x = c.x


def get_pcl_to_bbox(pointcloud_dict, N):
    """returns the bounding box of each object, if the object is not visible, the corresponding row is all zeros"""

    # pointcloud_dict is a dictionary of list of Coords
    bboxes = np.zeros((N, 6)) # [x0, x1, y0, y1, z0, z1]

    # for idx, pcl in pointcloud_dict.items():
    
    # keys starts from 1 for some reason
    for idx in range(N):
        # pcl: N x 3
        # pcl = pointcloud_dict.get(idx+1, None)
        
        if not (idx+1 in pointcloud_dict):
            bboxes[idx] = 0
        
        else:
            pcl = pointcloud_dict.get(idx+1)
            try:
                pcl_T = pcl.T
                mins = np.min(pcl_T, axis=1)
                maxs = np.max(pcl_T, axis=1)
            except:
                # print(f"Probably no point is selected", idx+1)
                bboxes[idx] = 0
                continue

            bboxes[idx] = np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]])

            if np.allclose(bboxes[idx], 0):
                print("pcl_T:", pcl_T)
                raise ValueError("Bounding box is all zeros")

    return bboxes


def get_relation_matrix(bbox_list, past_bbox_list, distance_equality_threshold):
    # bbox_list: N x 6 : [x0, x1, y0, y1, z0, z1]

    # Assumes that the objects are ordered in the same way in both bbox_list and past_bbox_list and they are the same objects. 
    # If an object is not visible, the corresponding row in bbox_list/past_bbox_list should be -1

    N=len(bbox_list)
    relation_matrix = np.zeros((N, N, len(RELATION_LIST)))
    distance_matrix = np.zeros((N, N))
    
    # Static relations
    for subject_idx in range(N):        
        for object_idx in range(subject_idx+1, N):
            
            subject_bb = bbox_list[subject_idx]
            object_bb = bbox_list[object_idx]

            # TODO: check if the -1 is the correct way to handle this
            if np.allclose(subject_bb, -1) or np.allclose(object_bb, -1):
                continue

            ### Static relations

            # contact:
            if IS_CONTACTING(subject_bb, object_bb):
                relation_matrix[subject_idx, object_idx, RELATION('contact')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('contact')] = 1
            
            # left of and right of:
            if subject_bb[1] < object_bb[0]:
                relation_matrix[subject_idx, object_idx, RELATION('right of')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('left of')] = 1

            elif subject_bb[0] > object_bb[1]:
                relation_matrix[subject_idx, object_idx, RELATION('left of')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('right of')] = 1

            # above and below:
            if subject_bb[3] < object_bb[2]:
                relation_matrix[subject_idx, object_idx, RELATION('above')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('below')] = 1

            elif subject_bb[2] > object_bb[3]:
                relation_matrix[subject_idx, object_idx, RELATION('below')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('above')] = 1
            
            # behind of and in front of:
            if subject_bb[5] < object_bb[4]:
                relation_matrix[subject_idx, object_idx, RELATION('in front of')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('behind of')] = 1
            
            elif subject_bb[4] > object_bb[5]:
                relation_matrix[subject_idx, object_idx, RELATION('behind of')] = 1
                relation_matrix[object_idx, subject_idx, RELATION('in front of')] = 1
            
            # inside and surround:
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


            ### Dynamic relations
            subject_past_bb = past_bbox_list[subject_idx]
            object_past_bb = past_bbox_list[object_idx]
            
            # TODO: check if the -1 is the correct way to handle this
            if np.allclose(subject_past_bb, -1) or np.allclose(object_past_bb, -1):
                continue

            delta_so=distance_btw_bb(subject_bb, object_bb)
            delta_so_past=distance_btw_bb(subject_past_bb, object_past_bb)

            distance_matrix[subject_idx, object_idx] = delta_so
            distance_matrix[object_idx, subject_idx] = delta_so

            # Şimdi de geçmişte de temas ediyorlar
            p1 = IS_CONTACTING(subject_bb, object_bb) and IS_CONTACTING(subject_past_bb, object_past_bb)

            # Şimdi de geçmişte de temas etmiyorlar
            p2 = not IS_CONTACTING(subject_bb, object_bb) and not IS_CONTACTING(subject_past_bb, object_past_bb)

            if p1:

                p3 = distance_btw_bb(object_bb, object_past_bb) < (distance_equality_threshold / 2)
                p4 = distance_btw_bb(subject_bb, subject_past_bb) < (distance_equality_threshold / 2)

                if p3 and p4:
                    relation_matrix[subject_idx, object_idx, RELATION('moving together')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('moving together')] = 1
                
                elif (not p3) and (not p4):
                    relation_matrix[subject_idx, object_idx, RELATION('halting together')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('halting together')] = 1
                
                elif p3 ^ p4:
                    relation_matrix[subject_idx, object_idx, RELATION('fixed moving together')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('fixed moving together')] = 1
            
            elif p2:
                if delta_so - delta_so_past < - distance_equality_threshold:
                    relation_matrix[subject_idx, object_idx, RELATION('getting close')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('getting close')] = 1
                
                elif delta_so - delta_so_past > distance_equality_threshold:
                    relation_matrix[subject_idx, object_idx, RELATION('moving apart')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('moving apart')] = 1
                
                else:
                    relation_matrix[subject_idx, object_idx, RELATION('stable')] = 1
                    relation_matrix[object_idx, subject_idx, RELATION('stable')] = 1


    return relation_matrix, distance_matrix


def get_relation_string(relation_matrix, obj_names):
    
    relation_string = ""

    for i in range(len(obj_names)):
        for j in range(i+1, len(obj_names)):
            for k in range(len(RELATION_LIST)):
                if relation_matrix[i, j, k] == 1:
                    relation_string += f"{obj_names[i]}-{RELATION_LIST_SHORT[k]}-{obj_names[j]}\n"
    
    return relation_string

