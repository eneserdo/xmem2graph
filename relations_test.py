import glob
import natsort 
from relations import get_relation_matrix
from os.path import join as opj
import json
import numpy as np

RELATION_LIST = ['contact', 'above', 'below', 'left of', 'right of', 'behind of', 'in front of',
                 'inside', 'surround', 'moving together', 'halting together', 'fixed moving together',
                 'getting close', 'moving apart', 'stable']

# RELATIONS_TO_CHECK = ['contact', 'above', 'below', 'left of', 'right of', 'behind of', 'in front of']
RELATIONS_TO_CHECK = ['moving apart']


def relation_tensor_to_set(relation_matrix, obj_names):
    
    relation_set = set()

    for i in range(len(obj_names)):
        for j in range(len(obj_names)):
            for k in range(len(RELATION_LIST)):
                if relation_matrix[i, j, k] == 1:
                    if RELATION_LIST[k] in RELATIONS_TO_CHECK:
                        relation_set.add(f"{obj_names[i]} - {RELATION_LIST[k]} - {obj_names[j]}")
                    
    return relation_set


def checkMyImplementation(base_dir, subject, task, take, i):
    # derived_dir=base_dir+"bimacs_derived_data/"


    derived_dir=opj(base_dir, "bimacs_derived_data")

    _3d_objects_dirs=opj(derived_dir, subject, task, take, "3d_objects")
    _3d_objects_json_list=natsort.natsorted(glob.glob(opj(_3d_objects_dirs, "*.json")))

    relation_json_dirs=opj(derived_dir, subject, task, take, "spatial_relations")
    relations_path=natsort.natsorted(glob.glob(opj(relation_json_dirs, "*.json")))


    with open(relations_path[i]) as f:
        relations = json.load(f)

    with open(_3d_objects_json_list[i]) as f:
        _3d = json.load(f)


    rels=set()
    for relation in relations:
        oi=relation["object_index"]
        si=relation["subject_index"]

        if relation["relation_name"] in RELATIONS_TO_CHECK:
            rels.add(_3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"])

    # get bounding boxes from _3d json
    bbox=np.zeros((len(_3d), 6))
    past_bbox=np.zeros((len(_3d), 6))
    obj_names=[]
    for i, obj in enumerate(_3d):
        obj_names.append(obj["class_name"])
        bb=obj["bounding_box"]
        bbox[i]=np.array([bb["x0"], bb["x1"], bb["y0"], bb["y1"], bb["z0"], bb["z1"]])
        pbb=obj["past_bounding_box"]
        past_bbox[i]=np.array([pbb["x0"], pbb["x1"], pbb["y0"], pbb["y1"], pbb["z0"], pbb["z1"]])
        
    rels_predicted_mat, _ = get_relation_matrix(bbox, past_bbox, 30, obj_names)
    rels_predicted=relation_tensor_to_set(rels_predicted_mat, obj_names)
    
    if set(rels_predicted) != rels:
        print("preds: ", "\n".join(list(rels_predicted)))
        print("*****")
        print("gt: ", "\n".join(list(rels)))
        print("*******************")
        return False
    else:
        print("True positive")
        print("gt: ", "\n".join(list(rels)))
        print("*******************")
        return True


checkMyImplementation("/home/cak/enes/bimanual/bimacs_rgbd_data", "subject_1", "task_1_k_cooking", "take_0", 10)
