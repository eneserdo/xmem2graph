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
RELATIONS_TO_CHECK = ['getting close', 'moving apart', 'stable']

THRESHOLD = 30

def crawl_dataset():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
                     'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
                     'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing']:
            for take in ['take_{}'.format(take_i) for take_i in range(10)]:
                print(".", end="", flush=True)
                yield subject, task, take
            
            print()


def relation_tensor_to_set(relation_matrix, obj_names):    
    relation_set = set()

    for i in range(len(obj_names)):
        for j in range(len(obj_names)):
            for k in range(len(RELATION_LIST)):
                if relation_matrix[i, j, k] == 1:
                    if RELATION_LIST[k] in RELATIONS_TO_CHECK:
                        relation_set.add(f"{obj_names[i]} - {RELATION_LIST[k]} - {obj_names[j]}")
                    
    return relation_set


def checkMyImplementation(base_dir, subject, task, take, frame_idx):
    # derived_dir=base_dir+"bimacs_derived_data/"


    derived_dir=opj(base_dir, "bimacs_derived_data")

    _3d_objects_dirs=opj(derived_dir, subject, task, take, "3d_objects")
    _3d_objects_json_list=natsort.natsorted(glob.glob(opj(_3d_objects_dirs, "*.json")))

    relation_json_dirs=opj(derived_dir, subject, task, take, "spatial_relations")
    relations_path=natsort.natsorted(glob.glob(opj(relation_json_dirs, "*.json")))


    with open(relations_path[frame_idx]) as f:
        relations = json.load(f)

    with open(_3d_objects_json_list[frame_idx]) as f:
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
    for obj_id, obj in enumerate(_3d):
        obj_names.append(obj["class_name"])
        bb=obj["bounding_box"]
        bbox[obj_id]=np.array([bb["x0"], bb["x1"], bb["y0"], bb["y1"], bb["z0"], bb["z1"]])
        pbb=obj["past_bounding_box"]
        past_bbox[obj_id]=np.array([pbb["x0"], pbb["x1"], pbb["y0"], pbb["y1"], pbb["z0"], pbb["z1"]])
        
    rels_predicted_mat, _ = get_relation_matrix(bbox, past_bbox, THRESHOLD)
    rels_predicted = relation_tensor_to_set(rels_predicted_mat, obj_names)
    
    if set(rels_predicted) != rels:
        print("\nfile: ", relations_path[frame_idx])
        print("preds: ", "\n".join(list(rels_predicted)))
        print("gt: ", "\n".join(list(rels)))
        
        print("**********")
        return False
    else:
        # print("OK")
        # print("**********")
        return True


if __name__ == "__main__":

    print("Checking: ", RELATIONS_TO_CHECK)
    err_count = 0
    err_limit = 1
    derived_dir="/home/enerd/bimanual/bimacs_derived_data"    
    for subject, task, take in crawl_dataset():
        relation_json_dirs=opj(derived_dir, subject, task, take, "spatial_relations")
        relations_path=glob.glob(opj(relation_json_dirs, "*.json"))

        for frame_idx in range(len(relations_path)):
            ret = checkMyImplementation("/home/enerd/bimanual", subject, task, take, frame_idx)
            err_count += 1 if ret == False else 0
            
            if err_count ==err_limit:
                exit(0)

    print("OK: ", RELATIONS_TO_CHECK)

    # if checkMyImplementation("/home/enerd/bimanual", "subject_1", "task_1_k_cooking", "take_0", 10):
    #     print("OK: ", RELATIONS_TO_CHECK)
