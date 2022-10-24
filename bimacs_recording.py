import json
from os import mkdir, path

import numpy as np
from utils import load_config


cfg=load_config("./config.yml") # buna gerek olmayabilir

bimacsMap={'bowl':1, 'knife':2, 'screwdriver':3, 'cuttingboard':4, 'whisk':5, 'hammer':6, 'bottle':7, 'cup':8,
           'banana':9, 'cereals':10, 'sponge':11, 'woodenwedge':12, 'saw':13, 'harddrive':14, 'LeftHand':15,
           'RightHand':16}


def get3dDict(class_name, instance_name):
    return {"bounding_box": { "x0": 0, "x1": 0, "y0": 0, "y1": 0, "z0": 0, "z1": 0},
            "certainty": 1,
            "class_index": bimacsMap[class_name],
            "class_name": class_name,
            "colour": [0,0,0],
            "instance_name": instance_name,
            "past_bounding_box": { "x0": 0, "x1": 0, "y0": 0, "y1": 0, "z0": 0, "z1": 0}}

    
def getRelationDict(oi, si):
    return {'object_index': oi, 'relation_name': 'contact', 'subject_index': si}


def writeForBimacs(relationMatrix, folder_name, frame_number):
    
    relationDict=[]
    d3Dict=[]

    # assume that last element is "table" always
    relationMatrix=relationMatrix[:-1, :-1]
    axis_h, axis_w = np.where(relationMatrix > 0)
    contacts = np.concatenate([axis_h.reshape(-1, 1), axis_w.reshape(-1, 1)], axis=1)
    
    for c in contacts:
        object_name=cfg.LABEL_MAP[str(c[0]+1)]
        subject_name=cfg.LABEL_MAP[str(c[1]+1)]
        object_i=bimacsMap[object_name]
        subject_i=bimacsMap[subject_name]

        relationDict.append(getRelationDict(object_i, subject_i))
        d3Dict.append(get3dDict(object_name, f"{object_name}_{str(c[0]+1)}"))

    spatialRelation=path.join(folder_name, "spatial_relations")
    d3_objects=path.join(folder_name, "3d_objects")

    if frame_number==0:
        mkdir(spatialRelation)
        mkdir(d3_objects)
    
    with open(path.join(spatialRelation, f"frame_{frame_number}.json"), 'w') as outfile:
        json.dump(relationDict, outfile, indent=4)

    with open(path.join(d3_objects, f"frame_{frame_number}.json"), 'w') as outfile:
        json.dump(d3Dict, outfile, indent=4)

rel=np.zeros((5,5))
rel[0,1]=1
rel[1,0]=1

rel[4,2]=1
rel[2,4]=1

rel[1,2]=1
rel[2,1]=1
print(rel)

writeForBimacs(rel, "samplef", 1)