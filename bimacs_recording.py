import json
from os import makedirs, mkdir, path

import numpy as np
from utils import load_config


cfg = load_config("./config.yml")  # buna gerek olmayabilir

bimacsMap = {
    "bowl": 0,
    "knife": 1,
    "screwdriver": 2,
    "cuttingboard": 3,
    "whisk": 4,
    "hammer": 5,
    "bottle": 6,
    "cup": 7,
    "banana": 8,
    "cereals": 9,
    "sponge": 10,
    "woodenwedge": 11,
    "saw": 12,
    "harddrive": 13,
    "LeftHand": 14,
    "RightHand": 15,
}


def _get3dDict(class_name, instance_name):
    return {
        "bounding_box": {"x0": 0, "x1": 0, "y0": 0, "y1": 0, "z0": 0, "z1": 0},
        "certainty": 1,
        "class_index": bimacsMap[class_name],
        "class_name": class_name,
        "colour": [0, 0, 0],
        "instance_name": instance_name,
        "past_bounding_box": {"x0": 0, "x1": 0, "y0": 0, "y1": 0, "z0": 0, "z1": 0},
    }


def _getRelationDict(oi, si):
    return {"object_index": oi, "relation_name": "contact", "subject_index": si}


def writeForBimacs(relationMatrix, folder_name, frame_number):

    relationDict = []
    d3Dict = []

    # assume that last element is "table" always
    relationMatrix = relationMatrix[:-1, :-1]

    if frame_number == 0:
        relationMatrix = np.ones_like(relationMatrix) - np.eye(relationMatrix.shape[0])

    axis_h, axis_w = np.where(relationMatrix > 0)
    contacts = np.concatenate([axis_h.reshape(-1, 1), axis_w.reshape(-1, 1)], axis=1)

    temp = {}
    i = 0
    for obj_i in range(relationMatrix.shape[0]):
        if obj_i in contacts[:, 0]:

            object_name = cfg.LABEL_MAP[str(obj_i + 1)]
            d3Dict.append(_get3dDict(object_name, f"{object_name}_{str(obj_i+1)}"))
            instance_name=object_name+str(obj_i+1)
            temp[instance_name] = i
            i += 1

    for c in contacts:
        object_instance_name = cfg.LABEL_MAP[str(c[0] + 1)]+str(c[0] + 1)
        subject_instance_name = cfg.LABEL_MAP[str(c[1] + 1)]+str(c[1] + 1)
        # object_i=bimacsMap[object_name]
        # subject_i=bimacsMap[subject_name]

        relationDict.append(_getRelationDict(temp[object_instance_name], temp[subject_instance_name]))
        # d3Dict.append(_get3dDict(object_name, f"{object_name}_{str(c[0]+1)}"))

    spatialRelation = path.join(folder_name, "spatial_relations")
    d3_objects = path.join(folder_name, "3d_objects")

    if frame_number == 0:
        print("Starts saving graphs...")
        makedirs(spatialRelation, exist_ok=True)
        makedirs(d3_objects, exist_ok=True)

    with open(path.join(spatialRelation, f"frame_{frame_number}.json"), "w") as outfile:
        json.dump(relationDict, outfile, indent=4)

    with open(path.join(d3_objects, f"frame_{frame_number}.json"), "w") as outfile:
        json.dump(d3Dict, outfile, indent=4)

# def writeForBimacs(relationMatrix, folder_name, frame_number):

#     relationDict = []
#     d3Dict = []

#     # assume that last element is "table" always
#     relationMatrix = relationMatrix[:-1, :-1]

#     if frame_number == 0:
#         relationMatrix = np.ones_like(relationMatrix) - np.eye(relationMatrix.shape[0])

#     axis_h, axis_w = np.where(relationMatrix > 0)
#     contacts = np.concatenate([axis_h.reshape(-1, 1), axis_w.reshape(-1, 1)], axis=1)

#     temp = {}
#     i = 0
#     for obj_i in range(relationMatrix.shape[0]):
#         if obj_i in contacts[:, 0]:
#             object_name = cfg.LABEL_MAP[str(obj_i + 1)]
#             d3Dict.append(_get3dDict(object_name, f"{object_name}_{str(obj_i+1)}"))
#             temp[object_name] = i
#             i += 1

#     for c in contacts:
#         object_name = cfg.LABEL_MAP[str(c[0] + 1)]
#         subject_name = cfg.LABEL_MAP[str(c[1] + 1)]
#         # object_i=bimacsMap[object_name]
#         # subject_i=bimacsMap[subject_name]

#         relationDict.append(_getRelationDict(temp[object_name], temp[subject_name]))
#         # d3Dict.append(_get3dDict(object_name, f"{object_name}_{str(c[0]+1)}"))

#     spatialRelation = path.join(folder_name, "spatial_relations")
#     d3_objects = path.join(folder_name, "3d_objects")

#     if frame_number == 0:
#         print("Starts saving graphs...")
#         makedirs(spatialRelation, exist_ok=True)
#         makedirs(d3_objects, exist_ok=True)

#     with open(path.join(spatialRelation, f"frame_{frame_number}.json"), "w") as outfile:
#         json.dump(relationDict, outfile, indent=4)

#     with open(path.join(d3_objects, f"frame_{frame_number}.json"), "w") as outfile:
#         json.dump(d3Dict, outfile, indent=4)


def test_func():
    rel = np.zeros((6, 6))
    # rel[0, 1] = 1
    # rel[1, 0] = 1

    # rel[4, 2] = 1
    # rel[2, 4] = 1

    # rel[1, 2] = 1
    # rel[2, 1] = 1
    rel = np.ones_like(rel) - np.eye(rel.shape[0])
    print(rel)

    writeForBimacs(rel, "samplef", 0)


# class oneD_recording:
#     def __init__(self) -> None:

#         pass

#     def add_matrix(self, relationMatrix):


#         # assume that last element is "table" always
#         relationMatrix = relationMatrix[:-1, :-1]
#         object_name = cfg.LABEL_MAP[str(obj_i + 1)]
        