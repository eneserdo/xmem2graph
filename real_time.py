import os
import sys
import cv2
import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
import numpy as np
from time import perf_counter
import subprocess

sys.path.append("./XMem")
# from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
# from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis


# My libs
from ros_capture import RosRun, shift, getBlended
from grapher import GraphDrawer, TableRelation
from utils import Timer, convert_from_uvd, Coord, load_config, argparser, add_text_to_image, dictToText
from Mask import mask2orderedCenters, adaptMask



ZOOM_COEF=1.2

cfg=load_config("./config.yml")

def putNames(graphImg, cntrs):
    for id, center in cntrs.items():
        try:
            cv2.putText(graphImg, cfg.LABEL_MAP[str(id)], (center.u - 10, center.v - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,60,200), 1)    
        except:
            cv2.putText(graphImg, "?", (center.u - 10, center.v - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,60,200), 1)    


prev_coord=None
def print_coord(event,x,y, flags, param):
    global prev_coord
    if event == cv2.EVENT_LBUTTONDOWN:
        x=x/ZOOM_COEF
        y=y/ZOOM_COEF
        print(round(x),round(y))
        x1,y1,z1=convert_from_uvd(x, y, depth_img.item((int(y),int(x))))
        print(f"({round(x1, 2)}, {round(y1, 2)}, {round(z1, 2)})")	
        if prev_coord is not None:
            print("dist: ", np.round(np.sqrt(np.sum((prev_coord-np.array([x1, y1, z1]))**2))))
            try:
                print("distToTable: ", tableRelation._distToPoint(x1,y1,z1))
            except:
                pass

        prev_coord=np.array([x1, y1, z1])



def nothing(x):
    pass


def main(args):
    global depth_img, tableRelation, cfg
    torch.set_grad_enabled(False)

    # default configuration for XMem
    config = {
        'top_k': 30,
        'mem_every': 5,
        'deep_update_every': -1,
        'enable_long_term': True,
        'enable_long_term_count_usage': True,
        'num_prototypes': 128,
        'min_mid_term_frames': 5,
        'max_mid_term_frames': 10,
        'max_long_term_elements': 10000,
    }


    assert torch.cuda.is_available()
    device="cuda"


    print("CUDA version: ",torch.version.cuda)
    network = XMem(config, './XMem/saves/XMem.pth').eval().to(device)

    torch.cuda.empty_cache()

    mask_ready = args.mask_ready
    set_mask = args.mask_ready or args.set_mask
    
    shift_x = args.shift_x
    shift_y = args.shift_y

    verbosity=args.verbosity_level
    
    # rosrunner = RosRun()
    record_frames=args.record_frames
    run_xmem=False
    show_blended=False
    graph_show_details=True
    fps=0
    timer=Timer()
    i=1
    import glob

    with torch.cuda.amp.autocast(enabled=True):

        # for temp_rgb, temp_depth in rosrunner.frameGenerator():
            # if cfg.CAMERA_NAME=="ZED":
                # temp_depth=np.where(np.isnan(temp_depth), 0, temp_depth)
                # temp_depth=np.where(temp_depth==np.inf, -1, temp_depth)
                # temp_depth=np.where(temp_depth==-1, np.max(temp_depth), temp_depth)
                # temp_rgb=temp_rgb[:,:,:-1] # To exclude alpha channel
        
        rgb_files=sorted(glob.glob("sampleVids/zed_conf50/rgb/*.npy"))
        depth_files=sorted(glob.glob("sampleVids/zed_conf50/depth/*.npy"))
        counter=0
        for rgb_path, depth_path in zip(rgb_files, depth_files):
            # temp_rgb=cv2.imread(rgb_path, -1)
            # temp_depth=cv2.imread(depth_path, -1)
            counter+=1
            print(counter, "/", len(rgb_files))

            temp_rgb=np.load(rgb_path)
            temp_depth=np.load(depth_path)

            timer.start()

            if temp_rgb is None:
                break

            rgb_img=shift(temp_rgb, shift_x, shift_y)
            depth_img=shift(temp_depth, -shift_x, -shift_y)

            if set_mask:
                cv2.destroyAllWindows()
                set_mask=False

                if not mask_ready:
                    print("Wait for the interactive segmentation screen")
                    cv2.imwrite("./ritm_interactive_segmentation/initial.png", rgb_img)
                    subprocess.run("cd ./ritm_interactive_segmentation && ./run.sh", shell=True)

                num_objects, mask = adaptMask('./ritm_interactive_segmentation/' + args.mask_name)
                processor = InferenceCore(network, config=config)
                processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
                
                frame_torch, _ = image_to_torch(rgb_img, device=device) # frame:=img1
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
                prediction = processor.step(frame_torch, mask_torch[1:])
                prediction = torch_prob_to_numpy_mask(prediction)
                visualization = overlay_davis(rgb_img, prediction)

                cv2.namedWindow("vis")
                cv2.imshow("vis", visualization)

                cv2.namedWindow("graphImage")
                cv2.setMouseCallback('graphImage', print_coord)
                cv2.createTrackbar('check_threshold','graphImage',500,750,nothing)
                cv2.createTrackbar('touch_threshold','graphImage',25,100,nothing)
                cv2.createTrackbar('table_threshold','graphImage',15,100,nothing)
                run_xmem=True

                tableRelation=TableRelation(rgb_img, depth_img, num_objects, use_existing=args.table_points_ready)
                graphDrawer=GraphDrawer(num_obj=num_objects, beta=0.95, tr=tableRelation)
            
            if show_blended:
                blended=getBlended(rgb_img, depth_img)
                blended = cv2.putText(blended, f"shift x:{2*shift_x}, y: {2*shift_y}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        
            # if record_frames:

                # np.save("/home/enerd/catkin_ws/rgb_frame_kinect.npy", img1)
                # cv2.imwrite("/home/enerd/catkin_ws/rgb_frame_kinect.png", img1)
                # np.save("/home/enerd/catkin_ws/depth_frame_kinect.npy", img2)
                # cv2.imwrite("/home/enerd/catkin_ws/depth_frame_kinect.png", img2)
                # np.save("/home/enerd/catkin_ws/prediction_frame_kinect.npy", prediction)
                # cv2.imwrite("/home/enerd/catkin_ws/prediction_frame_kinect.png", prediction)
                # print("saved")
                # record_frames=False
                # raise NotImplementedError
                # i=i+1
                # print("Recording: True")
                # cv2.imwrite(os.path.join(color_dir, f"color_{i:0>6}.png"), img1)
                # cv2.imwrite(os.path.join(depth_dir, f"depth_{i:0>6}.png"), img2)

            
            if run_xmem:
                frame_torch, _ = image_to_torch(rgb_img, device=device)
                prediction = processor.step(frame_torch)
                prediction = torch_prob_to_numpy_mask(prediction)
                centers=mask2orderedCenters(prediction, depth_img)
                # print("Centers:", centers)

                check_threshold = cv2.getTrackbarPos('check_threshold','graphImage')
                touch_threshold = cv2.getTrackbarPos('touch_threshold','graphImage')
                table_threshold = cv2.getTrackbarPos('table_threshold','graphImage')
                
                start_graph=perf_counter()
                graphImage, distance_dict=graphDrawer.getGraph(centers, prediction.copy(), check_threshold, touch_threshold, table_threshold, depth=depth_img, show_details=graph_show_details)
                
                # graphDrawer._testProximity(rgb_img, depth_img, (prediction==1)*1, touch_threshold)

                putNames(graphImage, centers)
                print("graph time:", round(perf_counter()-start_graph, 3))
                # graphDrawer.getTableRelations()
                
                visualization = overlay_davis(rgb_img, prediction)  # frame:=img1


                # visualization = cv2.putText(visualization, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                # tableRelation._test(table_threshold, rgb_img, depth_img)
                # _, prediction_mask = cv2.threshold(prediction*2, 1, 255, cv2.THRESH_BINARY)
                # masked_rgb = cv2.bitwise_and(graphImage, graphImage, mask=prediction_mask)
                # masked_depth = cv2.bitwise_and(depth_img, depth_img, mask=prediction_mask)
                # cv2.imshow("masked_rgb", masked_rgb)

                visualization = cv2.resize(visualization, None, fx=ZOOM_COEF, fy=ZOOM_COEF, interpolation=cv2.INTER_CUBIC)
                graphImage = cv2.resize(graphImage, None, fx=ZOOM_COEF, fy=ZOOM_COEF, interpolation=cv2.INTER_CUBIC)

                # Combines graph, simple graph and segmented image
                h1,w1,c1=graphImage.shape
                ssg=cv2.imread("simpleStableGraph.png")
                h2, w2, c2=ssg.shape
                combined=np.zeros((2*h1, w1+w2, c1), dtype=np.uint8)
                
                combined[:h1, :w1]=visualization
                combined[h1:, :w1]=graphImage
                combined[h1-h2//2 : h1-h2//2+h2, w1:]=ssg
                # combined=cv2.putText(combined, dictToText(distance_dict), (w1+5,h1-h2//2-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                combined=add_text_to_image(combined, dictToText(distance_dict), (w1+10, 10), 0.6, font_color_rgb=(255,255,255), bg_color_rgb=(0, 0, 0))
                
                cv2.imshow("combined", combined)

                if record_frames:
                    cv2.imwrite(f"results/combined_{i:0>5}.png", combined)
                    i+=1

            else:
                if cfg.CAMERA_NAME=="ZED":
                    depth_img_=((2**16-1)*depth_img/(np.max(depth_img)-np.min(depth_img))).astype(np.uint16)
                elif cfg.CAMERA_NAME=="KINECT":
                    depth_img_=depth_img*20

                depth_img_ = cv2.resize(depth_img_, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                rgb_img = cv2.resize(rgb_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                cv2.imshow('depth_img', depth_img_)
                cv2.imshow('rgb_img', rgb_img)
            
            fps=timer.end_and_print(no_print=True)

            if show_blended:
                blended = cv2.putText(blended, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow('blended', blended)
            
            
            
            key = cv2.waitKey(1) & 0xFF
            #
            if key  == ord('q'): 
                break 

            elif key==ord('d'):
                if shift_x<160:
                    shift_x=shift_x+1
            
            elif key==ord('a'):
                if shift_x>-160:
                    shift_x=shift_x-1
                    
            elif key==ord('s'):
                if shift_y<30:
                    shift_y=shift_y+1
                
            elif key==ord('w'):
                if shift_y>-30:
                    shift_y=shift_y-1
            
            elif key==ord('r'):
                record_frames = not record_frames

            elif key==ord('p'):
                cv2.imwrite("color_frame.png", graphImage+15)
                depth_img_=((2**16-1)*depth_img/(np.max(depth_img)-np.min(depth_img))).astype(np.uint16)
                cv2.imwrite("depth_frame.png", depth_img_)
                subprocess.run("/home/enerd/Desktop/playground/opencv/cv_test color_frame.png depth_frame.png && pcl_viewer pcl_file.pcd", shell=True)
                os.remove("./color_frame.png")
                os.remove("./depth_frame.png")
                os.remove("./pcl_file.pcd")

            elif key==ord('m'):
                set_mask = True
                mask_ready = False
            
            elif key==ord('u'):
                cfg=load_config("./config.yml")
                graphDrawer.update_cfg()
            
            elif key==ord('b'):
                show_blended = not show_blended
                if not show_blended:
                    cv2.destroyWindow("blended")

            elif key==ord('l'):
                graph_show_details = not graph_show_details

            elif key==ord('t'):
                shift_y=0
                shift_x=0
            
            elif key==32: # space keycode
                key=0
                print("press any key to continue")
                while key != 32:
                    key=cv2.waitKey(0) & 0xFF
                


        # if os.path.exists("./ritm_interactive_segmentation/initial.png"):
        #     os.remove("./ritm_interactive_segmentation/initial.png")

        # if os.path.exists("./ritm_interactive_segmentation/mask.png"):
        #     os.remove("./ritm_interactive_segmentation/mask.png")

        if os.path.exists("./simpleStableGraph.png"):
            os.remove("./simpleStableGraph.png")


if __name__ == '__main__':
    depth_img=None
    tableRelation=None
    args=argparser()

    assert not ((args.set_mask is True) and (args.mask_ready is True))
    
    main(args)
