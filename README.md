# xmem2graph

This code mainly depends on following works:
* [(RITM)](https://github.com/saic-vul/ritm_interactive_segmentation) Reviving Iterative Training with Mask Guidance for Interactive Segmentation [CoRR 2021]
* [XMem](https://github.com/hkchengrex/XMem/): Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model [ECCV 2022]


This repo creates a relation graph using selected objects from the scene in real-time. I used relations from enriched semantic event chain [1]:

Static: contact, above, below, left, right, front, behind, inside, surround
Dynamic: moving together, halting together, fixed moving together, getting close, moving apart, stable 



## Setting up the environment

Do not forget to install ROS.

```
# Create conda env
conda create -n x2g python=3.8
conda activate x2g

# Install CUDA and cuDNN, and add to the path
conda install -c conda-forge cudatoolkit=11.3 cudnn=8.4.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Install requirements
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

# (Optional) ROS related packages
conda install -c conda-forge ros-rospy rospkg
```
OR use yml file

```
conda env create -f environment.yml
```

To get pre-trained models
```
./download.sh  
```

Install ZED or Kinect (OpenNi or freenect) package from their sources.


## To run 

Do not forget to change camera type (kinect and ZED are available only. For others modify the ros_capture.py file)

```
python real_time.py
```

Press <kbd>m</kbd> to set the initial mask then the magic begins

## Note

We experience alignment problem in between rgb and depth data while using Kinect camera. So we preferred ZED camera, which has no such a problem due to stereo cameras.

If you are going to use ZED camera, please consider to change depth quality parameter to 4. Check out the following configuration file:
```
gedit path/to/workspace/zed-ros-wrapper/zed_wrapper/params/common.yaml
```
