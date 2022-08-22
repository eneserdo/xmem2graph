# xmem2graph

This code mainly depends on following works:
* [(RITM)](https://github.com/saic-vul/ritm_interactive_segmentation) Reviving Iterative Training with Mask Guidance for Interactive Segmentation [CoRR 2021]
* [XMem](https://github.com/hkchengrex/XMem/): Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model [ECCV 2022]


Then the code creates a relation graph. I used relations from [ref]:

Static: contact, above, below, left, right, front, behind, inside, surround
Dynamic: moving together, halting together, fixed moving together, getting close, moving apart, stable 



## Setting up the environment

Do not forget to install ROS.

```
# Create conda env
conda create -n x2g python=3.8
conda activate x2g

# Install CUDA and cuDNN, and ADD TO THE PATH!
conda install -c conda-forge cudatoolkit=11.3 cudnn=8.4.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Install requirements
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

# (Optional) ROS related packages
conda install -c conda-forge ros-rospy rospkg

# To get pre-trained models
./download.sh  
```

OR use yml file:

```
conda env create -f environment.yml
```

# To run 

Do not forget to change camera type (kinect and ZED are available only. For others modify the ros_capture.py file)

```
python real_time.py
```

Press <kbd>m</kbd> to set the initial mask then the magic begins


