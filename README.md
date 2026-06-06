# Watch your STEPP: Semantic Traversability Estimation using Pose Projected Features #
![demo](assets/front_page.png)

**Authors**: [Sebastian Aegidius*](https://rvl.cs.toronto.edu/), [Dennis Hadjivelichkov](https://dennisushi.github.io/), [Jianhao Jiao](https://gogojjh.github.io/), [Jonathan Embly-Riches](https://rpl-as-ucl.github.io/people/), [Dimitrios Kanoulas](https://dkanou.github.io/)

**ROS 2 migration contributors / AI assistance**: Codex, Claude Code, DeepSeek

<div style="text-align: center;">

[Project Page](https://rpl-cs-ucl.github.io/STEPP/)  [STEPP arXiv](https://arxiv.org/)

</div>


![demo](assets/outdoor_all_2.png)
![demo](assets/pre_train_pipeline.png)

## Installation and ROS 2 Humble setup ##

This repository now contains both the original ROS 1 package and a ROS 2 Humble
port:

- `STEPP_ros/`: original ROS 1/catkin package, kept for reference.
- `stepp_ros2_humble/`: ROS 2 Humble/ament package used for current testing.
- `STEPP/`: shared Python inference/training code.

The ROS 2 port has been validated for package build, custom message generation,
node startup, inference output, and depth projection output. The detailed
migration notes are kept in `docs/ROS2_MIGRATION.md`.

### 1. Create and activate the Python environment

Use Python 3.10 for ROS 2 Humble compatibility.

```bash
conda create -n stepp python=3.10 -y
conda activate stepp

# Example for x86_64 desktop CUDA builds.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# From the repository root.
pip install -e .

# ROS 2 Humble message generation helpers.
pip install empy==3.3.4 catkin_pkg lark
```

If you are doing a lightweight runtime-only install, make sure the ROS 2 inference
dependencies are also available, in particular `omegaconf`, `pytictac`, and
`fast-slic`.

### 2. Activate ROS 2

Activate the Conda environment first, then source ROS 2 Humble:

```bash
conda activate stepp
source /opt/ros/humble/setup.zsh   # zsh
# or: source /opt/ros/humble/setup.bash
```

### 3. Build the ROS 2 package

Create a ROS 2 workspace and link the package from this repository:

```bash
mkdir -p ~/Projects/stepp_ros2_humble_ws/src
ln -s /path/to/STEPP-Code/stepp_ros2_humble \
  ~/Projects/stepp_ros2_humble_ws/src/stepp_ros2_humble

cd ~/Projects/stepp_ros2_humble_ws
colcon build --packages-select stepp_ros2_humble --symlink-install
```

For each new terminal:

```bash
conda activate stepp
source /opt/ros/humble/setup.zsh
source ~/Projects/stepp_ros2_humble_ws/install/setup.zsh
```

Basic verification:

```bash
ros2 interface show stepp_ros2_humble/msg/Float32Stamped
ros2 pkg executables stepp_ros2_humble
ros2 launch stepp_ros2_humble STEPP_launch.py --show-args
```

### Jetson notes

The notes below are kept from the original project as Jetson/PyTorch reference
material. For ROS 2 Humble deployments, adapt the wheel and Python version to
your JetPack/L4T and ROS 2 environment.

For installation of JetPack, PyTorch, and Torchvision on your Jetson Platform: [Link](https://pytorch.org/audio/stable/build.jetson.html) and [Link](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
* Show JetPack version: ```apt-cache show nvidia-jetpack```
* [MUST] Create conda with python=3.8 and download wheel from this [link](https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl)
* And then ```pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl```
* Install Torchvision (check the compatible matrix with the corresponding PyTorch).
    * Check this [link](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048/1285?page=63) for this issue: ```ValueError: Unknown CUDA arch (8.7+PTX) or GPU not supported```
    * Command: 
    ```
    pip install numpy && \
    pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl && \
    cd torchvision/ && \
    export BUILD_VERSION=0.15.1 && \
    python setup.py install --user && \
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); import torchvision"
    ```
## Checkpoints ##

The following trained checkpoints are included in the repo:

| Modelname   | Dataset| Image resolutions| DINOv2 size |MLP architecture|
|-------------|--------|---------------------|-------------|---------|
| [`richmond_forest.pth`](\\wsl.localhost\Ubuntu-20.04\home\sebastian\code\STEPP-Code\checkpoints\richmond_forest_full_ViT_small_big_nn_checkpoint_20240821-1825.pth) |Richmond Forest| 700x700 | dinov2_vits14 |bin_nn|
| [`unreal_synthetic_data.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth)  |Unreal engine synthetic Data| 700x700 | dinov2_vits14 |big_nn|
| [`all_data.pth`](\checkpoints\unreal_full_ViT_small_big_nn_checkpoint_20240819-2003.pth)|Richmond Forest, Unreal synthetic Data | 700x700 | dinov2_vits14 |big_nn|

## Usage ##

Launch the ROS 2 port with a trained checkpoint:

```bash
ros2 launch stepp_ros2_humble STEPP_launch.py \
  model_path:=/path/to/checkpoint.pth \
  camera_type:=zed2 \
  cutoff:=0.45
```

The two nodes can also be started separately:

```bash
ros2 run stepp_ros2_humble inference_node.py --ros-args \
  -p model_path:=/path/to/checkpoint.pth \
  -p visualize:=false

ros2 run stepp_ros2_humble depth_projection_node.py --ros-args \
  -p camera_type:=zed2 \
  -p decay_time:=8.0
```

### ROS 2 launch arguments

- `model_path`: path to the chosen `.pth` checkpoint. Required for inference.
- `visualize`: publish an image overlay of the traversability cost. This slows inference.
- `ump`: use mixed precision for model inference.
- `cutoff`: maximum normalized reconstruction error.
- `camera_type`: camera intrinsics preset, one of `zed2`, `D455`, or `cmu_sim`.
- `decay_time`: how long the projected cost cloud is retained outside the active camera view.
- `rgb_topic`: compressed RGB image input topic.
- `depth_topic`: aligned raw depth image input topic.
- `odom_topic`: odometry input topic.

### Local planner integration

The original project was used with the CMU Falco local planner from the
[Autonomous Exploration Development Environment](https://www.cmu-exploration.com/).
The ROS 2 migration in this repository currently focuses on the STEPP inference
and depth projection pipeline. Integration with the CMU AEDE pipeline/local
planner will be validated separately and documented outside this README once the
planner-side interface is confirmed.

## Train Your Own STEPP inference model ##
to train your own STEPP traversability estimation model all you need is a dataset consisting of an image folder and an odometry pose folder. Here each SE(3) odometry pose has to relate to the exact location and rotation of the correlating image. With these two you can run the `extract_future_poses.py` script and obtain a json file containing the pixels that represent the cameras future poses in the given image frame. 

With this json file and the associated images you can run the `make_dataset.py` file to obtain a `.npy` of the DINOv2 feature averaged vectors of each segment that the future poses in each image from your dataset belonges to. this can in turn be used to train the STEPP model on using `training.py`

### Acknowledgement
https://github.com/leggedrobotics/wild_visual_navigation\
https://github.com/facebookresearch/dinov2\
https://github.com/HongbiaoZ/autonomous_exploration_development_environment

### Citation
If you think any of our work was useful, please consider citing it:

```bibtex
@inproceedings{aegidius2025stepp,
  author    = {Sebastian Ægidius and Dennis Hadjivelichkov and Jianhao Jiao and Jonathan Embley-Riches and Dimitrios Kanoulas},
  title     = {Watch Your STEPP: Semantic Traversability Estimation Using Pose Projected Features},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025},
  pages     = {2376--2382},
  doi       = {10.1109/ICRA55743.2025.11127781}
}
```
