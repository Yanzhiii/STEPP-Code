# STEPP ROS 2 Humble 迁移文档

> 迁移日期：2026-06-04 ~ 2026-06-05
>
> 迁移协作 / AI assistance：Codex、Claude Code、DeepSeek
>
> 原始项目：https://github.com/RPL-CS-UCL/STEPP-Code

## 一、项目概述

STEPP（Semantic Traversability Estimation using Pose Projected Features）是一套基于视觉的可通行性估计系统，出自 UCL RPL 实验室（ICRA 2025）。

### 原始架构（ROS 1）

```
STEPP_ros/                        ← ROS 1 Noetic catkin 包
├── scripts/inference_node.py     ← Python: DINOv2 + SLIC + MLP 推理
├── src/depth_projection_synchronized.cpp  ← C++: 深度投影 + 点云融合
├── msg/Float32Stamped.msg        ← 自定义消息
├── launch/STEPP.launch           ← XML launch
├── CMakeLists.txt / package.xml  ← catkin 构建

STEPP/                            ← 纯 Python (与 ROS 无关)
├── DINO/   (特征提取)
├── SLIC/   (超像素分割)
└── model/  (MLP 自编码器)
```

### 论文训练数据来源

| 数据集 | 类型 | 数量 | 采集设备 |
|--------|------|------|---------|
| Richmond Forest | 真实，人类手持行走 | 55,580 | 自定义相机-LiDAR 手持设备 |
| Indoor Lab | 真实 | 5,384 | 同上 |
| Unreal Engine 合成 | 合成 | 26,954 | UE C++ 插件 |
| **合计** | | **87,918** | |

- **训练阶段未使用 ZED 相机**。自定义手持设备相机型号论文未指明。
- **部署阶段**使用 ZED2（ANYmal-D 机器人），配合 Livox Mid-360 + FAST-LIO2 做里程计。
- DINOv2 工作在高维语义空间，跨相机泛化在论文中已验证。

---

## 二、迁移产物

### 2.1 新包结构

```
stepp_ros2_humble/                       ← ROS 2 Humble ament_cmake 包
├── CMakeLists.txt                ← ament_cmake + rosidl 消息生成
├── package.xml                   ← format 3
├── config/model_config.yaml      ← 参数模板
├── launch/STEPP_launch.py        ← Python launch (9 个可配置参数)
├── msg/Float32Stamped.msg        ← 不变 (ROS 1/2 兼容)
└── scripts/
    ├── inference_node.py         ← rospy → rclpy 迁移
    └── depth_projection_node.py  ← C++ → Python/NumPy 重写
```

### 2.2 原始文件状态

| 文件/目录 | 状态 |
|-----------|------|
| `STEPP_ros/` | **未修改**，保留 ROS 1 原始版本 |
| `STEPP/` | **未修改**，纯 Python 与 ROS 无关 |
| `standalone_inference.py` | **未修改**，无 ROS 依赖 |
| `setup.py` | **未修改** |

### 2.3 关键 API 变更

| ROS 1 | ROS 2 Humble |
|-------|-------------|
| `import rospy` | `import rclpy` + 继承 `rclpy.node.Node` |
| `rospy.Subscriber(t, M, cb)` | `self.create_subscription(M, t, cb, qos)` |
| `rospy.Publisher(t, M, q)` | `self.create_publisher(M, t, qos)` |
| `rospy.get_param('~p', d)` | `self.declare_parameter('p', d)` + `.get_parameter('p').value` |
| `rospy.Time.now()` | `self.get_clock().now().to_msg()` |
| `rospy.is_shutdown()` | `rclpy.ok()` |
| `rospy.spin()` | `rclpy.spin(node)` |
| XML `.launch` | Python `launch_ros.actions.Node` |
| `catkin` 构建 | `ament_cmake` + `colcon build` |

### 2.4 C++ → Python 重写

原始的 C++ 深度投影节点（`depth_projection_synchronized.cpp`，~370 行）被 Python/NumPy 版本替代：

- **深度投影**: C++ 逐像素循环 → NumPy 向量化 meshgrid 投影
- **体素滤波**: PCL VoxelGrid → NumPy `unique` + `add.at` 质心平均
- **坐标变换**: tf + Eigen → 手写 NumPy 4×4 齐次变换
- **消息同步**: `message_filters::ApproximateTime` → `message_filters.ApproximateTimeSynchronizer`（Python API）
- **点云发布**: 直接构建 `PointCloud2` 消息（无需 PCL 依赖）

### 2.5 QoS 配置

ROS 2 引入 QoS 机制，不匹配会导致无法建立连接：

```python
# 相机图像：BEST_EFFORT（匹配 ZED 驱动默认策略）
sensor_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)

# 推理输出：RELIABLE
reliable_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)
```

---

## 三、构建与使用

### 3.1 环境搭建（从零开始）

以下步骤在 Ubuntu 22.04 上从零搭建 STEPP 运行环境。

#### 3.1.1 创建 Conda 环境

```bash
# 创建 Python 3.10 环境（ROS 2 Humble 官方支持版本）
conda create -n stepp python=3.10 -y
conda activate stepp
```

#### 3.1.2 安装 PyTorch（CUDA 12.1）

```bash
# x86_64 桌面平台
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Jetson Orin 平台
# 参考: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# 下载对应 JetPack 版本的 wheel 后 pip install
```

验证安装：

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

#### 3.1.3 安装 STEPP Python 依赖

```bash
# 进入项目根目录
cd /path/to/STEPP-Code

# 以可编辑模式安装 STEPP 及其所有 Python 依赖
pip install -e .
```

#### 3.1.4 安装 ROS 2 构建依赖

ROS 2 Humble 的 `rosidl` 消息生成需要以下 Python 包：

```bash
pip install empy==3.3.4 catkin_pkg lark
```

> **注意**：`empy` 必须使用 3.x 版本。4.x 版本删除了 `BUFFERED_OPT` 属性，与 ROS 2 Humble 的 `rosidl_adapter` 不兼容。

#### 3.1.5 激活 ROS 2 环境

```bash
# Bash
source /opt/ros/humble/setup.bash

# Zsh
source /opt/ros/humble/setup.zsh
```

#### 3.1.6 完整激活流程（每次新终端）

```bash
# 1. 激活 conda 环境
conda activate stepp

# 2. 激活 ROS 2 Humble
source /opt/ros/humble/setup.zsh

# 3. 激活工作空间（构建后）
source ~/Projects/stepp_ros2_humble_ws/install/setup.zsh
```

### 3.2 构建

```bash
# 创建工作空间
mkdir -p ~/Projects/stepp_ros2_humble_ws/src

# 将 stepp_ros2_humble 包链接到工作空间
ln -s /path/to/STEPP-Code/stepp_ros2_humble ~/Projects/stepp_ros2_humble_ws/src/stepp_ros2_humble

# 构建
cd ~/Projects/stepp_ros2_humble_ws
colcon build --packages-select stepp_ros2_humble --symlink-install

# 后续只需 source
source install/setup.zsh
```

> **说明**：`--symlink-install` 将 Python 脚本符号链接到 build 目录，修改源码后无需重新构建即可生效。

### 3.3 运行

**方式 1：Launch 文件（推荐）**

```bash
ros2 launch stepp_ros2_humble STEPP_launch.py \
    model_path:=/path/to/checkpoint.pth \
    camera_type:=zed2 \
    cutoff:=0.45
```

**方式 2：分别启动**

```bash
# 终端 1：推理节点
ros2 run stepp_ros2_humble inference_node.py --ros-args \
    -p model_path:=/path/to/checkpoint.pth \
    -p visualize:=false

# 终端 2：深度投影节点
ros2 run stepp_ros2_humble depth_projection_node.py --ros-args \
    -p camera_type:=zed2 \
    -p decay_time:=8.0
```

### 3.4 Launch 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | `""` | MLP checkpoint 路径（**必填**） |
| `visualize` | `true` | 发布可通行性叠加图 |
| `ump` | `false` | 混合精度推理 |
| `cutoff` | `0.45` | 归一化重建误差上限 |
| `camera_type` | `zed2` | 相机内参预设：zed2 / D455 / cmu_sim |
| `decay_time` | `8.0` | 点云衰减时间（秒） |
| `rgb_topic` | `/camera/color/image_raw/compressed` | RGB 输入话题 |
| `depth_topic` | `/camera/aligned_depth_to_color/image_raw` | 深度输入话题 |
| `odom_topic` | `/state_estimation` | 里程计输入话题 |

### 3.5 验证结果

```bash
# 消息定义
$ ros2 interface show stepp_ros2_humble/msg/Float32Stamped
std_msgs/Header header
    builtin_interfaces/Time stamp
    string frame_id
std_msgs/Float32MultiArray data
    ...

# 可执行文件
$ ros2 pkg executables stepp_ros2_humble
stepp_ros2_humble depth_projection_node.py
stepp_ros2_humble inference_node.py

# 推理节点启动
$ ros2 run stepp_ros2_humble inference_node.py --ros-args \
    -p model_path:=checkpoints/all_ViT_small_input_700_big_nn_checkpoint_20240827-1935.pth
[INFO] [inference_node]: cutoff = 1.2
[INFO] [inference_node]: Inference node initialized

# 深度投影节点启动
$ ros2 run stepp_ros2_humble depth_projection_node.py
[INFO] [depth_projection]: Depth projection node initialized
  (camera=zed2, fx=534.4, fy=534.5, fovx=83.7°, fovy=53.6°)
```

---

## 四、ZED 2i 真机部署调研

### 4.1 需要额外安装的包

```bash
# ZED ROS 2 驱动
git clone https://github.com/stereolabs/zed-ros2-wrapper.git -b humble-v4.1.4

# 系统包
sudo apt install ros-humble-zed-msgs \
                 ros-humble-image-transport-plugins \
                 ros-humble-compressed-depth-image-transport

# ZED SDK（从 stereolabs.com 下载安装）
```

### 4.2 话题映射

ZED 驱动默认话题 → STEPP 期望话题（通过 launch 参数完成）：

```
/zed/zed_node/rgb/image_rect_color/compressed  → rgb_topic
/zed/zed_node/depth/depth_registered           → depth_topic
/zed/zed_node/odom                             → odom_topic
```

### 4.3 必须解决的硬件适配问题

| 优先级 | 问题 | 说明 |
|--------|------|------|
| 🔴 | 相机内参 | 当前硬编码为 540p ZED2 内参。建议改为从 `/camera_info` 动态读取 |
| 🔴 | 相机外参 | `CAMERA_TO_MAP` 矩阵硬编码了 ANYmal 的安装位置，必须重新标定 |
| 🔴 | 里程计方案 | 论文使用 Livox Mid-360 + FAST-LIO2。纯 ZED 内置 VIO 室外精度很差 |
| 🟡 | TF 树配置 | 必须禁用 ZED 驱动的 `publish_tf`，自行发布 `base_link → camera_link` |
| 🟡 | 领域迁移 | 训练数据来自手持设备，ZED 2i 推理可能存在分布差异 |

### 4.4 里程计方案对比

| 方案 | 传感器 | 精度 | 论文使用 |
|------|--------|------|---------|
| FAST-LIO2 | Livox Mid-360 LiDAR | 厘米级 | ✅ 论文方案 |
| RTAB-Map 立体 | ZED 双目 | 0.16m RMSE (室内) | ❌ |
| ZED 内置 VIO | ZED IMU+视觉 | 206m 漂移/514m (室外) | ❌ |

### 4.5 测试数据

**方案 1：Stereolabs 官方 SVO 样本（ZED 2，非 ZED 2i）**

```bash
# 下载 + 回放
wget https://download.stereolabs.com/assets/svo_samples/ZED2_Street_H264

ros2 launch zed_wrapper zed_camera.launch.py \
    camera_model:=zed2 svo_path:=./ZED2_Street_H264
```

**方案 2：MUSE/UnCal-Flight 数据集**

- 145 条 ZED 2i 采集的室内无人机轨迹
- 含双目图像、IMU、Vicon 真值
- ROS 1 bag 格式，需要转换为 ROS 2

### 4.6 CMU Nav Pipeline 集成方案

CMU Pipeline 有 `humble` 分支。两者共存构建：

```bash
cd ~/cmu_ws
git clone https://github.com/HongbiaoZ/autonomous_exploration_development_environment.git -b humble src/
ln -s /path/to/STEPP-Code/stepp_ros2_humble src/stepp_ros2_humble
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

**关键适配点：**

| 项目 | STEPP | CMU Pipeline | 适配方式 |
|------|-------|-------------|---------|
| 输出话题 | `/depth_projection` | 订阅 `/terrain_map` | launch 中 remap |
| 点云帧 | `odom` | `odom` | ✅ 一致 |
| cost 语义 | MLP 重建误差 (0~1) | 高度代价 | 调整 `costHeightThre`、`costScore` |
| 导航栈 | 无 | Falco 局部规划器 | 禁用 CMU 自带的 `terrain_analysis` |

---

## 五、已知限制

1. **模型文件路径固定**：`model_path` 参数必须手动指定，未做搜索逻辑
2. **相机内参硬编码**：不支持从 `/camera_info` 动态读取
3. **相机外参硬编码**：`CAMERA_TO_MAP` 矩阵必须重新标定才能上真机
4. **Python 体素滤波性能**：纯 NumPy 实现，大场景可能不如 PCL C++ 优化版
5. **消息版本兼容**：`Float32Stamped.msg` 的 `Float32MultiArray` 字段语义与 CMU 的 `/terrain_map` 不完全一致
6. **SVO 回放需 GPU**：ZED SDK 的深度计算基于 CUDA，无 GPU 的机器无法用 SVO 测试

---

## 六、文件清单

```
STEPP-Code/
├── stepp_ros2_humble/                          # [新增] ROS 2 Humble 包
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── config/model_config.yaml
│   ├── launch/STEPP_launch.py
│   ├── msg/Float32Stamped.msg
│   └── scripts/
│       ├── inference_node.py
│       └── depth_projection_node.py
├── STEPP_ros/                           # [保留] 原始 ROS 1 包，未修改
├── STEPP/                               # [保留] 纯 Python，未修改
├── standalone_inference.py              # [保留] 无 ROS 依赖的推理脚本
├── setup.py                             # [保留] 原始 setup.py
└── docs/ROS2_MIGRATION.md               # [新增] 本文档
```

### 工作空间

```
~/Projects/stepp_ros2_humble_ws/               # [新增] ROS 2 构建与测试工作空间
├── src/stepp_ros2_humble → ../../STEPP-Code/stepp_ros2_humble/
├── build/
└── install/
```
