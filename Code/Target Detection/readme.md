# Target Detection


目标检测，也叫目标提取，是一种基于目标几何和统计特征的图像分割，它将目标的分割和识别合二为一，其准确性和实时性是整个系统的一项重要能力。尤其是在复杂场景中，需要对多个目标进行实时处理时，目标自动提取和识别就显得特别重要。 随着计算机技术的发展和计算机视觉原理的广泛应用，利用计算机图像处理技术对目标进行实时跟踪研究越来越热门，对目标进行动态实时跟踪定位在智能化交通系统、智能监控系统、军事目标检测及医学导航手术中手术器械定位等方面具有广泛的应用价值。

yolov5目标检测案例给出了简单的训练、验证和预测示例，可以通过下载yolov5模型对各种来源的数据，图片等进行识别，训练和分析。我们可以通过yolov5模型来实现目标检测的功能。
Point cloud classification with PointNet：
实现了无序3D点集（即点云）的分类，检测和分割。作者https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdgriffiths3.github.io，实现了Qi等人在2017年发布的开创性深度学习论文。

## 介绍

### 项目目标

目标检测要解决的核心问题是： 1.目标可能出现在图像的任何位置。 2.目标有各种不同的大小。 3.目标可能有各种不同的形状。 如果用矩形框来定义目标，则矩形有不同的宽高比。由于目标的宽高比不同，因此采用经典的滑动窗口+图像缩放的方案解决通用目标检测问题的成本太高。

yolov5目标检测项目通过自动下载的模型对给定的图像，数据等进行识别检测，通过大量的图像等对该模型进行训练，提高模型对于图像中目标检测识别的准确率。可应用于多种场景下对给定图像进行目标检测识别，找出图像中出现的目标。
Point cloud classification with PointNet：该算法实现了无序3D点集（即点云）的分类，检测和分割。可以应用于从点云信息归类识别出各种物体，比如分析识别通过激光雷达扫描的点云信息构建环境的3维地图。

### 背景

计算机视觉领域有四大主要任务，分别是图像分类、目标检测、目标跟踪、图像分割。
目标检测的有两个主要任务，分别是物体分类和定位，具体来讲就是在图像中用边界框(bounding box)标定多个目标的位置和类别。

## 功能

### 主要功能

Point cloud classification with PointNet：
该算法通过trimesh库读取并可视化.off文件。再用.sample（） 执行 unifrom 随机采样，把网格文件转化为点云。最后输入到网络中让模型学习分类点云的能力。
特点是此案例使用的网络由两个核心组件组成：主MLP网络和T-net。T-net旨在通过自己的迷你网络学习仿射变换矩阵。T网使用两次。首次将输入要素 （n， 3） 转换为规范制图表达。第二个是在特征空间 （n， 3） 中对齐的仿射变换。

YOLOv5 Tutorial：
YOLO算法将输入图像划分为SxS个网格并对每个网格预测B个边界框，对每个网格中的物体分别进行预测。
每个预测的结果包括边界框的位置、尺寸、置信度以及边界框中物体属于各个类别的概率。
这种划分网格的方法，避免了大量的重复计算，使得YOLO算法达到较快的检测速度。
YOLOV5相比于YOLOV4，在数据增强使用了缩放、色彩空间调整和马赛克增强，且加入了自适应锚定框。


### 支持的输入类型

Point cloud classification with PointNet：
输入.off 网格文件，通过trimesh包可以读取并可视化。
物体文件格式（.off）文件用于通过指定模型表面的多边形来表示模型的几何形状。多边形可以具有任意数量的顶点。除了trimesh、还可以使用vedo、open3D、py3dviewer来读取和可视化.off文件。

yolov5目标检测项目：
1.detect: 数据集：data-COCO128.yaml 模型：yolov5s  图像大小：640*640
weights=['yolov5s.pt'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640]
2.validate: 数据集：data-COCO128.yaml 模型：yolov5s
data=/content/yolov5/data/coco.yaml, weights=['yolov5s.pt'],
3.train:在COCO128数据集上使用--data-COCO128.yaml训练YOLOv5s模型，从预训练的--weights YOLOv5s.pt开始，或从随机初始化的--weight''-cfg YOLOv5s.yaml开始。



## 开始

### 安装要求

Point cloud classification with PointNet：
数据集：ModelNet10模型数据集
需要的库：os、glob、trimesh、numpy、tensorflow
yolov5 目标检测案例setup
Clone GitHub https://github.com/ultralytics/yolov5, install https://github.com/ultralytics/yolov5/blob/master/requirements.txt and check PyTorch and GPU.

### 快速开始

Point cloud classification with PointNet：https://colab.research.google.com/drive/1hOp_nO3X0fxO8TImdDq9FiYO3ovL4vWF
YOLOv5：https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb#scrollTo=X58w8JLpMnjH


### 代码运行效果
Point cloud classification with PointNet运行效果：
通过trimesh库读取并可视化.off文件。

通过.sample（） 执行 unifrom 随机采样，把网格文件转化为点云

生成数据集，并且设置采样和批量大小的点数。

数据增强

构建模型
每个卷积和全连接层（端层除外）由卷积/dense -> 批量归一化 -> ReLU 激活组成。

定义模型后，可以使用 .compile（） 和 .fit（）训练

通过matplotlib 来可视化我们训练好的模型性能。


YOLOv5案例代码运行效果
1.detect检测识别
对给定数据来源中的数据输入模型进行检测
detect: weights=['yolov5s.pt'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_csv=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
YOLOv5 🚀 v7.0-214-g8c30c58 Python-3.10.12 torch-2.0.1+cu118 CUDA:0 (Tesla T4, 15102MiB)


YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients
image 1/2 /content/yolov5/data/images/bus.jpg: 640x480 4 persons, 1 bus, 40.9ms
image 2/2 /content/yolov5/data/images/zidane.jpg: 384x640 2 persons, 2 ties, 41.3ms
Speed: 0.5ms pre-process, 41.1ms inference, 127.6ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/exp

2.validate 验证
对模型对数据集的分析结果进行验证，得到准确率
val: data=/content/yolov5/data/coco.yaml, weights=['yolov5s.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
YOLOv5 🚀 v7.0-214-g8c30c58 Python-3.10.12 torch-2.0.1+cu118 CUDA:0 (Tesla T4, 15102MiB)
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients
val: Scanning /content/datasets/coco/val2017... 4952 images, 48 backgrounds, 0 corrupt: 100% 5000/5000 [00:02<00:00, 1993.69it/s]
val: New cache created: /content/datasets/coco/val2017.cache
Class Images  Instances    P      R      mAP50   mAP50-95:100% 157/157[01:18<00:00,2.00it/s]
all    5000     36335    0.671  0.519    0.566      0.371


3.train 训练
在COCO128数据集上使用--data-COCO128.yaml训练YOLOv5s模型，从预训练的--weights YOLOv5s.pt开始，或从随机初始化的--weight''-cfg YOLOv5s.yaml开始。得到模型对不同目标检测识别的准确率。


Comet Logging and Visualization
数据分析结果




