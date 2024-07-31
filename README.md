# Introduction

This repository is created to manage and save my personal code that I wrote during my internship,7.2024  

此仓库用于管理和保存在202407实习期间个人编写的代码

## Content

1.create_image_dataset

2.README.md

# How to use

## create_image_dataset

### Content

1.crop.py

2.index_to_image_info.json

3.load_image.py

3.requirements.txt

4.retrieve_image.py

### Implementation

主要使用了[DINOv2模型](https://hf-mirror.com/facebook/dinov2-small)和Faiss库完成了创建图像特征向量，并通过图像检索的功能。  

对于load_image.py，DINOv2可以从图像中提取通用特征向量，降维后通过Faiss保存在index文件中，并且构建一个索引和图像路径、类别相对应的字典，保存在json文件中。  

对于retrieve_image.py，首先将待检索的图像提取出特征向量，降维后计算余弦相似度通过Knn最近邻搜索出最相似的图片，并返回索引和图像。

### History

#### 2024年7月30日

feat：新增了代码文件crop.py，同事实现了根据.../cable_rarity1300/Annotations的xml文件中的boundbox切割../cable_rarity1300/images图像的功能，将处理后的图片存入上一级目录../cropped_images；新增了保存图像索引、路径、标签的字典index_to_image_info.json文件。

docs：修改了代码文件load_image.py和retrieve_image.py，修改后load_image.py读取裁剪后的数据集实现了构建一个索引、路径、标签的字典的功能，其中标签依然同上的xml文件中读取；同样的，实现了在检索图像后也能输出其标签的功能；修改后通过同一个上级目录下的文件夹加载模型；删除了一些不必要的注释；更新了requirements.txt文件。 

#### 2024年7月31日
docs：修改了代码文件load_image.py和retrieve_image.py，加入了PCA，在存入向量库之前首先进行了降维，使得向量长度大大缩减，生成的index文件大小为原来的十分之一；上传了index文件，即image_features.index；更新了json文件；更新了requirements.txt文件。

### Details

1.没有上传原始数据集文件夹，即cable_rarity1300。若有，应与文件夹create_image_dataset在同一目录下，通过运行crop.py会创建处理后的数据集文件夹，同在create_image_dataset在同一目录下。

2.没有上传保存图像向量的向量库文件，即image_features.index文件。通过运行load_image.py文件会自动读取./cropped_images下的数据集，并在当前目录创建image_features.index文件和index_to_image_info.json文件，即与load_image.py和retrieve_image.py在同一目录下。

3.没有上传模型文件，即dinov2-small文件夹。下载后应与文件夹create_image_dataset在同一目录下。





