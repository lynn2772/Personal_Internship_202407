# Introduction

This repository is created to manage and save my personal code that I wrote during my internship,7.2024  

此仓库用于管理和保存在202407实习期间个人编写的代码

## Content

1.create_image_dataset

# How to use

## create_image_dataset

### Content

文件夹中主要包含两个python代码文件，保存了图像特征向量的index文件，保存了索引与图像路径的字典json文件。

### Implementation Details

主要使用了[DINOv2模型](https://hf-mirror.com/facebook/dinov2-small)和Faiss库完成了创建图像特征向量，并通过图像检索的功能。  
对于load_image.py，DINOv2可以从图像中提取通用特征向量，通过Faiss保存在index文件中，并且构建一个索引和图像路径相对应的字典，保存在json文件中。  
对于retrieve_image.py，首先将待检索的图像提取出特征向量，计算余弦相似度通过Knn最近邻搜索出最相似的图片，并返回索引和图像。
