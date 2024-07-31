from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import faiss
import logging
import json
import time
from sklearn.decomposition import PCA
from xml.etree import ElementTree as ET

start_time = time.time()
logging.basicConfig(level=logging.INFO, format=('%(levelname)s-%(message)s'))
logger = logging.getLogger(__name__)

# 设置路径
# image_folder_path = '../cable_rarity1300/images' #使用原数据集
image_folder_path = '../cropped_images'  # 使用裁剪后的数据集
annotation_folder_path = '../cable_rarity1300/Annotations'
index_filename = "image_features.index"
dict_filename = "index_to_image_info.json"
pca_dimension = 44 # 设置pca维数
patch = 257 # 设置dinov模型patch数

if os.path.exists(index_filename):
    os.remove(index_filename)
    logger.info("Removing old index file")
if os.path.exists(dict_filename):
    os.remove(dict_filename)
    logger.info("Removing old dictionary file")

if not os.listdir(image_folder_path):
    logging.error(f"The folder {image_folder_path} is empty.")
    exit(1)
else:
    # 初始化AutoImageProcessor和AutoModel
    # 如果本地没有下载dinov2-small模型，则：
    # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    # model = AutoModel.from_pretrained('facebook/dinov2-small')

    # 如果本地已经下载了dinov2-small模型，则输入模型所在的文件夹：
    processor = AutoImageProcessor.from_pretrained('../dinov2-small')
    model = AutoModel.from_pretrained('../dinov2-small')
    # 初始化Faiss索引，这里使用FlatL2索引，基于L2距离
    # d = 257 * 384

    d = patch * pca_dimension
    index = faiss.IndexFlatIP(d)

    ''' 用于计算pca降低至多少维合适
    # 统计此数据集各张图需要多少维向量能够使得方差占比超过0.9
    count = []
    # 统计各张图片在保留上述平均值（此数据集为22）维向量时方差占比达到多少
    ave_ratio = []
    # 统计各张图片在保留上述最大值（此数据集为22）维向量时方差占比达到多少
    max_ratio = []
    '''
    index_to_image_info = {}
    # 遍历文件夹中的所有图片
    for image_filename in os.listdir(image_folder_path):
        if image_filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder_path, image_filename)

            # 解析对应的XML文件，获取类别信息
            annotation_filename = image_filename.replace('.jpg', '.xml')  # 获取annotation文件名 默认xml文件仅后缀不同
            annotation_path = os.path.join(annotation_folder_path, annotation_filename)  # 获取annotation文件路径
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            label = root.find('object/name').text  # 获取类别名称

            # 提取图片特征向量
            image = Image.open(image_path)

            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            features = outputs.last_hidden_state[0].detach().numpy()

            # 如果不使用pca
            # reshaped_feature = features.reshape(1, -1)
            # 如果使用pca
            pca = PCA(n_components=pca_dimension) # 经过计算，此数据集在保留44维向量时，所有图像的新特征向量方差占比超过0.9。index文件大小为原来的十分之一
            pca_feature = pca.fit_transform(features)
            reshaped_feature = pca_feature.reshape(1, -1)

            '''用于计算pca降低至多少维合适
            sum_ratio = 0
            i = 0
            # 计算方差百分比占比
            for ratio in pca.explained_variance_ratio_:
                if sum_ratio >= 0.9:
                    count.append(i)
                    break
                else:
                    sum_ratio += ratio
                    i += 1
            ave_ratio.append(sum(pca.explained_variance_ratio_[0:21]))
            max_ratio.append(sum(pca.explained_variance_ratio_[0:43]))
            '''
            # 将重新塑形的特征添加到索引
            index.add(reshaped_feature)
            # 获取当前图片的索引ID
            index_id = index.ntotal - 1
            # 将索引ID、图片路径和类别保存到字典中
            index_to_image_info[index_id] = {
                'path': image_path,
                'label': label
            }

            logging.info('Successfully processed image: {}'.format(image_filename))
        else:
            logging.warning('Skip unrecognized image filename: {}'.format(image_filename))

    # 保存索引到文件
    faiss.write_index(index, index_filename)
    script_time = time.time() - start_time
    logging.info(f'Finished saving index, {index.ntotal} images in total, and took {script_time} seconds')

    with open(dict_filename, 'w') as f:
        json.dump(index_to_image_info, f)
    logging.info('Finished saving dictionary to file')
