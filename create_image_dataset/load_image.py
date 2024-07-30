from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import faiss
import logging
import json
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format=('%(levelname)s-%(message)s'))
logger = logging.getLogger(__name__)

# 设置路径
# image_folder_path = '../cable_rarity1300/images' #使用原数据集
image_folder_path = '../cropped_images'  # 使用裁剪后的数据集
annotation_folder_path = '../cable_rarity1300/Annotations'
index_filename = "image_features.index"
dict_filename = "index_to_image_info.json"

if os.path.exists(index_filename):
    os.remove(index_filename)
    logger.info("Removing old index file")
if os.path.exists(dict_filename):
    os.remove(dict_filename)
    logger.info("Removing old dictionary file")

if not os.listdir(image_folder_path):
    logging.error(f"The folder '{image_folder_path}' is empty.")
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
    d = 257*384
    index = faiss.IndexFlatIP(d)

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
            reshaped_feature = features.reshape(1, -1)

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
    logging.info('Finished saving index')

    with open(dict_filename, 'w') as f:
        json.dump(index_to_image_info, f)
    logging.info('Finished saving dictionary to file')