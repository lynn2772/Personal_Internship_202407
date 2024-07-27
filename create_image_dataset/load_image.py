from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import faiss
import logging
import json

logging.basicConfig(level=logging.INFO, format=('%(levelname)s-%(message)s'))
logger = logging.getLogger(__name__)
# 设置图片文件夹路径
image_folder_path = '../image'
index_filename = "image_features.index"
dict_filename = "index_to_image_path.json"
if os.path.exists(index_filename):
    os.remove(index_filename)
    logger.info("Removing old index file")
if os.path.exists(dict_filename):
    os.remove(dict_filename)
    logger.info("Removing old dictionary file")
if not os.listdir(image_folder_path):
    logging.error(f"The folder '{image_folder_path}' is empty.")
    exit(1)  # 退出脚本并返回状态码1，表示出错
else:
    # 初始化AutoImageProcessor和AutoModel
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small')
    # 初始化Faiss索引，这里使用FlatL2索引，基于L2距离
    d = 257*384
    index = faiss.IndexFlatIP(d)

    index_to_image_path = {}
    # 遍历文件夹中的所有图片
    for image_filename in os.listdir(image_folder_path):
        if image_filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder_path, image_filename)

            # 提取图片特征向量
            image = Image.open(image_path)
            # 处理图片并获取模型输出
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            features = outputs.last_hidden_state[0].detach().numpy()
            reshaped_feature = features.reshape(1, -1)

            # 将重新塑形的特征添加到索引
            index.add(reshaped_feature)
            index_to_image_path[index.ntotal - 1] = image_path
            logging.info('Successfully processed image: {}'.format(image_filename))
        else:
            logging.warning('Skip unrecognized image filename: {}'.format(image_filename))

    # 保存索引到文件
    faiss.write_index(index, "image_features.index")
    logging.info('Finished saving index')
    # num_vectors = index.ntotal
    # logging.info('%d', num_vectors)

    with open(dict_filename, 'w') as f:
        json.dump(index_to_image_path, f)
    logging.info('Finished saving dictionary to file')