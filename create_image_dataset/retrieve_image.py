import os
import faiss
import logging
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import time  # 导入time模块

start_time = time.time()  # 记录开始时间
# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# 初始化AutoImageProcessor和AutoModel
# 如果本地没有下载dinov2-small模型，则：
# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
# model = AutoModel.from_pretrained('facebook/dinov2-small')
# 如果本地已经下载了dinov2-small模型，则输入模型所在的文件夹：
processor = AutoImageProcessor.from_pretrained('../dinov2-small')
model = AutoModel.from_pretrained('../dinov2-small')

process_model_time = time.time() - start_time
logging.info("Processing model took time: {}".format(process_model_time))
# 加载Faiss索引
index_path = "image_features.index"
index = faiss.read_index(index_path)
#字典路径
dict_path = "index_to_image_info.json"
#设置输入图片路径
input_image_path = "../cropped_images/ra_1.jpg"
def knn_search(image_path, k=1):
    """使用Faiss进行k-最近邻搜索"""

    image = Image.open(image_path)
    # 处理图片并获取模型输出
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    features = outputs.last_hidden_state[0].detach().numpy()
    features = features.reshape(1, -1)

    D, I = index.search(features, k)
    return D[0],I[0]  # 返回最近邻的索引数组

def retrieve_similar_images(input_image_path, k=1):
    """检索最相似的图片并显示"""
    logging.info(f"Processing input image: {input_image_path}")
    distances, indices = knn_search(input_image_path, k)
    logging.info(f"Distances: {distances}")
    logging.info(f"Indices: {indices}")
    for idx in indices:
        str_idx=str(idx)
        if str_idx in index_to_image_info:
            image_info = index_to_image_info[str_idx]
            image_path = image_info["path"]
            image_label = image_info["label"]
            logging.info(f"Retrieved image: index:{idx}, path: {image_path}, Label: {image_label}")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image.show()
            else:
                logging.warning(f"Image file '{image_path}' does not exist.")
        else:
            logging.warning(f"Index {idx} does not exist in the dictionary.")
    return

if __name__ == "__main__":
    image = Image.open(input_image_path)
    # image.show(title='imput_image')

    if not os.path.exists(input_image_path):
        logging.error("The input image path does not exist.")
        exit(1)
    # 读取字典文件
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            index_to_image_info = json.load(f)
        logging.info("Dictionary loaded successfully.")
    else:
        logging.error(f"Dictionary file '{dict_path}' does not exist.")
        exit(1)

    retrieve_similar_images(input_image_path, k=3)

    script_time = time.time() - start_time
    logging.info(f"Script took {script_time:.4f} seconds")