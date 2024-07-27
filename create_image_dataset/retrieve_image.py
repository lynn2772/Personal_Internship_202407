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
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')

# 加载Faiss索引
index_path = "image_features.index"
index = faiss.read_index(index_path)
#字典路径
dict_path = "index_to_image_path.json"
def knn_search(image_path, k=1):
    """使用Faiss进行k-最近邻搜索"""
    knn_start_time = time.time()

    image = Image.open(image_path)
    # 处理图片并获取模型输出
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    features = outputs.last_hidden_state[0].detach().numpy()
    features = features.reshape(1, -1)

    D, I = index.search(features, k)

    knn_time = time.time() - knn_start_time
    logging.info(f"Knn took {knn_time:.4f} seconds")
    return D[0],I[0]  # 返回最近邻的索引数组

def retrieve_similar_images(input_image_path, k=1):
    """检索最相似的图片并显示"""
    logging.info(f"Processing input image: {input_image_path}")
    distances, indices = knn_search(input_image_path, k)
    logging.info(f"Distances: {distances}")
    logging.info(f"Indices: {indices}")
    for idx in indices:
        str_idx=str(idx)
        if str_idx in index_to_image_path:
            image_path = index_to_image_path[str_idx]
            logging.info(f"Displaying image for index {idx}: {image_path}")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image.show()
            else:
                logging.warning(f"Image file '{image_path}' does not exist.")
        else:
            logging.warning(f"Index {idx} does not exist in the dictionary.")
    return
    # image_dir = os.path.dirname(input_image_path)
    # if not similar_indices.size:
    #     logging.warning("No similar images found.")
    #     return
    #
    # for idx in similar_indices:
    #     similar_image_filename = os.listdir(image_dir)[idx]
    #     similar_image_path = os.path.join(image_dir, similar_image_filename)
    #     logging.info(f"Similar image: {similar_image_path}")
    #     similar_image = Image.open(similar_image_path)
    #     similar_image.show()  # 显示图片

if __name__ == "__main__":
    input_image_path = "../image/20240715-193837.jpg"
    image = Image.open(input_image_path)
    # image.show(title='imput_image')

    if not os.path.exists(input_image_path):
        logging.error("The input image path does not exist.")
        exit(1)
    # 读取字典文件
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            index_to_image_path = json.load(f)
        logging.info("Dictionary loaded successfully.")
    else:
        logging.error(f"Dictionary file '{dict_path}' does not exist.")
        exit(1)

    retrieve_similar_images(input_image_path, k=3)

    time = time.time() - start_time
    logging.info(f"Script took {time:.4f} seconds")