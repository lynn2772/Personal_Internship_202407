import os
import xml.etree.ElementTree as ET
from PIL import Image


def crop_images(annotations_dir, images_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        base_filename = os.path.splitext(xml_file)[0]
        image_path = os.path.join(images_dir, f"{base_filename}.jpg")

        if not os.path.exists(image_path):
            print(f"Image file {base_filename}.jpg not found in {images_dir}, skipping.")
            continue

        image = Image.open(image_path)
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            cropped_filename = f"{base_filename}.jpg"
            cropped_image.save(os.path.join(output_dir, cropped_filename))
            print(f"Cropped image saved as {cropped_filename}")


# 指定路径
annotations_dir = '../cable_rarity1300/Annotations'
images_dir = '../cable_rarity1300/images'
output_dir = '../cropped_images'

# 调用函数裁剪并保存图片
crop_images(annotations_dir, images_dir, output_dir)
