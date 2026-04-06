import os
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from local_datasets.cityscapes import Cityscapes
from local_datasets.ade20k import ADE20K
from local_datasets.mapillary import Mapillary

# 数据集名称
dataset_name = 'mapillary'
# 原始图像目录
src_image_dir = '../gsam-data/0623'
# 语义分割结果色块图目录
seg_color_dir = '../gsam-data-output/0623/class'
# 保存提取结果的目录
extracted_dir = '../gsam-data-output/0623/ext'
# 需要提取的类别
classes_to_extract = ['Building']
# 进程数，保持就行
num_workers = 10

def process_image(image, src_image_dir, seg_color_dir, extracted_dir, colors):
    image_name = image.stem
    seg_color_path = os.path.join(seg_color_dir, image_name + '.png')
    out_path = os.path.join(extracted_dir, image_name + '.png')
    
    if not os.path.exists(seg_color_path):
        print(f'{seg_color_path} not exists, skip')
        return
    
    src_image = np.array(Image.open(image))
    seg_color = np.array(Image.open(seg_color_path))

    # 生成掩码：对 `colors` 中的每种颜色进行匹配，并合并所有掩码
    masks = np.zeros(seg_color.shape[:2], dtype=bool)
    for color in colors:
        masks |= np.all(seg_color == color, axis=-1)

    extracted_image = np.zeros_like(src_image)
    extracted_image[masks] = src_image[masks]

    # 保存提取结果
    if extracted_image.size > 0:
        Image.fromarray(extracted_image).save(out_path)
        print(f'extracted from {image_name}')

def main(dataset_name, src_image_dir, seg_color_dir, extracted_dir, classes_to_extract, num_workers):
    if dataset_name == 'cityscapes':
        decode_fn = Cityscapes.decode_target
        id_dict = Cityscapes.name_id_color_dict
    elif dataset_name == 'ade20k':
        decode_fn = ADE20K.decode_target
        id_dict = ADE20K.name_id_color_dict
    elif dataset_name == 'mapillary':
        decode_fn = Mapillary.decode_target
        id_dict = Mapillary.name_id_color_dict
    
    extracted_dir = Path(extracted_dir)
    extracted_dir.mkdir(exist_ok=True)

    src_images = list(Path(src_image_dir).glob('*.jpg'))
    colors = np.array([id_dict[c][1] for c in classes_to_extract])

    # 使用多进程加速处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for image in src_images:
            futures.append(executor.submit(process_image, image, src_image_dir, seg_color_dir, extracted_dir, colors))
        for future in as_completed(futures):
            future.result()

if __name__ == '__main__':
    main(dataset_name, src_image_dir, seg_color_dir, extracted_dir, classes_to_extract, num_workers)
