import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from tqdm import tqdm
from seg_core import get_files_abs_paths, Seger

import os
# 模型名称，mask2former或segformer。
# mask2former和segformer速度、效果差不多
# 建议优先用mask2former
# 更详尽的模型选择原则请参考配套的PPT
model_type = 'mask2former'
# 数据集名称，cityscapes, mapillary, ade20k。
# 来自百度、谷歌等地图的街道场景街景建议使用前二者（尤其是mapillary）；自己拍照得到的公园、景区等场景的街景建议使用后者
dataset_name = 'mapillary'

# 输入文件夹名称。一般在/root/autodl-tmp文件夹下，比如：/root/autodl-tmp/imgs
input_dir = r'example_data'
# 输出文件夹名称，必须在/root/autodl-tmp下，比如：/root/autodl-tmp/output
out_dir = r'example_data_output'
# 输入的图片是否为全景图（True或者False）
is_panorama = True


##################如果不熟悉代码请不要修改以下部分##################
seger = Seger(dataset_name = dataset_name,
              out_dir = out_dir,
              is_panorama = is_panorama,
              model_type = model_type)
# 获取所有图片
img_paths = get_files_abs_paths(input_dir)
# 遍历每张图片，分割
for img in tqdm(img_paths):
    try:
        seger.seg_batch_images([img])
    except Exception as e:
        print('分割错误：', str(e))
