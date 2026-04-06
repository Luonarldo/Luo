import os, sys
import numpy as np
# import pandas as pd
import math
from typing import List
import warnings
warnings.filterwarnings("ignore")
import csv
import logging

from PIL import Image
from glob import glob
from my_datasets.cityscapes import Cityscapes
from my_datasets.ade20k import ADE20K
from my_datasets.mapillary import Mapillary

# import mmseg

# from pillow_heif import register_heif_opener
# register_heif_opener()

from pathlib import Path
# from mmseg.apis import init_model, inference_model
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation



logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS 
    else:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


def get_img_mask(img: np.ndarray, class_arr: np.ndarray) -> np.ndarray:
    img2 = img.copy()
    img2 = 0.3*img2 + 0.7*class_arr
    return img2

def create_new_panorama(single_class_mask: np.ndarray):
    height, width = single_class_mask.shape[:2]
    yrow, xcolumn = np.where(single_class_mask == True)
    r = width/(2*math.pi) #球面的半径
    x0, y0 = xcolumn, height/2 - yrow
    x1, y1 = xcolumn + 1, height/2 - (yrow + 1)
    theta0, theta1= y0/r, y1/r
    fi0, fi1 = x0/r, x1/r
    area = (fi1 - fi0)*(np.sin(theta1) - np.sin(theta0))/(4*math.pi)
    area = area.reshape(height, width)
    return np.abs(area)

def combine_gsam(gsam_arr: np.ndarray, 
                 segformer_arr: np.ndarray) -> np.ndarray:
    gsam_sky_mask = np.where(np.all(gsam_arr == [0,0,255], axis=-1) == True)
    segformer_arr[gsam_sky_mask] = 10
    return segformer_arr



model_config_dict = {
    'mask2former': {
        'cityscapes': {
            'config': '.mim/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py',
            'checkpoint': 'checkpoints/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221202_141901-28ad20f1.pth'
        },
        'ade20k': {
            'config': '.mim/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py',
            'checkpoint': 'checkpoints/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth'
        },
        'mapillary': {
        }
    },
    
    'segformer': {
        'cityscapes': {
            'config': '.mim/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
            'checkpoint': 'checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
        },
        'ade20k': {
            'config': '.mim/configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py',
            'checkpoint': 'checkpoints/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'
        }
    },
    
    'depth_anything': {
        'cityscapes': {
            'config': '.mim/configs/depth_anything/depth_anything_large_mask2former_16xb1_80k_cityscapes_896x896.py',
            'checkpoint': 'checkpoints/cityscapes_vitl_mIoU_86.4.pth'
        },
        'ade20k': {
            'config': '.mim/configs/depth_anything/depth_anything_large_mask2former_16xb1_160k_ade20k_896x896.py',
            'checkpoint': 'checkpoints/ade20k_vitl_mIoU_59.4.pth'
        },
    },
    
    'deeplabv3plus': {
        'cityscapes': {
            'config': '.mim/configs/deeplabv3plus/deeplabv3plus_r101-d16-mg124_4xb2-80k_cityscapes-512x1024.py',
            'checkpoint': 'checkpoints/deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes_20200908_005644-ee6158e0.pth'
        },
        'ade20k': {
            'config': '.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py',
            'checkpoint': 'checkpoints/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth'
        }
    }
    
}

class Seger():
    def __init__(self, 
                 dataset_name: str,
                 out_dir: str,
                 is_panorama: bool = False,
                 model_type: str = 'segformer'):

        self.dataset_name = dataset_name
        self.is_panorama = is_panorama
        self.device = 'cuda'
        
        self.templates = {}
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.output_class = out_dir / 'class'
        self.output_mask = out_dir /  'mask'
        self.output_csv = out_dir /   'result.csv'
        self.output_class.mkdir(exist_ok=True, parents=True)
        self.output_mask.mkdir(exist_ok=True, parents=True)

        if dataset_name == 'cityscapes':
            self.decode_fn = Cityscapes.decode_target
            self.id_dict = Cityscapes.name_id_color_dict
        elif dataset_name == 'ade20k':
            self.decode_fn = ADE20K.decode_target
            self.id_dict = ADE20K.name_id_color_dict
        elif dataset_name == 'mapillary':
            self.decode_fn = Mapillary.decode_target
            self.id_dict = Mapillary.name_id_color_dict
        else:
            raise RuntimeError('数据集名称必须为cityscapes、ade20k或mapillary')
 
        car_arr_1024 = np.array(Image.open(get_path("resources/car_label_1024_2048.png")))
        car_arr_512  = np.array(Image.open(get_path("resources/car_label_1024_2048.png")).resize((1024, 512)))
        self.car_mask_1024 = np.where(car_arr_1024==1) 
        self.car_mask_512 = np.where(car_arr_512==1) 

        self.model_type = model_type
        if self.dataset_name == 'mapillary':   
            assert model_type == 'mask2former', 'mapillary数据集仅支持mask2former模型'
            self.seg_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
            self.seg_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic").to('cuda')

        elif self.dataset_name in ['cityscapes', 'ade20k']:
            if model_type in model_config_dict and dataset_name in model_config_dict[model_type]:
                config_path = model_config_dict[model_type][dataset_name]['config']
                config_path = os.path.join(os.path.dirname(mmseg.__file__), config_path)
                checkpoint_path = model_config_dict[model_type][dataset_name]['checkpoint']
                self.seg_model = init_model(config_path, checkpoint_path, device='cuda:0')
            else:
                raise RuntimeError(f'模型名称必须为mask2former或segformer')
        else:
            raise RuntimeError(f'不支持的数据集: {self.dataset_name}')
        
        try:
            self.csv_header = ['file', 'fileBaseName'] + list(self.id_dict.keys())
            if not os.path.exists(self.output_csv):
                with open(self.output_csv, 'w', encoding='utf-8-sig') as f:
                    csv_writer = csv.writer(f, 
                                        delimiter=',', quotechar='"')
                    csv_writer.writerow(self.csv_header)
        except Exception as e:
            raise ValueError('无法创建csv文件', e)        
        
        
    def check_template(self, height, width):
        # 创建不同分辨率的统计模版
        hw_str = f'{height}_{width}'
        if hw_str not in self.templates.keys():
            if self.is_panorama:
                self.templates[hw_str] = create_new_panorama(np.ones((height,width)))
            else:
                self.templates[hw_str] = np.ones((height, width)) / (height * width)
        return hw_str
        
    def stats_class(self, pred_arr: np.ndarray) -> list:
        height, width = pred_arr.shape[:2]
        hw_str = self.check_template(height, width)
        template = self.templates[hw_str]

        class_ids = [idd for idd, _ in self.id_dict.values()]
        flat_pred = pred_arr.flatten()
        flat_template = template.flatten()

        max_id = max(class_ids)
        proportions = np.zeros(max_id + 1)
        np.add.at(proportions, flat_pred, flat_template)
        return [proportions[idd] for idd in class_ids]


  
    def call_model(self, img_paths: list) -> List:
        try:
            res_arrs = inference_model(self.seg_model, img_paths)
            new_arr = res_arrs[0].pred_sem_seg.data.cpu().detach().numpy()            
        except Exception as e:
            raise RuntimeError(f'{img_paths[0]}, 模型推理失败: {e}')
        return new_arr

    def call_model_mapillary(self, img_paths: list) -> List:
        try:
            imgs = []
            for img in img_paths:
                image = Image.open(img)
                imgs.append(image)
                image_size = image.size[::-1]
            # imgs= np.array(imgs).transpose(0,3,1,2)
            inputs  = self.seg_processor(images=imgs, return_tensors="pt", padding=True).to('cuda')
            outputs = self.seg_model(**inputs)
            new_arr = self.seg_processor.post_process_semantic_segmentation(outputs, target_sizes=[image_size for i in range(len(img_paths))])
            new_arr = [r.cpu().detach().numpy() for r in new_arr]
        except Exception as e:
            raise RuntimeError(f'{img_paths[0]}, 模型推理失败: {e}')
        return new_arr
    
    def process_batch_result(self, 
                             img_paths: List[str], 
                             pred_arrs: List[np.ndarray]) -> list:
        tmp_res = []
        for img_path, pred_arr in zip(img_paths, pred_arrs):
            # 上色、统计
            replace_idx = 6 if self.dataset_name == 'ade20k' else 0 if self.dataset_name == 'cityscapes' else 13
            
            if self.is_panorama and pred_arr.shape[0] == 1024 and pred_arr.shape[1] == 2048:
                pred_arr[self.car_mask_1024] = replace_idx
            elif self.is_panorama and pred_arr.shape[0] == 512 and pred_arr.shape[1] == 1024:
                pred_arr[self.car_mask_512] = replace_idx

            color_arr = self.decode_fn(pred_arr).astype('uint8')
            props = self.stats_class(pred_arr)
            masked_arr = get_img_mask(np.asarray(Image.open(img_path)), color_arr).astype(np.uint8)
            # 保存结果
            img_name = os.path.basename(img_path)
            out_class = os.path.join(self.output_class, img_name).replace('.jpg', '.png')
            out_mask = os.path.join(self.output_mask, img_name).replace('.png', '.jpg')
            try:
                Image.fromarray(masked_arr).save(out_mask)
                Image.fromarray(color_arr).save(out_class)
            except Exception as e:
                raise ValueError(f'{img_path}, 保存图片结果失败: {e}')
            img_base_name = os.path.splitext(os.path.basename(img_path))[0]
            tmp_res.append([img_path, img_base_name] + props)
            
            try:
                with open(self.output_csv, 'a', encoding='utf-8-sig') as f:
                    csv_writer = csv.writer(f, 
                                        delimiter=',', quotechar='"')
                    csv_writer.writerow(tmp_res[-1])
            except Exception as e:
                raise ValueError(f'{img_path}, 无法写入csv文件: {e}')
        return tmp_res
        
    def seg_batch_images(self, img_paths: List[str]) -> list:
        if self.dataset_name == 'mapillary':
            pred_arrs = self.call_model_mapillary(img_paths)
        else:
            pred_arrs = self.call_model(img_paths)
        tmp_res = self.process_batch_result(img_paths, pred_arrs)
        return tmp_res


def get_files_abs_paths(directory: str) -> list:
    img_files_abs_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'checkpoint' in file:
                continue
            if file.lower().endswith('.png') or file.lower().endswith('.jpg'):
                abs_path = os.path.join(root, file)
                img_files_abs_paths.append(abs_path)
    if img_files_abs_paths == []:
        print('没有获取到有效的jpg png图像，请检查输入文件夹路径是否正确')
    return img_files_abs_paths
