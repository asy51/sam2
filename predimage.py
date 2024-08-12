import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import glob
import pandas as pd
import matplotlib
import ast

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='l')
opt = parser.parse_args()

if opt.model == 't':
    sam2_checkpoint = "/home/asy51/repos/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
if opt.model == 's':
    sam2_checkpoint = "/home/asy51/repos/segment-anything-2/checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
if opt.model == 'l':
    sam2_checkpoint = "/home/asy51/repos/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
if opt.model == 'b':
    sam2_checkpoint = "/home/asy51/repos/segment-anything-2/checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"

pred_name = f'predbone_{opt.model}_image'
print(pred_name)

predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))

df = pd.read_csv('/home/asy51/repos/fetch/sam2/centermass.csv')
df['femur'] = df['femur'].str.replace('nan', '-1')
df['tibia'] = df['tibia'].str.replace('nan', '-1')
df['femur'] = df['femur'].apply(lambda x: np.array(ast.literal_eval(x)))
df['tibia'] = df['tibia'].apply(lambda x: np.array(ast.literal_eval(x)))
df = df.drop(columns='Unnamed: 0')
df = df.rename(columns={'idstr': 'IMAGE_PATH_SAG_3D_DESS'})
df['jpgdir'] = df['IMAGE_PATH_SAG_3D_DESS'].str.replace('.tar.gz','_jpg')

for row_ndx, row in df.iterrows():
    pred = {}
    for BONE in ['femur', 'tibia']:
        pred[BONE] = []
        for i in range(160):
            prompt = np.expand_dims(np.array([row[BONE][i][1], row[BONE][i][0], ]), 0)
            if -1 in prompt:
                pred[BONE].append(np.zeros((1,384,384)))
                continue
            img = np.array(Image.open(f'{row["jpgdir"]}/{i:03d}.jpg').convert('RGB'))
            predictor.set_image(img)
            masks, _, _, = predictor.predict(
                point_coords=prompt,
                point_labels=np.array([1]),
                multimask_output=False
            )
            pred[BONE].append(masks)
        pred[BONE] = np.concat(pred[BONE])
    pred = np.stack([pred['femur'], pred['tibia']])
    pred_path = f"{row['jpgdir']}/{pred_name}.npz"
    np.savez_compressed(pred_path, pred)
    print(pred_path)