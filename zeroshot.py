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
import time
import sys

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='l')
# parser.add_argument('-i', '--image', action='store_true')
parser.add_argument('-n', '--negatives', action='store_true')
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

pred_name = f'predbone_{opt.model}{"_neg" if opt.negatives else ""}.npz'
print(pred_name)

# if opt.image:
    # predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
# else:
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


FRAME_NDX = 80
FEMUR_ID = 1
PATELLA_ID = 2
TIBIA_ID = 3

COLOR = {
    FEMUR_ID: 'r',
    PATELLA_ID: 'y',
    TIBIA_ID: 'g',
}

VIS_FRAME_STRIDE = 10

preds = {}

df = pd.read_csv('/home/asy51/repos/segment-anything-2/prompts/coords.csv')
df['jpgdir'] = df['jpg'].str.replace('/080.jpg','')
for row_ndx, row in df.iterrows():
    # plt.close("all")

    loc = {}
    loc[FEMUR_ID] = [row['femur_x'], row['femur_y']]
    loc[PATELLA_ID] = [row['patella_x'], row['patella_y']]
    loc[TIBIA_ID] = [row['tibia_x'], row['tibia_y']]

    ### SHOW PROMPTS
    frames = sorted(glob.glob(f'{row["jpgdir"]}/*.jpg'))
    inference_state = predictor.init_state(video_path=row["jpgdir"])
    predictor.reset_state(inference_state)

    # fig, ax = plt.subplots()
    # ax.matshow(np.array(Image.open(row['jpg'])), cmap='gray')

    for BONE_ID in range(1,4):
        if opt.negatives:
            labels = [0,0,0]
            labels[BONE_ID-1] = 1
            labels = np.array(labels, dtype=np.int32)
            points = np.array([loc[bone_id] for bone_id in range(1,4)], dtype=np.float32)
        else:
            labels = np.array([1], dtype=np.int32)
            points = np.array([loc[BONE_ID]], dtype=np.float32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=FRAME_NDX,
            obj_id=BONE_ID,
            points=points,
            labels=labels,
        )
        # ax.scatter(*loc[BONE_ID], c=COLOR[BONE_ID])
        # show_mask((out_mask_logits[BONE_ID-1] > 0.0).cpu().numpy(), ax, color=COLOR[BONE_ID])
    # plt.show()

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=FRAME_NDX):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=FRAME_NDX, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    # for out_frame_idx in range(0, len(frames), VIS_FRAME_STRIDE):
    #     fig, ax = plt.subplots(figsize=(4,4))
    #     ax.set_title(f"frame {out_frame_idx}")
    #     ax.matshow(np.array(Image.open(os.path.join(row['jpgdir'], frames[out_frame_idx]))), cmap='gray')
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         show_mask(out_mask, ax, color=COLOR[out_obj_id])
    #     plt.show() 

    pred = {BONE_ID: np.concat([video_segments[idx][BONE_ID] for idx in range(len(frames))]) for BONE_ID in range(1,4)}
    pred_path = f"{row['jpgdir']}/{pred_name}.npz"
    np.savez_compressed(pred_path, np.stack([pred[1], pred[2], pred[3]]))
    print(pred_path)
    # y = None
    # dice_fn = lambda _: 1.0
    # dice = dice_fn(pred, y)