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
import monai.transforms as MT
import ast
import SimpleITK as sitk
from sam2.utils.misc import mask_to_box
import sys

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

CENTER_NDX=80
N_SLC=160
FEMUR=0
TIBIA=1
PATELLA=2

COLOR = {
    FEMUR: 'r',
    PATELLA: 'y',
    TIBIA: 'g',
}

VIS_FRAME_STRIDE = 10

class JPEGDS(torch.utils.data.Dataset):
    def __init__(self):
        df = pd.read_csv('img_and_mask.csv')
        df['jpgdir'] = df['IMAGE_PATH_SAG_3D_DESS'].str.replace('.tar.gz','_jpg')

        ef = pd.read_csv('/home/asy51/repos/fetch/sam2/centermass.csv')
        ef['femur'] = ef['femur'].str.replace('nan', '-1')
        ef['tibia'] = ef['tibia'].str.replace('nan', '-1')
        ef['femur'] = ef['femur'].apply(lambda x: np.array(ast.literal_eval(x)))
        ef['tibia'] = ef['tibia'].apply(lambda x: np.array(ast.literal_eval(x)))
        ef = ef.drop(columns=['Unnamed: 0',])
        ef = ef.rename(columns={'idstr': 'IMAGE_PATH_SAG_3D_DESS'})

        ff = pd.read_csv('/home/asy51/repos/segment-anything-2/prompts/coords.csv')

        df = df.merge(ef, on='IMAGE_PATH_SAG_3D_DESS', validate='1:1')
        df = df.merge(ff, on='IMAGE_PATH_SAG_3D_DESS', validate='1:1')
        self.df = df.drop(columns=['jpg', 'ndx'])

        self.mask_tx = MT.Compose([
            MT.Lambda(lambda x: torch.from_numpy(x).swapaxes(-1, 0).rot90(k=1, dims=[-2,-1])),
            MT.Lambda(lambda x: x.unsqueeze(0)),
            MT.AsDiscrete(to_onehot=5),
            MT.Lambda(lambda x: x[[1, 3]]),
            MT.ToTensor(track_meta=False),
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ndx):
        ret = self.df.iloc[ndx].to_dict()
        ret['mask'] = self.mask_tx(sitk.GetArrayFromImage(sitk.ReadImage(ret['mask'])))
        ret['jpg'] = sorted(glob.glob(f'{ret["jpgdir"]}/*.jpg'))
        ret['center'] = {}
        ret['center'][FEMUR] = [ret['femur_x'], ret['femur_y']]
        ret['center'][TIBIA] = [ret['tibia_x'], ret['tibia_y']]
        ret['center'][PATELLA] = [ret['patella_x'], ret['patella_y']]
        return ret
    
def make_pred(predictor, d, dims=3, prompt='point', negatives=False, **kwargs):
    if dims == 2:
        image_batch = [Image.open(f'{d["jpgdir"]}/{i:03d}.jpg').convert('RGB') for i in range(N_SLC)]
        pred = {}
        for BONE, bone_str in enumerate(['femur', 'tibia']):
            pred[BONE] = []
            for slc_ndx in range(N_SLC):
                # TODO: 
                point_coords = np.expand_dims([d[bone_str][slc_ndx][1], d[bone_str][slc_ndx][0]], axis=0)
                if -1 in point_coords:
                    pred[BONE].append(np.zeros((1, 384, 384)))
                    continue
                mask_input = d['mask'][BONE][slc_ndx][None,None].to(bool)
                box = mask_to_box(mask_input)
                predictor.set_image(image_batch[slc_ndx])
                masks, _, _ = predictor.predict(
                    point_coords=point_coords if prompt == 'point' else None,
                    point_labels=np.array([1]) if prompt == 'point' else None,
                    box=box if prompt == 'box' else None,
                    mask_input=mask_input if prompt == 'mask' else None,
                    multimask_output=False,
                )
                pred[BONE].append(masks)
            pred[BONE] = np.concat(pred[BONE])
        pred = np.stack([pred[FEMUR], pred[TIBIA]], dtype=bool)

    elif dims == 3:
        inference_state = predictor.init_state(video_path=d["jpgdir"])
        predictor.reset_state(inference_state)
        if prompt == 'mask':
            for BONE in range(2):
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=CENTER_NDX,
                    obj_id=BONE,
                    mask=d['mask'][BONE][CENTER_NDX],
                )

        elif prompt == 'point':
            for BONE in range(3):
                if negatives:
                    labels = [0,0,0]
                    labels[BONE] = 1
                    labels = np.array(labels, dtype=np.int32)
                    points = np.array([d['center'][bone] for bone in range(3)], dtype=np.float32)
                else:
                    labels = np.array([1], dtype=np.int32)
                    points = np.array([d['center'][BONE]], dtype=np.float32)

                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=CENTER_NDX,
                    obj_id=BONE,
                    points=points,
                    labels=labels,
                )

        elif prompt == 'box':
            for BONE in range(2):
                labels = np.array([1], dtype=np.int32)
                box = mask_to_box(d['mask'][BONE][CENTER_NDX][None,None].to(bool))
                points = np.array([d['center'][BONE]], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=CENTER_NDX,
                    obj_id=BONE,
                    # points=points,
                    # labels=labels,
                    box=box
                )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=CENTER_NDX):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=CENTER_NDX, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        pred = np.stack([np.concat([video_segments[idx][BONE] for idx in range(N_SLC)]) for BONE in sorted(video_segments[0].keys())])

    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='l')
    parser.add_argument('-p', '--prompt', type=str, default='point')
    parser.add_argument('-n', '--negatives', action='store_true')
    parser.add_argument('-d', '--dims', type=int, default=3)
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

    pred_name = f'predbone_{opt.model}{"_neg" if opt.negatives else ""}_{opt.prompt}{"_img" if opt.dims == 2 else "_vid"}'
    print(pred_name)

    if opt.dims == 2:
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
    else:
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    ds = JPEGDS()
    preds = {}
    for d in ds:
        pred = make_pred(predictor, d, **vars(opt))
        pred_path = f"{d['jpgdir']}/{pred_name}.npz"
        np.savez_compressed(pred_path, pred)
        print(pred_path)

    # fig, ax = plt.subplots()
    # ax.matshow(np.array(Image.open(row['jpg'])), cmap='gray')

    # for BONE_ID in range(1,4):
        # ax.scatter(*loc[BONE_ID], c=COLOR[BONE_ID])
        # show_mask((out_mask_logits[BONE_ID-1] > 0.0).cpu().numpy(), ax, color=COLOR[BONE_ID])
    # plt.show()

    # for out_frame_idx in range(0, len(frames), VIS_FRAME_STRIDE):
    #     fig, ax = plt.subplots(figsize=(4,4))
    #     ax.set_title(f"frame {out_frame_idx}")
    #     ax.matshow(np.array(Image.open(os.path.join(row['jpgdir'], frames[out_frame_idx]))), cmap='gray')
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         show_mask(out_mask, ax, color=COLOR[out_obj_id])
    #     plt.show() 
