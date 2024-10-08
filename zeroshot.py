import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.utils.misc import mask_to_box
from sam2.sam2_image_predictor import SAM2ImagePredictor
import glob
import pandas as pd
import matplotlib
import time
import sys
import ast
import monai.transforms as MT
import SimpleITK as sitk
from monai.visualize import matshow3d as m3d
import tempfile
from skimage.morphology import skeletonize
from matplotlib.colors import to_rgb

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

N_SLC=160

BONE_ID = {
    'background': 0,
    'femur': 1,
    'femur_cartilage': 2,
    'tibia': 3,
    'tibia_cartilage': 4,
    'patella': 5,
}

COLOR = {
    'femur': to_rgb('tab:blue'),
    'femur_cartilage': to_rgb('tab:orange'),
    'tibia': to_rgb('tab:green'),
    'tibia_cartilage': to_rgb('tab:red'),
}

def mask_to_centroid(mask):
    if mask.dim() == 3:
        centroids = []
        for i in range(mask.size(0)):
            slice_mask = mask[i]
            slice_mask = slice_mask.float()
            indices = torch.nonzero(slice_mask, as_tuple=False).float()
            centroid = indices.mean(dim=0)
            centroids.append(centroid.flip(dims=[0]))
        return torch.stack(centroids)
    elif mask.dim() == 2:
        mask = mask.float()
        indices = torch.nonzero(mask, as_tuple=False).float()
        centroid = indices.mean(dim=0)
        return centroid.flip(dims=[0])
    raise ValueError

def mask_to_skeleton_centroid(mask):
    if mask.sum() == 0: return np.array([np.nan, np.nan])
    skeleton_mask = skeletonize(np.array(mask))
    skeleton_coords = np.flip(np.argwhere(skeleton_mask))
    skeleton_centroid = skeleton_coords.mean(axis=0)
    distances = np.sqrt((skeleton_coords[:, 0] - skeleton_centroid[0]) ** 2 + (skeleton_coords[:, 1] - skeleton_centroid[1]) ** 2)
    min_index = np.argmin(distances)
    return skeleton_coords[min_index]

def dice_fn(a,b):
    a = a.numpy().astype(bool) if isinstance(a, torch.Tensor) else a.astype(bool)
    b = b.numpy().astype(bool) if isinstance(b, torch.Tensor) else b.astype(bool)
    if a.sum() == 0 and b.sum() == 0: return np.nan
    return (2 * (a & b).sum() / (a.sum() + b.sum())).item()

class JPEGDS(torch.utils.data.Dataset):
    def __init__(self):
        df = pd.read_csv('/home/asy51/repos/segment-anything-2/img_and_mask.csv')
        df['jpgdir'] = df['IMAGE_PATH_SAG_3D_DESS'].str.replace('.tar.gz','_jpg')

        ff = pd.read_csv('/home/asy51/repos/segment-anything-2/prompts/coords.csv') # manual prompts
        df = df.merge(ff, on='IMAGE_PATH_SAG_3D_DESS', validate='1:1')
        self.df = df.drop(columns=['jpg', 'ndx'])

        self.mask_tx = MT.Compose([
            MT.Lambda(lambda x: torch.from_numpy(x).swapaxes(-1, 0).rot90(k=1, dims=[-2,-1])),
            MT.Lambda(lambda x: x.unsqueeze(0)),
            MT.AsDiscrete(to_onehot=5),
            # MT.Lambda(lambda x: x[[1, 3]]),
            MT.Lambda(lambda x: {'femur': x[1], 'femur_cartilage': x[2], 'tibia': x[3], 'tibia_cartilage': x[4]}),
            MT.ToTensor(track_meta=False),
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ndx):
        ret = self.df.iloc[ndx].to_dict()
        ret['mask'] = self.mask_tx(sitk.GetArrayFromImage(sitk.ReadImage(ret['mask'])))
        ret['jpg'] = sorted(glob.glob(f'{ret["jpgdir"]}/*.jpg'))
        # ret['centroid'] = {k: mask_to_centroid(ret['mask'][k]) for k in ['femur', 'femur_cartilage', 'tibia', 'tibia_cartilage']}
        return ret
    
def make_pred(predictor, d, dims=3, prompt='point', components=['femur', 'femur_cartilage', 'tibia', 'tibia_cartilage',],
              start_ndx=40, mask_to_point=mask_to_skeleton_centroid, **kwargs):
    if dims == 2:
        image_batch = [Image.open(f'{d["jpgdir"]}/{i:03d}.jpg').convert('RGB') for i in range(N_SLC)]
        pred = {}
        for bone_ndx, bone in enumerate(components):
            pred[bone] = []
            for slc_ndx in range(N_SLC):
                mask = d['mask'][bone][slc_ndx]
                if mask.sum() == 0:
                    pred[bone].append(np.zeros((1, 384, 384)))
                    continue
                predictor.set_image(image_batch[slc_ndx])
                points = labels = box = None
                if prompt == 'pointneg':
                    points = np.array([mask_to_point(d['mask'][b][start_ndx]) for b in components], dtype=np.float32)
                    labels = np.array([0] * len(components), dtype=np.int32)
                    labels[bone_ndx] = 1
                elif prompt == 'point':
                    points = np.array([mask_to_point(d['mask'][bone][start_ndx])], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                elif prompt == 'box':
                    box = mask_to_box(d['mask'][bone][slc_ndx][None,None].to(bool))

                masks, _, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box,
                    mask_input=None,
                    multimask_output=False,
                )
                pred[bone].append(masks)
            pred[bone] = np.concatenate(pred[bone]).astype(bool)

    elif dims == 3:
        inference_state = predictor.init_state(video_path=d["jpgdir"])
        predictor.reset_state(inference_state)
        if prompt == 'mask':
            for bone in components:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=start_ndx,
                    obj_id=BONE_ID[bone],
                    mask=d['mask'][bone][start_ndx],
                )

        elif prompt == 'box' or prompt == 'point' or prompt == 'pointneg':
            for bone_ndx, bone in enumerate(components):

                points = labels = box = None
                if prompt == 'pointneg':
                    points = np.array([mask_to_point(d['mask'][b][start_ndx]) for b in components], dtype=np.float32)
                    labels = np.array([0] * len(components), dtype=np.int32)
                    labels[bone_ndx] = 1
                elif prompt == 'point':
                    points = np.array([mask_to_point(d['mask'][bone][start_ndx])], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                elif prompt == 'box':
                    box = mask_to_box(d['mask'][bone][start_ndx][None,None].to(bool))
                else: raise ValueError
                
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=start_ndx,
                    obj_id=BONE_ID[bone],
                    points=points,
                    labels=labels,
                    box=box,
                )
        else: raise ValueError

        pred = {}
        for slc_ndx, obj_ids, pred_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_ndx):
            pred[slc_ndx] = (pred_logits > 0.0).cpu().numpy() # binarize pred_logits
            pred[slc_ndx] = {next((k for k, v in BONE_ID.items() if v == obj_id)): pred[slc_ndx][i] for i, obj_id in enumerate(obj_ids)} # pred_ndx => obj_id => obj_str

        for slc_ndx, obj_ids, pred_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_ndx, reverse=True):
            pred[slc_ndx] = (pred_logits > 0.0).cpu().numpy() # binarize pred_logits
            pred[slc_ndx] = {next((k for k, v in BONE_ID.items() if v == obj_id)): pred[slc_ndx][i] for i, obj_id in enumerate(obj_ids)} # pred_ndx => obj_id => obj_str

        pred = {bone: np.concatenate([pred[slc_ndx][bone] for slc_ndx in range(N_SLC)]) for bone in components}

    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='l')
    parser.add_argument('-p', '--prompt', type=str, default='pointneg')
    parser.add_argument('-d', '--dims', type=int, default=3)
    opt = parser.parse_args()
    opt.components = ['femur', 'femur_cartilage', 'tibia', 'tibia_cartilage',]

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

    predname = f'{opt.model}_{opt.prompt}{"_img" if opt.dims == 2 else "_vid"}'
    print(predname)
    if os.path.exists(f'results/skeltroid/{predname}.csv'):
        print('already done! :)')
        sys.exit(0)
    if predname in ['t_box_vid', 't_box_img', 'b_box_img', 's_box_vid', 'b_box_vid', 'l_box_img', 'l_box_vid', 's_box_img']:
        print('already running D:')
        sys.exit(0)

    if opt.dims == 2:
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
    else:
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    ds = JPEGDS()
    dice = {}
    for i, d in enumerate(ds):
        pred = make_pred(predictor, d, **vars(opt))
        gt = {k: d['mask'][k].numpy() for k in d['mask']}
        idstr = d['IMAGE_PATH_SAG_3D_DESS'].split('00m/')[-1].replace('.tar.gz','')
        dice[idstr] = {k: dice_fn(pred[k], gt[k]) for k in opt.components}
        dice[idstr]['combined'] = dice_fn(np.concatenate([pred[k] for k in opt.components]), np.concatenate([gt[k] for k in opt.components]))
        # pred_path = f"{d['jpgdir']}/{predname}.npz"
        # np.savez_compressed(pred_path, pred)
        print(idstr)
    pd.DataFrame(dice).T.reset_index(drop=False).rename(columns={'index': 'idstr'}).to_csv(f'results/skeltroid/{predname}.csv', index=False)
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
