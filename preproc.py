from PIL import Image
import glob
import monai.transforms as MT
import torch
import tqdm
import os
from sam2.build_sam import build_sam2_video_predictor
import argparse
import numpy as np

tx = MT.Compose([
    MT.LoadImage(image_only=True),
    MT.ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True, relative=False),
    # MT.Lambda(lambda img: img.fliplr().rot90(k=-1, dims=[1,2])),
    MT.ToTensor(track_meta=False),
])

DIR = {
    0: 'sag',
    1: 'cor',
    2: 'ax',
}

def proc_folder(folder):
    """e.g. /home/yua4/for_patrick/ccf/10199/knee_contra"""
    ret = {}
    ret['site'], ret['pid'], ret['side'] = folder.split('/')[-3:]
    ret['root'] = folder
    ret['img_path'] = f'{folder}/SPACE.nii'
    ret['mask_path'] = f'{folder}/SPACE_mask.nii'
    ret['img'] = tx(ret['img_path'])
    ret['mask'] = tx(ret['mask_path'])

    ret['sag_ndx'] = ret['mask'].sum(dim=[1, 2]).argmax().item()
    ret['cor_ndx'] = ret['mask'].sum(dim=[2, 0]).argmax().item()
    ret['ax_ndx'] = ret['mask'].sum(dim=[0, 1]).argmax().item()

    ret['sag_mask'] = ret['mask'].moveaxis(0,0)[ret['sag_ndx']]
    ret['cor_mask'] = ret['mask'].moveaxis(1,0)[ret['cor_ndx']]
    ret['ax_mask'] = ret['mask'].moveaxis(2,0)[ret['ax_ndx']]
    return ret

def save_images(img, slc_dim, dst, ext='jpg'):
    for slc_ndx in range(img.shape[slc_dim]):

        slc_indexer = [slice(None)] * len(img.shape)
        slc_indexer[slc_dim] = slc_ndx

        slc = img[tuple(slc_indexer)]
        slc = (slc * 255).to(torch.uint8)
        slc_pil = Image.fromarray(slc.numpy(), mode='L')
        slc_pil.save(f'{dst}/{DIR[slc_dim]}/{slc_ndx:03d}.{ext}')

def make_pred(predictor, src, prompt, prompt_ndx):
    inference_state = predictor.init_state(video_path=src)
    predictor.reset_state(inference_state)
    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=prompt_ndx,
        obj_id=0,
        mask=prompt,
    )
    pred = {}
    for slc_ndx, obj_ids, pred_logits in predictor.propagate_in_video(inference_state, start_frame_idx=prompt_ndx):
        pred[slc_ndx] = (pred_logits > 0.0).cpu().numpy() # binarize pred_logits

    for slc_ndx, obj_ids, pred_logits in predictor.propagate_in_video(inference_state, start_frame_idx=prompt_ndx, reverse=True):
        pred[slc_ndx] = (pred_logits > 0.0).cpu().numpy() # binarize pred_logits

    pred = torch.concat([pred[slc_ndx] for slc_ndx in sorted(pred.keys())])
    return pred

if __name__ == '__main__':
    root = '/home/yua4/for_patrick/ccf'
    root = '/home/asy51/tmp/ccf'
    knees = glob.glob(f'{root}/*/*')
    for k in tqdm.tqdm(knees):
        ret = proc_folder(k)
        for slc_dim in range(3):
            os.makedirs(f'{k}/{DIR[slc_dim]}', exist_ok=True)
            save_images(ret['img'], slc_dim, k)
        print(k)

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='l')
opt = parser.parse_args(['-m', 'l'])

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

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

src = f"{ret['root']}/ax"
prompt_ndx = ret['ax_ndx']
prompt = ret['ax_mask']

inference_state = predictor.init_state(video_path=src)
predictor.reset_state(inference_state)
predictor.add_new_mask(
    inference_state=inference_state,
    frame_idx=prompt_ndx,
    obj_id=0,
    mask=prompt,
)
pred = {}
for slc_ndx, obj_ids, pred_logits in predictor.propagate_in_video(inference_state, start_frame_idx=prompt_ndx):
    pred[slc_ndx] = (pred_logits > 0.0).cpu().numpy() # binarize pred_logits

for slc_ndx, obj_ids, pred_logits in predictor.propagate_in_video(inference_state, start_frame_idx=prompt_ndx, reverse=True):
    pred[slc_ndx] = (pred_logits > 0.0).cpu().numpy() # binarize pred_logits

pred = np.concatenate([pred[slc_ndx] for slc_ndx in sorted(pred.keys())]).squeeze()