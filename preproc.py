from PIL import Image
import glob
import monai.transforms as MT
import torch
import tqdm

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
    ret['img_path'] = f'{folder}/SPACE.nii'
    ret['mask_path'] = f'{folder}/SPACE_mask.nii'
    ret['img'] = tx(ret['img_path'])
    ret['mask'] = tx(ret['mask_path'])

    ret['sag_ndx'] = ret['mask'].sum(dim=[1, 2]).argmax().item()
    ret['cor_ndx'] = ret['mask'].sum(dim=[2, 0]).argmax().item()
    ret['ax_ndx'] = ret['mask'].sum(dim=[0, 1]).argmax().item()
    return ret

def save_images(img, slc_dim, dst, ext='jpg'):
    for slc_ndx in range(img.shape[slc_dim]):

        slc_indexer = [slice(None)] * len(img.shape)
        slc_indexer[slc_dim] = slc_ndx

        slc = img[tuple(slc_indexer)]
        slc = (slc * 255).to(torch.uint8)
        slc_pil = Image.fromarray(slc.numpy(), mode='L')
        slc_pil.save(f'{dst}/{DIR[slc_dim]}_{slc_ndx:03d}.{ext}')

if __name__ == '__main__':
    root = '/home/yua4/for_patrick/ccf'
    knees = glob.glob(f'{root}/*/*')
    for k in tqdm.tqdm(knees):
        ret = proc_folder(k)
        for slc_dim in range(3):
            save_images(ret['img'], slc_dim, k)
        print(k)