import os
import time
import h5py
import argparse

import torch
import openslide
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from src import clip
from src.feature_extraction.vision_transformers import vit_small
from src.feature_extraction.resnet_trunc import resnet50_trunc_baseline
from src.wsi_core.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from src.wsi_core.utils import save_hdf5, print_network, collate_features

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
clip_RN50_path = ""
clip_ViTB32_path = "ViT-B/32"

def eval_transforms_clip(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((224, 224)),
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val

def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val


def load_model(enc_name, assets_dir):
    if enc_name == 'resnet50_trunc':
        model = resnet50_trunc_baseline(pretrained=True)
        eval_t = eval_transforms(pretrained=True)

    elif enc_name == 'clip_RN50':
        model, preprocess = clip.load(clip_RN50_path)
        eval_t = eval_transforms_clip(pretrained=True)

    elif enc_name == 'clip_ViTB32':
        model, preprocess = clip.load(clip_ViTB32_path)
        eval_t = eval_transforms_clip(pretrained=True)

    elif enc_name == 'model_dino' or enc_name == 'dino_HIPT':
        ckpt_path = os.path.join(assets_dir, enc_name + '.pth')
        assert os.path.isfile(ckpt_path)
        model = vit_small(patch_size=16)
        state_dict = torch.load(ckpt_path, map_location="cpu")['teacher']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # print("Missing Keys:", missing_keys)
        # print("Unexpected Keys:", unexpected_keys)
        eval_t = eval_transforms(pretrained=False)


    elif enc_name == 'model_simclr_histo_res18':
        ckpt_path = os.path.join(assets_dir, enc_name + '.ckpt')
        assert os.path.isfile(ckpt_path)

        model = torchvision.models.__dict__['resnet18'](pretrained=False)

        state = torch.load(ckpt_path, map_location='cpu')
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        eval_t = eval_transforms(pretrained=True)
    else:
        raise ValueError("Invalid encoder name")

    model = model.to(device)
    if (enc_name != 'clip_ViTB32' and enc_name != 'clip_RN50'):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    model.eval()

    return model, eval_t


def compute_w_loader(file_path, output_path, wsi, model, enc_name,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, target_patch_size=-1, custom_transforms=None):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size, custom_transforms=custom_transforms)
    x, y = dataset[0]
    kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            if (enc_name != 'clip_RN50' and enc_name != 'clip_ViTB32'):
                features = model(batch).cpu().numpy()
            else:
                features = model.encode_image(batch).cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'

    return output_path

def get_args():
    parser = argparse.ArgumentParser(description='Feature Extraction for Whole Slide Images')
    parser.add_argument('--enc_name', type=str, default="clip_ViTB32", help="Name of the encoder model to use (e.g., 'resnet50_trunc', 'clip_RN50', 'clip_ViTB32').")
    parser.add_argument('--assets_dir', type=str, default=None, help="Directory containing model assets such as checkpoint files.")
    parser.add_argument('--data_root', type=str, default="/home/auwqh/dataset/HER2/patch/", help="Root directory for input data.")
    parser.add_argument('--data_slide_dir', type=str, default="/home/auwqh/dataset/HER2/WSI/Testing/WSI/", help="Directory containing the whole slide image files.")
    parser.add_argument('--slide_ext', type=str, default='.tiff', help="File extension of the whole slide images (default: '.tiff').")
    parser.add_argument('--csv_path', type=str, default="/home/auwqh/dataset/HER2/patch/process_list_autogen.csv", help="Path to the CSV file listing all bags.")
    parser.add_argument('--feat_dir', type=str, default=None, help="Directory to save the extracted features.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for computing features (default: 256).")
    parser.add_argument('--no_auto_skip', default=False, action='store_true', help="Disable automatic skipping of already processed files.")
    parser.add_argument('--custom_downsample', type=int, default=1, help="Custom downscale factor for image patches (default: 1).")
    parser.add_argument('--target_patch_size', type=int, default=256, help="Custom target patch size before embedding (default: 256).")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print('initializing dataset')
    args = get_args()
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError
    bags_dataset = Dataset_All_Bags(csv_path)

    if args.feat_dir is None:
        args.feat_dir = args.enc_name
    save_dir = os.path.join(args.data_root, args.feat_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, f'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, f'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(save_dir, f'pt_files'))

    print('loading model checkpoint')
    model, custom_transforms = load_model(args.enc_name, args.assets_dir)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id, slide_ext = os.path.splitext(bags_dataset[bag_candidate_idx])
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_root, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        if not os.path.exists(slide_file_path) or not os.path.exists(h5_file_path):
            continue

        output_path = os.path.join(save_dir, f'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi, enc_name=args.enc_name,
                                            model=model, batch_size=args.batch_size, verbose=1, print_every=20,
                                            custom_downsample=args.custom_downsample,
                                            target_patch_size=args.target_patch_size,
                                            custom_transforms=custom_transforms)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(save_dir, f'pt_files', bag_base + '.pt'))



