import json
import os
import glob
import numpy as np
from PIL import Image
from skimage import color

def is_file_size_less_than_50kb(file_path):
    file_size = os.path.getsize(file_path)

    size_limit = 50 * 1024

    return file_size > size_limit

def is_tumor(label_path):
    label = np.load(label_path)
    return label.sum() / (label.shape[0] * label.shape[1]) > 0.2

def is_stained(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image, dtype=np.uint8)
    image = color.rgb2hed(image)
    image_dab = image[:, :, 2]
    return (image_dab > 0.2).sum() / (image.shape[0] * image.shape[1]) > 0.01

def get_tumor_image_list(data_root, save_path):
    save_image_list = []
    label_list = []

    image_list = glob.glob(os.path.join(data_root, 'Images', "*", "*.png"))
    for image_path in image_list:
        if is_file_size_less_than_50kb(image_path):
            save_image_list.append(image_path)
            label_path = image_path.replace("Images", "Labels").replace("png", "npy")
            if is_tumor(label_path):
                label_list.append(1)
            else:
                label_list.append(0)
    print("image_list:", len(save_image_list))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(json.dumps({"image_list": save_image_list, "label_list": label_list}, indent=2))
        f.close()

def get_stained_image_list(data_root, save_path):
    save_image_list = []
    label_list = []

    image_list = glob.glob(os.path.join(data_root, 'Images', "*", "*.png"))
    for image_path in image_list:
        if is_file_size_less_than_50kb(image_path):
            save_image_list.append(image_path)
            if is_stained(image_path):
                label_list.append(1)
            else:
                label_list.append(0)
    print("image_list:", len(save_image_list))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(json.dumps({"image_list": save_image_list, "label_list": label_list}, indent=2))
        f.close()


if __name__ == '__main__':
    # get_tumor_image_list("/home/auwqh/dataset/PDL1/DatasetTissue2classes/", "/home/auwqh/code/CLIP-MIL/data/tumor_dataset.json")
    get_stained_image_list("/home/auwqh/dataset/PDL1/DatasetTissue2classes/", "/home/auwqh/code/CLIP-MIL/data/stained_dataset.json")