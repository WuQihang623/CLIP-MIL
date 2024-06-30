import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def is_file_size_less_than_50kb(file_path):
    file_size = os.path.getsize(file_path)

    size_limit = 50 * 1024

    return file_size > size_limit

def is_tumor(label_path):
    label = np.load(label_path)
    return label.sum() / (label.shape[0] * label.shape[1]) > 0.2

class CustomDataset(Dataset):
    """
    Args:
        data_root(str): a path to the data root
        template(list): a template to get the image path
        processor(CLIPProcessor): packing image and text pairs
    """
    def __init__(self, image_list, label_list, processor):
        self.image_list = image_list
        self.label_list = label_list
        self.processor = processor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        label = self.label_list[idx]
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=[image], return_tensors="pt", padding=True)

        input_item = {key: val.squeeze(0) for key, val in inputs.items()}

        return {"input": input_item, "label": label}

