from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
import config

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        print(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.files[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image = input_image, image0 = target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.tranform_only_input(image = input_image)["image"]
        target_image = config.transform_only_mask(image = target_image)["image"]
        return input_image, target_image