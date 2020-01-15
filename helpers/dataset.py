import torch
import os
import cv2
from torchvision import transforms


class IconsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, imread_mode=cv2.IMREAD_COLOR):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = os.listdir(root_dir)
        self.mode = imread_mode

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.filenames[idx])
        image = cv2.imread(filename, self.mode)

        if self.transform:
            image = self.transform(image)

        return image

    
basic_img_transform = transforms.Compose([transforms.ToTensor()])