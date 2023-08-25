import cv2
import torch
from torch.utils.data import DataLoader, Dataset


def create_coords(h, w):
    grid_y, grid_x = torch.meshgrid(
        [torch.linspace(0, 1, steps=h), torch.linspace(0, 1, steps=w)]
    )
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


class ImageDataset(Dataset):
    def __init__(self, image_path, img_dim, trainset_size):
        self.trainset_size = trainset_size
        self.img_dim = (img_dim, img_dim) if isinstance(img_dim, int) else img_dim

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        left_w = int((w - h) / 2)

        image = image[:, left_w : left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)

        self.coords = create_coords(image.shape[0], image.shape[1])

        self.img = image

    def __getitem__(self, idx):
        image = self.img / 255
        return self.coords, torch.tensor(image, dtype=torch.float32)

    def __len__(self):
        return self.trainset_size


def get_loader(image_path, img_dim, trainset_size):
    return DataLoader(
        ImageDataset(image_path, img_dim, trainset_size),
        batch_size=1,
        num_workers=0,
    )
