import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import trimesh
import numpy as np

def create_coords(h, w, k):
    grid_y, grid_x, grid_z = torch.meshgrid(
        [torch.linspace(0, 1, steps=h), torch.linspace(0, 1, steps=w),torch.linspace(0, 1, steps=k)]
    )
    grid = torch.stack([grid_y, grid_x, grid_z], dim=-1)
    return grid

class PCDDataset(Dataset):
    def __init__(self, pcd_path, trainset_size):
        self.trainset_size = trainset_size
  
        
        pcd = trimesh.load(pcd_path)
        vertices, colors = np.array(pcd.vertices), np.array(pcd.colors)
        gt_occ = 1*(colors[:,0] > 0) + (-1)*(colors[:,1] > 0)
        self.num_points = len(gt_occ)
        self.vertices = vertices
        self.occ = gt_occ
        self.coord = create_coords(512, 512, 512)
    
    def __getitem__(self, index):
        rand_indices = np.random.choice(self.num_points, self.trainset_size)
        return self.vertices[rand_indices,:], self.occ[rand_indices]
        
    def __len__(self):
        return self.trainset_size
    
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

def get_point_loader(pcd_path, trainset_size):
    return DataLoader(
        PCDDataset(pcd_path,trainset_size), batch_size = 128,
        num_workers=0,)
