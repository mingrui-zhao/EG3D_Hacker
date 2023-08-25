from scipy.spatial import KDTree
import sys
sys.path.append("/Users/mingruizhao/Desktop/EG3D")
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import einsum, nn
from tqdm import tqdm
from einops import rearrange
import trimesh
from torch.utils.data import DataLoader, Dataset
from exp_3D.dataloader import get_loader
from exp_3D.utils import bilinear_interpolation, show_model_stats, visualize, psnr_score, visualize_pcd, triplane_interpolation
from argparse import ArgumentParser
import exp_3D.reconstruction as recon
import time

device = 'cpu'

class TriPlane(nn.Module):
    def __init__(self, transform, feat_dim = 48, res=512):
        super().__init__()
        # Feature dimension 48
        self.feat_dim = feat_dim
        self.codebook = nn.ParameterList([])
        self.transform = transform
        self.res = res
        self.init_feature_structure()
        
    def init_feature_structure(self):
        #XY plane, XZ plane, YZ plane
        fts = nn.Parameter(torch.zeros(self.res**2*3, self.feat_dim))
        torch.nn.init.normal_(fts, 0, 0.01)
        self.codebook.append(fts)
    
    def forward(self, pts):
        feats = []
        if len(pts.shape) > 2:
            pts = rearrange(pts, 'b h c -> (b h) c')
        # Iterate in every level of detail resolution
        pts = (torch.linalg.inv(self.transform[:3,:3]) @ (pts.T - self.transform[:3,3:4])).T
        features = triplane_interpolation(self.res, self.codebook[0], pts)
        return features
    
class DenseGrid(nn.Module):
    def __init__(self, transform, feat_dim = 18, base_lod=7, num_lod=1, interpolation_type="bilinear", Rfield = False, concate = False):
        super().__init__()
        self.feat_dim = feat_dim  # feature dim size
        self.codebook = nn.ParameterList([])
        self.interpolation_type = interpolation_type  # bilinear
        self.transform = transform
        self.LODS = [2**L for L in range(base_lod, base_lod + num_lod)]
        print("LODS:", self.LODS)
        self.init_feature_structure()
        self.Rfield = Rfield
        self.concate = concate

    def init_feature_structure(self):
        for LOD in self.LODS:
         
            fts = nn.Parameter(torch.zeros(LOD**3, self.feat_dim))
            torch.nn.init.normal_(fts, 0, 0.01)
            self.codebook.append(fts)

    def forward(self, pts):
        feats = []
        if len(pts.shape) > 2:
            pts = rearrange(pts, 'b h c -> (b h) c')
        # Iterate in every level of detail resolution
        pts = (torch.linalg.inv(self.transform[:3,:3]) @ (pts.T - self.transform[:3,3:4])).T
        for i, res in enumerate(self.LODS):

            if self.interpolation_type == "closest":
                
                x = torch.floor((pts[:,0])*(res-1))
                y = torch.floor((pts[:,1])*(res-1))
                z = torch.floor((pts[:,2])*(res-1))
               
                features = self.codebook[i][(x + y * res + z * res**2).long()]
            elif self.interpolation_type == "bilinear":

                features = bilinear_interpolation(res, self.codebook[i], pts, grid_type = "NGLOD")

            else:
                raise NotImplementedError

            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        all_features = all_features.sum(-1)
        return all_features
   
    
class PCDDataset(Dataset):
    def __init__(self, pcd_path, trainset_size):
        self.trainset_size = trainset_size
        pcd = trimesh.load(pcd_path)
        vertices, colors = np.array(pcd.vertices), np.array(pcd.colors)
        gt_occ = 1*(colors[:,0] > 0) + (-1)*(colors[:,1] > 0)
        self.num_points = len(gt_occ)
        self.vertices = vertices
        self.occ = gt_occ

    def __getitem__(self, index):
        # rand_indices = np.random.choice(self.num_points, self.trainset_size)
        # return self.vertices[rand_indices,:], self.occ[rand_indices]
        # return self.vertices[[index]], self.occ[[index]]
        return self.vertices, self.occ
    
    def __len__(self):
        # return int(self.num_points/ self.trainset_size)
        # return self.num_points
        return 1
    
def get_point_loader(pcd_path, trainset_size):
    return DataLoader(
        PCDDataset(pcd_path, trainset_size), batch_size = 1, shuffle = True,
        num_workers=0,)

class Single_LOD(nn.Module):
    def __init__(self, grid_structure, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super().__init__()
        self.module_list = torch.nn.ModuleList()
        self.first_hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.last_hidden_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.intermdediate_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.module_list = torch.nn.ModuleList()
        self.activation_final = torch.nn.Sigmoid()
        for i in range(num_hidden_layers):
            if i == 0:
                self.module_list.append(self.first_hidden_layer)
            elif i == num_hidden_layers-1:
                self.module_list.append(self.activation)
                self.module_list.append(self.last_hidden_layer)
                self.module_list.append(self.activation_final)
            else:
                self.module_list.append(self.activation)
                self.module_list.append(self.intermdediate_layer)
        ############ END OF YOUR CODE ############
        self.model = torch.nn.Sequential(*self.module_list)
        self.grid_structure = grid_structure

    def forward(self, coords):
        feat = self.grid_structure(coords).float()
        out = self.model(feat)
        return out

def train_epoch(epoch, model, dataloader, loss_fn, optimizer,config, N_reg_points = 5000):
    """Train the model for one epoch.

    Args:
        epoch (int): epoch index.
        model : model to be trained
        dataloader : dataloader
        loss_fn : loss function
        N_reg_points (int): number of point used to determine the total variation loss. Defaults to 5000.

    Returns:
        model: updated model
    """
    for batch_id, data in enumerate(dataloader):
        coords, values = data[0].squeeze(), data[1].squeeze()
        coords = coords.to(device)
        values = values.to(device)
        output = model(coords).squeeze().float().to(device)
        values = values.reshape(output.shape).float().to(device)
        loss_gt = loss_fn(output, values)

        reg_coords = (torch.rand(N_reg_points, 3) * 2 - 1).to(device)
        shift = (torch.randn(N_reg_points, 1) * 0.01).to(device)
        reg_pred = model(reg_coords)
        shift_reg_pred = model(torch.clip(reg_coords + shift, -1, 1))
        reg_loss = torch.mean(torch.abs(reg_pred - shift_reg_pred))
        reg_weight = config.reg_weight  # weight of regularization term. A hyperparameter to be tuned
        loss = (
            loss_gt + reg_loss * reg_weight
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loop.set_description(f"Epoch: {epoch}")
        # loop.set_postfix_str(
        #     f"PSNR: {psnr_score(output, values).item():.2f}; Loss: {loss.item():.5f}; Batch id: {batch_id}"
        # )
    return loss

def get_args():
    parser = ArgumentParser(description = "CMPT764 HW3 model training parameters")

    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--epochs', type = int, default= 100)
    parser.add_argument('--feature_dim', type = int, default = 18)
    parser.add_argument('--num_hidden_layers', type = int, default = 4)
    parser.add_argument('--single_batch', type = bool, default = True)
    parser.add_argument('--trainset_size', type = int, default = 8192)
    parser.add_argument('--interpolation_type', type = str, default = "bilinear")
    parser.add_argument('--grid_type', type = str, default = "DenseGrid")
    parser.add_argument('--reg_weight', type = float, default = 0.01)
    parser.add_argument('--num_lod', type = int, default = 1)
    parser.add_argument('--base_lod', type = int, default = 7)
    parser.add_argument('--min_grid_res', type = int, default = 32)
    parser.add_argument('--max_grid_res', type = int, default = 128)
    parser.add_argument('--bandwidth', type = int, default = 10)
    parser.add_argument('--latent_size', type = int, default = 64)
    parser.add_argument('--Rfield', type = bool, default = False)
    parser.add_argument('--concate', type = bool, default = False)
    args = parser.parse_args()
    return args

def train_object_set(config, folder_name, evaluation = True):
    for cur_obj in os.listdir("./exp_3D/processed"):
            # Load data
            pc = trimesh.load(f"./exp_3D/processed/{cur_obj}")
            verts = np.array(pc.vertices)
            gt_occ = np.array(pc.visual.vertex_colors)[:, 0]
            gt_occ = (gt_occ == 0).astype("float32") * -1 + 1

            # model = Baseline(verts, gt_occ)
            _, feature_trans = recon.generate_grid(verts, 1)
            feature_trans = torch.tensor(feature_trans).to(device)
            
            # Train
            max_epochs = config.epochs
            learning_rate = config.lr

            data_loader = recon.get_point_loader(verts, gt_occ, config.trainset_size)
            # smart_grid = m.DenseGrid(feature_trans,  interpolation_type="closest").to(device)  # "closest" or "bilinear"
            if config.grid_type == "DenseGrid":
                smart_grid = DenseGrid(feature_trans, feat_dim = config.feature_dim, base_lod=config.base_lod, num_lod=config.num_lod, interpolation_type = config.interpolation_type, Rfield= config.Rfield, concate = config.concate).to(device)
            elif config.grid_type == "TriPlane":
                smart_grid = TriPlane(feature_trans).to(device)
            
            model = Single_LOD(smart_grid, config.feature_dim, config.latent_size, 1, config.num_hidden_layers).to(device)

            loss_fn = torch.nn.BCELoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
            show_model_stats(model)

            loop = tqdm(range(max_epochs))
            start = time.time()
            for epoch in loop:
                model.train()
                loss = train_epoch(epoch, model, data_loader,loss_fn, optimizer,config)
            end = time.time()
            time_spent = end - start
            # Inference
            resolution = 128
            grid, transform = recon.generate_grid(verts, res=resolution)
            grid = torch.tensor(grid).to(device)
    
            model.to(device)

            reconstr_dir = f"{folder_name}/{config.grid_type}_{config.num_lod}_{config.epochs}_{config.reg_weight}"
            reconstr_path = reconstr_dir + f"/{cur_obj}"
            os.makedirs(os.path.dirname(reconstr_path), exist_ok=True)
            try:
                rec_verts, rec_faces = recon.reconstruct(model, grid, resolution, transform)
            except Exception:
                pass
                print("Reconstruction failed")
                continue

            print("Time spent: ", time_spent)
            trimesh.Trimesh(rec_verts, rec_faces).export(reconstr_path)

            if evaluation:
                gt_path = f"data/{cur_obj}"

                chamfer_dist, hausdorff_dist = recon.compute_metrics(
                    reconstr_path, gt_path, num_samples=1000000
                )
                txt_file = os.path.join(reconstr_dir,"evaluation.txt")
                if os.path.isfile(txt_file):
                    f = open(os.path.join(reconstr_dir,"evaluation.txt"), "a")
                else:
                    f = open(os.path.join(reconstr_dir,"evaluation.txt"), "w")
                f.write(f"{cur_obj}\n")
                f.write(f"Chamfer distance: {chamfer_dist:.4f}\n")
                f.write(f"Hausdorff distance: {hausdorff_dist:.4f}\n")
                f.write(f"Training time: {time_spent:.4f}\n")
                f.close()
                print(cur_obj)
                print(f"Chamfer distance: {chamfer_dist:.4f}")
                print(f"Hausdorff distance: {hausdorff_dist:.4f}")
                print("##################")
                
if __name__ == "__main__":
    config = get_args()
    config.feature_dim = 48
    config.grid_type = "TriPlane"
    train_object_set(config, "single_lod", evaluation = False)
    