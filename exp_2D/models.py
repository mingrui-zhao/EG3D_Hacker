# %%[markdown] #########################################
# # Execute once (no edits needed)
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append("/Users/mingruizhao/Desktop/EG3D_Hacker/exp_2D")
from torch import einsum, nn
from tqdm import tqdm
from einops import rearrange
from dataloader import get_loader
from utils import bilinear_interpolation, show_model_stats, visualize, psnr_score
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

class feature_array_generator(nn.Module):
    def __init__(self, in_channel=3, out_channel=96):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, out_channel, kernel_size=3, stride=1, padding=0),
            nn.Flatten()
            # nn.ReLU(),
            # nn.Conv2d(64, self.out_channel, kernel_size=3, stride=1, padding=0)
        )
    
    def forward(self, image):
        return self.cnn(image)

class dual_array(nn.Module):
    def __init__(self, feature_gen, feature_dim = 48):
        super().__init__()
        self.feat_dim = feature_dim
        self.feature_gen = feature_gen
        
    def forward(self, pts, image):
        self.feature_array = self.feature_gen(image).permute(1,0)
        res = self.feature_array.shape[0]
        x = pts[:,0]*(res-1)
        y = pts[:,1]*(res-1)
        x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).long()
        y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).long()

        x2 = torch.clip(x1 + 1, 0, res - 1).long()
        y2 = torch.clip(y1 + 1, 0, res - 1).long()

        # Compute the weights for each of the four points
        x_weight1 =  x2 - x
        x_weight2 =  x - x1
        y_weight1 =  y2 - y
        y_weight2 =  y - y1
        
        features = x_weight1[:,None] * self.feature_array[x1][:,:self.feat_dim] + x_weight2[:,None] * self.feature_array[x2][:,:self.feat_dim] + y_weight1[:,None] * self.feature_array[y1][:,self.feat_dim:] + y_weight2[:,None] * self.feature_array[y2][:,self.feat_dim:]
        return features
    
class dual_plane(nn.Module):
    def __init__(self, base_lod=9, num_lod=1, resolution_pow = 18, feature_dim = 48):
        super().__init__()
        self.feat_dim = feature_dim
        self.codebook = nn.ParameterList([])
        self.resolution_pow = resolution_pow
        self.LODS = [2**L for L in range(base_lod, base_lod + num_lod)]
        self.init_feature_structure()
        
    def init_feature_structure(self):
        for LOD in self.LODS:
            fts = nn.Parameter(torch.zeros(LOD, 2*self.feat_dim))
            fts = torch.normal(fts, std = torch.tensor([0.1]))
            self.codebook.append(fts)
    
    def forward(self, pts):
        feats = []
        # Iterate in every level of detail resolution
        for i, res in enumerate(self.LODS):
            x = pts[:,0]*(res-1)
            y = pts[:,1]*(res-1)
            x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).long()
            y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).long()

            x2 = torch.clip(x1 + 1, 0, res - 1).long()
            y2 = torch.clip(y1 + 1, 0, res - 1).long()

            # Compute the weights for each of the four points
            x_weight1 =  x2 - x
            x_weight2 =  x - x1
            y_weight1 =  y2 - y
            y_weight2 =  y - y1
            
            features = x_weight1[:,None] * self.codebook[i][x1][:,:self.feat_dim] + x_weight2[:,None] * self.codebook[i][x2][:,:self.feat_dim] + y_weight1[:,None] * self.codebook[i][y1][:,self.feat_dim:] + y_weight2[:,None] * self.codebook[i][y2][:,self.feat_dim:]
            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)

class Baselinegrid(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pts):
        return pts
    
    
class DenseGrid(nn.Module):
    def __init__(self, base_lod=7, num_lod=1, interpolation_type="closest"):
        super().__init__()
        self.feat_dim = 3  # feature dim size
        self.codebook = nn.ParameterList([])
        self.interpolation_type = interpolation_type  # bilinear

        self.LODS = [2**L for L in range(base_lod, base_lod + num_lod)]
        print("LODS:", self.LODS)
        self.init_feature_structure()

    def init_feature_structure(self):
        for LOD in self.LODS:

            fts = nn.Parameter(torch.zeros(LOD**2, self.feat_dim))
            fts = torch.normal(fts, std = torch.tensor([0.1]))
 
            self.codebook.append(fts)

    def forward(self, pts):
        feats = []
        # Iterate in every level of detail resolution
        for i, res in enumerate(self.LODS):

            if self.interpolation_type == "closest":

                x = torch.floor(pts[:,0]*(res-1))
                y = torch.floor(pts[:,1]*(res-1))
        

                features = self.codebook[i][(x + y * res).long()]
            elif self.interpolation_type == "bilinear":

                features = bilinear_interpolation(res, self.codebook[i], pts, grid_type = "NGLOD")


            else:
                raise NotImplementedError

            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)


class SimpleModel(nn.Module):
    def __init__(
        self, grid_structure, input_dim, hidden_dim, output_dim, num_hidden_layers=3, pe_order=8
    ):
        super().__init__()
        self.module_list = torch.nn.ModuleList()
        self.first_hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.last_hidden_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.intermdediate_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activation = torch.nn.Softplus()
        for i in range(num_hidden_layers):
            if i == 0:
                self.module_list.append(self.first_hidden_layer)
                self.module_list.append(self.activation)
            elif i == num_hidden_layers-1:
                self.module_list.append(self.last_hidden_layer)
            else:
                self.module_list.append(self.intermdediate_layer)
                self.module_list.append(self.activation)
   
        self.model = torch.nn.Sequential(*self.module_list)
        self.grid_structure = grid_structure
        self.pe_order = pe_order
    
    def positional_encoding(self, coords):
        t = torch.zeros(coords.shape[0], self.pe_order*4)
        for i in range(self.pe_order):
            t[:, 4*i:4*i+2] = torch.sin(2**i*coords).squeeze()
            t[:, 4*i+2:4*i+4] = torch.cos(2**i*coords).squeeze()
        return t
    
    def forward(self, coords):
        h, w, c = coords.shape
        coords = torch.reshape(coords, (h*w,2))
        # coords = self.positional_encoding(coords)
        feat = self.grid_structure(coords)
        out = self.model(feat)
        out = torch.reshape(out,(h,w,3))

        return out


def main():
    # Set training params
    max_epochs = 1000
    learning_rate = 5.0e-3

    data_path = "./exp_2D/data/mong_tea.png"
    img_size = 256
    num_imgs_per_iter = 2
    data_loader = get_loader(data_path, img_size, num_imgs_per_iter)
    
    # feature_gen = feature_array_generator()
    # smart_grid = dual_plane()
    # smart_grid = DenseGrid(interpolation_type="bilinear")  # "closest" or "bilinear"
    smart_grid = Baselinegrid()
    # smart_grid = dual_array(feature_gen)
    model = SimpleModel(smart_grid, 2, 128, 3, 4)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    show_model_stats(model)

    loop = tqdm(range(max_epochs))
    for epoch in loop:
        for data in data_loader:
            coords, values = data[0][0], data[1][0]

            output = model(coords)

            loss = loss_fn(output, values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output_reshaped = torch.clip(output.permute(2,0,1).unsqueeze(0),0,1)
            values_reshaped = torch.clip(values.permute(2,0,1).unsqueeze(0),0,1)
            loop.set_description(f"Epoch: {epoch}")
            loop.set_postfix_str(
                f"PSNR: {psnr_score(output, values).item():.2f}; LPIPS:{lpips(output_reshaped*2-1, values_reshaped*2-1).item():.2f}; SSIM:{ssim(output_reshaped, values_reshaped).item():.2f};Loss: {loss.item():.5f}"
            )

            visualize(output, values)


if __name__ == "__main__":
    main()
# %%
