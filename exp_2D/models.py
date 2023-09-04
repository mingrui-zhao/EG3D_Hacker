"""Image reconstruction model with various types of feature space architecture."""
import os
import numpy as np
import torch
import sys
sys.path.append("/Users/mingruizhao/Desktop/EG3D_Hacker/exp_2D")
from torch import nn
from tqdm import tqdm
from dataloader import get_loader
from utils import bilinear_interpolation, show_model_stats, visualize, psnr
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
import cv2
from torch.utils.tensorboard import SummaryWriter 
import time
from argparse import ArgumentParser
import torch.nn.functional as F
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to("cuda")
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")

def get_args():
    """Get argument parser."""
    parser = ArgumentParser(description='EG3D_hacker')
    # Devices
    parser.add_argument('--device', type=str, default="cuda:0")
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--feature_dim', type=int, default=48, help="feature channels for dual arrays.")
    parser.add_argument('--exp_name', type=str, default="test_code", help="name of the experiment.")
    parser.add_argument('--dual_array_dim', type=int, default=9, help="log 2 dimension of dula arrays")
    parser.add_argument('--dense_grid_dim', type=int, default=5, help="log 2 dimension of dense grid")
    parser.add_argument('--hidden_dim', type=int, default=128, help="hidden layer dim for the decoder")
    parser.add_argument('--output_dim', type=int, default=3, help="output layer dim for the decoder")
    parser.add_argument('--num_layers', type=int, default=3, help="number of layers in the decoder")
    parser.add_argument('--concat', type=bool, default=False, help="wheter to use concatenation over summation to combine features.")
    parser.add_argument('--exp_setup', type=int, default=3, help="integer code for use of different feature spaces/")
    
    args = parser.parse_args()
    return args
    
class FeatureExtractor(nn.Module):
    """Extractor that extract features from images."""
    
    def __init__(self, baselod, feature_dim):
        """Initilaise conv layers."""
        super(FeatureExtractor, self).__init__()
        
        # Input shape: [batch_size, 3, 256, 256]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 512)
        
        # The output shape would be [2**baselod, feature_dim * 2]
        self.fc2 = nn.Linear(512, 2**baselod * feature_dim * 2)
        
    def forward(self, x):
        """Extract features of the image."""
        x = x.unsqueeze(0).permute(0,3,1,2)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # shape: [batch_size, 64, 128, 128]
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # shape: [batch_size, 128, 64, 64]
        
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        
        return x

class dual_array_gen_feature(nn.Module):
    """Dual array method with feature space created by Conv2D Network."""
    
    def __init__(self, feature_extractor, feature_dim = 48):
        super().__init__()
        self.feat_dim = feature_dim
        self.feature_extractor = feature_extractor
        
    def forward(self, pts, image):
        self.feature_array = self.feature_extractor(image).reshape(-1,self.feat_dim*2)
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
    
class dual_array_fit_feature(nn.Module):
    """Dual array method with feature space learnt from scratch."""
    
    def __init__(self, base_lod=7, feature_dim = 48, concat=False):
        """Initialise feature space hyperparameters."""
        super().__init__()
        self.feat_dim = feature_dim
        self.codebook = nn.ParameterList([])
        self.LODS = [2**L for L in range(base_lod, base_lod + 1)] #Each feature array is in length 2^base_lod
        self.init_feature_structure()
        self.concat = concat
        
    def init_feature_structure(self):
        """Initialise two feature arrays."""
        for LOD in self.LODS:
            fts = nn.Parameter(torch.zeros(LOD, 2*self.feat_dim)) #The parameter has LOD rows, first half columns for x features and second half for y features.
            fts = torch.normal(fts, std = torch.tensor([0.1])) # Initliase with some non-zero numbers.
            self.codebook.append(fts)
    
    def forward(self, pts):
        """Query features in dual array based on input point coordinate."""
        feats = []
        # Iterate in every level of detail resolution, in this case just one.
        for i, res in enumerate(self.LODS):
            # Find float x,y position
            x = pts[:,0]*(res-1)
            y = pts[:,1]*(res-1)
            
            # Find the lower integer
            x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).long()
            y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).long()
            
            # Find the upper integer
            x2 = torch.clip(x1 + 1, 0, res - 1).long()
            y2 = torch.clip(y1 + 1, 0, res - 1).long()

            # Compute the weights for each of the four points
            x_weight1 =  x2 - x
            x_weight2 =  x - x1
            y_weight1 =  y2 - y
            y_weight2 =  y - y1
            
            # Interpolate on x and y feature arrays, respectively
            x_features = x_weight1[:,None] * self.codebook[i][x1][:,:self.feat_dim] + x_weight2[:,None] * self.codebook[i][x2][:,:self.feat_dim]
            y_features = y_weight1[:,None] * self.codebook[i][y1][:,self.feat_dim:] + y_weight2[:,None] * self.codebook[i][y2][:,self.feat_dim:]
            
            # Sum them up
            if self.concat:
                features = torch.concat((x_features,y_features), dim=1)
            else:
                features = x_features + y_features
            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)

class Baselinegrid(nn.Module):
    """Baseline MLP that does not use any feature grid."""
    
    def __init__(self):
        """Init as nn module."""
        super().__init__()
    
    def forward(self, pts):
        """Directly pass point coordinates."""
        return pts
    
    
class DenseGrid(nn.Module):
    """Dense grid feature space with single level of details."""
    
    def __init__(self, base_lod=5, feat_dim=48):
        """Initialise feature grid hyper parameters."""
        super().__init__()
        self.feat_dim = feat_dim  # feature dimension
        self.codebook = nn.ParameterList([])
        self.LODS = [2**L for L in range(base_lod, base_lod + 1)]
        print("LODS:", self.LODS)
        self.init_feature_structure()

    def init_feature_structure(self):
        """Initlialse feature grid plane."""
        for LOD in self.LODS:
            fts = nn.Parameter(torch.zeros(LOD**2, self.feat_dim)) # Size is (2^base_lod)^2
            fts = torch.normal(fts, std = torch.tensor([0.1])) # Initialise with some non-zero numbers
            self.codebook.append(fts)

    def forward(self, pts):
        """Query feature in feature grid based on input point coordinate."""
        feats = []
        for i, res in enumerate(self.LODS):
            # Bilinear interpolation on feature grid.
            features = bilinear_interpolation(res, self.codebook[i], pts, grid_type = "NGLOD")
            feats.append((torch.unsqueeze(features, dim=-1)))
        # Sum across all levels of details, in this case we only have 1.
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)

class Decoder(nn.Module):
    """Decoder for feature decoding."""
    
    def __init__(
        self, grid_structure, input_dim, hidden_dim, output_dim, num_hidden_layers=3, pe_order=8, pe=False, feature_extractor=False
    ):
        """Initialise decoder parameters."""
        super().__init__()
        self.module_list = torch.nn.ModuleList()
        self.first_hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.last_hidden_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.intermdediate_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activation = torch.nn.Softplus() # Activation function is softplus as specified in the paper.
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
        self.grid_structure = grid_structure # The type of feature sapce.
        self.pe_order = pe_order # The degree of positional encoding.
        self.pe = pe # Whether use positional encoding
        self.FeatureExtractor = feature_extractor
    def positional_encoding(self, coords):
        """Positional encoding on 2D inputs."""
        t = torch.zeros(coords.shape[0], self.pe_order*4).to("cuda")
        for i in range(self.pe_order):
            t[:, 4*i:4*i+2] = torch.sin(2**i*coords).squeeze()
            t[:, 4*i+2:4*i+4] = torch.cos(2**i*coords).squeeze()
        return t
    
    def forward(self, coords, image=None):
        """Decode learnt features to image space."""
        h, w, c = coords.shape
        coords = torch.reshape(coords, (h*w,2))
        if self.pe:
            coords = self.positional_encoding(coords)
        if self.FeatureExtractor:
            feat = self.grid_structure(coords, image)
        else:
            feat = self.grid_structure(coords)
        out = self.model(feat)
        out = torch.reshape(out,(h,w,3))

        return out


def main():
    config = get_args()
    
    # Set training params
    device = config.device
    max_epochs = config.epochs
    learning_rate = 5.0e-3
    exp_name = config.exp_name
    exp_setup = config.exp_setup
    
    output_path = "./exp_2D/result/" + exp_name
    os.makedirs(output_path,exist_ok=True)
    
    # Load input data
    data_path = "./exp_2D/data/mong_tea.png"
    img_size = 256
    num_imgs_per_iter = 2
    data_loader = get_loader(data_path, img_size, num_imgs_per_iter)
    
    writer = SummaryWriter("./runs/" + exp_name)
    
    exp_setup = config.exp_setup
    
    if exp_setup == 0: # Plain MLP, baseline
        smart_grid = Baselinegrid()
        model = Decoder(smart_grid, config.feature_dim, config.hidden_dim, config.output_dim, config.num_layers).to(device)
    elif exp_setup == 1: # MLP with PE
        smart_grid = Baselinegrid()
        model = Decoder(smart_grid, config.feature_dim, config.hidden_dim, config.output_dim, config.num_layers, pe=True).to(device)
    elif exp_setup == 2: # Dense grid
        smart_grid = DenseGrid(base_lod=config.dense_grid_dim, feat_dim=config.feature_dim)
        model = Decoder(smart_grid, config.feature_dim, config.hidden_dim, config.output_dim, config.num_layers).to(device)
    elif exp_setup == 3: # Dual array, with feature fitted by parameters
        smart_grid = dual_array_fit_feature(base_lod=config.dual_array_dim, feature_dim=config.feature_dim, concat=config.concat)
        if config.concat:
            model = Decoder(smart_grid, config.feature_dim*2, config.hidden_dim, config.output_dim, config.num_layers).to(device)
        else:
            model = Decoder(smart_grid, config.feature_dim, config.hidden_dim, config.output_dim, config.num_layers).to(device)
    elif exp_setup == 4: # Dual array, with feature generated from ConvNet
        feature_gen = FeatureExtractor(baselod = config.dual_array_dim, feature_dim=config.feature_dim)
        smart_grid = dual_array_gen_feature(feature_gen, feature_dim = config.feature_dim)
        model = Decoder(smart_grid, config.feature_dim, config.hidden_dim, config.output_dim, config.num_layers, feature_extractor=True).to(device)
    else:
        raise NotImplementedError
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    show_model_stats(model)

    loop = tqdm(range(max_epochs))
    best_loss = 10000
    
    start_time = time.time()
    for epoch in loop:
        for data in data_loader:
            coords, values = data[0][0], data[1][0]
            coords, values = coords.to(device), values.to(device)
            output = model(coords, values)

            loss = loss_fn(output, values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output_reshaped = torch.clip(output.permute(2,0,1).unsqueeze(0),0,1).to(device)
            values_reshaped = torch.clip(values.permute(2,0,1).unsqueeze(0),0,1).to(device)
            psnr_score = psnr(output, values).item()
            lpips_score = lpips(output_reshaped*2-1, values_reshaped*2-1).item()
            ssim_score = ssim(output_reshaped, values_reshaped).item()
            loop.set_description(f"Epoch: {epoch}")
            loop.set_postfix_str(
                f"PSNR: {psnr_score:.2f}; LPIPS:{lpips_score:.2f}; SSIM:{ssim_score:.2f};Loss: {loss.item():.5f}"
            )
            if loss < best_loss:
                best_psnr = psnr_score
                best_lpips = lpips_score
                best_ssim = ssim_score
                best_loss = loss.item()
            writer.add_scalar('Loss', loss.item(), epoch)
            writer.add_scalar('PSNR', psnr_score, epoch)
            writer.add_scalar('LPIPS', lpips_score, epoch)
            writer.add_scalar('SSIM', ssim_score, epoch)

                
            visualize(output, values)
        if epoch % 100 == 0:
            pred_true = np.hstack((torch.clip(output,0,1).detach().cpu().numpy(), torch.clip(values,0,1).detach().cpu().numpy()))

            pred_true = (pred_true * 255).astype("uint8")
            pred_true = cv2.cvtColor(pred_true, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(output_path, f"epoch_{epoch}.png"), pred_true)
    end_time = time.time()
    elapsed_time = end_time-start_time
    output_str = f"PSNR: {best_psnr:.2f}; LPIPS:{best_lpips:.2f}; SSIM:{best_ssim:.2f};Loss: {best_loss:.5f}"
    num_params = sum([param.nelement() for param in model.parameters()])
    grad_params = sum(
            [
                param.nelement()
                for param in model.parameters()
                if param.requires_grad is True
            ]
        )
    param_detail = f"Num params: {num_params}, Num params need gradients: {grad_params}"
    with open(os.path.join(output_path, "output_metrics.txt"), "w") as f:
        f.write(output_str + "\n")
        f.write(f"Training time: {elapsed_time}" + "\n")
        f.write(str(model) + '\n')
        f.write(param_detail)

    video_name = exp_name + '.avi'

    images = [img for img in os.listdir(output_path) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.png')[0]))

    frame = cv2.imread(os.path.join(output_path, images[0]))

    # Setting the frame width, height width
    # the width, height of an image (assuming all images are the same size)
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 24, (width, height))

    for i, image in enumerate(images):
        epoch_number = int(image.split('_')[1].split('.png')[0])
        
        # Only include images at every 400-epoch interval
        if epoch_number % 400 == 0:
            video.write(cv2.imread(os.path.join(output_path, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()
