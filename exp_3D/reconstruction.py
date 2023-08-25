import os
import torch
import numpy as np
import trimesh
from scipy.spatial import KDTree
from skimage import measure
from tqdm import tqdm
from utils import bilinear_interpolation, show_model_stats, visualize, psnr_score, visualize_pcd
from torch.utils.data import DataLoader, Dataset

def generate_grid(point_cloud, res):
    """Generate grid over the point cloud with given resolution
    Args:
        point_cloud (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
    Returns:
        coords (np.array, [res*res*res, 3]): grid vertices
        coords_matrix (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]
    """
    b_min = np.min(point_cloud, axis=0)
    b_max = np.max(point_cloud, axis=0)

    coords = np.mgrid[:res, :res, :res]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    length += length / res
    coords_matrix[0, 0] = length[0] / res
    coords_matrix[1, 1] = length[1] / res
    coords_matrix[2, 2] = length[2] / res
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    coords = coords.T

    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples):
    """Predict occupancy of values batch-wise
    Args:
        points (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_samples (int): number of points to evaluate at once
    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """

    num_pts = points.shape[0]
    occ = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        occ[i * num_samples : i * num_samples + num_samples] = eval_func(
            points[i * num_samples : i * num_samples + num_samples]
        ).squeeze().detach().cpu().numpy()
        # occ[i * num_samples : i * num_samples + num_samples] = eval_func(
        #     points[i * num_samples : i * num_samples + num_samples]
        # )
    if num_pts % num_samples:
        occ[num_batches * num_samples :] = eval_func(
            points[num_batches * num_samples :]
        ).squeeze().detach().cpu().numpy()
        # occ[num_batches * num_samples :] = eval_func(
        #     points[num_batches * num_samples :]
        # )

    # occ[occ > 0] = 1.
    # occ[occ < 0] = -1.
    return occ


def eval_grid(coords, eval_func, num_per_sample=1024):
    """Predict occupancy of values on a grid
    Args:
        coords (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_per_sample (int): number of points to evaluate at once

    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """
    coords = coords.reshape([-1, 3])
    occ = batch_eval(coords, eval_func, num_samples=num_per_sample)
    return occ


def reconstruct(model, grid, res, transform):
    """Reconstruct mesh by predicting occupancy values on a grid
    Args:
        model (function): function that takes a batch of points and returns occupancy values
        grid (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
        transform (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]

    Returns:
        verts (np.array, [M, 3]): 3D coordinates of M vertices
        faces (np.array, [K, 3]): indices of K faces
    """

    occ = eval_grid(grid, model)
    occ = occ.reshape([res, res, res])

    verts, faces, normals, values = measure.marching_cubes(occ, 0.5)
    verts = np.matmul(transform[:3, :3], verts.T) + transform[:3, 3:4]
    verts = verts.T

    return verts, faces


def compute_metrics(reconstr_path, gt_path, num_samples=1000000):
    """Compute chamfer and hausdorff distances between the reconstructed mesh and the ground truth mesh
    Args:
        reconstr_path (str): path to the reconstructed mesh
        gt_path (str): path to the ground truth mesh
        num_samples (int): number of points to sample from each mesh

    Returns:
        chamfer_dist (float): chamfer distance between the two meshes
        hausdorff_dist (float): hausdorff distance between the two meshes
    """
    reconstr = trimesh.load(reconstr_path)
    gt = trimesh.load(gt_path)

    # sample points on the mesh surfaces using trimesh
    reconstr_pts = reconstr.sample(num_samples)
    gt_pts = gt.sample(num_samples)

    # compute chamfer distance between the two point clouds
    reconstr_tree = KDTree(reconstr_pts)
    gt_tree = KDTree(gt_pts)
    dist1, _ = reconstr_tree.query(gt_pts)
    dist2, _ = gt_tree.query(reconstr_pts)
    chamfer_dist = (dist1.mean() + dist2.mean()) / 2
    hausdorff_dist = max(dist1.max(), dist2.max())

    return chamfer_dist, hausdorff_dist

class PCDDataset(Dataset):
    def __init__(self, verts, gt_occ, trainset_size):
        self.vertices = verts
        self.occ = gt_occ
        self.trainset_size = trainset_size
        self.num_points = len(gt_occ)

    def __getitem__(self, index):
        # rand_indices = np.random.choice(self.num_points, self.trainset_size)
        # return self.vertices[rand_indices,:], self.occ[rand_indices]
        # return self.vertices[[index]], self.occ[[index]]
        return self.vertices, self.occ
    
    def __len__(self):
        # return 2*int(self.num_points/ self.trainset_size)
        # return self.num_points
        return 1
    
def get_point_loader(verts, gt_occ, trainset_size):
    return DataLoader(
        PCDDataset(verts, gt_occ,trainset_size=trainset_size), batch_size = 1, shuffle = False,
        num_workers=0,)

if __name__ == "__main__":
    from model import Baseline
    import model as m

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="HW3-SingleLOD",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.00001,
        "architecture": "MLP",
        "epochs": 100,
        }
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for cur_obj in os.listdir("processed"):
        pc = trimesh.load(f"processed/{cur_obj}")
        verts = np.array(pc.vertices)
        gt_occ = np.array(pc.visual.vertex_colors)[:, 0]
        gt_occ = (gt_occ == 0).astype("float32") * -1 + 1
        # model = Baseline(verts, gt_occ)
        _, feature_trans = generate_grid(verts, 1)
        feature_trans = torch.tensor(feature_trans).to(device)
        # train model
        max_epochs = 100
        learning_rate = 1.0e-3

        data_path = f"processed/{cur_obj}"
        trainset_size = 8192
        # data_loader = m.get_point_loader(data_path, num_imgs_per_iter)
        data_loader = get_point_loader(verts, gt_occ, trainset_size)
        smart_grid = m.DenseGrid(feature_trans,  interpolation_type="closest").to(device)  # "closest" or "bilinear"
        model = m.Single_LOD(smart_grid, 8, 64, 1, 5).to(device)
        # loss_fn = torch.nn.MSELoss()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        show_model_stats(model)

        loop = tqdm(range(max_epochs))
        for epoch in loop:
            for batch_id, data in enumerate(data_loader):
                coords, values = data[0].squeeze(), data[1].squeeze()
                coords = coords.to(device)
                values = values.to(device)
                output = model(coords).squeeze().float().to(device)
                values = values.reshape(output.shape).float().to(device)
                loss = loss_fn(output, values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({"loss":loss.item(), "data_length":data[0].shape[0]})
                loop.set_description(f"Epoch: {epoch}")
                loop.set_postfix_str(
                    f"PSNR: {psnr_score(output, values).item():.2f}; Loss: {loss.item():.5f}; Batch id: {batch_id}"
                )

        resolution = 128
        grid, transform = generate_grid(verts, res=resolution)
        grid = torch.tensor(grid).to(device)
        # transform = torch.tensor(transform).to(device)
        model.to(device)

        rec_verts, rec_faces = reconstruct(model, grid, resolution, transform)

        reconstr_path = f"reconstructions/{cur_obj}"
        os.makedirs(os.path.dirname(reconstr_path), exist_ok=True)
        trimesh.Trimesh(rec_verts, rec_faces).export(reconstr_path)

        recon_pcd_path = f"reconstructions_pcd/{cur_obj}"
        os.makedirs(os.path.dirname(recon_pcd_path), exist_ok=True)
        occ = eval_grid(grid, model)
        colors = np.zeros_like(grid.detach().cpu().numpy())
        colors[occ > 0] = [1, 0, 0]
        colors[occ < 0] = [0, 1, 0]
        trimesh.points.PointCloud(grid.detach().cpu().numpy(), colors=colors).export(recon_pcd_path)
        gt_path = f"data/{cur_obj}"

        chamfer_dist, hausdorff_dist = compute_metrics(
            reconstr_path, gt_path, num_samples=1000000
        )

        print(cur_obj)
        print(f"Chamfer distance: {chamfer_dist:.4f}")
        print(f"Hausdorff distance: {hausdorff_dist:.4f}")
        print("##################")
