import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh
import torch
import cv2

def plot_points(path):
    ax = plt.figure().add_subplot(projection="3d")
    obj = trimesh.load(path)
    x, y, z = obj.vertices[:, 0], obj.vertices[:, 1], obj.vertices[:, 2]
    mask = obj.colors[:, 1] == 255
    ax.scatter(
        x[mask], y[mask], zs=z[mask], zdir="y", alpha=1, c=obj.colors[mask] / 255
    )
    ax.scatter(
        x[~mask], y[~mask], zs=z[~mask], zdir="y", alpha=0.01, c=obj.colors[~mask] / 255
    )
    plt.show()


def download_data():
    import gdown

    if not os.path.exists("./data"):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1EKWU_daQL3pxFkjFUomGs25_qekyfeAd",
            quiet=False,
        )

    if not os.path.exists("./processed"):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/175_LtuWh1LknbbMjUumPjGzeSzgQ4ett",
            quiet=False,
        )



def show_model_stats(model):
    print(model)
    print("Num of params:", sum([param.nelement() for param in model.parameters()]))
    print(
        "Num of params require grad:",
        sum(
            [
                param.nelement()
                for param in model.parameters()
                if param.requires_grad is True
            ]
        ),
    )


def psnr_score(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def visualize(output, img, save=False):
    out = output.detach().cpu().numpy()
    out = (out - out.min()) / (out.max() - out.min())

    pred_true = np.hstack((out, img.detach().cpu().numpy()))

    pred_true = (pred_true * 255).astype("uint8")
    pred_true = cv2.cvtColor(pred_true, cv2.COLOR_RGB2BGR)

    if save:
        cv2.imwrite(f"temp/pred_true.jpg", pred_true)

    cv2.imshow("Pred_true", pred_true)
    cv2.waitKey(1)

def visualize_pcd(output, pcd, save=False):
    out = output.detach().cpu().numpy()
    color_code = np.zeros_like(pcd)
    color_code[out > 0] = np.tile(np.array([1,0,0]), (sum(out>0),1))
    color_code[out < 0] = np.tile(np.array([0,1,0]), (sum(out<0),1))
   
    if save:
       trimesh.points.PointCloud(pcd, colors = color_code).export("temp/pred.obj")

   
def triplane_interpolation(res, grid, points):
    # Get the dimensions of the grid
    grid_size,  feat_size = grid.shape
    points = points[None]
    _, N, _ = points.shape
    # Get the x and y coordinates of the four nearest points for each input point
    x = points[:, :, 0] * (res - 1)
    y = points[:, :, 1] * (res - 1)
    z = points[:, :, 2] * (res - 1)
    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()
    z1 = torch.floor(torch.clip(z, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()
    z2 = torch.clip(z1 + 1, 0, res - 1).int()

    # XY interpolatioin
    w1 = (x2 - x) * (y2 - y)
    w2 = (x - x1) * (y2 - y)
    w3 = (x2 - x) * (y - y1)
    w4 = (x - x1) * (y - y1)
    id1 = (x1 + y1 * res).long()
    id2 = (y1 * res + x2).long()
    id3 = (y2 * res + x1).long()
    id4 = (y2 * res + x2).long()
    feature_xy = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
    )
    # XZ interpolation
    w5 = (x2 - x) * (z2 - z)
    w6 = (x - x1) * (z2 - z)
    w7 = (x2 - x) * (z - z1)
    w8 = (x - x1) * (z - z1)
    id5 = (x1 + z1 * res + res ** 2).long()
    id6 = (z1 * res + x2 + res ** 2).long()
    id7 = (z2 * res + x1 + res ** 2).long()
    id8 = (z2 * res + x2 + res ** 2).long()
    feature_xz = (
        torch.einsum("ab,abc->abc", w5, grid[(id5).long()])
        + torch.einsum("ab,abc->abc", w6, grid[(id6).long()])
        + torch.einsum("ab,abc->abc", w7, grid[(id7).long()])
        + torch.einsum("ab,abc->abc", w8, grid[(id8).long()])
    )
    # YZ interpolation
    w9 = (y2 - y) * (z2 - z)
    w10 = (y - y1) * (z2 - z)
    w11 = (y2 - y) * (z - z1)
    w12 = (y - y1) * (z - z1)
    id9 = (y1 + z1 * res + 2 * res ** 2).long()
    id10 = (z1 * res + y2 + 2 * res ** 2).long()
    id11 = (z2 * res + y1 + 2 * res ** 2).long()
    id12 = (z2 * res + y2 + 2 * res ** 2).long()
    feature_yz = (
        torch.einsum("ab,abc->abc", w9, grid[(id9).long()])
        + torch.einsum("ab,abc->abc", w10, grid[(id10).long()])
        + torch.einsum("ab,abc->abc", w11, grid[(id11).long()])
        + torch.einsum("ab,abc->abc", w12, grid[(id12).long()])
    )
    return feature_xy[0] + feature_xz[0] + feature_yz[0]


def bilinear_interpolation(res, grid, points, grid_type):
    """
    Performs bilinear interpolation of points with respect to a grid.

    Parameters:
        grid (numpy.ndarray): A 2D numpy array representing the grid.
        points (numpy.ndarray): A 2D numpy array of shape (n, 2) representing
            the points to interpolate.

    Returns:
        numpy.ndarray: A 1D numpy array of shape (n,) representing the interpolated
            values at the given points.
    """
    PRIMES = [1, 265443567, 805459861]

    # Get the dimensions of the grid
    grid_size,  feat_size = grid.shape
    points = points[None]
    _, N, _ = points.shape
    # Get the x and y coordinates of the four nearest points for each input point
    x = points[:, :, 0] * (res - 1)
    y = points[:, :, 1] * (res - 1)
    z = points[:, :, 2] * (res - 1)

    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()
    z1 = torch.floor(torch.clip(z, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()
    z2 = torch.clip(z1 + 1, 0, res - 1).int()

    # Compute the weights for each of the four points
    w1 = (x2 - x) * (y2 - y) * (z2 - z)
    w2 = (x - x1) * (y2 - y) * (z2 - z)
    w3 = (x2 - x) * (y - y1) * (z2 - z)
    w4 = (x - x1) * (y - y1) * (z2 - z)

    w5 = (x2 - x) * (y2 - y) * (z - z1)
    w6 = (x - x1) * (y2 - y) * (z - z1)
    w7 = (x2 - x) * (y - y1) * (z - z1)
    w8 = (x - x1) * (y - y1) * (z - z1)
    if grid_type == "NGLOD":
        id1 = (x1 + y1*res + z1* (res**2)).long() # x1, y1, z1
        id2 = (x2 + y1*res + z1* (res**2)).long() # x2, y1, z1
        id3 = (x1 + y2*res + z1* (res**2)).long() # x1, y2, z1
        id4 = (x2 + y2*res + z1* (res**2)).long() # x2, y2, z1
        id5 = (x1 + y1*res + z2* (res**2)).long() # x1, y1, z2
        id6 = (x2 + y1*res + z2* (res**2)).long() # x2, y1, z2
        id7 = (x1 + y2*res + z2* (res**2)).long() # x1, y2, z2
        id8 = (x2 + y2*res + z2* (res**2)).long() # x2, y2, z2

    elif grid_type == "HASH":
        npts = res**3
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id2 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id3 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id4 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id5 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id6 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id7 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id8 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size

        else:
            id1 = (x1 + y1*res + z1* (res**2)).long() # x1, y1, z1
            id2 = (x2 + y1*res + z1* (res**2)).long() # x2, y1, z1
            id3 = (x1 + y2*res + z1* (res**2)).long() # x1, y2, z1
            id4 = (x2 + y2*res + z1* (res**2)).long() # x2, y2, z1
            id5 = (x1 + y1*res + z2* (res**2)).long() # x1, y1, z2
            id6 = (x2 + y1*res + z2* (res**2)).long() # x2, y1, z2
            id7 = (x1 + y2*res + z2* (res**2)).long() # x1, y2, z2
            id8 = (x2 + y2*res + z2* (res**2)).long() # x2, y2, z2

    else:
        print("NOT IMPLEMENTED")

    values = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
        + torch.einsum("ab,abc->abc", w5, grid[(id5).long()])
        + torch.einsum("ab,abc->abc", w6, grid[(id6).long()])
        + torch.einsum("ab,abc->abc", w7, grid[(id7).long()])
        + torch.einsum("ab,abc->abc", w8, grid[(id8).long()])
    )
    
    return values[0].float()
