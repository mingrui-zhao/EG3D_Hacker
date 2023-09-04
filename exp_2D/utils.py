import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import lpips
from PIL import Image
from skimage import metrics

lpips_model = lpips.LPIPS(net="alex")

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


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def lpips_score(img1, img2):
    image1_tensor = torch.tensor(img1).permute(2,0,1).unsqueeze(0).float()/255.0
    image2_tensor = torch.tensor(img2).permute(2,0,1).unsqueeze(0).float()/255.0
    distance = lpips_model(image1_tensor, image2_tensor)
    return distance


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
    grid_size, feat_size = grid.shape
    points = points[None]
    _, N, _ = points.shape
    # Get the x and y coordinates of the four nearest points for each input point
    x = points[:, :, 0] * (res - 1)
    y = points[:, :, 1] * (res - 1)

    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()

    # Compute the weights for each of the four points
    w1 = (x2 - x) * (y2 - y)
    w2 = (x - x1) * (y2 - y)
    w3 = (x2 - x) * (y - y1)
    w4 = (x - x1) * (y - y1)

    if grid_type == "NGLOD":
        # Interpolate the values for each point
        id1 = (x1 + y1 * res).long()
        id2 = (y1 * res + x2).long()
        id3 = (y2 * res + x1).long()
        id4 = (y2 * res + x2).long()

    elif grid_type == "HASH":
        npts = res**2
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1])) % grid_size
            id2 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1])) % grid_size
            id3 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1])) % grid_size
            id4 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1])) % grid_size
        else:
            id1 = (x1 + y1 * res).long()
            id2 = (y1 * res + x2).long()
            id3 = (y2 * res + x1).long()
            id4 = (y2 * res + x2).long()
    else:
        print("NOT IMPLEMENTED")

    values = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
    )
    return values[0]



