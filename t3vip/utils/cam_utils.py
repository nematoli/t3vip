import sys
import numpy as np
import torch
import torch.nn.functional as F


def xyz2uv(xyz, proj_mats):
    B, _, H, W = xyz.shape
    proj_pts = torch.matmul(proj_mats, xyz.view(B, 3, -1))
    pixels_mat = proj_pts.div(proj_pts[:, 2:3, :] + 1e-8)[:, 0:2, :]

    uv = pixels_mat.view(B, 2, H, W)

    return uv


def pxlcoords2flow(pxlcoords):
    B, _, H, W = pxlcoords.shape
    dtype = pxlcoords.dtype
    device = pxlcoords.device

    grid_uv = shape2pxlcoords(B=B, H=H, W=W, dtype=dtype, device=device)
    flow = pxlcoords - grid_uv

    return flow


def shape2pxlcoords(B, H, W, dtype, device):
    grid_v, grid_u = torch.meshgrid(
        [
            torch.arange(0.0, H, dtype=dtype, device=device),
            torch.arange(0.0, W, dtype=dtype, device=device),
        ]
    )

    grid_uv = torch.stack((grid_u, grid_v), dim=0)
    # 2xHxW

    if B != 0:
        grid_uv = grid_uv.unsqueeze(0).repeat(repeats=(B, 1, 1, 1))
        # Bx2xHxW

    return grid_uv


# project sceneflow into image space
def get2Dflow(fwdpts, proj_mats):
    pxlcoords_fwdwrpd = xyz2uv(fwdpts, proj_mats)
    oflow = pxlcoords2flow(pxlcoords_fwdwrpd)

    return oflow


def get_prj_mat(intrinsics):
    prj_mat = torch.zeros([3, 3], dtype=torch.float32)
    prj_mat[0, 0], prj_mat[0, 2], prj_mat[1, 1], prj_mat[1, 2], prj_mat[2, 2] = (
        intrinsics["fx"],
        intrinsics["cx"],
        intrinsics["fy"],
        intrinsics["cy"],
        1.0,
    )
    return prj_mat


def batch_seq_view(tensor, size=None):
    if size is None:
        B, S, C, H, W = tensor.size()
    else:
        B, S, C, H, W = size
    return tensor.contiguous().view(B * S, C, H, W)


def gradient_x(img):
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx


def gradient_y(img):
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy


def gradient_x_2nd(img):
    img_l = F.pad(img, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
    img_r = F.pad(img, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
    gx = img_l + img_r - 2 * img
    return gx


def gradient_y_2nd(img):
    img_t = F.pad(img, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]
    img_b = F.pad(img, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]
    gy = img_t + img_b - 2 * img
    return gy


# edge-aware smoothness
def motion_smoothness(flow, img, order=1, beta=1):
    # first order smoothness
    if order == 1:
        f_grad_x = gradient_x(flow)
        f_grad_y = gradient_y(flow)
    # second order smoothness
    elif order == 2:
        f_grad_x = gradient_x_2nd(flow)
        f_grad_y = gradient_y_2nd(flow)
    else:
        raise ValueError

    img_grad_x = gradient_x(img)
    img_grad_y = gradient_y(img)
    weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x), 1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y), 1, keepdim=True) * beta)

    smoothness_x = f_grad_x * weights_x
    smoothness_y = f_grad_y * weights_y

    return smoothness_x.abs() + smoothness_y.abs()


def make_color_wheel():

    #  color encoding scheme

    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])  # RGB

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col : YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col : YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col : GC + col, 1] = 255
    colorwheel[col : GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    # CB
    colorwheel[col : CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col : CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col : BM + col, 2] = 255
    colorwheel[col : BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    # MR
    colorwheel[col : MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col : MR + col, 0] = 255

    return colorwheel


def compute_color(u, v):

    colorwheel = make_color_wheel()
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)  # 1, 2, ..., ncols
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[:, :, i] = np.floor(255 * col).astype(np.uint8)  # RGB
    # img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

    return img.astype(np.uint8)


def flow_to_rgb(flow):

    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e10  # 1e9
    UNKNOWN_FLOW = 1e10

    flow = flow.squeeze()
    flow = flow.permute(1, 2, 0).cpu().numpy()
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    # fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    # less_u =  np.where(u < -UNKNOWN_FLOW_THRESH)
    # less_v =  np.where(v < -UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0

    # u[less_u] = 0
    # u[less_v] = 0
    # v[less_u] = 0
    # v[less_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([maxrad, np.amax(rad)])
    # print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    img = compute_color(u, v)

    img = img[np.newaxis, :]

    img = np.moveaxis(img, 3, 1)

    return img
