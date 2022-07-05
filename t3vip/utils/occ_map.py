import torch


def project_points(tfm_ptc, intrinsics):
    B, C, H, W = tfm_ptc.size()

    occ_map = torch.zeros([B, 1, H, W], dtype=torch.float32, device=tfm_ptc.device).fill_(float("inf"))

    x = tfm_ptc[:, 0].unsqueeze(1)
    y = tfm_ptc[:, 1].unsqueeze(1)
    z = tfm_ptc[:, 2].unsqueeze(1)

    xpix = torch.round(((x / z) * intrinsics["fx"]) + intrinsics["cx"]).long()
    ypix = torch.round(((y / z) * intrinsics["fy"]) + intrinsics["cy"]).long()

    xpix = torch.where(xpix < 0, 0, xpix)
    xpix = torch.where(xpix >= W, W - 1, xpix)
    ypix = torch.where(ypix < 0, 0, ypix)
    ypix = torch.where(ypix >= H, H - 1, ypix)

    bs_coord = torch.arange(B).reshape(B, 1, 1, 1).expand(xpix.shape).reshape(-1)
    new_xpix = xpix.reshape(-1)
    new_ypix = ypix.reshape(-1)

    old_z = occ_map[bs_coord, 0, new_xpix, new_ypix]
    new_z = z.reshape(-1)
    occ_map[bs_coord, 0, new_ypix, new_xpix] = torch.where(new_z < old_z, new_z, old_z)

    occ_map = occ_map.ge(float("inf")).float()
    return occ_map
