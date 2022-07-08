import torch
import torch.nn.functional as F
from t3vip.helpers.KNN import KNN
from t3vip.utils.cam_utils import batch_seq_view, motion_smoothness

knn = KNN(search=15)


def rec_loss(alpha, pred, real, L=2):
    if L == 2:
        recloss = alpha * F.mse_loss(pred, real.detach())
    else:
        recloss = alpha * F.l1_loss(pred, real.detach())

    return recloss


def ptc_knn_loss(alpha_knn, ptc_2, tfmptc_1):
    dist1 = knn(tfmptc_1, ptc_2.detach())
    dist2 = knn(ptc_2.detach(), tfmptc_1)
    dist_mean = torch.mean(dist1) + torch.mean(dist2)

    knnloss = alpha_knn * dist_mean

    return knnloss


def smooth_loss(alpha_fs, rgb_1, flow):
    fs_loss = alpha_fs * motion_smoothness(flow, rgb_1, order=2).mean()

    return fs_loss


def calc_3d_loss(alpha_rcd, alpha_knn, alpha_sfs, alpha_l, dpt_2, ptc_2, rgb_1, nxtdpts, tfmptc_1, sflow):
    rcd_loss, knn_loss, sfs_loss = (
        torch.tensor(0.0).to(rgb_1.device),
        torch.tensor(0.0).to(rgb_1.device),
        torch.tensor(0.0).to(rgb_1.device),
    )

    size = ptc_2.size()
    rgb_1, ptc_2 = batch_seq_view(rgb_1, size), batch_seq_view(ptc_2, size)
    tfmptc_1, sflow = batch_seq_view(tfmptc_1, size), batch_seq_view(sflow, size)
    dpt_2, nxtdpts = batch_seq_view(dpt_2), batch_seq_view(nxtdpts)

    if alpha_rcd != 0:
        rcd_loss = rec_loss(alpha_rcd, nxtdpts, dpt_2, alpha_l)

    if alpha_knn != 0:
        knn_loss = ptc_knn_loss(alpha_knn, ptc_2, tfmptc_1)

    if alpha_sfs != 0:
        sfs_loss = smooth_loss(alpha_sfs, rgb_1, sflow)

    return rcd_loss, knn_loss, sfs_loss


def calc_2d_loss(alpha_rcr, alpha_ofs, alpha_l, rgb_1, rgb_2, nxtrgbs, oflow):
    rcr_loss, ofs_loss = (
        torch.tensor(0.0).to(rgb_1.device),
        torch.tensor(0.0).to(rgb_1.device),
    )

    size = rgb_1.size()
    rgb_1, rgb_2, nxtrgbs = batch_seq_view(rgb_1, size), batch_seq_view(rgb_2, size), batch_seq_view(nxtrgbs, size)

    if alpha_rcr != 0:
        rcr_loss = rec_loss(alpha_rcr, nxtrgbs, rgb_2, alpha_l)
    if alpha_ofs != 0:
        oflow = batch_seq_view(oflow)
        ofs_loss = smooth_loss(alpha_ofs, rgb_1, oflow)

    return rcr_loss, ofs_loss
