import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.autograd import Variable, grad
import torch.nn.functional as F


def save_model(state, directory='./checkpoints', filename=None):
    if os.path.isdir(directory):
        pkl_filename = os.path.join(directory, filename)
        torch.save(state, pkl_filename)
        print('Save "{:}" in {:} successful'.format(pkl_filename, directory))
    else:
        print(' "{:}" directory is not exsits!'.format(directory))

def scale(x, dim=0):
    if isinstance(x, np.ndarray):
        x_max = x.max(axis=dim)
        x_min = x.min(axis=dim)
    else:
        x_max = x.max(dim=dim, keepdim=True)[0]
        x_min = x.min(dim=dim, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min + 1e-12)

def binary(img, thresh=0.2):
    _img = img
    _img[_img>thresh] = 1
    _img[_img<thresh] = 0
    return _img

def hook(module, grad_input, grad_output, grad_feature_maps_blob):
    b, c, h, w = grad_output[0].shape
    # grad_input[0].data = grad_input[0].data.clamp(max=0)#
    # grad_output[0].data = grad_output[0].data.clamp(max=0)
    grad_feature_maps_blob.append(grad_output[0])

def get_mask_fms(grad_fms_blob, fms_blob):
    for l, (grad_fms, fms) in enumerate(zip(grad_fms_blob, fms_blob[::-1])):
        b, c, h, w = grad_fms.shape
        grad_fm = grad_fms.data.norm(dim=1, p=2, keepdim=True)
        fm = scale((fms.norm(dim=1, p=2, keepdim=True)).view(b, 1, h * w), dim=2).view(b, 1, h, w)
        # grad_fm, fm = Variable(grad_fm, requires_grad=False), Variable(fm, requires_grad=False)
        if l == 0:
            mask_fms = grad_fm * fm
        else:
            if grad_fm.shape != mask_fms.shape:
                mask_fms = F.avg_pool2d(Variable(grad_fm * fm) + F.upsample(mask_fms, scale_factor=(2, 2),
                                                                  mode='bilinear'), kernel_size=3,
                                        stride=1, padding=1).data
            else:
                mask_fms = grad_fm * fm + mask_fms  # shape:[b, 1, h, w]
    mask_fms = scale(mask_fms.view(b, 1, h * w), dim=2).view(b, 1, h, w).cuda()
    return mask_fms

def onehot_mask(mask, num_cls=21):
    """
    :param mask: label of a image. tensor shape:[h, w]
    :param num_cls: number of class. int
    :return: onehot encoding mask. tensor shape:[num_cls, h, w]
    """
    b, h, w = mask.shape
    mask_o = torch.zeros([b, num_cls, h, w]).cuda()
    mask_o = mask_o.scatter_(1, mask.long().unsqueeze(1), 1)
    return mask_o

def mask_cluster(prob):
    """

    :param prob: Tensor[c, h, w]
    :return: Onehot encoding mask. tensor shape:[num_cls, h, w]
    """
    b, C_all, h, w = prob.shape
    cluster = onehot_mask(prob.min(1)[1], num_cls=C_all)
    return cluster

def gradient_penality(critic, feature_S, feature_T, cuda=True):
    # Gradient penality: caculate the norm of critic gradients, which are restrict to 1.
    alpha = torch.rand(feature_S.size(0), 1, 1, 1)
    alpha = alpha.cuda() if cuda else alpha
    interpolates = alpha * feature_S.data[0] + ((1 - alpha) * feature_T.data[0])
    if cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = critic(interpolates)
    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda(
                     ) if cuda else torch.ones(disc_interpolates.size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    LAMBDA = 10
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

u = {}
v = {}

def spectral_normed_init(Model, mode_num):
    u[mode_num] = {}
    v[mode_num] = {}
    i = 0
    for param in zip(Model.parameters()):
        if param[0].data.shape.__len__() >= 2:
            i = i + 1
            W = param[0].data
            u[mode_num][i] = torch.rand(1,W.size(0)).cuda()

def _l2normalize(v, eps=1e-8):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)

def spectral_normed_weight(Model, mode_num):
    i = 0
    for param in zip(Model.parameters()):
        if param[0].data.shape.__len__() >= 2:
            i = i + 1
            W = param[0].data
            W = W.view(W.size(0), -1)
            v[mode_num][i] = _l2normalize(torch.matmul(u[mode_num][i], W))
            u[mode_num][i] = _l2normalize(torch.matmul(v[mode_num][i], W.t()))
            sigma = torch.matmul(torch.matmul(v[mode_num][i], W.t()), u[mode_num][i].t())
            #if sigma.cpu().numpy()>1:
            param[0].data = param[0].data / sigma

def get_miou(mpred, mture):
    # mpred[b,C+1,h/4,w/4]
    # mture[b,h,w]


    b, C, _, _ = mpred.shape
    mpred = F.upsample(mpred, scale_factor=4, mode='bilinear').max(dim=1)[1].data.cpu()
    count = np.zeros((C))
    tureC = 0

    for j in range(C):
        x = np.where(mture == j)
        T_idx_j = set(zip(x[0].tolist(), x[1].tolist(), x[2].tolist()))

        if len(T_idx_j) != 0:
            tureC += 1
            x = np.where(mpred == j)
            p_idx_j = set(zip(x[0].tolist(), x[1].tolist(), x[2].tolist()))

            n_jj = set.intersection(p_idx_j, T_idx_j)
            u_jj = set.union(p_idx_j, T_idx_j)

            count[j] = float(len(n_jj)) / float(len(u_jj))

    mIOU = np.sum(count[:]) / float(tureC)

    return mIOU

def MTL(list_z):
    """
    :param list_z: shape[T, b, c]
    :return: grad: shape[b, c]
    """
    all_grad_z = torch.stack(list_z).permute(1, 0, 2)  # [b, T, c]
    b, T, c = all_grad_z.shape
    all_z = all_grad_z.mean(0)  # [T, c]
    device = all_z.device
    M = torch.matmul(all_z, all_z.transpose(0, 1))  # [T, T]

    alpha = torch.ones(T, 1).to(device) / T  # [T, 1]
    NUM_ITER_MAX = 20
    for i in range(NUM_ITER_MAX):
        _, t_hat = torch.max(torch.matmul(M, alpha), 0)  # max([T, 1])

        e = torch.zeros([T, 1]).to(device).scatter_(0, t_hat.long().unsqueeze(1), 1)

        t_hat_idx = (t_hat * T + t_hat)
        Mtt = torch.index_select(M.view(-1), 0, t_hat_idx)
        am = torch.matmul(alpha.transpose(0, 1), M)  # [1, T]
        ama = torch.matmul(am, alpha)  # [1]
        ame = torch.matmul(am, e)  # [1]

        gamma_a = Mtt + ama - 2 * ame
        gamma_b = 2 * ame - 2 * ama
        gamma_hat = (-gamma_b / (2 * gamma_a + 1e-8)).view(1, 1)

        alpha = (torch.Tensor([1]).to(device) - gamma_hat) * alpha + gamma_hat * e  # [T, 1]

        if gamma_hat.mean().abs() < 0.001 or i == (NUM_ITER_MAX - 1):
            break
    alpha_expand = alpha.unsqueeze(0).expand(b, T, 1)
    grad = (alpha_expand * all_grad_z).sum(1)  # b, c
    return grad

def set_parameters_grad(networks, isgrad=True):
    for n in networks:
        for p in n.parameters():
            if isgrad:
                p.requires_grad=True
            else:
                p.requires_grad=False