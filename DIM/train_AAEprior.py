import os
import argparse
import logging
from pathlib import Path

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import visdom
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
from model import Convnet, Classifier, Discriminator_local, Critic
from utils import gradient_penality

parser = argparse.ArgumentParser(description='DeepMax-Info')
parser.add_argument('--batch-size', type=int, default=1024,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Global DIM (default: 0.5)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='Global DIM (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='Global DIM (default: 0.1)')
args = parser.parse_args()

experiment_name = 'test5_DIM_KL_a{:}b{:}g{:}'.format(args.alpha, args.beta, args.gamma)
path_log = Path('./log/' + experiment_name + '.log')
try:
    if path_log.exists():
        raise FileExistsError
except FileExistsError:
    print("Already exist log file: {}".format(path_log))
    raise
else:
    logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                                datefmt='%a, %d %b %Y %H:%M:%S',
                                filename=path_log.__str__(),
                                filemode='w'
                                )
    print('Create log file: {}'.format(path_log))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                           transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                                           transforms.ToTensor()])

CIFAR10_training_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True, transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

CIFAR10_test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, download=True, transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

dim = 32
convnet = Convnet(conv_dim=dim)
classifier = Classifier(in_features=dim*8)
discriminator_local = Discriminator_local(in_features=dim*4, out_features=dim*8) # 4-128 5-256layer
discriminator_global = Discriminator_local(in_features=dim*8, out_features=dim*8) # 4-128 5-256layer

convnet.cuda()
classifier.cuda()
discriminator_local.cuda()
discriminator_global.cuda()

def sampling(z_mu, z_log_var):
    epsilon = torch.randn(z_mu.shape).cuda()
    return z_mu + epsilon * torch.exp(z_log_var / 2)

def train(epoch, optimizer, data_loader):
    convnet.train()
    iteration = 0
    correct_lbl = 0
    for batch_idx, (imgs, labels) in enumerate(data_loader):
        iteration += 1
        imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())

        # fms, out_features = convnet(imgs)
        fms, z_mu, z_log_var = convnet(imgs)
        loss_prior_kl = - 0.5 * torch.mean(1 + z_log_var - z_mu**2 - torch.exp(z_log_var))
        out_features = sampling(z_mu, z_log_var)

        out = classifier(out_features.detach())

        # Classification Loss
        one_hot_labels = torch.zeros_like(out).scatter_(1, labels.view(-1, 1), 1)
        loss_cls = F.binary_cross_entropy_with_logits(out, one_hot_labels)

        # DIM global
        layer_idx = 4
        b, c, h, w = fms[layer_idx].shape
        shuffle_idx = torch.randperm(b).cuda()
        real_score_maps = discriminator_global(out_features.unsqueeze(2).unsqueeze(3), out_features)
        fake_score_maps = discriminator_global(out_features[shuffle_idx].unsqueeze(2).unsqueeze(3), out_features)
        loss_DIM_global = discriminator_global.loss_DIM(real_score_maps, fake_score_maps)

        # local
        shuffle_idx = torch.randperm(b).cuda()
        real_score_maps = discriminator_local(fms[layer_idx], out_features)
        fake_score_maps = discriminator_local(fms[layer_idx][shuffle_idx], out_features)
        loss_DIM_local = discriminator_local.loss_DIM(real_score_maps, fake_score_maps)

        # Update model
        model_name = [convnet, classifier, discriminator_local, discriminator_global]
        for m in model_name:
            m.zero_grad()
        (loss_cls + args.alpha*loss_DIM_global + args.beta*loss_DIM_local + args.gamma*loss_prior_kl).backward()
        # (loss_cls + args.alpha*loss_DIM_global + args.beta*loss_DIM_local).backward()
        model_name = ['convnet', 'classifier', 'discriminator_local', 'discriminator_global']
        for m in model_name:
            optimizer[m].step()

        # results
        pred_lbl = out.data.max(1, keepdim=True)[1]
        correct_lbl += pred_lbl.eq(labels.data.view_as(pred_lbl)).cpu().sum().item()
        acc = correct_lbl/(batch_size*iteration)
        if (batch_idx+1) % 20 == 0:
            logging.info(
                "Train Epoch:{}, Acc:{:.4f}%, loss_cls:{:.4f}, loss_DIM_global:{:.4f}, loss_DIM_local:{:.4f}, loss_prior_kl:{:.4f}, z_mu,z_sigma:{:.4f} {:.4f}".format(
                    epoch, acc * 100, loss_cls, loss_DIM_global, loss_DIM_local, loss_prior_kl, out_features.mean().data[0], out_features.std().data[0]))
        # if (batch_idx+1) % 20 == 0:
        #     logging.info(
        #         "Train Epoch:{}, Acc:{:.4f}%, loss_cls:{:.4f}, loss_DIM_global:{:.4f}, loss_DIM_local:{:.4f}".format(
        #             epoch, acc * 100, loss_cls, loss_DIM_global, loss_DIM_local))
def test(data_loader):
    convnet.eval()
    iteration = 0
    correct_lbl = 0
    for batch_idx, (imgs, labels) in enumerate(data_loader):
        iteration += 1
        imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())

        fms, z_mu, z_log_var = convnet(imgs)
        out = classifier(z_mu.detach())

        # fms, out_features = convnet(imgs)
        # out = classifier(out_features.detach())

        # results
        pred_lbl = out.data.max(1, keepdim=True)[1]
        correct_lbl += pred_lbl.eq(labels.data.view_as(pred_lbl)).cpu().sum().item()
        acc = correct_lbl/(batch_size*iteration)

        if (batch_idx + 1) % 20 == 0:
            logging.info("Test Acc:{:.4f}%".format(acc * 100))

optimizer = {'convnet': optim.Adam(convnet.parameters(), lr=0.001),
             'classifier': optim.Adam(classifier.parameters(), lr=0.001),
             'discriminator_local': optim.Adam(discriminator_local.parameters(), lr=0.001),
             'discriminator_global': optim.Adam(discriminator_global.parameters(), lr=0.001)}
batch_size = args.batch_size

for epoch in range(args.epochs+1):
    logging.info('Epoch:{:}'.format(epoch))
    train(epoch, optimizer, CIFAR10_training_loader)
    if epoch % 10 == 0:
        test(CIFAR10_training_loader)
