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
from model import Convnet_AAE, Classifier, Discriminator_local, Critic, Encoder
from utils import gradient_penality,MTL,set_parameters_grad

parser = argparse.ArgumentParser(description='DeepMax-Info-MTL')
parser.add_argument('--batch-size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Global DIM (default: 0.5)')
parser.add_argument('--beta', type=float, default=1.5,
                    help='Global DIM (default: 1.5)')
parser.add_argument('--gamma', type=float, default=0.01,
                    help='Global DIM (default: 0.01)')
args = parser.parse_args()

experiment_name = '1105MTL_DIM_fmsdetach_a{:}b{:}g{:}_l1_NOcritic'.format(args.alpha, args.beta, args.gamma)
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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
convnet = Convnet_AAE(conv_dim=dim)
classifier = Classifier(in_features=dim*16)
encoder = Encoder(conv_dim=dim)
discriminator_local = Discriminator_local(in_features=dim*2, out_features=dim*16) #  0-32 12-64 34-128 5-256layer
# discriminator_global = Discriminator_local(in_features=dim*16, out_features=dim*16) # 4-128 5-256layer
critic = Critic(in_features=dim*16)

convnet.cuda()
classifier.cuda()
encoder.cuda()
discriminator_local.cuda()
# discriminator_global.cuda()
critic.cuda()

def long_tail_noise_sample(shape, a=1, b=2):
    y = torch.rand(shape)
    x = -torch.log(y/a + 1e-8)/b
    return x

def long_tail_sample(alpha):
    y = Variable(torch.rand(alpha.shape)).cuda()
    x = -alpha * torch.log((y + 1e-8).clamp(0, 1))
    return x

def train(epoch, optimizer, data_loader):
    convnet.train()
    iteration = 0
    correct_lbl = 0
    for batch_idx, (imgs, labels) in enumerate(data_loader):
        iteration += 1
        imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
        fms, prev_fc, out_features = convnet(imgs)

        out = classifier(prev_fc)

        # Classification Loss
        one_hot_labels = torch.zeros_like(out).scatter_(1, labels.view(-1, 1), 1)
        loss_cls = F.cross_entropy(out, labels)

        # DIM GLOBAL

        # shuffle_idx = torch.randperm(b).cuda()
        # real_score_maps = discriminator_global(out_features.unsqueeze(2).unsqueeze(3), out_features)
        # fake_score_maps = discriminator_global(out_features[shuffle_idx].unsqueeze(2).unsqueeze(3), out_features)
        # loss_DIM_global = discriminator_global.loss_DIM(real_score_maps, fake_score_maps)

        # LOCAL
        z_feature = encoder(prev_fc)
        layer_idx = 1
        b, c, h, w = fms[layer_idx].shape

        shuffle_idx = torch.randperm(b).cuda()
        real_score_maps = discriminator_local(fms[layer_idx].detach(), z_feature)
        fake_score_maps = discriminator_local(fms[layer_idx][shuffle_idx].detach(), z_feature)
        loss_DIM_local = discriminator_local.loss_DIM(real_score_maps, fake_score_maps)

        # WGAN-GP
        for i in range(1):
            # fv_real = Variable(long_tail_noise_sample(out_features.shape)).cuda()
            fv_real = Variable(torch.randn(z_feature.shape)).cuda()
            fv_fake = z_feature

            critic_real = critic(fv_real.detach())
            critic_fake = critic(fv_fake.detach())

            critic.zero_grad()
            Wasserstein_critic = (critic_real - critic_fake).mean()

            # WGAN-GP loss
            loss_critic = args.gamma * ((critic_fake - critic_real).mean())
            # loss_critic = args.gamma * (F.binary_cross_entropy_with_logits(critic_fake, torch.zeros_like(critic_fake)) +
            #                             F.binary_cross_entropy_with_logits(critic_real, torch.ones_like(critic_fake)))
            loss_critic.backward(retain_graph=False)
            optimizer['critic'].step()

        # Update model_fe
        critic_fake = critic(fv_fake)
        loss_WGAN = -critic_fake.mean()
        # Update model

        model_name = [classifier, discriminator_local, encoder]
        for m in model_name:
            m.zero_grad()

        set_parameters_grad([convnet], isgrad=False)
        (loss_cls + args.beta*loss_DIM_local + args.gamma*loss_WGAN).backward(retain_graph=True)
        set_parameters_grad([convnet], isgrad=True)

        # prev_fc MTL
        grad_output_blob = []
        grad_output_blob.append(torch.autograd.grad(
            F.binary_cross_entropy_with_logits(out, one_hot_labels),
            prev_fc, retain_graph=True)[0].data)
        grad_output_blob.append(torch.autograd.grad(
            args.beta * discriminator_local.loss_DIM(real_score_maps, fake_score_maps),
            prev_fc, retain_graph=True)[0].data)
        # grad_output_blob.append(torch.autograd.grad(
        #     args.gamma * loss_WGAN,
        #     prev_fc, retain_graph=True)[0].data)

        grad_on_shared_params = MTL(grad_output_blob)
        prev_fc.backward(gradient=grad_on_shared_params)

        # # fms MTL
        # grad_output_blob = []
        # grad_output_blob.append(torch.autograd.grad(
        #     grad_on_shared_params,
        #     fms[layer_idx], retain_graph=True)[0].data)
        # grad_output_blob.append(torch.autograd.grad(
        #     grad_on_shared_params,
        #     fms[layer_idx], retain_graph=True)[0].data)
        #
        # grad_on_shared_params = MTL(grad_output_blob)
        # fms[layer_idx].backward(gradient=grad_on_shared_params)
        model_name = ['convnet', 'classifier', 'discriminator_local', 'encoder']
        for m in model_name:
            optimizer[m].step()

        # results
        pred_lbl = out.data.max(1, keepdim=True)[1]
        correct_lbl += pred_lbl.eq(labels.data.view_as(pred_lbl)).cpu().sum().item()
        if (batch_idx+1) % 20 == 0:
            acc = correct_lbl / (batch_size * iteration)
            logging.info(
                "Train Epoch:{}, Acc:{:.4f}%, loss_cls:{:.4f}, loss_DIM_local:{:.4f},, loss_critic:{:.4f} loss_fake:{:.4f}, Wasserstein_critic:{:.4f}, z_mu,z_sigma:{:.4f} {:.4f}".format(
                    epoch, acc * 100, loss_cls, loss_DIM_local, loss_critic, loss_WGAN, Wasserstein_critic.item(), out_features.mean().item(), out_features.std().item()))
def test(data_loader):
    convnet.eval()
    iteration = 0
    correct_lbl = 0
    for batch_idx, (imgs, labels) in enumerate(data_loader):
        iteration += 1
        imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
        fms, prev_fc, out_features = convnet(imgs)
        out = classifier(prev_fc.detach())

        # results
        pred_lbl = out.data.max(1, keepdim=True)[1]
        correct_lbl += pred_lbl.eq(labels.data.view_as(pred_lbl)).cpu().sum().item()

        if (batch_idx + 1) % 10 == 0:
            acc = correct_lbl / (batch_size * iteration)
            logging.info("Test Acc:{:.4f}%".format(acc * 100))

optimizer = {'convnet': optim.Adam(convnet.parameters(), lr=1e-4),
             'classifier': optim.Adam(classifier.parameters(), lr=1e-4),
             'discriminator_local': optim.Adam(discriminator_local.parameters(), lr=1e-4),
             'encoder': optim.Adam(encoder.parameters(), lr=1e-4),
             'critic': optim.Adam(critic.parameters(), lr=1e-4)}
batch_size = args.batch_size

for epoch in range(args.epochs+1):
    logging.info('Epoch:{:}'.format(epoch))
    train(epoch, optimizer, CIFAR10_training_loader)
    if epoch % 10 == 0:
        test(CIFAR10_test_loader)