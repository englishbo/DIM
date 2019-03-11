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
from model import Convnet_AAE, Classifier, Discriminator_local, Critic
from utils import gradient_penality, save_model

parser = argparse.ArgumentParser(description='DeepMax-Info')
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

experiment_name = 'test_DIM_fmsdetach_a{:}b{:}g{:}_l0'.format(args.alpha, args.beta, args.gamma)
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
convnet = Convnet_AAE(conv_dim=dim)
classifier = Classifier(in_features=dim*16)
discriminator_local = Discriminator_local(in_features=dim, out_features=dim*16) #  0-32 12-64 34-128 5-256layer
discriminator_global = Discriminator_local(in_features=dim*16, out_features=dim*16) # 4-128 5-256layer
critic = Critic(in_features=dim*16)
torch.load
convnet.cuda()
classifier.cuda()
discriminator_local.cuda()
discriminator_global.cuda()
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

        # out_features = long_tail_sample(F.relu(out_features))
        out = classifier(prev_fc.detach())

        # Classification Loss
        one_hot_labels = torch.zeros_like(out).scatter_(1, labels.view(-1, 1), 1)
        loss_cls = F.binary_cross_entropy_with_logits(out, one_hot_labels)

        # DIM GLOBAL
        layer_idx = 0
        b, c, h, w = fms[layer_idx].shape
        shuffle_idx = torch.randperm(b).cuda()
        real_score_maps = discriminator_global(out_features.unsqueeze(2).unsqueeze(3), out_features)
        fake_score_maps = discriminator_global(out_features[shuffle_idx].unsqueeze(2).unsqueeze(3), out_features)
        loss_DIM_global = discriminator_global.loss_DIM(real_score_maps, fake_score_maps)

        # LOCAL
        shuffle_idx = torch.randperm(b).cuda()
        real_score_maps = discriminator_local(fms[layer_idx].detach(), out_features)
        fake_score_maps = discriminator_local(fms[layer_idx][shuffle_idx].detach(), out_features)
        loss_DIM_local = discriminator_local.loss_DIM(real_score_maps, fake_score_maps)

        # WGAN-GP
        for i in range(1):
            # fv_real = Variable(long_tail_noise_sample(out_features.shape)).cuda()
            fv_real = Variable(torch.randn(out_features.shape)).cuda()
            fv_fake = out_features


            critic_real = critic(fv_real.detach())
            critic_fake = critic(fv_fake.detach())

            critic.zero_grad()
            # gp = gradient_penality(critic, fv_real.detach(), fv_fake.detach())
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
        model_name = [convnet, classifier, discriminator_local, discriminator_global]
        for m in model_name:
            m.zero_grad()
        (loss_cls + args.alpha*loss_DIM_global + args.beta*loss_DIM_local + args.gamma*loss_WGAN).backward()
        model_name = ['convnet', 'classifier', 'discriminator_local', 'discriminator_global']
        for m in model_name:
            optimizer[m].step()

        # results
        pred_lbl = out.data.max(1, keepdim=True)[1]
        correct_lbl += pred_lbl.eq(labels.data.view_as(pred_lbl)).cpu().sum().item()
        if (batch_idx+1) % 20 == 0:
            acc = correct_lbl / (batch_size * iteration)
            logging.info(
                "Train Epoch:{}, Acc:{:.4f}%, loss_cls:{:.4f}, loss_DIM_global:{:.4f}, loss_DIM_local:{:.4f},, loss_critic:{:.4f} loss_fake:{:.4f}, Wasserstein_critic:{:.4f}, z_mu,z_sigma:{:.4f} {:.4f}".format(
                    epoch, acc * 100, loss_cls, loss_DIM_global, loss_DIM_local, loss_critic, loss_WGAN, Wasserstein_critic.item(), out_features.mean().item(), out_features.std().item()))
def test(data_loader):
    convnet.eval()
    iteration = 0
    correct_lbl = 0
    best_acc = 0
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

    if acc > best_acc:
        logging.info('==>Save model, best_acc:{:.2f}%'.format(acc*100))
        best_acc = acc
        state = {'epoch': epoch,
                 'best_acc': best_acc,
                 'convnet': convnet.state_dict(),
                 'classifier': classifier.state_dict(),
                 'discriminator_local': discriminator_local.state_dict(),
                 'discriminator_global': discriminator_global.state_dict(),
                 'critic': critic.state_dict(),
                 'optimizer': optimizer,
                 }
        save_model(state, directory='./checkpoints', filename=experiment_name+'.pkl')

optimizer = {'convnet': optim.Adam(convnet.parameters(), lr=1e-4),
             'classifier': optim.Adam(classifier.parameters(), lr=1e-4),
             'discriminator_local': optim.Adam(discriminator_local.parameters(), lr=1e-4),
            'discriminator_global': optim.Adam(discriminator_global.parameters(), lr=1e-4),
             'critic': optim.Adam(critic.parameters(), lr=1e-4)}
batch_size = args.batch_size

for epoch in range(args.epochs+1):
    logging.info('Epoch:{:}'.format(epoch))
    train(epoch, optimizer, CIFAR10_training_loader)
    if epoch % 10 == 0:
        test(CIFAR10_test_loader)
