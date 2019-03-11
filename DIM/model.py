import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from gn import GroupNorm2d
from sn import SpectralNorm

class Convnet(nn.Module):
    def __init__(self, in_features=3, conv_dim=32, num_groups=16, leaky=0.2):
        super(Convnet, self).__init__()
        self.conv1 = nn.Conv2d(in_features, conv_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1)

        self.fc_mu = nn.Linear(conv_dim*8, conv_dim*8)
        self.fc_log_var = nn.Linear(conv_dim*8, conv_dim*8)
        self.fc = nn.Linear(conv_dim*8, conv_dim*8)


        # self.gn1 = GroupNorm2d(conv_dim, num_groups=num_groups, affine=True)
        # self.gn2 = GroupNorm2d(conv_dim*2, num_groups=num_groups, affine=True)
        # self.gn3 = GroupNorm2d(conv_dim*2, num_groups=num_groups, affine=True)
        # self.gn4 = GroupNorm2d(conv_dim*4, num_groups=num_groups, affine=True)
        # self.gn5 = GroupNorm2d(conv_dim*4, num_groups=num_groups, affine=True)
        # self.gn6 = GroupNorm2d(conv_dim*8, num_groups=num_groups, affine=True)
        # self.gn7 = GroupNorm2d(conv_dim*8, num_groups=num_groups, affine=True)

        self.gn1 = nn.BatchNorm2d(conv_dim, affine=True)
        self.gn2 = nn.BatchNorm2d(conv_dim*2, affine=True)
        self.gn3 = nn.BatchNorm2d(conv_dim*2, affine=True)
        self.gn4 = nn.BatchNorm2d(conv_dim*4, affine=True)
        self.gn5 = nn.BatchNorm2d(conv_dim*4, affine=True)
        self.gn6 = nn.BatchNorm2d(conv_dim*8, affine=True)
        self.gn7 = nn.BatchNorm2d(conv_dim*8, affine=True)

        self.relu = nn.LeakyReLU(leaky, inplace=True)
    def forward(self, x):
        l1 = self.relu(self.gn1(self.conv1(x)))
        l2 = self.relu(self.gn2(self.conv2(l1)))
        l3 = self.relu(self.gn3(self.conv3(l2)))
        l4 = self.relu(self.gn4(self.conv4(l3)))
        l5 = self.relu(self.gn5(self.conv5(l4)))
        l6 = self.relu(self.gn6(self.conv6(l5)))
        l7 = self.relu(self.gn7(self.conv7(l6)))

        # out_features = self.fc(nn.AvgPool2d(l7.shape[2:])(l7).view(l7.shape[0], -1))

        z_global_avg = nn.AvgPool2d(l7.shape[2:])(l7).view(l7.shape[0], -1)
        z_mu = self.fc_mu(z_global_avg)
        z_log_var = self.fc_log_var(z_global_avg)

        fms = [l1, l2, l3, l4, l5, l6, l7]
        # return fms, out_features
        return fms, z_mu, z_log_var

# class Convnet(nn.Module):
#     def __init__(self, in_features=3, conv_dim=32, leaky=0.1):
#         super(Convnet, self).__init__()
#         self.conv1 = nn.Conv2d(in_features, conv_dim*2, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1)
#         self.conv5 = nn.Conv2d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1)
#         self.conv6 = nn.Conv2d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1)
#
#         self.fc = nn.Linear(conv_dim*16, conv_dim*16)
#
#         self.bn1 = nn.BatchNorm2d(conv_dim*2, affine=True)
#         self.bn2 = nn.BatchNorm2d(conv_dim*2, affine=True)
#         self.bn3 = nn.BatchNorm2d(conv_dim*2, affine=True)
#         self.bn4 = nn.BatchNorm2d(conv_dim*4, affine=True)
#         self.bn5 = nn.BatchNorm2d(conv_dim*4, affine=True)
#         self.bn6 = nn.BatchNorm2d(conv_dim*4, affine=True)
#
#         self.relu = nn.LeakyReLU(leaky, inplace=True)
#
#     def forward(self, x):
#         fms_blob = []
#         l1 = self.relu(self.bn1(self.conv1(x)))
#         l2 = self.relu(self.bn2(self.conv2(l1)))
#         l3 = self.relu(self.bn3(self.conv3(l2)))
#         l4 = self.relu(self.bn4(self.conv4(l3)))
#         l5 = self.relu(self.bn5(self.conv5(l4)))
#         l6 = self.relu(self.bn6(self.conv6(l5)))
#         fms_blob.append([l1, l2, l3, l4, l5, l6])
#
#         prev_fc = nn.AvgPool2d(l6.shape[2:])(l6).view(l6.shape[0], -1)
#         out_features = self.fc(prev_fc)
#
#         return fms_blob, prev_fc, out_features

class Convnet_AAE(nn.Module):
    def __init__(self, in_features=3, conv_dim=32, num_groups=16, leaky=0.2):
        super(Convnet_AAE, self).__init__()
        self.conv1 = nn.Conv2d(in_features, conv_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(conv_dim*8, conv_dim*16, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(conv_dim*16, conv_dim*16)

        self.gn1 = nn.BatchNorm2d(conv_dim, affine=True)
        self.gn2 = nn.BatchNorm2d(conv_dim*2, affine=True)
        self.gn3 = nn.BatchNorm2d(conv_dim*2, affine=True)
        self.gn4 = nn.BatchNorm2d(conv_dim*4, affine=True)
        self.gn5 = nn.BatchNorm2d(conv_dim*4, affine=True)
        self.gn6 = nn.BatchNorm2d(conv_dim*8, affine=True)
        self.gn7 = nn.BatchNorm2d(conv_dim*8, affine=True)

        self.relu = nn.LeakyReLU(leaky, inplace=True)
    def forward(self, x):
        l1 = self.relu(self.gn1(self.conv1(x)))
        l2 = self.relu(self.gn2(self.conv2(l1)))
        l3 = self.relu(self.gn3(self.conv3(l2)))
        l4 = self.relu(self.gn4(self.conv4(l3)))
        l5 = self.relu(self.gn5(self.conv5(l4)))
        l6 = self.relu(self.gn6(self.conv6(l5)))
        l7 = self.relu(self.gn7(self.conv7(l6)))
        l8 = self.conv8(l7)

        prev_fc = nn.AvgPool2d(l8.shape[2:])(l8).view(l8.shape[0], -1)
        out_features = self.fc(prev_fc)

        fms = [l1, l2, l3, l4, l5, l6, l7, l8]
        return fms, prev_fc, out_features

class Discriminator_local(nn.Module):
    def __init__(self, in_features=256, out_features=256, leaky=0):
        super(Discriminator_local, self).__init__()
        self.out_features = out_features
        dim = in_features + out_features
        self.conv1 = nn.Conv2d(dim, int(dim/2), kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(int(dim/2), int(dim/4), kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(int(dim/4), 1, kernel_size=1, stride=1, padding=0)

        self.leaky_relu = nn.LeakyReLU(leaky, inplace=True)

    def forward(self, fms, fv):
        '''
        :param fms: [b, 512, h, w]
        :param fv: [b, 2048]
        :return: score maps
        '''
        b, c, h, w = fms.shape

        # fv_: shape [b, 2048, h, w]
        fv_ = fv.unsqueeze(2).unsqueeze(3).expand(b, self.out_features, h, w)

        # fms_fvs: shape [b, 512+2048, h, w]
        fms_fvs = torch.cat([fms, fv_], dim=1)

        # score_maps: shape [b, 1, h, w]
        # score_maps = self.conv2(self.leaky_relu(self.conv1(fms_fvs)))
        score_maps = self.conv3(self.leaky_relu(self.conv2(self.leaky_relu(self.conv1(fms_fvs)))))

        return score_maps

    def loss_DIM(self, real, fake):
        b, c, h, w = real.shape

        # logits_real = -F.softplus(-real)
        # logits_fake = F.softplus(fake)

        logits_real = F.binary_cross_entropy_with_logits(real, torch.ones_like(real).cuda())
        logits_fake = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake).cuda())
        # logits_real = F.sigmoid(real)
        # logits_fake = 1 - F.sigmoid(fake)
        # logits_real = torch.log(logits_real + 1e-8)
        # logits_fake = torch.log(logits_fake + 1e-8)
        # shape: [b, 1, h, w]
        # loss = -(logits_real + logits_fake).view(b, -1).mean()
        loss = (logits_real + logits_fake)

        return loss

class Critic(nn.Module):
    def __init__(self, in_features=256, leaky=0.1):
        """
        Critic is the Component of Wasserstein-GAN.
        :param in_features: The number of features of feature vector
        :param leaky: Use Leaky relu since we want the gradients can easily pass all the network.
        """
        super(Critic, self).__init__()
        self.leaky = leaky

        self.fc1 = SpectralNorm(nn.Linear(in_features, int(in_features/2)))
        self.fc2 = SpectralNorm(nn.Linear(int(in_features/2), int(in_features/4)))
        self.fc3 = SpectralNorm(nn.Linear(int(in_features/4), 1))
        # self.fc1 = nn.Linear(in_features, int(in_features / 2))
        # self.fc2 = nn.Linear(int(in_features / 2), int(in_features / 4))
        # self.fc3 = nn.Linear(int(in_features / 4), 1)

        self.leaky_relu = nn.LeakyReLU(leaky, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Classifier(nn.Module):
    def __init__(self, in_features=256, num_class=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, int(in_features/2))
        self.fc2 = nn.Linear(int(in_features/2), num_class)

    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))

        return x

class Encoder(nn.Module):
    """
    Caculate the representation z.
    """
    def __init__(self, conv_dim=32):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(conv_dim*16, conv_dim*16)

    def forward(self, x):
        z_feature = self.fc(x)
        return z_feature
