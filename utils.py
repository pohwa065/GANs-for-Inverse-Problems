"""
The test code for:
"Adversarial Training for Solving Inverse Problems"
using Tensorflow.

With this project, you can train a model to solve the following
inverse problems:
- on MNIST and CIFAR-10 datasets for separating superimposed images.
- image denoising on MNIST
- remove speckle and streak noise in CAPTCHAs
All the above tasks are trained w/ or w/o the help of pair-wise supervision.

"""

import numpy as np
import os
import tensorflow as tf
import cv2
import initializer as init
import string
characters = string.digits + string.ascii_uppercase
import random
from captcha.image import ImageCaptcha
Image = ImageCaptcha()

# All parameters used in this file
Params = init.TrainingParamInitialization()


def get_batch(DATA, batch_size, mode):

    if mode is 'X_domain':
        n, h, w, c = DATA['Source1'].shape
        idx = np.random.choice(range(n), batch_size, replace=False)
        batch = DATA['Source1'][idx, :, :, :]

        return batch

    if mode is 'Y_domain':

        if Params.task_name in ['unmixing_mnist_cifar', 'unmixing_mnist_mnist']:

            n, h, w, c = DATA['Source1'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source1'][idx, :, :, :]

            # # image mixture
            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            z = DATA['Source2'][idx, :, :, :]
            batch = batch + z

            return batch

        if Params.task_name is 'denoising':

            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source2'][idx, :, :, :]

            # # add noise
            z = 1.0 * np.random.randn(batch_size, h, w, c)
            batch = batch + z

            return batch

        if Params.task_name is 'captcha':

            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source2'][idx, :, :, :]

            return batch



    if mode is 'XY_pair':

        if Params.task_name in ['unmixing_mnist_cifar', 'unmixing_mnist_mnist']:
            n, h, w, c = DATA['Source1'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source1'][idx, :, :, :]

            # # image mixture
            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            z = DATA['Source2'][idx, :, :, :]

            return batch, batch + z

        if Params.task_name is 'denoising':
            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch = DATA['Source2'][idx, :, :, :]

            # # add noise
            z = 1.0 * np.random.randn(batch_size, h, w, c)

            return batch, batch + z

        if Params.task_name is 'captcha':
            n, h, w, c = DATA['Source2'].shape
            idx = np.random.choice(range(n), batch_size, replace=False)
            batch_x = DATA['Source1'][idx, :, :, :]
            batch_y = DATA['Source2'][idx, :, :, :]

            return batch_x, batch_y





def plot2x2(samples):

    IMG_SIZE = samples.shape[1]

    if Params.task_name in ['unmixing_mnist_mnist', 'denoising']:
        n_channels = 1
    else:
        n_channels = 3

    img_grid = np.zeros((2 * IMG_SIZE, 2 * IMG_SIZE, n_channels))

    for i in range(4):
        py, px = IMG_SIZE * int(i / 2), IMG_SIZE * (i % 2)
        this_img = samples[i, :, :, :]
        img_grid[py:py + IMG_SIZE, px:px + IMG_SIZE, :] = this_img

    if n_channels == 1:
        img_grid = img_grid[:,:,0]

    if Params.task_name is 'captcha':
        img_grid = 1 - cv2.resize(img_grid, (320, 120))

    return img_grid




def load_historical_model(sess, checkpoint_dir='checkpoints'):

    # check and create model dir
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)

    if 'checkpoint' in os.listdir(checkpoint_dir):
        # training from the last checkpoint
        print('loading model from the last checkpoint ...')
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, latest_checkpoint)
        print(latest_checkpoint)
        print('loading finished!')
    else:
        print('no historical model found, start training from scratch!')




def load_and_resize_mnist_data(is_training):

    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.mnist.load_data()
    if is_training is True:
        data = x_train/255.
    else:
        data = x_test/255.

    m = data.shape[0]
    data = np.reshape(data, [m, 28, 28])
    if Params.task_name in ['unmixing_mnist_mnist', 'denoising']:
        n_channels = 1
    else:
        n_channels = 3
    data_rs = np.zeros((m, Params.IMG_SIZE, Params.IMG_SIZE, n_channels))

    for i in range(m):
        img = data[i, :, :]
        img_rs = cv2.resize(img, dsize=(Params.IMG_SIZE, Params.IMG_SIZE))
        if Params.task_name in ['unmixing_mnist_mnist', 'denoising']:
            img_rs = np.expand_dims(img_rs, axis=-1)
        else:
            img_rs = np.stack([img_rs, img_rs, img_rs], axis=-1)
        data_rs[i, :, :, :] = img_rs

    return data_rs




def load_and_resize_cifar10_data(is_training):

    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.cifar10.load_data()

    if is_training is True:
        data = x_train/255.
    else:
        data = x_test/255.

    m = data.shape[0]
    n_channels = 3
    data_rs = np.zeros((m, Params.IMG_SIZE, Params.IMG_SIZE, n_channels))

    for i in range(m):
        img = data[i, :, :, :]
        img_rs = cv2.resize(img, dsize=(Params.IMG_SIZE, Params.IMG_SIZE),
                            interpolation=cv2.INTER_CUBIC)
        data_rs[i, :, :, :] = img_rs

    return data_rs




def load_captcha(m_samples=10000, color=False):

    if color is True:
        data_x = np.zeros(
            (m_samples, Params.IMG_SIZE, Params.IMG_SIZE, 3))
        data_y = np.zeros(
            (m_samples, Params.IMG_SIZE, Params.IMG_SIZE, 3))
    else:
        data_x = np.zeros(
            (m_samples, Params.IMG_SIZE, Params.IMG_SIZE, 1))
        data_y = np.zeros(
            (m_samples, Params.IMG_SIZE, Params.IMG_SIZE, 1))

    labels = []

    for i in range(m_samples):

        random_str = ''.join([random.choice(characters) for j in range(4)])
        band_id = np.random.randint(3)
        img_clean, img_ns = Image.generate_image_pair(random_str)

        img_clean = np.array(img_clean, dtype=np.uint8)
        img_clean = 1 - img_clean / 255.
        img_clean = cv2.resize(
            img_clean, (Params.IMG_SIZE, Params.IMG_SIZE))
        if color is not True:
            img_clean = np.expand_dims(img_clean[:,:,band_id], axis=-1)
        data_x[i, :, :, :] = img_clean

        img_ns = np.array(img_ns, dtype=np.uint8)
        img_ns = 1 - img_ns / 255.
        img_ns = cv2.resize(
            img_ns, (Params.IMG_SIZE, Params.IMG_SIZE))
        if color is not True:
            img_ns = np.expand_dims(img_ns[:,:,band_id], axis=-1)
        data_y[i, :, :, :] = img_ns

        labels.append(random_str)

        if np.mod(i, 200) == 0:
            print('generating captchas: ' + str(i) + ' / ' + str(m_samples))

    return data_x, data_y, labels



def load_data(is_training):

    DATA = {'Source1': 0, 'Source2': 0}

    if Params.task_name is 'unmixing_mnist_cifar':
        print('loading cifar10 data...')
        data_cifar10 = load_and_resize_cifar10_data(is_training=is_training)
        print('loading mnist data...')
        data_mnist = load_and_resize_mnist_data(is_training=is_training)
        DATA['Source1'] = data_cifar10
        DATA['Source2'] = data_mnist

    if Params.task_name is 'unmixing_mnist_mnist':
        print('loading mnist data...')
        data_mnist = load_and_resize_mnist_data(is_training=is_training)
        DATA['Source1'] = data_mnist
        DATA['Source2'] = data_mnist

    if Params.task_name is 'denoising':
        print('loading mnist data...')
        data_mnist = load_and_resize_mnist_data(is_training=is_training)
        DATA['Source1'] = data_mnist
        DATA['Source2'] = np.copy(data_mnist)

    if Params.task_name is 'captcha':
        print('generating captchas...')
        data_captcha_clean, data_captcha_ns, _ = load_captcha(m_samples=10000)
        DATA['Source1'] = data_captcha_clean
        DATA['Source2'] = data_captcha_ns

    print('loading data finished')

    return DATA
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################
# 1. Fixed Handcrafted Matching Filter
###############################
class FixedFilter(nn.Module):
    """
    A fixed (non-trainable) matching filter applied depthwise.
    For example, a 3x3 Laplacian kernel or a configurable averaging filter.
    """
    def __init__(self, in_channels, kernel_size=3, filter_type='laplacian'):
        super(FixedFilter, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Depthwise convolution: each channel is filtered independently.
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding,
                              groups=in_channels, bias=False)

        # Define the kernel.
        if kernel_size == 3 and filter_type == 'laplacian':
            # A common Laplacian kernel: emphasizes edges.
            kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32)
        else:
            # Default to an averaging filter.
            kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
            kernel /= kernel.sum()

        # Expand kernel dimensions and copy to convolution weights.
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(in_channels, 1, 1, 1)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False  # Freeze weights.

    def forward(self, x):
        return self.conv(x)

###############################
# 2. Squeeze-and-Excitation (SE) Block for Channel Attention
###############################
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

###############################
# 3. Weighted Addition for Feature Fusion
###############################
class WeightedAdd(nn.Module):
    """
    Dynamically fuses a list of feature maps with learnable nonnegative weights.
    """
    def __init__(self, num_inputs):
        super(WeightedAdd, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, inputs):
        w = F.relu(self.weights)
        w = w / (torch.sum(w) + 1e-4)
        out = sum(w[i] * inputs[i] for i in range(len(inputs)))
        return out

###############################
# 4. BiFPN Layer with Bidirectional Fusion and Attention
###############################
class BiFPNLayer(nn.Module):
    def __init__(self, channels, num_levels=3):
        super(BiFPNLayer, self).__init__()
        self.num_levels = num_levels

        # Convolution blocks after fusion.
        self.conv_after_td = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            for _ in range(num_levels)
        ])
        self.conv_after_bu = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            for _ in range(num_levels)
        ])

        # Weighted addition modules for top-down (TD) and bottom-up (BU) fusion.
        self.wadd_td = nn.ModuleList([WeightedAdd(2) for _ in range(num_levels - 1)])
        self.wadd_bu = nn.ModuleList([WeightedAdd(2) for _ in range(num_levels - 1)])

        # Channel attention modules.
        self.attn_td = nn.ModuleList([SEBlock(channels) for _ in range(num_levels)])
        self.attn_bu = nn.ModuleList([SEBlock(channels) for _ in range(num_levels)])

    def forward(self, feats):
        # Assume feats is a list ordered from high to low resolution.
        # Top-down pathway:
        td_feats = [None] * self.num_levels
        td_feats[-1] = feats[-1]
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(td_feats[i+1], size=feats[i].shape[-2:], mode='nearest')
            fused = self.wadd_td[i]([feats[i], upsampled])
            fused = self.conv_after_td[i](fused)
            fused = self.attn_td[i](fused)
            td_feats[i] = fused

        # Bottom-up pathway:
        bu_feats = [None] * self.num_levels
        bu_feats[0] = td_feats[0]
        for i in range(1, self.num_levels):
            downsampled = F.max_pool2d(bu_feats[i-1], kernel_size=2, stride=2)
            fused = self.wadd_bu[i-1]([td_feats[i], downsampled])
            fused = self.conv_after_bu[i](fused)
            fused = self.attn_bu[i](fused)
            bu_feats[i] = fused

        return bu_feats

###############################
# 5. EMSA-BiFPN: Stacking Layers and Integrating Fixed Filter
###############################
class EMSA_BiFPN(nn.Module):
    def __init__(self, channels, num_levels=3, num_layers=3,
                 use_fixed_filter=True, fixed_filter_size=3):
        """
        channels      : Number of channels per feature map.
        num_levels    : Number of resolutions (e.g., 3).
        num_layers    : Number of BiFPN layers to stack.
        use_fixed_filter : Whether to apply a fixed (handcrafted) filter.
        fixed_filter_size: Kernel size for the handcrafted filter.
        """
        super(EMSA_BiFPN, self).__init__()
        self.num_layers = num_layers
        self.bifpn_layers = nn.ModuleList(
            [BiFPNLayer(channels, num_levels) for _ in range(num_layers)]
        )
        self.use_fixed_filter = use_fixed_filter
        if use_fixed_filter:
            self.fixed_filters = nn.ModuleList([
                FixedFilter(channels, kernel_size=fixed_filter_size)
                for _ in range(num_levels)
            ])
        # Final fusion: upsample all features to the highest resolution.
        self.final_conv = nn.Conv2d(channels * num_levels, channels, kernel_size=3, padding=1)

    def forward(self, feats):
        for layer in self.bifpn_layers:
            feats = layer(feats)
        if self.use_fixed_filter:
            feats = [filt(feat) for filt, feat in zip(self.fixed_filters, feats)]
        target_size = feats[0].shape[-2:]
        upsampled_feats = [feats[0]] + [
            F.interpolate(feat, size=target_size, mode='nearest')
            for feat in feats[1:]
        ]
        concat = torch.cat(upsampled_feats, dim=1)
        out = self.final_conv(concat)
        return out

###############################
# 6. Simple Backbone for Multi-scale Feature Extraction
###############################
class SimpleBackbone(nn.Module):
    """
    A simple CNN backbone that extracts three levels of features.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Stage 1: High resolution.
        self.stage1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        # Stage 2: Medium resolution.
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        # Stage 3: Low resolution.
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        feat1 = self.stage1(x)    # High resolution
        feat2 = self.stage2(feat1)  # Medium resolution
        feat3 = self.stage3(feat2)  # Low resolution
        return [feat1, feat2, feat3]

###############################
# 7. Full EMSA-BiFPN Network (Backbone + BiFPN + Head)
###############################
class EMSA_BiFPN_Network(nn.Module):
    def __init__(self, in_channels=3, base_channels=64,
                 num_levels=3, num_layers=3,
                 use_fixed_filter=True, fixed_filter_size=3):
        super(EMSA_BiFPN_Network, self).__init__()
        self.backbone = SimpleBackbone(in_channels, base_channels)
        self.emsa_bifpn = EMSA_BiFPN(base_channels, num_levels, num_layers,
                                     use_fixed_filter, fixed_filter_size)
        # A simple head that outputs a refined feature map.
        self.head = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feats = self.backbone(x)
        fused = self.emsa_bifpn(feats)
        out = self.head(fused)
        return out

###############################
# 8. Testing the Network
###############################
if __name__ == '__main__':
    model = EMSA_BiFPN_Network(in_channels=3, base_channels=64,
                               num_levels=3, num_layers=3,
                               use_fixed_filter=True, fixed_filter_size=3)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print("Output shape:", output.shape)



