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
    For example, a 3x3 Laplacian kernel (or a configurable averaging filter).
    """
    def __init__(self, in_channels, kernel_size=3, filter_type='laplacian'):
        super(FixedFilter, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding,
                              groups=in_channels, bias=False)

        if kernel_size == 3 and filter_type == 'laplacian':
            kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32)
        else:
            kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
            kernel /= kernel.sum()

        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(in_channels, 1, 1, 1)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False

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
    def __init__(self, channels, num_levels=5):
        super(BiFPNLayer, self).__init__()
        self.num_levels = num_levels

        self.conv_after_td = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            for _ in range(num_levels)
        ])
        self.conv_after_bu = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            for _ in range(num_levels)
        ])

        self.wadd_td = nn.ModuleList([WeightedAdd(2) for _ in range(num_levels - 1)])
        self.wadd_bu = nn.ModuleList([WeightedAdd(2) for _ in range(num_levels - 1)])

        self.attn_td = nn.ModuleList([SEBlock(channels) for _ in range(num_levels)])
        self.attn_bu = nn.ModuleList([SEBlock(channels) for _ in range(num_levels)])

    def forward(self, feats):
        # Assume feats is a list of feature maps ordered from highest resolution to lowest.
        td_feats = [None] * self.num_levels
        td_feats[-1] = feats[-1]
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(td_feats[i + 1], size=feats[i].shape[-2:], mode='nearest')
            fused = self.wadd_td[i]([feats[i], upsampled])
            fused = self.conv_after_td[i](fused)
            fused = self.attn_td[i](fused)
            td_feats[i] = fused

        bu_feats = [None] * self.num_levels
        bu_feats[0] = td_feats[0]
        for i in range(1, self.num_levels):
            downsampled = F.max_pool2d(bu_feats[i - 1], kernel_size=2, stride=2)
            fused = self.wadd_bu[i - 1]([td_feats[i], downsampled])
            fused = self.conv_after_bu[i](fused)
            fused = self.attn_bu[i](fused)
            bu_feats[i] = fused

        return bu_feats

###############################
# 5. EMSA-BiFPN: Stacking Layers and Integrating Fixed Filter
###############################
class EMSA_BiFPN(nn.Module):
    def __init__(self, channels, num_levels=5, num_layers=3,
                 use_fixed_filter=True, fixed_filter_size=3):
        """
        channels: number of channels per feature map.
        num_levels: number of resolution levels (here, 5).
        num_layers: number of BiFPN layers.
        use_fixed_filter: whether to apply a fixed matching filter.
        fixed_filter_size: kernel size for the fixed filter.
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
# 6. MultiResolution Backbone with Five Stages
###############################
class MultiResolutionBackbone(nn.Module):
    """
    Extracts five levels of features using different convolution kernel sizes.
    For an input of (B, 3, 96, 96):
      - Stage 1: 1x1 convolution, stride 1 → 96x96.
      - Stage 2: 3x3 convolution, stride 2 → 48x48.
      - Stage 3: 5x5 convolution, stride 2 → 24x24.
      - Stage 4: 7x7 convolution, stride 2 → 12x12.
      - Stage 5: 9x9 convolution, stride 2 → 6x6.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(MultiResolutionBackbone, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.stage1(x)    # 96x96
        feat2 = self.stage2(feat1)  # 48x48
        feat3 = self.stage3(feat2)  # 24x24
        feat4 = self.stage4(feat3)  # 12x12
        feat5 = self.stage5(feat4)  # 6x6
        return [feat1, feat2, feat3, feat4, feat5]

###############################
# 7. Full EMSA-BiFPN Network (Backbone + BiFPN + Head)
###############################
class EMSA_BiFPN_Network(nn.Module):
    def __init__(self, in_channels=3, base_channels=64,
                 num_levels=5, num_layers=3,
                 use_fixed_filter=True, fixed_filter_size=3):
        super(EMSA_BiFPN_Network, self).__init__()
        self.backbone = MultiResolutionBackbone(in_channels, base_channels)
        self.emsa_bifpn = EMSA_BiFPN(base_channels, num_levels, num_layers,
                                     use_fixed_filter, fixed_filter_size)
        # A simple head to output a refined feature map.
        self.head = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feats = self.backbone(x)  # Produces 5 feature maps.
        fused = self.emsa_bifpn(feats)
        out = self.head(fused)
        return out  # Expected shape: (B, base_channels, 96, 96)

###############################
# 8. Detection Head for Bounding Box Regression and Classification
###############################
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        in_channels: number of channels from the concatenated input.
                     (e.g., EMSA_BiFPN output channels + SNR map channels).
        num_classes: user-defined number of classes.
        """
        super(DetectionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global pooling.
        self.fc_bbox = nn.Linear(128, 4)       # Bounding box: (x_min, y_min, x_max, y_max)
        self.fc_cls = nn.Linear(128, num_classes)  # Classification logits.

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        bbox = self.fc_bbox(x)
        cls_logits = self.fc_cls(x)
        return bbox, cls_logits

###############################
# 9. Utility Function: Draw Bounding Box from Center Coordinates
###############################
def draw_bbox_from_center(center, m, image_shape=(96, 96)):
    """
    Given center coordinates (x, y) and a hyperparameter m (box side length),
    returns bounding box coordinates (x_min, y_min, x_max, y_max) such that the box
    is centered at (x, y) and has size (m, m). The coordinates are clamped to the
    image boundaries (image_shape: (H, W)).
    """
    x_center, y_center = center
    half = m // 2  # Use integer division for pixel indices.
    x_min = max(0, x_center - half)
    y_min = max(0, y_center - half)
    x_max = min(image_shape[1] - 1, x_center + half)
    y_max = min(image_shape[0] - 1, y_center + half)
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

###############################
# 10. Object Detector: Combining Feature Extraction, SNR Map, and Detection Head
###############################
class ObjectDetector(nn.Module):
    def __init__(self, num_classes, base_channels=64):
        """
        num_classes: number of object classes.
        The detector uses the EMSA_BiFPN_Network as feature extractor.
        It then concatenates the resulting feature map with the SNR map (assumed to have 3 channels),
        so that the detection head receives (base_channels + 3) channels.
        """
        super(ObjectDetector, self).__init__()
        self.feature_extractor = EMSA_BiFPN_Network(in_channels=3, base_channels=base_channels,
                                                     num_levels=5, num_layers=3,
                                                     use_fixed_filter=True, fixed_filter_size=3)
        # The feature extractor outputs (B, base_channels, 96, 96).
        # The SNR map is (B, 3, 96, 96) → concatenated: (B, base_channels+3, 96, 96).
        self.detection_head = DetectionHead(in_channels=base_channels + 3, num_classes=num_classes)

    def forward(self, x, snr_map):
        # x: input image, shape (B, 3, 96, 96)
        # snr_map: precomputed SNR map, shape (B, 3, 96, 96)
        feature_map = self.feature_extractor(x)  # (B, base_channels, 96, 96)
        # Concatenate along channel dimension.
        concat = torch.cat([feature_map, snr_map], dim=1)  # (B, base_channels+3, 96, 96)
        bbox, cls_logits = self.detection_head(concat)
        return bbox, cls_logits

###############################
# 11. Example Testing of the Object Detector and Bounding Box Drawing Function
###############################
if __name__ == '__main__':
    num_classes = 5  # For example, 5 object classes.
    model = ObjectDetector(num_classes=num_classes, base_channels=64)
    
    # Example input image (B, 3, 96, 96) and a corresponding SNR map (B, 3, 96, 96).
    x = torch.randn(1, 3, 96, 96)
    snr_map = torch.randn(1, 3, 96, 96)
    
    # Forward pass through the detection model.
    pred_bbox, pred_cls_logits = model(x, snr_map)
    print("Predicted bounding box (regression output):", pred_bbox)
    print("Predicted class logits:", pred_cls_logits)
    
    # Demonstration of drawing a bounding box using a provided center.
    # Suppose we have computed (or received) a center coordinate from the SNR map.
    center = (48, 48)  # For example, center at (48,48) in a 96x96 feature map.
    m = 20  # Hyperparameter: side length of the bounding box.
    drawn_bbox = draw_bbox_from_center(center, m, image_shape=(96, 96))
    print("Drawn bounding box from center {} with side length {}:".format(center, m), drawn_bbox)
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################
# 1. CNN/Inception–Style Feature Extractor
##########################################
class InceptionBlock(nn.Module):
    """
    A simple Inception–style block with four parallel branches:
      - Branch 1: 1×1 convolution (preserves resolution)
      - Branch 2: 3×3 convolution (with padding=1)
      - Branch 3: 5×5 convolution (with padding=2)
      - Branch 4: 3×3 max pooling followed by a 1×1 convolution
    The outputs of the branches are concatenated along the channel dimension.
    """
    def __init__(self, in_channels, branch_channels):
        super(InceptionBlock, self).__init__()
        # Branch 1: 1x1 conv
        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0)
        # Branch 2: 3x3 conv with same resolution (padding=1)
        self.branch2 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        # Branch 3: 5x5 conv with same resolution (padding=2)
        self.branch3 = nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2)
        # Branch 4: max pool then 1x1 conv
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        b3 = self.relu(self.branch3(x))
        b4 = self.relu(self.branch4_conv(self.branch4_pool(x)))
        # Concatenate along channels. If branch_channels is C, output will have 4×C channels.
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return out

class CNNInceptionFeatureExtractor(nn.Module):
    """
    A baseline CNN feature extractor built around an Inception–style module.
    It begins with a 1×1 convolution (to adjust channel depth), passes the result through an Inception block,
    and finally uses a 1×1 convolution to fuse the concatenated features back to the desired output channels.
    The spatial resolution is preserved (i.e. no downsampling) so that for an input of (B, 3, 96, 96) the output
    feature map remains (B, out_channels, 96, 96).
    """
    def __init__(self, in_channels=3, out_channels=64):
        super(CNNInceptionFeatureExtractor, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        # Use an Inception block where each branch produces out_channels//4 channels.
        self.inception_block = InceptionBlock(out_channels, branch_channels=out_channels // 4)
        # After concatenation the number of channels is out_channels (because 4*(out_channels//4)==out_channels).
        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.initial_conv(x))
        x = self.inception_block(x)
        x = self.relu(self.conv_1x1(x))
        return x

##########################################
# 2. Vision Transformer (ViT)–Style Feature Extractor
##########################################
class PatchEmbed1x1(nn.Module):
    """
    A patch embedding module that uses a 1×1 convolution to map each pixel (i.e. patch of size 1×1)
    to a vector of dimension embed_dim. This preserves the original spatial resolution.
    """
    def __init__(self, in_channels=3, embed_dim=64):
        super(PatchEmbed1x1, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        x = self.proj(x)  # (B, embed_dim, H, W)
        B, C, H, W = x.shape
        # Flatten the spatial dimensions and transpose: (B, H*W, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)

class TransformerBlock(nn.Module):
    """
    A standard Transformer block with multi-head self-attention and an MLP.
    Uses LayerNorm before the attention and MLP sub-blocks and applies residual connections.
    """
    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention sub-block.
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP sub-block.
        x = x + self.mlp(self.norm2(x))
        return x

class ViTFeatureExtractor(nn.Module):
    """
    A ViT–style feature extractor that embeds each input pixel (using a 1×1 patch embedding)
    into an embedding vector. A series of Transformer blocks (with attention and MLP sub-blocks)
    process the tokens. Finally, the token sequence is reshaped back to a feature map of shape
    (B, embed_dim, H, W), preserving the original resolution.
    """
    def __init__(self, in_channels=3, embed_dim=64, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super(ViTFeatureExtractor, self).__init__()
        self.patch_embed = PatchEmbed1x1(in_channels, embed_dim)
        # Positional embedding is created dynamically based on input spatial dimensions.
        self.pos_embed = None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        x, (H, W) = self.patch_embed(x)  # x: (B, H*W, embed_dim)
        B, N, C = x.shape
        # Create or update positional embeddings if needed.
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != N):
            self.pos_embed = nn.Parameter(torch.zeros(1, N, C), requires_grad=True)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        x = x + self.pos_embed
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Reshape back to (B, embed_dim, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

##########################################
# 3. Testing the Baseline Feature Extractors
##########################################
if __name__ == '__main__':
    # Example input: (B, 3, 96, 96)
    x = torch.randn(1, 3, 96, 96)
    print("Input shape:", x.shape)
    
    # Test the CNN/Inception-based Feature Extractor.
    cnn_extractor = CNNInceptionFeatureExtractor(in_channels=3, out_channels=64)
    cnn_features = cnn_extractor(x)
    print("CNN/Inception Feature Extractor output shape:", cnn_features.shape)
    
    # Test the ViT-based Feature Extractor.
    vit_extractor = ViTFeatureExtractor(in_channels=3, embed_dim=64, depth=6, num_heads=4)
    vit_features = vit_extractor(x)
    print("ViT Feature Extractor output shape:", vit_features.shape)


import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Set up the simulation: grid, pupil, and random phase
# ---------------------------------------------------
N = 512             # grid size (number of samples per dimension)
L = 5e-3            # physical size of the pupil plane (meters)
dx = L / N          # spatial sampling interval

# Create spatial coordinate arrays (centered at zero)
x = np.linspace(-L/2, L/2, N)
y = x.copy()
X, Y = np.meshgrid(x, y)

# Define a circular pupil (aperture) of radius R
R = 1e-3            # pupil radius (meters)
pupil = np.zeros((N, N))
pupil[np.sqrt(X**2 + Y**2) <= R] = 1.0

# Impose a random phase over the pupil
np.random.seed(0)   # for reproducibility
random_phase = np.random.uniform(0, 2*np.pi, (N, N))
E_pupil = pupil * np.exp(1j * random_phase)

# ---------------------------------------------------
# 2. Compute the image (speckle) field via Fourier Transform
# ---------------------------------------------------
# The imaging plane field is the Fourier transform of the pupil field.
E_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil)))
I_image = np.abs(E_image)**2  # intensity in the image (speckle) plane

# Define spatial frequency coordinates for the image plane (u,v)
u = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
v = u.copy()  # square grid

# ---------------------------------------------------
# 3. Identify the bright spot in the image plane
# ---------------------------------------------------
max_idx = np.unravel_index(np.argmax(I_image), I_image.shape)
u0 = u[max_idx[1]]
v0 = v[max_idx[0]]
print(f"Bright spot located at: u0 = {u0:.2e}, v0 = {v0:.2e}")

# ---------------------------------------------------
# 4. Demodulate the pupil field to reveal the residual phase
# ---------------------------------------------------
# Multiply by the conjugate Fourier kernel corresponding to (u0, v0)
E_pupil_demod = E_pupil * np.exp(1j * 2 * np.pi * (u0 * X + v0 * Y))
phase_residual = np.angle(E_pupil_demod)

# Consider only points inside the pupil.
pupil_mask = (pupil > 0)

# Compute the average phase in the pupil (using complex averaging to avoid 2π issues)
phase_mean = np.angle(np.mean(np.exp(1j * phase_residual[pupil_mask])))

# Compute the phase difference relative to the average
phase_diff = np.angle(np.exp(1j * (phase_residual - phase_mean)))

# Define a threshold (in radians) to decide which regions are “in-phase”
threshold = 0.2  # adjust as needed; smaller threshold means stricter condition
contrib_mask = (np.abs(phase_diff) < threshold) & pupil_mask

# ---------------------------------------------------
# 5. Approach 1: Add a Phase Plate to Disturb the Coherent Contribution
# ---------------------------------------------------
# Instead of blocking the contributions, add an extra phase (e.g. π/2)
# over the pupil regions that are nearly in phase.
phase_plate = np.zeros((N, N))
phase_plate[contrib_mask] = np.pi / 2  # you can choose other values or even a spatially varying pattern

# Create a modified pupil field with the phase plate added
E_pupil_modified = E_pupil * np.exp(1j * phase_plate)

# Compute the new image field and intensity
E_image_modified = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil_modified)))
I_image_modified = np.abs(E_image_modified)**2

# ---------------------------------------------------
# 6. Approach 2: Alternative Idea – Apodization
# ---------------------------------------------------
# An alternative to altering the phase is to apply an amplitude taper (apodization)
# to smooth out the abrupt phase/amplitude changes. Here, we use a Gaussian taper.
apodization = np.exp(-((X/(0.7*R))**2 + (Y/(0.7*R))**2))
E_pupil_apod = pupil * np.exp(1j * random_phase) * apodization
E_image_apod = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil_apod)))
I_image_apod = np.abs(E_image_apod)**2

# ---------------------------------------------------
# 7. Plot the Results
# ---------------------------------------------------
plt.figure(figsize=(16, 12))

# (a) Original Image Intensity (Speckle Pattern)
plt.subplot(2, 3, 1)
plt.imshow(I_image, extent=[u[0], u[-1], v[0], v[-1]], cmap='inferno')
plt.title("Original Image Intensity")
plt.xlabel("u (spatial freq.)")
plt.ylabel("v (spatial freq.)")
plt.colorbar()

# (b) Residual Phase in the Pupil (after Demodulation)
plt.subplot(2, 3, 2)
plt.imshow(phase_residual, extent=[x[0], x[-1], y[0], y[-1]], cmap='hsv')
plt.title("Residual Phase (Demodulated)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.colorbar()

# (c) Contributing Region in the Pupil (mask)
plt.subplot(2, 3, 3)
plt.imshow(contrib_mask, extent=[x[0], x[-1], y[0], y[-1]], cmap='gray')
plt.title("Pupil Regions Contributing Coherently")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.colorbar()

# (d) Image with the Phase Plate Added (Bright spot suppressed)
plt.subplot(2, 3, 4)
plt.imshow(I_image_modified, extent=[u[0], u[-1], v[0], v[-1]], cmap='inferno')
plt.title("Image with Phase Plate")
plt.xlabel("u (spatial freq.)")
plt.ylabel("v (spatial freq.)")
plt.colorbar()

# (e) The Phase Plate Pattern Applied to the Pupil
plt.subplot(2, 3, 5)
plt.imshow(phase_plate, extent=[x[0], x[-1], y[0], y[-1]], cmap='jet')
plt.title("Phase Plate Pattern")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.colorbar()

# (f) Alternative: Image with Apodization (Another way to suppress intensity peaks)
plt.subplot(2, 3, 6)
plt.imshow(I_image_apod, extent=[u[0], u[-1], v[0], v[-1]], cmap='inferno')
plt.title("Image with Apodization")
plt.xlabel("u (spatial freq.)")
plt.ylabel("v (spatial freq.)")
plt.colorbar()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Set up the simulation: grid, pupil, and random phase
# ---------------------------------------------------
N = 512             # grid size (number of samples per dimension)
L = 5e-3            # physical size of the pupil plane (meters)
dx = L / N          # spatial sampling interval

# Create spatial coordinate arrays (centered at zero)
x = np.linspace(-L/2, L/2, N)
y = x.copy()
X, Y = np.meshgrid(x, y)

# Define a circular pupil (aperture) of radius R
R = 1e-3            # pupil radius (meters)
pupil = np.zeros((N, N))
pupil[np.sqrt(X**2 + Y**2) <= R] = 1.0

# Impose a random phase over the pupil
np.random.seed(0)   # for reproducibility
random_phase = np.random.uniform(0, 2*np.pi, (N, N))
E_pupil = pupil * np.exp(1j * random_phase)

# ---------------------------------------------------
# 2. Create a Soft Mask (Apodization Function)
# ---------------------------------------------------
# Instead of a hard binary mask, we use a Gaussian taper that smoothly decreases from the center.
# The Gaussian width is chosen relative to the pupil radius.
sigma = R / 2.0  # adjust sigma for a suitable smoothness
soft_mask = np.exp(-((X**2 + Y**2) / (2 * sigma**2)))

# Multiply the pupil by the soft mask (note that the soft mask here modulates amplitude)
E_pupil_soft = pupil * soft_mask * np.exp(1j * random_phase)

# ---------------------------------------------------
# 3. Compute the Image Field (Speckle Pattern)
# ---------------------------------------------------
def compute_image(E):
    # Compute the Fourier transform (image plane field) and intensity
    E_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E)))
    I_image = np.abs(E_image)**2
    return E_image, I_image

# Original (hard-edged) pupil image
E_image_orig, I_image_orig = compute_image(E_pupil)

# Soft mask (apodized) pupil image
E_image_soft, I_image_soft = compute_image(E_pupil_soft)

# ---------------------------------------------------
# 4. Plot the Results
# ---------------------------------------------------
plt.figure(figsize=(14, 10))

# (a) Pupil Intensity (hard-edged)
plt.subplot(2, 3, 1)
plt.imshow(np.abs(E_pupil)**2, extent=[x[0], x[-1], y[0], y[-1]], cmap='gray')
plt.title("Original Pupil Intensity (Hard Mask)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.colorbar()

# (b) Soft Mask (Apodization Function)
plt.subplot(2, 3, 2)
plt.imshow(soft_mask, extent=[x[0], x[-1], y[0], y[-1]], cmap='viridis')
plt.title("Soft Mask (Gaussian Apodization)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.colorbar()

# (c) Pupil Intensity with Soft Mask Applied
plt.subplot(2, 3, 3)
plt.imshow(np.abs(E_pupil_soft)**2, extent=[x[0], x[-1], y[0], y[-1]], cmap='gray')
plt.title("Pupil Intensity with Soft Mask")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.colorbar()

# (d) Image Intensity from Original (Hard Mask) Pupil
plt.subplot(2, 3, 4)
plt.imshow(I_image_orig, extent=[-1e3, 1e3, -1e3, 1e3], cmap='inferno')
plt.title("Image from Hard-edged Pupil")
plt.xlabel("u (spatial freq.)")
plt.ylabel("v (spatial freq.)")
plt.colorbar()

# (e) Image Intensity from Soft Masked Pupil
plt.subplot(2, 3, 5)
plt.imshow(I_image_soft, extent=[-1e3, 1e3, -1e3, 1e3], cmap='inferno')
plt.title("Image from Soft (Apodized) Pupil")
plt.xlabel("u (spatial freq.)")
plt.ylabel("v (spatial freq.)")
plt.colorbar()

plt.tight_layout()
plt.show()


# 4. Demodulate the pupil field to reveal the residual phase
# ---------------------------------------------------
# Multiply by the conjugate Fourier kernel corresponding to (u0, v0)
E_pupil_demod = E_pupil * np.exp(1j * 2 * np.pi * (u0 * X + v0 * Y))
phase_residual = np.angle(E_pupil_demod)

# Consider only points inside the pupil.
pupil_mask = (pupil > 0)

# Compute the average phase in the pupil (using complex averaging to avoid 2π issues)
phase_mean = np.angle(np.mean(np.exp(1j * phase_residual[pupil_mask])))

# Compute the phase difference relative to the average
phase_diff = np.angle(np.exp(1j * (phase_residual - phase_mean)))

# Define a threshold (in radians) for the in-phase condition
threshold = 0.2  # This value controls the "width" of the in-phase region.

# Instead of a binary (hard) mask, we build a soft mask using a Gaussian function.
# We choose sigma such that at |phase_diff| = threshold, the mask value is 0.5.
import numpy as np
sigma = threshold / np.sqrt(2 * np.log(2))  # because exp(-threshold^2/(2*sigma^2)) = 0.5

# soft_mask_phase is 1 when phase_diff=0 and decays continuously as |phase_diff| increases.
soft_mask_phase = np.exp(- (np.abs(phase_diff)**2) / (2 * sigma**2))

# Apply the pupil mask to ensure the soft mask is defined only inside the pupil.
soft_mask_phase = soft_mask_phase * pupil_mask.astype(float)

# Option 1: If you want to suppress the contributions gradually, you can reduce the pupil field
# by multiplying by (1 - soft_mask_phase).
E_pupil_modified = E_pupil * (1 - soft_mask_phase)

# Option 2: Alternatively, if you want to simply weight the contributions later (e.g., for analysis),
# you can use soft_mask_phase directly as a weighting factor.

# For example, computing a modified image:
E_image_modified = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil_modified)))
I_image_modified = np.abs(E_image_modified)**2

# (Now you can compare I_image_modified to the original I_image.)

import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
N = 512                  # grid size (samples per dimension)
L = 5e-3                 # physical size of the pupil plane (meters)
dx = L / N
num_realizations = 50    # number of independent speckle realizations

# Spatial coordinates (centered at zero)
x = np.linspace(-L/2, L/2, N)
y = x.copy()
X, Y = np.meshgrid(x, y)

# Define a circular pupil of radius R
R = 1e-3
pupil = np.zeros((N, N))
pupil[np.sqrt(X**2 + Y**2) <= R] = 1.0

def generate_speckle(E_pupil):
    """Compute the intensity speckle pattern from a pupil field via FFT."""
    E_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil)))
    I_image = np.abs(E_image)**2
    return I_image

def ensemble_speckle(E_pupil_func, num_realizations):
    """
    Compute the ensemble-averaged intensity and speckle contrast
    for a given pupil modification function E_pupil_func.
    """
    I_accum = np.zeros((N, N))
    all_contrasts = []
    for _ in range(num_realizations):
        # Generate a new random phase for each realization
        random_phase = np.random.uniform(0, 2*np.pi, (N, N))
        E_pupil = pupil * np.exp(1j * random_phase)
        # Apply the pupil modification (e.g. waveplate or soft mask)
        E_mod = E_pupil_func(E_pupil, X, Y)
        I_image = generate_speckle(E_mod)
        I_accum += I_image
        # Contrast for this realization:
        contrast = np.std(I_image) / np.mean(I_image)
        all_contrasts.append(contrast)
    I_avg = I_accum / num_realizations
    # Contrast computed from the ensemble-averaged intensity:
    contrast_ensemble = np.std(I_avg) / np.mean(I_avg)
    return I_avg, np.mean(all_contrasts), contrast_ensemble

# ---------------------------
# Define the pupil modification functions
# ---------------------------

def identity(E_pupil, X, Y):
    """No modification; return the pupil field as is."""
    return E_pupil

def waveplate_mod(E_pupil, X, Y):
    """
    Waveplate modification:
      Add a constant phase shift (π/2) to the region where X > 0.
    This shifts the phase of that region relative to the rest.
    """
    phase_shift = np.pi / 2
    mask_waveplate = np.where(X > 0, 1, 0)
    return E_pupil * np.exp(1j * phase_shift * mask_waveplate)

def soft_mask_mod(E_pupil, X, Y):
    """
    Soft mask modification:
      Gradually suppress the pupil transmission for X > 0 using a smooth transition.
    We use a tanh function to create a smooth transition from full transmission to suppression.
    """
    # Define a transition width (adjustable)
    T = L / 20
    # The tanh function yields values from -1 to 1; converting to [0,1]:
    mask = (1 + np.tanh(X / (T/5))) / 2
    # Here, the mask is ~0 for X << 0 and ~1 for X >> 0.
    # To suppress the contributions in the X > 0 region smoothly, multiply by (1 - mask).
    return E_pupil * (1 - mask)

# ---------------------------
# Compute ensemble averages for each modification
# ---------------------------

I_avg_id, contrast_id_avg, contrast_id_ensemble = ensemble_speckle(identity, num_realizations)
I_avg_wp, contrast_wp_avg, contrast_wp_ensemble = ensemble_speckle(waveplate_mod, num_realizations)
I_avg_sm, contrast_sm_avg, contrast_sm_ensemble = ensemble_speckle(soft_mask_mod, num_realizations)

print("Ensemble-averaged speckle contrast (mean of individual contrasts):")
print("  Identity (no modification): {:.2f}".format(contrast_id_avg))
print("  Waveplate modification:      {:.2f}".format(contrast_wp_avg))
print("  Soft mask modification:      {:.2f}".format(contrast_sm_avg))

print("\nContrast computed from ensemble-averaged intensity:")
print("  Identity (no modification): {:.2f}".format(contrast_id_ensemble))
print("  Waveplate modification:      {:.2f}".format(contrast_wp_ensemble))
print("  Soft mask modification:      {:.2f}".format(contrast_sm_ensemble))

# ---------------------------
# Plot the ensemble-averaged speckle patterns
# ---------------------------
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.imshow(I_avg_id, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3],
           cmap='inferno', origin='lower')
plt.title("Ensemble Averaged (Identity)\nContrast: {:.2f}".format(contrast_id_ensemble))
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(I_avg_wp, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3],
           cmap='inferno', origin='lower')
plt.title("Ensemble Averaged (Waveplate)\nContrast: {:.2f}".format(contrast_wp_ensemble))
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(I_avg_sm, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3],
           cmap='inferno', origin='lower')
plt.title("Ensemble Averaged (Soft Mask)\nContrast: {:.2f}".format(contrast_sm_ensemble))
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# 1. Setup: Define grid, pupil, and simulation parameters.
# =============================================================================
N = 512               # grid size (samples per dimension)
L = 5e-3              # physical size of the pupil plane in meters
dx = L / N            # spatial sampling interval

# Spatial coordinates (centered at zero)
x = np.linspace(-L/2, L/2, N)
y = x.copy()
X, Y = np.meshgrid(x, y)

# Define a circular pupil (aperture) of radius R
R = 1e-3              # pupil radius (meters)
pupil = np.zeros((N, N))
pupil[np.sqrt(X**2 + Y**2) <= R] = 1.0

num_realizations = 50  # number of independent speckle realizations

# =============================================================================
# 2. Generate ensemble of speckle images and store pupil fields.
# =============================================================================
I_orig_accum = np.zeros((N, N))
E_pupil_list = []  # store each realization's pupil field

for i in range(num_realizations):
    # Generate a pupil field with a random phase.
    random_phase = np.random.uniform(0, 2*np.pi, (N, N))
    E_pupil_i = pupil * np.exp(1j * random_phase)
    E_pupil_list.append(E_pupil_i)
    
    # Propagate to the image plane via FFT.
    E_img_i = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil_i)))
    I_img_i = np.abs(E_img_i)**2
    I_orig_accum += I_img_i

# Ensemble-averaged original speckle image.
I_orig_avg = I_orig_accum / num_realizations

# Define spatial frequency axes for the image plane.
u = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
v = u.copy()

# =============================================================================
# 3. Find the ensemble-averaged bright spot (maximum intensity) and its (u₀,v₀).
# =============================================================================
max_idx = np.unravel_index(np.argmax(I_orig_avg), I_orig_avg.shape)
u0 = u[max_idx[1]]   # Column index corresponds to u
v0 = v[max_idx[0]]   # Row index corresponds to v
print("Ensemble-averaged bright spot at (u0, v0) =", u0, v0)

# =============================================================================
# 4. For each realization, demodulate the pupil field and compute a mask
#    based on the residual phase at the bright spot.
# =============================================================================
# We use the same threshold (in radians) for all realizations.
threshold = 0.2  # radians
# Choose sigma so that at |phase_diff| == threshold, Gaussian value ~0.5.
sigma = threshold / np.sqrt(np.log(2))
phase_shift_const = np.pi / 2  # for waveplate modification

# Initialize accumulators for the modified images.
I_hard_accum = np.zeros((N, N))
I_soft_accum = np.zeros((N, N))
I_wave_accum = np.zeros((N, N))

# Also accumulate contrast values from each realization.
contrasts_hard = []
contrasts_soft = []
contrasts_wave = []

for E_pupil_i in E_pupil_list:
    
    # Demodulate the pupil field: multiply by the conjugate Fourier kernel corresponding to (u0,v0).
    E_demod = E_pupil_i * np.exp(1j * 2 * np.pi * (u0 * X + v0 * Y))
    phase_res = np.angle(E_demod)
    
    # Consider only points inside the pupil.
    mask_pupil = (pupil > 0)
    
    # Compute average residual phase over the pupil (using only pupil points).
    phi_avg = np.angle(np.mean(np.exp(1j * phase_res[mask_pupil])))
    
    # Compute the residual phase difference relative to the average.
    phase_diff = np.angle(np.exp(1j * (phase_res - phi_avg)))
    
    # Generate a binary (hard) mask: coherent if |phase_diff| < threshold.
    mask_binary = (np.abs(phase_diff) < threshold) & mask_pupil
    
    # Generate a soft mask using a Gaussian function of |phase_diff|.
    soft_mask = np.exp(- (np.abs(phase_diff)**2) / (2 * sigma**2))
    soft_mask = soft_mask * mask_pupil.astype(float)
    
    # =============================================================================
    # 5. Pupil modifications over the coherent (masked) region.
    # =============================================================================
    # (a) Hard mask modification: zero out the pupil field where mask_binary is True.
    E_pupil_hard = E_pupil_i * (1 - mask_binary.astype(float))
    
    # (b) Soft mask modification: attenuate the pupil field amplitude gradually.
    E_pupil_soft = E_pupil_i * (1 - soft_mask)
    
    # (c) Waveplate modification: add a constant phase shift in the binary mask region.
    E_pupil_wave = E_pupil_i.copy()
    E_pupil_wave[mask_binary] *= np.exp(1j * phase_shift_const)
    
    # Propagate each modified pupil field to get the corresponding speckle image.
    I_img_hard = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil_hard))))**2
    I_img_soft = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil_soft))))**2
    I_img_wave = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil_wave))))**2
    
    I_hard_accum += I_img_hard
    I_soft_accum += I_img_soft
    I_wave_accum += I_img_wave
    
    # Compute speckle contrast (std/mean) for this realization.
    c_hard = np.std(I_img_hard) / np.mean(I_img_hard)
    c_soft = np.std(I_img_soft) / np.mean(I_img_soft)
    c_wave = np.std(I_img_wave) / np.mean(I_img_wave)
    contrasts_hard.append(c_hard)
    contrasts_soft.append(c_soft)
    contrasts_wave.append(c_wave)

# Ensemble-averaged modified images.
I_hard_avg = I_hard_accum / num_realizations
I_soft_avg = I_soft_accum / num_realizations
I_wave_avg = I_wave_accum / num_realizations

# Compute ensemble speckle contrast from the averaged images.
def speckle_contrast(I):
    return np.std(I) / np.mean(I)

contrast_orig = speckle_contrast(I_orig_avg)
contrast_hard = speckle_contrast(I_hard_avg)
contrast_soft = speckle_contrast(I_soft_avg)
contrast_wave = speckle_contrast(I_wave_avg)

print("\nEnsemble-averaged speckle contrast:")
print("  Original:      {:.2f}".format(contrast_orig))
print("  Hard mask:     {:.2f}".format(contrast_hard))
print("  Soft mask:     {:.2f}".format(contrast_soft))
print("  Waveplate:     {:.2f}".format(contrast_wave))

# =============================================================================
# 6. Plot the results.
# =============================================================================
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# (a) Ensemble-averaged original speckle image with bright spot circled.
axs[0, 0].imshow(I_orig_avg, extent=[u[0], u[-1], v[0], v[-1]], cmap='inferno', origin='lower')
axs[0, 0].set_title("Ensemble Averaged Original\nContrast: {:.2f}".format(contrast_orig))
axs[0, 0].set_xlabel("u (spatial freq.)")
axs[0, 0].set_ylabel("v (spatial freq.)")
# Circle the bright spot.
radius_circle = (u[-1]-u[0]) * 0.05
circle = patches.Circle((u0, v0), radius=radius_circle, edgecolor='cyan', facecolor='none', linewidth=2)
axs[0, 0].add_patch(circle)
plt.colorbar(ax=axs[0, 0])

# (b) Show one instance of the binary mask on the pupil (from the last realization).
axs[0, 1].imshow(mask_binary, extent=[x[0], x[-1], y[0], y[-1]], cmap='gray', origin='lower')
axs[0, 1].set_title("Binary Mask (Coherent Region)")
axs[0, 1].set_xlabel("x (m)")
axs[0, 1].set_ylabel("y (m)")
plt.colorbar(ax=axs[0, 1])

# (c) Show one instance of the soft mask on the pupil (from the last realization).
axs[0, 2].imshow(soft_mask, extent=[x[0], x[-1], y[0], y[-1]], cmap='viridis', origin='lower')
axs[0, 2].set_title("Soft Mask on Pupil")
axs[0, 2].set_xlabel("x (m)")
axs[0, 2].set_ylabel("y (m)")
plt.colorbar(ax=axs[0, 2])

# (d) Ensemble-averaged speckle image with Hard Mask modification.
axs[1, 0].imshow(I_hard_avg, extent=[u[0], u[-1], v[0], v[-1]], cmap='inferno', origin='lower')
axs[1, 0].set_title("Hard Mask Modified\nContrast: {:.2f}".format(contrast_hard))
axs[1, 0].set_xlabel("u (spatial freq.)")
axs[1, 0].set_ylabel("v (spatial freq.)")
plt.colorbar(ax=axs[1, 0])

# (e) Ensemble-averaged speckle image with Soft Mask modification.
axs[1, 1].imshow(I_soft_avg, extent=[u[0], u[-1], v[0], v[-1]], cmap='inferno', origin='lower')
axs[1, 1].set_title("Soft Mask Modified\nContrast: {:.2f}".format(contrast_soft))
axs[1, 1].set_xlabel("u (spatial freq.)")
axs[1, 1].set_ylabel("v (spatial freq.)")
plt.colorbar(ax=axs[1, 1])

# (f) Ensemble-averaged speckle image with Waveplate modification.
axs[1, 2].imshow(I_wave_avg, extent=[u[0], u[-1], v[0], v[-1]], cmap='inferno', origin='lower')
axs[1, 2].set_title("Waveplate Modified\nContrast: {:.2f}".format(contrast_wave))
axs[1, 2].set_xlabel("u (spatial freq.)")
axs[1, 2].set_ylabel("v (spatial freq.)")
plt.colorbar(ax=axs[1, 2])

plt.tight_layout()
plt.show()
