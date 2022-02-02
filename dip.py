# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

from DrawingInterface import DrawingInterface

import sys
import subprocess
import os
import os.path
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
import math
from torchvision.utils import save_image
from util import wget_file
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
from pathlib import Path

import gc
import math
import random
import sys
import time

import cv2
from einops import rearrange
import imageio
from IPython import display
import kornia.augmentation as K
from madgrad import MADGRAD
import numpy as np
import torch.optim
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.utils import save_image

# sys.path.append('./deep-image-prior')

from DIP.models import *
from DIP.utils.sr_utils import *

print(get_net)

class DecorrelatedColorsToRGB(nn.Module):
    """From https://github.com/eps696/aphantasia."""

    def __init__(self, inv_color_scale=1.):
        super().__init__()
        color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
        color_correlation_svd_sqrt /= torch.tensor([inv_color_scale, 1., 1.])  # saturate, empirical
        max_norm_svd_sqrt = color_correlation_svd_sqrt.norm(dim=0).max()
        color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
        self.register_buffer('colcorr_t', color_correlation_normalized.T)

    def inverse(self, image):
        colcorr_t_inv = torch.linalg.inv(self.colcorr_t)
        return torch.einsum('nchw,cd->ndhw', image, colcorr_t_inv)

    def forward(self, image):
        return torch.einsum('nchw,cd->ndhw', image, self.colcorr_t)


class CaptureOutput:
    """Captures a layer's output activations using a forward hook."""

    def __init__(self, module):
        self.output = None
        self.handle = module.register_forward_hook(self)

    def __call__(self, module, input, output):
        self.output = output

    def __del__(self):
        self.handle.remove()

    def get_output(self):
        return self.output


class CLIPActivationLoss(nn.Module):
    """Maximizes or minimizes a single neuron's activations."""

    def __init__(self, module, neuron, class_token=False, maximize=True):
        super().__init__()
        self.capture = CaptureOutput(module)
        self.neuron = neuron
        self.class_token = class_token
        self.maximize = maximize

    def forward(self):
        activations = self.capture.get_output()
        if self.class_token:
            loss = activations[0, :, self.neuron].mean()
        else:
            loss = activations[1:, :, self.neuron].mean()
        return -loss if self.maximize else loss


def optimize_network(num_iterations, optimizer_type, lr):
    global itt
    itt = 0

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    make_cutouts = MakeCutouts(clip_models[clip_model].visual.input_resolution, cutn)
    loss_fn = CLIPActivationLoss(clip_models[clip_model].visual.transformer.resblocks[layer],
                                 neuron, class_token, maximize)

    # Initialize DIP skip network
    input_depth = 32
    net = get_net(
        input_depth, 'skip',
        pad='reflection',
        skip_n33d=128, skip_n33u=128,
        skip_n11=4, num_scales=7,  # If you decrease the output size to 256x256 you might want to use num_scales=6
        upsample_mode='bilinear',
        downsample_mode='lanczos2',
    )
    net = get_hq_skip_net(input_depth, offset_type='full')

    # Modify DIP to operate in a decorrelated color space
    net = net[:-1]  # remove the sigmoid at the end
    net.add(DecorrelatedColorsToRGB(inv_color_scale))
    net.add(nn.Sigmoid())

    net = net.to(device)

    # Initialize input noise
    net_input = torch.zeros([1, input_depth, sideY, sideX], device=device).normal_().div(10).detach()

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr)
    elif optimizer_type == 'MADGRAD':
        optimizer = MADGRAD(net.parameters(), lr, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    try:
        for _ in range(num_iterations):
            optimizer.zero_grad(set_to_none=True)
    
            with torch.cuda.amp.autocast():
                out = net(net_input).float()
            cutouts = make_cutouts(out)
            image_embeds = clip_models[clip_model].encode_image(clip_normalize(cutouts))
            loss = loss_fn()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            itt += 1

            if itt % display_rate == 0 or save_progress_video:
                with torch.inference_mode():
                    image = TF.to_pil_image(out[0].clamp(0, 1))
                    if itt % display_rate == 0:
                        display.clear_output(wait=True)
                        display.display(image)
                        if display_augs:
                            aug_grid = torchvision.utils.make_grid(cutouts, nrow=math.ceil(math.sqrt(cutn)))
                            display.display(TF.to_pil_image(aug_grid.clamp(0, 1)))
                    if save_progress_video and itt > 15:
                        video_writer.append_data(np.asarray(image))

            if anneal_lr:
                optimizer.param_groups[0]['lr'] = max(0.00001, .99 * optimizer.param_groups[0]['lr'])

            print(f'Iteration {itt} of {num_iterations}, loss: {loss.item():g}')
    
    except KeyboardInterrupt:
        pass

    return TF.to_pil_image(net(net_input)[0])



class DIPDrawer(DrawingInterface):
    @staticmethod
    def add_settings(parser):
        # parser.add_argument("--vdiff_model", type=str, help="VDIFF model from [yfcc_2, yfcc_1, cc12m_1,cc12m_1_cfg]", default='yfcc_2', dest='vdiff_model')
        # parser.add_argument("--vdiff_init_skip", type=float, help="skip steps (step power) when init", default=0.9, dest='vdiff_init_skip')
        # parser.add_argument("--vqgan_config", type=str, help="VQGAN config", default=None, dest='vqgan_config')
        # parser.add_argument("--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=None, dest='vqgan_checkpoint')
        parser.add_argument("--input_opt", type=bool,  default=False, dest='input_opt')
        parser.add_argument("--input_depth", type=int,  default=32, dest='input_depth')
        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()
        os.makedirs("models",exist_ok=True)
        self.canvas_width = settings.size[0]
        self.canvas_height = settings.size[1]
        self.lr = settings.learning_rate
        self.input_opt = settings.input_opt
        self.input_depth = settings.input_depth


    def load_model(self, settings, device):
        # Initialize DIP skip network
        inv_color_scale = 1.6
        input_depth = self.input_depth
        print(inv_color_scale,input_depth)
        net = get_hq_skip_net(input_depth, offset_type='full',decorr_rgb=False)

        self.input_depth = input_depth

        # Modify DIP to operate in a decorrelated color space
        # net = net[:-1]  # remove the sigmoid at the end
        # net.add(DecorrelatedColorsToRGB(inv_color_scale))
        # net.add(nn.Sigmoid())

        self.net = net.to(device)
        self.device = device


    def get_opts(self, decay_divisor):
        params = [{'params': get_non_offset_params(self.net), 'lr': self.lr},
                {'params': get_offset_params(self.net), 'lr': self.lr / 10}]
        if self.input_opt:
            opt_in = torch.optim.Adam([self.net_input], lr=self.lr)
            opt_out = torch.optim.Adam(params)
            self.opts = [opt_in,opt_out]
        else:
            self.opts = [torch.optim.Adam(params)]
        return self.opts

    def rand_init(self, toksX, toksY):
        # legacy init
        return None

    def init_from_tensor(self, init_tensor):
        self.net_input = torch.zeros([1, self.input_depth,  self.canvas_height,  self.canvas_width], device=self.device).normal_().div(10).detach()

    def reapply_from_tensor(self, new_tensor):
        return None

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        return None

    def synth(self, cur_iteration):
        out = self.net(self.net_input).float()
        save_image(out,f"tmpim/this{cur_iteration}.png")
        return out

    @torch.no_grad()
    def to_image(self):
        out = self.synth(None)
        return TF.to_pil_image(out[0].cpu())

    def clip_z(self):
        return None

    def get_z(self):
        return None

    def set_z(self, new_z):
        with torch.no_grad():
            # return self.net.parameters().copy_(new_z)
            return torch.tensor(0.0)

    def get_z_copy(self):
        # return self.net.parameters().clone()
        return torch.tensor(0.0)
