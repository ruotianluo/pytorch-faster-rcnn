# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network import Network
from model.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models

class vgg16(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)

  def _init_modules(self):
    self.vgg = models.vgg16()
    # Remove fc8
    self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.vgg.features[layer].parameters(): p.requires_grad = False

    # not using the last maxpool layer
    self._layers['head'] = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])

    # rpn
    self.rpn_net = nn.Conv2d(512, 512, [3, 3], padding=1)

    self.rpn_cls_score_net = nn.Conv2d(512, self._num_anchors * 2, [1, 1])
    
    self.rpn_bbox_pred_net = nn.Conv2d(512, self._num_anchors * 4, [1, 1])

    self.cls_score_net = nn.Linear(4096, self._num_classes)
    self.bbox_pred_net = nn.Linear(4096, self._num_classes * 4)

    self.init_weights()

  def _image_to_head(self):
    net_conv = self._layers['head'](self._image)
    self._act_summaries['conv'] = net_conv
    
    return net_conv

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.vgg.classifier(pool5_flat)

    return fc7

  def load_pretrained_cnn(self, state_dict):
    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})