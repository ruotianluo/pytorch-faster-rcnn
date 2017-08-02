# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

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

  def _build_network(self):
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

  def forward_prediction(self, mode):
    net_conv = self._layers['head'](self._image)
    self._act_summaries['conv']['value'] = net_conv

    # build the anchors for the image
    self._anchor_component(net_conv.size(2), net_conv.size(3))
   
    rois = self._region_proposal(net_conv)
    if cfg.POOLING_MODE == 'crop':
      pool5 = self._crop_pool_layer(net_conv, rois)
    else:
      pool5 = self._roi_pool_layer(net_conv, rois)

    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.vgg.classifier(pool5_flat)

    cls_prob, bbox_pred = self._region_classification(fc7)
    
    for k in self._predictions.keys():
      self._score_summaries[k]['value'] = self._predictions[k]

    return rois, cls_prob, bbox_pred

  def load_pretrained_cnn(self, state_dict):
    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})