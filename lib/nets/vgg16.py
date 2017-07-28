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

  def build_network(self):
    self.vgg = models.vgg16()
    # Remove fc8
    self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.vgg.features[layer].parameters(): p.requires_grad = False

    self._layers['head'] = self.vgg.features

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

    rpn = F.relu(self.rpn_net(net_conv))
    self._act_summaries['rpn']['value'] = rpn

    rpn_bbox_pred = self.rpn_bbox_pred_net(rpn) 
    rpn_cls_score = self.rpn_cls_score_net(rpn) # batch * (num_anchors * 2) * h * w

    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = rpn_cls_score.view(self._batch_size, 2, -1, rpn_cls_score.size()[-1]) # batch * 2 * (num_anchors*h) * w
    rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
    
    # Move channel to the last dimenstion, to fit the input of python functions
    rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score_reshape = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2
    rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

    if mode == 'TRAIN':
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred) # rois, roi_scores are numpy
      rpn_labels = self._anchor_target_layer(rpn_cls_score)
      rois, _ = self._proposal_target_layer(rois, roi_scores)
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
      else:
        raise NotImplementedError

    rois = Variable(torch.from_numpy(rois).float().cuda())
    # rcnn
    pool5 = self._roi_pool_layer(net_conv, rois)

    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.vgg.classifier(pool5_flat)

    cls_score = self.cls_score_net(fc7)
    cls_prob = F.softmax(cls_score)
    bbox_pred = self.bbox_pred_net(fc7)

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["cls_score"] = cls_score
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred
    self._predictions["rois"] = rois

    for k in self._predictions.keys():
      self._score_summaries[k]['value'] = self._predictions[k]

    return rois, cls_prob, bbox_pred

  def load_pretrained_cnn(self, state_dict):
    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})