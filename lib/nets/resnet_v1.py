# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np

from nets.network import Network
from model.config import cfg

import utils.timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnetv1(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._num_layers = num_layers

  def build_network(self):
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      self.resnet = resnet50()

    elif self._num_layers == 101:
      self.resnet = resnet101()

    elif self._num_layers == 152:
      self.resnet = resnet152()

    else:
      # other numbers are not supported
      raise NotImplementedError

    # Fix blocks 
    for p in self.resnet.bn1.parameters(): p.requires_grad=False
    for p in self.resnet.conv1.parameters(): p.requires_grad=False
    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.resnet.layer3.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.resnet.layer2.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.resnet.layer1.parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    if not cfg.RESNET.BN_TRAIN:
      self.resnet.apply(set_bn_fix)

    # Build resnet.
    self._layers['head'] = nn.Sequential(self.resnet.conv1, self.resnet.bn1,self.resnet.relu, 
      self.resnet.maxpool,self.resnet.layer1,self.resnet.layer2,self.resnet.layer3)

    # rpn
    self.rpn_net = nn.Conv2d(1024, 512, [3, 3], padding=1)

    self.rpn_cls_score_net = nn.Conv2d(512, self._num_anchors * 2, [1, 1])
    
    self.rpn_bbox_pred_net = nn.Conv2d(512, self._num_anchors * 4, [1, 1])

    self.cls_score_net = nn.Linear(2048, self._num_classes)
    self.bbox_pred_net = nn.Linear(2048, self._num_classes * 4)

    self.init_weights()

  def forward_prediction(self, mode):
    # Set fixed blocks to be in eval mode
    self.resnet.eval()
    self.resnet.layer1.train()
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      self.resnet.layer2.train()
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      self.resnet.layer3.train()
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      self.resnet.layer4.train()

    def set_bn_eval(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        m.eval()

    if not cfg.RESNET.BN_TRAIN:
      self.resnet.apply(set_bn_eval)

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
    pool5 = self._roi_pool_layer(net_conv, rois)

    fc7 = self.resnet.layer4(pool5).mean(3).squeeze(3).mean(2).squeeze(2) # average pooling after layer4

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
    self.resnet.load_state_dict(state_dict)
