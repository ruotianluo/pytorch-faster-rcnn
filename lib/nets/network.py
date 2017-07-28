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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.timer

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer

from layer_utils.roi_pooling.roi_pool import RoIPoolFunction

from model.config import cfg

class Network(nn.Module):
  def __init__(self, batch_size=1):
    nn.Module.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / 16., ]
    self._batch_size = batch_size
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = {}
    self._event_summaries = {}
    self._image_gt_summaries = {}
    self._variables_to_fix = {}

  def _add_image_summary(self, image, boxes, name):
    # add back mean
    image += cfg.PIXEL_MEANS
    # bgr to rgb (opencv uses bgr)
    channels = tf.unstack (image, axis=-1)
    image    = tf.stack ([channels[2], channels[1], channels[0]], axis=-1)
    # dims for normalization
    width  = tf.to_float(tf.shape(image)[2])
    height = tf.to_float(tf.shape(image)[1])
    # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
    cols = tf.unstack(boxes, axis=1)
    boxes = tf.stack([cols[1] / height,
                      cols[0] / width,
                      cols[3] / height,
                      cols[2] / width], axis=1)
    # add batch dimension (assume batch_size==1)
    assert image.get_shape()[0] == 1
    boxes = tf.expand_dims(boxes, dim=0)
    image = tf.image.draw_bounding_boxes(image, boxes)
    
    return tf.summary.image(name, image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_top_layer(\
                                    rpn_cls_prob.data.cpu().numpy(), rpn_bbox_pred.data.cpu().numpy(), self._im_info,
                                     self._feat_stride, self._anchors, self._num_anchors)
    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_layer(\
                                    rpn_cls_prob.data.cpu().numpy(), rpn_bbox_pred.data.cpu().numpy(), self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors)
    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bottom, rois):
    return RoIPoolFunction(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16.)(bottom, rois)

  def _anchor_target_layer(self, rpn_cls_score):
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      anchor_target_layer(
      rpn_cls_score.data.cpu().numpy(), self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors)

    rpn_labels = Variable(torch.from_numpy(rpn_labels).float().cuda()) #.set_shape([1, 1, None, None])
    rpn_bbox_targets = Variable(torch.from_numpy(rpn_bbox_targets).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_inside_weights = Variable(torch.from_numpy(rpn_bbox_inside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_outside_weights = Variable(torch.from_numpy(rpn_bbox_outside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])

    rpn_labels = rpn_labels.long()
    self._anchor_targets['rpn_labels'] = rpn_labels
    self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
    self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
    self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

    for k in self._anchor_targets.keys():
      self._score_summaries[k]['value'] = self._anchor_targets[k]

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores):
    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
      proposal_target_layer(
      rois, roi_scores, self._gt_boxes, self._num_classes)

    self._proposal_targets['rois'] = Variable(torch.from_numpy(rois).float().cuda())
    self._proposal_targets['labels'] = Variable(torch.from_numpy(labels).long().cuda())
    self._proposal_targets['bbox_targets'] = Variable(torch.from_numpy(bbox_targets).float().cuda())
    self._proposal_targets['bbox_inside_weights'] = Variable(torch.from_numpy(bbox_inside_weights).float().cuda())
    self._proposal_targets['bbox_outside_weights'] = Variable(torch.from_numpy(bbox_outside_weights).float().cuda())

    for k in self._proposal_targets.keys():
      self._score_summaries[k]['value'] = self._proposal_targets[k]

    return rois, roi_scores

  def _anchor_component(self, height, width):
    # just to get the shape right
    #height = int(math.ceil(self._im_info.data[0, 0] / self._feat_stride[0]))
    #width = int(math.ceil(self._im_info.data[0, 1] / self._feat_stride[0]))
    anchors, anchor_length = generate_anchors_pre(\
                                          height, width,
                                           self._feat_stride, self._anchor_scales, self._anchor_ratios)
    self._anchors = anchors
    self._anchor_length = anchor_length

  def build_network(self):
    raise NotImplementedError

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

  def _add_losses(self, sigma_rpn=3.0):
    # RPN, class loss
    rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)
    rpn_label = self._anchor_targets['rpn_labels'].view(-1)
    rpn_select = Variable((rpn_label.data != -1).nonzero().view(-1))
    rpn_cls_score = rpn_cls_score.index_select(0, rpn_select).contiguous().view(-1, 2)
    rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)
    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

    # RPN, bbox loss
    rpn_bbox_pred = self._predictions['rpn_bbox_pred']
    rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
    rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
    rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

    rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

    # RCNN, class loss
    cls_score = self._predictions["cls_score"]
    label = self._proposal_targets["labels"].view(-1)

    cross_entropy = F.cross_entropy(cls_score.view(-1, self._num_classes), label)

    # RCNN, bbox loss
    bbox_pred = self._predictions['bbox_pred']
    bbox_targets = self._proposal_targets['bbox_targets']
    bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
    bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

    loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    self._losses['cross_entropy'] = cross_entropy
    self._losses['loss_box'] = loss_box
    self._losses['rpn_cross_entropy'] = rpn_cross_entropy
    self._losses['rpn_loss_box'] = rpn_loss_box

    loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
    self._losses['total_loss'] = loss

    for k in self._losses.keys():
      self._event_summaries[k]['value'] = self._losses[k]

    return loss

  def create_architecture(self, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._tag = tag

    self._num_classes = num_classes
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    assert tag != None

    # Initialize layers
    self.build_network()
    self.init_summary_op()

  def init_summary_op(self):
    """
    Handle summaries Notes:
    Here we still use original tensorflow tensorboard to do summary.
    The way we send our result to summary, is we create placeholders for the values that needs summarized, and
    created summary operators of these placeholders
    Then during forwarding, we save the values.
    To send it to the tensorboard, we run the summary operator by feeding the placeholders with 
    the saved values and get the summary.

    """

    # Here we first create placeholders, and create summary operation.

    # Manually add losses to event_summaries
    for key in ['cross_entropy','loss_box','rpn_cross_entropy','rpn_loss_box','total_loss']:
      self._event_summaries[key] = {'placeholder': tf.placeholder(tf.float32, shape=(), name=key)}

    # Manually add losses to score_summaries
    score_summaries_keys = []
    # _anchor_targets
    score_summaries_keys += ['rpn_labels','rpn_bbox_targets', 'rpn_bbox_inside_weights', 'rpn_bbox_outside_weights']
    #_proposal_targets
    score_summaries_keys += ['rois', 'labels', 'bbox_targets', 'bbox_inside_weights', 'bbox_outside_weights']
    #_predictions
    score_summaries_keys += ["rpn_cls_score", "rpn_cls_score_reshape", "rpn_cls_prob", "rpn_bbox_pred", \
                "cls_score", "cls_prob", "bbox_pred", "rois"]
    for key in score_summaries_keys:
      self._score_summaries[key] = {'placeholder': tf.placeholder(tf.float32, name=key)}

    # Manually add act_summaries
    self._act_summaries = {'conv':{'placeholder': tf.placeholder(tf.float32, name='conv')}, 
                            'rpn':{'placeholder': tf.placeholder(tf.float32, name='rpn')}}


    self._image_gt_summaries = {'image':{'placeholder': tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])},\
                                'gt_boxes':{'placeholder': tf.placeholder(tf.float32, shape=[None, 5])}}

    # Add train summaries
    for k, var in dict(self.named_parameters()).items():
      if var.requires_grad:
        self._train_summaries[k] = {'placeholder': tf.placeholder(tf.float32, name=k)}

    val_summaries = []
    with tf.device("/cpu:0"):
      val_summaries.append(self._add_image_summary(self._image_gt_summaries['image']['placeholder'], 
        self._image_gt_summaries['gt_boxes']['placeholder'], 'ground truth'))
      for key, var in self._event_summaries.items():
        val_summaries.append(tf.summary.scalar(key, var['placeholder']))
      for key, var in self._score_summaries.items():
        self._add_score_summary(key, var['placeholder'])
      for var in self._act_summaries.values():
        self._add_act_summary(var['placeholder'])
      for var in self._train_summaries.values():
        self._add_train_summary(var['placeholder'])

    self._summary_op = tf.summary.merge_all()
    self._summary_op_val = tf.summary.merge(val_summaries)


  def run_summary_op(self, sess, val=False):
    """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
    def delete_summaries_values(d):
      # Delete the saved values to save memory, in case we have references of these computational graphs.
      for _ in d.values(): del _['value']

    feed_dict = {}
    # Add image gt
    feed_dict.update({_['placeholder']:_['value'] for _ in self._image_gt_summaries.values()})
    delete_summaries_values(self._image_gt_summaries)
    # Add event_summaries
    feed_dict.update({_['placeholder']:_['value'].data[0] for _ in self._event_summaries.values()})
    delete_summaries_values(self._event_summaries)
    if not val:
      # Add score summaries
      feed_dict.update({_['placeholder']:_['value'].data.cpu().numpy() for _ in self._score_summaries.values()})
      delete_summaries_values(self._score_summaries)
      # Add act summaries
      feed_dict.update({_['placeholder']:_['value'].data.cpu().numpy() for _ in self._act_summaries.values()})
      delete_summaries_values(self._act_summaries)
      # Add train summaries
      for k, var in dict(self.named_parameters()).items():
        if var.requires_grad:
          feed_dict.update({self._train_summaries[k]['placeholder']: var.data.cpu().numpy()})
      return sess.run(self._summary_op, feed_dict=feed_dict)
    else:
      return sess.run(self._summary_op_val, feed_dict=feed_dict)

  def forward(self, image, im_info, gt_boxes=None, mode='TRAIN'):
    self._image_gt_summaries['image']['value'] = image
    self._image_gt_summaries['gt_boxes']['value'] = gt_boxes

    self._image = Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=mode == 'TEST')
    self._im_info = im_info # no nned to conver to variable
    self._gt_boxes = gt_boxes # no need to convert to variable

    self._mode = mode

    rois, cls_prob, bbox_pred = self.forward_prediction(mode)

    if mode == 'TEST':
      stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      self._predictions["bbox_pred"] = bbox_pred.mul(Variable(stds)).add(Variable(means))
    else:
      self._add_losses() # compute losses

  def init_weights(self):
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev)
      
    normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.cls_score_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
    normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, image):
    feat = self._layers["head"](Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=True))
    return feat

  # only useful during testing mode
  def test_image(self, image, im_info):
    self.eval()
    self.forward(image, im_info, None, mode='TEST')
    cls_score, cls_prob, bbox_pred, rois = self._predictions["cls_score"].data.cpu().numpy(), \
                                                     self._predictions['cls_prob'].data.cpu().numpy(), \
                                                     self._predictions['bbox_pred'].data.cpu().numpy(), \
                                                     self._predictions['rois'].data.cpu().numpy()
    return cls_score, cls_prob, bbox_pred, rois

  def delete_intermediate_states(self):
    # Delete intermediate result to save memory
    for d in [self._losses, self._predictions, self._anchor_targets, self._proposal_targets]:
      for k in d.keys():
        del d[k]

  def get_summary(self, sess, blobs):
    self.eval()
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
    self.train()
    summary = self.run_summary_op(sess, True)

    return summary

  def train_step(self, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].data[0], \
                                                                        self._losses['rpn_loss_box'].data[0], \
                                                                        self._losses['cross_entropy'].data[0], \
                                                                        self._losses['loss_box'].data[0], \
                                                                        self._losses['total_loss'].data[0]
    #utils.timer.timer.tic('backward')
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    #utils.timer.timer.toc('backward')
    train_op.step()

    self.delete_intermediate_states()

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  def train_step_with_summary(self, sess, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].data[0], \
                                                                        self._losses['rpn_loss_box'].data[0], \
                                                                        self._losses['cross_entropy'].data[0], \
                                                                        self._losses['loss_box'].data[0], \
                                                                        self._losses['total_loss'].data[0]
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    summary = self.run_summary_op(sess)

    self.delete_intermediate_states()

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

  def train_step_no_return(self, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    self.delete_intermediate_states()

