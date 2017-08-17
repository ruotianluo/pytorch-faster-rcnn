import numpy as np
import torch
import time

def get_random_boxes(num):
  dets = torch.randn(num,4)

  new_dets = dets.clone()
  new_dets[:, 0] = torch.min(dets[:, 0], dets[:, 2])
  new_dets[:, 1] = torch.min(dets[:, 1], dets[:, 3])
  new_dets[:, 2] = torch.max(dets[:, 0], dets[:, 2])
  new_dets[:, 3] = torch.max(dets[:, 1], dets[:, 3])

  return new_dets.double().numpy()

def test(fn):
  total = 0
  for i in range(1000):
    boxes, query_boxes = get_random_boxes(10000), get_random_boxes(20)
    start = time.time()
    fn(boxes, query_boxes)
    total += time.time() - start
  return total


import cython_bbox
import bbox

print 'cython:', test(cython_bbox.bbox_overlaps)
#print 'torch:', test(bbox.bbox_overlaps)
