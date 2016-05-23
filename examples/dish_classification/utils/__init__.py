import os
import cv2
import re
import sys
import argparse
import numpy as np
import caffe
import apollocaffe
from scipy.misc import imread, imresize, imsave
from munkres import Munkres, print_matrix, make_cost_matrix

def load_image_mean_from_binproto(binproto_path):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( binproto_path, 'rb' ).read()
    blob.ParseFromString(data)
    image_mean = np.array( caffe.io.blobproto_to_array(blob))[0]
    image_mean = np.transpose(image_mean, (1, 2, 0))
    return image_mean

def image_to_h5(I, data_mean, crop_size=227, image_scaling = 1.0):

    # normalization as needed for ipython notebook
    I = I.astype(np.float32) / image_scaling - data_mean
    w = I.shape(0)
    h = I.shape(1)
    I = I[w/2-crop_size/2:crop_size+(w/2-crop_size/2), w/2-crop_size/2:crop_size+(w/2-crop_size/2), :]

    # MA: model expects BGR ordering
    I = I[:, :, (2, 1, 0)]

    data_shape = (1, I.shape[2], I.shape[0], I.shape[1])
    h5_image = np.transpose(I, (2,0,1)).reshape(data_shape) 
    return h5_image


def image_jitter(img_name, jitter_scale_min=0.9, jitter_scale_max=1.1, jitter_offset=16, jitter_shift=128, target_width=256, target_height=256):

    jitter_scale = np.random.uniform(jitter_scale_min, jitter_scale_max)

    jitter_flip = np.random.random_integers(0, 1)

    I = imread(img_name)
    I = cv2.resize(I, (256, 256))

    if jitter_flip == 1:
        I = np.fliplr(I)

    I1 = cv2.resize(I, None, fx=jitter_scale, fy=jitter_scale, interpolation = cv2.INTER_CUBIC)

    jitter_offset_x = np.random.random_integers(-jitter_offset, jitter_offset)
    jitter_offset_y = np.random.random_integers(-jitter_offset, jitter_offset)

    rescaled_width = I1.shape[1]
    rescaled_height = I1.shape[0]

    px = round(0.5*(target_width)) - round(0.5*(rescaled_width)) + jitter_offset_x
    py = round(0.5*(target_height)) - round(0.5*(rescaled_height)) + jitter_offset_y

    I2 = np.zeros((target_height, target_width, 3), dtype=I1.dtype)

    x1 = max(0, px)
    y1 = max(0, py)
    x2 = min(rescaled_width, target_width - x1)
    y2 = min(rescaled_height, target_height - y1)

    I2[0:(y2 - y1), 0:(x2 - x1), :] = I1[y1:y2, x1:x2, :]

    return I2
