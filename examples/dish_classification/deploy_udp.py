#!/usr/bin/python  
#coding=gbk


import json
import cv2
import os
import random
import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import caffe
import apollocaffe
import time
from utils import (image_jitter, image_to_h5, load_image_mean_from_binproto)
from train import forward, get_max_index
import socket

# # # #  interface


def classify(net, impath, regions_list, image_mean):
    im = imread(impath)
    batch_size = len(regions_list)/4
    image_batch = np.zeros((batch_size, 3, 227,227))
    for i in range(batch_size):
        x0 = int(regions_list[i*4+0])
        y0 = int(regions_list[i*4+1])
        x1 = int(regions_list[i*4+2])
        y1 = int(regions_list[i*4+3])
        sub_image = im[x0:x1,y0:y1,:]
        jit_image = cv2.resize(sub_image, (256, 256))
        jit_image = jit_image[256/2-227/2:227+(256/2-227/2),256/2-227/2:227+(256/2-227/2),:]
        image = image_to_h5(jit_image, image_mean, crop_size=227, image_scaling=1.0)
        image_batch[i,:,:,:] = image
    input_en = {"image":image_batch}
    probs = forward(net, input_en, deploy=True)
    res = ''
    for i in range(probs.shape[0]):
        index, value = get_max_index(probs[i,:])
        res += "%d %.03f"%(index, value)
    return res

# # # #

def run_socket(net, port, image_mean):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
    sock.bind(('', port))       #       
    print 'socket setup. port: %d' % port

    while True:  
        revcData, (remoteHost, remotePort) = sock.recvfrom(1024)  
        print("[%s:%s] connect" % (remoteHost, remotePort))     # 
        print "revcData: ", revcData  

        recv_splits = revcData.strip().split()
        impath = recv_splits[0]
        regions_list = recv_splits[1:]
        if os.path.exists(impath) and impath[-4:]=='.jpg':
            sendstr = classify(net, impath, regions_list, image_mean)
        else:
            sendstr = 'unknown path %s'% impath 
          
        sock.sendto(sendstr, (remoteHost, remotePort))  
        print "sendData: ", sendstr  
          
    sock.close()  


        

def main():
    """Sets up all the configurations for apollocaffe, and ReInspect
    and runs the trainer."""
    parser = apollocaffe.base_parser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    if args.weights is not None:
        config["solver"]["weights"] = args.weights
    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    net = apollocaffe.ApolloNet()
    image_mean = load_image_mean_from_binproto(config['data']["idl_mean"])
    fake_input_en = {"image": 
                                np.zeros((1,3,227, 227))}

    forward(net, fake_input_en, deploy=True)

    if config["solver"]["weights"]:
        net.load(config["solver"]["weights"])
    else:
        raise Exception('weights file is not provided!')

    run_socket(net, 13502, image_mean)

if __name__ == "__main__":
    main()


# python deploy_udp.py --gpu=0 --config=config.json --weights=./tmp/reinspect_20000.h5 