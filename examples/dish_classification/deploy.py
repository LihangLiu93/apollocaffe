#!/usr/bin/python  
#coding=gbk


import json
import cv2
import os
import random
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import caffe
import apollocaffe
import time
from utils import (image_jitter, image_to_h5, load_image_mean_from_binproto)
from train import forward, get_max_index

# # # #  interface

class ClassNet:

def setup(config, device_gpu):
    apollocaffe.set_device(device_gpu)
    net = apollocaffe.ApolloNet()

    image_mean = load_image_mean_from_binproto(config["idl_mean"])
    fake_input_en = {"image": 
                                np.zeros((config['new_width'], config['new_height']))}

    forward(net, fake_input_en, deploy=True)
    net.draw_to_file(logging["schematic_path"])

    if solver["weights"]:
        net.load(config["weights"])
    else:
        raise Exception('weights file is not provided!')

    return net

def classify(net, im, config):
    image_mean = load_image_mean_from_binproto(config["idl_mean"])
    jit_image = cv2.resize(im, (config['new_width'], config['new_height']))
    image = image_to_h5(jit_image, data_mean, crop_size=config['crop_size'], image_scaling=1.0)
    input_en = {"image":image}
    probs = forward(net, input_en, deploy=True)
    pred_class, value = get_max_index(probs[0,:])
    return pred_class, value

# # # #

def load_txt(txtfile, net_config, data_mean):
    lines = open(txtfile, 'r').readlines()
    for line in lines:
        img_name = line.strip().split()[0]
        img_name = 'dishes_315/' + img_name
        jit_image = imread(img_name)
        jit_image = cv2.resize(jit_image, (net_config['new_width'], net_config['new_height']))
        image = image_to_h5(jit_image, data_mean, crop_size=net_config['crop_size'], image_scaling=1.0)
        yield {"imname": img_name, "raw": jit_image, "image": image}

def save_image(img, impath, imname):
    if not os.path.exists(impath):
        os.makedirs(impath)
    imsave(os.path.join(impath,imname), img)

def deploy(config):
    """Trains the ReInspect model using SGD with momentum
    and prints out the logging information."""

    net = apollocaffe.ApolloNet()

    data_config = config["data"]
    solver = config["solver"]
    logging = config["logging"]
    net_config = config['net']

    image_mean = load_image_mean_from_binproto(data_config["idl_mean"])

    input_gen = load_txt(data_config["deploy_idl"], net_config, image_mean)
    classes_names = [str(name.strip()) for name in open(logging['classes_file'])]
    print classes_names

    forward(net, input_gen.next(), deploy=True)
    net.draw_to_file(logging["schematic_path"])

    if solver["weights"]:
        net.load(solver["weights"])
    else:
        raise Exception('weights file is not provided!')

    for input_en in input_gen:
            start = time.time()
            net.phase = 'test'
            test_accuracy = []
            probs = forward(net, input_en, deploy=True)
            pred_class, value = get_max_index(probs[0,:])
            print input_en['imname'], pred_class
            # plt.imshow(input_en['raw'])
            # plt.show()
            if value < 0.5:
                class_name = 'unknown'
            else:
                class_name = classes_names[pred_class]

            save_image(input_en['raw'], 
            		os.path.join(logging['deploy_out_dir'], class_name), 
            		input_en['imname'].split('/')[-1])
            # print time.time()-start
        

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

    deploy(config)

if __name__ == "__main__":
    main()


# python deploy.py --gpu=0 --config=config.json --weights=./tmp/reinspect_20000.h5 