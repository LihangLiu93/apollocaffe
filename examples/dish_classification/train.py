"""train.py is used to generate and train the
ReInspect deep network architecture."""

import numpy as np
from numpy import unravel_index
import json
import os
import cv2
import random
import operator
from scipy.misc import imread
import matplotlib.pyplot as plt
import caffe
import apollocaffe
from apollocaffe.models import alexnet
from apollocaffe.layers import (Power, LstmUnit, Convolution, NumpyData,
                                Transpose, Filler, SoftmaxWithLoss, Accuracy,
                                Softmax, Concat, Dropout, InnerProduct)

from utils import (image_jitter, image_to_h5, load_image_mean_from_binproto)

def get_max_index(lst):
    index, value = max(enumerate(lst), key=operator.itemgetter(1))
    return index, value

def get_accuracy(net, input_en):
    probs = forward(net, input_en, deploy=True)
    labels = input_en['label']
    right = 0
    for batch_id in range(probs.shape[0]):
        index, value = get_max_index(probs[batch_id,:])
        if index == labels[batch_id]:                            
            right += 1
    return right/float(probs.shape[0])

def load_txt(txtfile, net_config, data_mean, jitter=True):
    """Take the idlfile, data mean and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""
    
    lines = open(txtfile, 'r').readlines()

    img_name_batch = []
    raw_image_batch = np.zeros((net_config['batch_size'], 
        net_config['new_width'], net_config['new_height'], 3))
    image_batch = np.zeros((net_config['batch_size'], 3, 
        net_config['crop_size'], net_config['crop_size']))
    label_batch = np.zeros((net_config['batch_size'], 1))
    cnt = 0
    while True:
        random.shuffle(lines)
        for line in lines:
            line_segs = line.strip().split()
            img_name = ' '.join(line_segs[:-1])
            img_label = int(line_segs[-1])
            # print img_name, img_label
            if jitter:
                jit_image = image_jitter(img_name,
                    target_width=net_config['new_width'], target_height=net_config['new_height'])
                # plt.imshow(jit_image)
                # plt.show()
            else:
                jit_image = imread(img_name)
                jit_image = cv2.resize(jit_image, (net_config['new_width'], net_config['new_height']))
            image = image_to_h5(jit_image, data_mean, crop_size=net_config['crop_size'], image_scaling=1.0)

            img_name_batch.append(img_name)
            raw_image_batch[cnt, :, :, :] = jit_image
            image_batch[cnt, :, :, :] = image
            label_batch[cnt,:] = img_label
            cnt += 1
            if cnt == net_config['batch_size']:
                yield {"imname": img_name_batch, "raw": raw_image_batch, "image": image_batch,
                   "label": label_batch}
                img_name_batch = []
                cnt = 0

def generate_decapitated_alexnet(net):
    """Generates the googlenet layers until the inception_5b/output.
    The output feature map is then used to feed into the lstm layers."""

    alex_layers = alexnet.alexnet_layers()
    alex_layers[0].p.bottom[0] = "image"
    for layer in alex_layers:
        if layer.p.name == "fc8":
            break
        net.f(layer)

def forward(net, input_data, deploy=False):
    """Defines and creates the ReInspect network given the net, input data
    and configurations."""

    net.clear_forward()
    if deploy:
        image = np.array(input_data["image"])
    else:
        image = np.array(input_data["image"])
        label = np.array(input_data["label"])
        net.f(NumpyData("label", data=label))

    net.f(NumpyData("image", data=image))
    generate_decapitated_alexnet(net)
    net.f(InnerProduct(name="fc8_dish", bottoms=["fc7"], param_lr_mults=[1.0*10, 2.0*10],
            param_decay_mults=[1.0, 0.0],
            weight_filler=Filler("gaussian", 0.01),
            bias_filler=Filler("constant", 0.0), num_output=128))
            
    net.f(Softmax("dish_probs", bottoms=["fc8_dish"]))

    if not deploy:
        net.f(SoftmaxWithLoss(name="loss", bottoms=["fc8_dish", "label"]))
        # net.f(Accuracy(name="dish_accuracy",bottoms=["fc8_dish_23", "label"]))
        
    if deploy:
        probs = np.array(net.blobs["dish_probs"].data) 
        return probs
    else:
        return None

def train(config):
    """Trains the ReInspect model using SGD with momentum
    and prints out the logging information."""

    net = apollocaffe.ApolloNet()

    data_config = config["data"]
    solver = config["solver"]
    logging = config["logging"]
    net_config = config['net']

    image_mean = load_image_mean_from_binproto(data_config["idl_mean"])

    input_gen = load_txt(data_config["train_idl"], net_config, image_mean)
    input_gen_test = load_txt(data_config["test_idl"], net_config, image_mean, jitter=False)

    forward(net, input_gen.next())
    net.draw_to_file(logging["schematic_path"])

    if solver["weights"]:
        net.load(solver["weights"])
    else:
        net.load(alexnet.weights_file())

    loss_hist = {"train": [], "test": []}
    loggers = [
        apollocaffe.loggers.TrainLogger(logging["display_interval"],
                                        logging["log_file"]),
        apollocaffe.loggers.TestLogger(solver["test_interval"],
                                       logging["log_file"]),
        apollocaffe.loggers.SnapshotLogger(logging["snapshot_interval"],
                                           logging["snapshot_prefix"]),
        ]
    for i in range(solver["start_iter"], solver["max_iter"]):
        if i % solver["test_interval"] == 0:
            net.phase = 'test'
            test_loss = []
            test_accuracy = []
            for _ in range(solver["test_iter"]):
                input_en = input_gen_test.next()
                forward(net, input_en, False)
                test_loss.append(net.loss)
                test_accuracy.append(get_accuracy(net, input_en))
            loss_hist["test"].append(np.mean(test_loss))
            print 'accuracy', np.mean(test_accuracy)
            net.phase = 'train'
        forward(net, input_gen.next())
        loss_hist["train"].append(net.loss)
        net.backward()
        learning_rate = (solver["base_lr"] *
                         (solver["gamma"])**(i // solver["stepsize"]))
        net.update(lr=learning_rate, momentum=solver["momentum"])
        for logger in loggers:
            logger.log(i, {'train_loss': loss_hist["train"],
                           'test_loss': loss_hist["test"],
                           'apollo_net': net, 'start_iter': 0})

def main():
    """Sets up all the configurations for apollocaffe, and ReInspect
    and runs the trainer."""
    parser = apollocaffe.base_parser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    if args.weights is not None:
        config["solver"]["weights"] = args.weights
    config["solver"]["start_iter"] = args.start_iter
    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    train(config)

if __name__ == "__main__":
    main()


# python train.py --gpu=0 --config=config.json --weights=/home/apexgpu/Desktop/caffe/caffe/DownloadedModels/bvlc_alexnet/bvlc_alexnet.caffemodel
# python train.py --gpu=0 --config=config.json --weights=./tmp/reinspect_20000.h5 
