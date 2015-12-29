"""train.py is used to generate and train the
ReInspect deep network architecture."""

import numpy as np
import json
import os
import random
from scipy.misc import imread
import apollocaffe
from apollocaffe.models import googlenet
from apollocaffe.layers import (LstmUnit, NumpyData, Slice, Reshape,
                                Transpose, Filler, SoftmaxWithLoss,
                                Softmax, Concat, Dropout, InnerProduct)


def load_indice_file(file, net_config):
    """Take the idlfile, data mean and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""
    indice_list = []
    with open(file, 'r') as f:
        while True:
            line = f.readline()
            if line:
                indice = line.strip().split()
                indice = [int(i) for i in indice]
                indice_list.append(indice)
            else:
                break
    print file, len(indice_list)
    
    batch_input_indice = np.zeros((net_config["batch_size"], net_config["max_len"], 1, 1))
    batch_output_indice = np.zeros((net_config["batch_size"], net_config["max_len"], 1, 1))
    batch_wordvec_layer = np.zeros((net_config["batch_size"], net_config["vocab_size"], net_config["max_len"], 1))
    batch_id = 0
    while True:
        random.shuffle(indice_list)
        for indice in indice_list:
            output_indice = indice[:net_config['max_len']] + \
                          [net_config['zero_symbol']] * (net_config['max_len']-len(indice[:net_config['max_len']]))
            input_indice = [net_config["start_symbol"]] + output_indice[:-1]

            batch_input_indice[batch_id, :, 0, 0] = input_indice
            batch_output_indice[batch_id, :, 0, 0] = output_indice
            for i in range(net_config["max_len"]):
                ii = input_indice[i]
                vec = [0]*net_config["vocab_size"]
                vec[ii] = 1
                batch_wordvec_layer[batch_id, :, i, 0] = vec                
            
            if batch_id == net_config["batch_size"]-1:
                yield {"input_words":batch_input_indice, "target_words":batch_output_indice, 
                        "wordvec_layer":batch_wordvec_layer}
                batch_id = 0
            else:
                batch_id += 1


def forward(net, input_data, net_config, deploy=False):
    """Defines and creates the ReInspect network given the net, input data
    and configurations."""

    net.clear_forward()           

    net.f(NumpyData("wordvec_layer", data=np.array(input_data["wordvec_layer"])))    # 128*38*100*1
    net.f(NumpyData("target_words", data=np.array(input_data["target_words"])))      # 128*100*1*1

    tops = []
    slice_point = []
    for i in range(net_config['max_len']):
        tops.append('label%d' % i)
        if i != 0:
            slice_point.append(i)
    net.f(Slice("label_slice_layer", slice_dim = 1,
                    bottoms = ["target_words"], tops = tops, slice_point = slice_point))
    
    tops = []
    slice_point = []
    for i in range(net_config['max_len']):
        tops.append('target_wordvec%d_4d' % i)
        if i != 0:
            slice_point.append(i)
    net.f(Slice("wordvec_slice_layer", slice_dim = 2,
                    bottoms = ['wordvec_layer'], tops = tops, slice_point = slice_point))

    for i in range(net_config["max_len"]):              # 128*38*1*1 -> 128*38
        net.f("""
            name: "target_wordvec%d"
            type: "Reshape"
            bottom: "target_wordvec%d_4d"
            top: "target_wordvec%d"
            reshape_param {
              shape {
                dim: 0  # copy the dimension from below
                dim: -1
              }
            }
            """%(i, i,i))
        #net.f(Reshape('target_wordvec%d'%i, bottoms = ['target_wordvec%d_4d'%i], shape = [0,-1]))

    filler = Filler("uniform", net_config["init_range"])
    for i in range(net_config['max_len']):
        if i == 0:
            net.f(NumpyData("dummy_layer", 
                            np.zeros((net_config["batch_size"], net_config["lstm_num_cells"]))))       
            net.f(NumpyData("dummy_mem_cell", 
                            np.zeros((net_config["batch_size"], net_config["lstm_num_cells"]))))

        for j in range(net_config['lstm_num_stacks']):
            bottoms = []
            if j == 0:
                bottoms.append('target_wordvec%d' % i)
            if j >= 1:
                bottoms.append('dropout%d_%d' % (j - 1, i))
            if i == 0:
                bottoms.append("dummy_layer")
            else:
                bottoms.append('lstm%d_hidden%d' % (j, i - 1))
            net.f(Concat('concat%d_layer%d' % (j, i), bottoms = bottoms))

            param_names = []
            for k in range(4):
                param_names.append('lstm%d_param_%d' % (j, k))
            bottoms = ['concat%d_layer%d' % (j, i)]
            if i == 0:
                bottoms.append('dummy_mem_cell')
            else:
                bottoms.append('lstm%d_mem_cell%d' % (j, i - 1))
            net.f(LstmUnit('lstm%d_layer%d' % (j, i), net_config["lstm_num_cells"],
                       weight_filler=filler, 
                       param_names=param_names,
                       bottoms=bottoms,
                       tops=['lstm%d_hidden%d' % (j, i), 'lstm%d_mem_cell%d' % (j, i)]))

            net.f(Dropout('dropout%d_%d' % (j, i), net_config["dropout_ratio"],
                  bottoms=['lstm%d_hidden%d' % (j, i)]))

    bottoms = []
    for i in range(net_config['max_len']):
        bottoms.append('dropout%d_%d' % (net_config['lstm_num_stacks'] - 1, i))
    net.f(Concat('hidden_concat', bottoms = bottoms, concat_dim = 0))

    net.f(InnerProduct("inner_product", net_config['vocab_size'], bottoms=["hidden_concat"], 
                       weight_filler=filler))

    bottoms = []
    for i in range(net_config['max_len']):
        bottoms.append('label%d' % i)
    net.f(Concat('label_concat', bottoms = bottoms, concat_dim = 0))

    if deploy:
        net.f(Softmax("word_probs", bottoms=["inner_product"]))
    else:
        net.f(SoftmaxWithLoss("word_loss",
                          bottoms=["inner_product", "label_concat"], 
                          ignore_label = net_config['zero_symbol']))



def train(config):
    """Trains the ReInspect model using SGD with momentum
    and prints out the logging information."""

    net = apollocaffe.ApolloNet()

    net_config = config["net"]
    data_config = config["data"]
    solver = config["solver"]
    logging = config["logging"]


    input_gen = load_indice_file(data_config["train_file"], net_config) 
    input_gen_test = load_indice_file(data_config["test_file"], net_config)

    forward(net, input_gen.next(), config["net"])
    # net.draw_to_file(logging["schematic_path"])         # !!!!

    if solver["weights"]:
        net.load(solver["weights"])
    # else:
    #     net.load(googlenet.weights_file())

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
            for _ in range(solver["test_iter"]):
                forward(net, input_gen_test.next(), config["net"], False)
                test_loss.append(net.loss)
            loss_hist["test"].append(np.mean(test_loss))
            net.phase = 'train'
        forward(net, input_gen.next(), config["net"])
        loss_hist["train"].append(net.loss)
        net.backward()
        learning_rate = (solver["base_lr"] *
                         (solver["gamma"])**(i // solver["stepsize"]))
        net.update(lr=learning_rate, momentum=solver["momentum"],
                   clip_gradients=solver["clip_gradients"])
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
