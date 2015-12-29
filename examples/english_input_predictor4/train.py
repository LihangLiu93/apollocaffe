"""train.py is used to generate and train the
ReInspect deep network architecture."""

import numpy as np
import json
import os
import random
import pickle
from scipy.misc import imread
import apollocaffe
from apollocaffe.models import googlenet
from apollocaffe.layers import (LstmUnit, NumpyData, Slice, Reshape,
                                Transpose, Filler, SoftmaxWithLoss,
                                Softmax, Concat, Dropout, InnerProduct)


# # # # # # #  test accuracy   # # # # #

vocab = pickle.load(open(r'/home/pig/apollocaffe-master/data/english_input_predictor/vocab.pkl', 'rb'))
inv = lambda d: {v:k for k,v in d.iteritems()}
vocab_inv = inv(vocab)

def indice_2_string(indice, net_config):
    string = ''
    for i in indice:
        i = int(i)
        if i == net_config['unknown_symbol']:
            string += '#'
        elif i == net_config['start_symbol']:
            string += '$'
        elif i == net_config['zero_symbol']:
            string += ' '
        else:
            string += vocab_inv[i]
    return string

def get_max_indice(probs, net_config):
    indice = []
    for i in range(net_config['max_len']):
        peakIndex = np.argmax(probs[i,:])
        indice.append(peakIndex)
    return indice

def if_the_same(pred_indice, target_indice, net_config):
    for i in range(len(target_indice)):
        if target_indice[i]!=net_config['zero_symbol'] and target_indice[i]!=pred_indice[i]:
            return False
    return True

def get_net_accuracy(net, net_config):
    accuracy = 0.0
    for batch_id in range(net_config['batch_size']):
        data = net.blobs['target_words'].data
        target_indice = data[batch_id,:,0,0]

        probs = net.blobs['word_probs'].data
        out_prob = np.zeros((net_config['max_len'], net_config['vocab_size']))
        for i in range(net_config['max_len']):
            out_prob[i,:] = probs[batch_id+i*net_config['batch_size'],:]
        output_indice = get_max_indice(out_prob, net_config)
        if if_the_same(output_indice, target_indice, net_config):
            accuracy += 1
    #print accuracy/net_config['batch_size']
    return accuracy/net_config['batch_size']

# # # # # # # # # # # #

def list_find_last_index(list, item):
    for i in range(len(list)):
        ii = len(list)-i-1
        if list[ii] == item:
            return ii
    return -1

def load_indice_file(file, net_config, phase = 'train'):
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
    batch_ws_i = np.zeros((net_config["batch_size"]))
    batch_id = 0
    while True:
        random.shuffle(indice_list)
        for indice in indice_list:
            indice = indice[:net_config["max_len"]]
            ws_i = list_find_last_index(indice, net_config["whitespace_symbol"])
            if ws_i < 0:
                print "error", indice
                continue

            former_output_indice = indice + [net_config['zero_symbol']]*(net_config['max_len']-len(indice))
            output_indice = [net_config['zero_symbol']]*(ws_i+1) + former_output_indice[ws_i+1:net_config['max_len']] 
            output_indice[len(indice)] = net_config['whitespace_symbol']
            if phase == 'train':
                input_indice = [net_config["start_symbol"]] + former_output_indice[:-1]
            elif phase == 'test':
                input_indice = [net_config["start_symbol"]] + former_output_indice[:ws_i] +\
                            [net_config['zero_symbol']]*(net_config['max_len']-ws_i-1)
                input_indice[ws_i+1] = net_config['whitespace_symbol']

            batch_ws_i[batch_id] = ws_i + 1                                                         # note: 1 is due to start symbol inserted
            batch_input_indice[batch_id, :, 0, 0] = input_indice
            batch_output_indice[batch_id, :, 0, 0] = output_indice
            for i in range(net_config["max_len"]):
                ii = input_indice[i]
                vec = [0]*net_config["vocab_size"]
                vec[ii] = 1
                batch_wordvec_layer[batch_id, :, i, 0] = vec                
            
            if batch_id == net_config["batch_size"]-1:
                yield {"input_words":batch_input_indice, "target_words":batch_output_indice, 
                        "wordvec_layer":batch_wordvec_layer, "ws_i": batch_ws_i}                      # note: ws_i is according to input
                batch_id = 0
            else:
                batch_id += 1

def forward(net, input_data, net_config, phase = 'train', deploy=False):
    """Defines and creates the ReInspect network given the net, input data
    and configurations."""

    net.clear_forward()           

    batch_ws_i = input_data["ws_i"]         
    batch_stop_i = [net_config['max_len']]*net_config['batch_size']               
    wordvec_layer = input_data["wordvec_layer"]                                     # 128*38*100*1
    net.f(NumpyData("target_words", data=np.array(input_data["target_words"])))      # 128*100*1*1

    tops = []
    slice_point = []
    for i in range(net_config['max_len']):
        tops.append('label%d' % i)
        if i != 0:
            slice_point.append(i)
    net.f(Slice("label_slice_layer", slice_dim = 1,
                    bottoms = ["target_words"], tops = tops, slice_point = slice_point))

    net.f(NumpyData("target_wordvec%d" % 0, data=wordvec_layer[:,:,0,0]))  # start symbol, 128*38 

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

        net.f(InnerProduct("ip%d" % i, net_config['vocab_size'], 
                            bottoms=['dropout%d_%d' % (net_config['lstm_num_stacks'] - 1, i)], 
                            weight_filler=filler))

        
        if i < net_config['max_len']-1:
            tar_wordvec = np.array(wordvec_layer[:,:,i+1,0])        # 128*38
            if phase == 'test':
                net.f(Softmax("word_probs%d" % i, bottoms=["ip%d" % i]))
                probs = net.blobs["word_probs%d" % i].data
                for bi in range(net_config['batch_size']):
                    if i >= batch_ws_i[bi] and i < batch_stop_i[bi]:
                        vec = [0]*net_config["vocab_size"]
                        peakIndex = np.argmax(probs[bi,:])
                        if peakIndex == net_config['whitespace_symbol']:
                            batch_stop_i[bi] = i+1
                        vec[peakIndex] = 1
                        tar_wordvec[bi, :] = vec
            net.f(NumpyData("target_wordvec%d" % (i+1), data=tar_wordvec))


    bottoms = []
    for i in range(net_config['max_len']):
        bottoms.append("ip%d" % i)
    net.f(Concat('ip_concat', bottoms = bottoms, concat_dim = 0))

    bottoms = []
    for i in range(net_config['max_len']):
        bottoms.append('label%d' % i)
    net.f(Concat('label_concat', bottoms = bottoms, concat_dim = 0))

    if deploy:
        net.f(Softmax("word_probs", bottoms=["ip_concat"]))
    
    net.f(SoftmaxWithLoss("word_loss",
                          bottoms=["ip_concat", "label_concat"], 
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
    input_gen_test = load_indice_file(data_config["test_file"], net_config, phase = 'test')

    forward(net, input_gen.next(), config["net"])
    # net.draw_to_file(logging["schematic_path"])         # !!!!

    if solver["weights"]:
        net.load(solver["weights"])
    # else:
    #     net.load(googlenet.weights_file())

    loss_hist = {"train": [], "test": [], "test_accuracy": []}
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
            # test loss
            net.phase = 'test'
            test_loss = []
            test_accuracy = []
            for _ in range(solver["test_iter"]):
                forward(net, input_gen_test.next(), config["net"], phase = 'test', deploy = True)
                test_loss.append(net.loss)
                c_accuracy = get_net_accuracy(net, config['net'])
                test_accuracy.append(c_accuracy)
            loss_hist["test"].append(np.mean(test_loss))
            loss_hist["test_accuracy"].append(np.mean(test_accuracy))
            print 'test accuracy:',loss_hist["test_accuracy"][-1]
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
                           'test_accuracy': loss_hist["test_accuracy"],
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
