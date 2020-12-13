# /usr/bin/python
import itertools

from ann import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

##########################
# Shubham Swami
# Your A02315672
# Write your code at the end of
# this file in the provided
# function stubs.
##########################

#### Libraries
from mnist_loader import load_data_wrapper

train_d, valid_d, test_d = load_data_wrapper()

eta_list = [0.25, 1]
lambda_list = [5, 8]
d = {}
d1 = {}


#### auxiliary functions
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of ann.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = ann(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### plotting costs and accuracies
def plot_costs(eval_costs, train_costs, num_epochs):
    plt.title('Evaluation Cost (EC) and Training Cost (TC)')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_costs, label='EC', c='g')
    plt.plot(epochs, train_costs, label='TC', c='b')
    plt.grid()
    plt.autoscale(tight=True)
    plt.legend(loc='best')
    plt.show()


def plot_accuracies(eval_accs, train_accs, num_epochs, fileName):
    plt.title('Evaluation Acc (EA) and Training Acc (TC)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_accs, label='EA', c='g')
    plt.plot(epochs, train_accs, label='TA', c='b')
    plt.grid()
    plt.autoscale(tight=True)
    plt.legend(loc='best')
    # plt.show()
    save_results_to = r'A:\USU\Assignments\IntelligentSystems\hw04\graphs'
    plt.savefig(save_results_to + fileName)


def collect_1_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data, cost_function):
    global d
    while lower_num_hidden_nodes <= upper_num_hidden_nodes:
        print("training for: ")
        print("lower_num_hidden_nodes ", lower_num_hidden_nodes)
        print("eta ", eta)
        print("lmbda ", lmbda)

        net = ann([784, lower_num_hidden_nodes, 10], cost_function)
        net_stats = net.mini_batch_sgd(train_data,
                                       num_epochs, mbs, eta, lmbda,
                                       eval_data,
                                       monitor_evaluation_cost=True,
                                       monitor_evaluation_accuracy=True,
                                       monitor_training_cost=True,
                                       monitor_training_accuracy=True)
        d[lower_num_hidden_nodes] = net_stats
        lower_num_hidden_nodes += 10
    return d


def collect_2_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    global d1
    print("training for: ")
    print("eta ", eta)
    print("lmbda ", lmbda)
    arr = [i for i in range(lower_num_hidden_nodes + 10, upper_num_hidden_nodes + 1, 10)]
    l = [[lower_num_hidden_nodes], arr]
    res = list(itertools.product(*l))
    while lower_num_hidden_nodes < upper_num_hidden_nodes:
        for i in range(len(res)):
            n1 = res[i][0]
            n2 = res[i][1]
            print("n1  ", n1)
            print("n2  ", n2)
            net = ann([784, n1, n2, 10], cost_function)
            net_stats = net.mini_batch_sgd(train_data,
                                           num_epochs, mbs, eta, lmbda,
                                           eval_data,
                                           monitor_evaluation_cost=True,
                                           monitor_evaluation_accuracy=True,
                                           monitor_training_cost=True,
                                           monitor_training_accuracy=True)
            key = str(n1) + "_" + str(n2)
            d1[key] = net_stats
    lower_num_hidden_nodes += 10


def collect_3_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data, cost_function):
    global d1
    arr = [i for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 10)]
    l = [[lower_num_hidden_nodes], arr, arr]
    res = list(itertools.product(*l))
    while lower_num_hidden_nodes < upper_num_hidden_nodes:
        for i in range(len(res)):
            print("training for: ")
            print("eta ", eta)
            print("lmbda ", lmbda)


            n1 = res[i][0]
            n2 = res[i][1]
            n3 = res[i][2]
            print("n1  ", n1)
            print("n2  ", n2)
            print("n2  ", n3)
            net = ann([784, n1, n2, 10], cost_function)
            net_stats = net.mini_batch_sgd(train_data,
                                           num_epochs, mbs, eta, lmbda,
                                           eval_data,
                                           monitor_evaluation_cost=True,
                                           monitor_evaluation_accuracy=True,
                                           monitor_training_cost=True,
                                           monitor_training_accuracy=True)
            key = str(n1) + "_" + str(n2)
            d1[key] = net_stats
    lower_num_hidden_nodes += 10


def get_best_ann():
    global eta_list
    global lambda_list
    combination_list = [eta_list, lambda_list]
    res = list(itertools.product(*combination_list))
    #all combinations of eta and lambda
    print(res)
    for i in range(len(res)):
        lower_num_hidden_nodes = 30
        upper_num_hidden_nodes = 100
        eta = res[i][0]
        lmbda = res[i][1]
        list.append( collect_3_hidden_layer_net_stats(lower_num_hidden_nodes, upper_num_hidden_nodes, 30, 10, eta, lmbda, train_d,
                                         valid_d, CrossEntropyCost))


net=ann([784, 30, 80, 40, 10])
net_stats = net.mini_batch_sgd(train_d, 30, 10, 0.25, 5,
                               valid_d,
                               monitor_evaluation_cost=True,
                               monitor_evaluation_accuracy=True,
                               monitor_training_cost=True,
                               monitor_training_accuracy=True)
net.save("net3.json")
plot_costs(net_stats[0], net_stats[2], 30)
plt.figure()
plot_accuracies(net_stats[1], net_stats[3], 30)
print("evaluation cost of best ann", net_stats[0][29])
print("training cost of best ann", net_stats[2][29])
print("evaluation Accuracy of best ann ", net_stats[1][29])
print("training Accuracy of best ann ", net_stats[3][29])