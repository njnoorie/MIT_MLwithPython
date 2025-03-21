#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

num_classes1 = 128

option ={
    "initial": {
        "batch_size": 32, 
        "num_classes": num_classes1, 
        "learning_rate": 0.1, 
        "momentum": 0
    },
    "batch_64":{
        "batch_size": 64, 
        "num_classes": num_classes1, 
        "learning_rate": 0.1, 
        "momentum": 0

    },
    "lr_01":{
        "batch_size": 32, 
        "num_classes": num_classes1, 
        "learning_rate": 0.01, 
        "momentum": 0
    },
    "momentum_9":{
        "batch_size": 32, 
        "num_classes": num_classes1, 
        "learning_rate": 0.1, 
        "momentum": 0.9
    }

}

def main(batch_size, num_classes, lr, momentum):
    # Load the dataset
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    model = nn.Sequential(
              nn.Linear(784, 10),
              nn.ReLU(),
              nn.Linear(10, 10),
            )
    
    ##################################

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    opt = option["initial"]
    # opt = option["batch_64"]
    # opt = option["lr_01"]
    # opt = option["momentum_9"]
    main(opt["batch_size"], opt["num_classes"], opt["learning_rate"], opt["momentum"])

    #Base                   Val loss:   0.228388 | Val accuracy:   0.932487
    #                       Loss on test set:0.26722689906809816 Accuracy on test set: 0.9204727564102564
    #batch64                Val loss:   0.208171 | Val accuracy:   0.940020
    #                       Loss on test set:0.24238466640086606 Accuracy on test set: 0.9314903846153846
    # learning rate 0.01    Val loss:   0.233266 | Val accuracy:   0.934659
    #                       Loss on test set:0.27886608655516726 Accuracy on test set: 0.9206730769230769
    # momentum 0.9          Val loss:   0.489616 | Val accuracy:   0.851939
    #                       Loss on test set:0.5052944285174211 Accuracy on test set: 0.854667467948718
    # LeakyReLU activation  Val loss:   0.227615 | Val accuracy:   0.931985
    #                       Loss on test set:0.26892605127068236 Accuracy on test set: 0.9207732371794872


    #128 classes            
    #Base                   Val loss:   0.072770 | Val accuracy:   0.978108
                            #Loss on test set:0.07475928590356955 Accuracy on test set: 0.9770633012820513
    #batch64                Val loss:   0.079501 | Val accuracy:   0.976815
                            #Loss on test set:0.08358146542629513 Accuracy on test set: 0.9743589743589743
    # learning rate 0.01    Val loss:   0.169063 | Val accuracy:   0.955047
                            #Loss on test set:0.1977690439014576 Accuracy on test set: 0.9427083333333334
    # momentum 0.9          Val loss:   0.183256 | Val accuracy:   0.971090
                            #Loss on test set:0.19123652955591014 Accuracy on test set: 0.9658453525641025
    # LeakyReLU activation  Val loss:   0.072255 | Val accuracy:   0.977941
                            #Loss on test set:0.07432884118138762 Accuracy on test set: 0.9765625