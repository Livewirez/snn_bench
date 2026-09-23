from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import neuron, encoding, functional, surrogate
from spikingjelly import visualizing
from matplotlib import pyplot as plt

import torch.nn.functional as F

from typing import Callable, Optional, Union, Iterable, Tuple, Dict, Protocol, runtime_checkable

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Sequence, Mapping, Any

from ..config import Config
from . import CriterionProtocol, OptimizerFactory


Encoder = Union[
    spikingjelly.activation_based.encoding.StatelessEncoder,
    spikingjelly.activation_based.encoding.StatefulEncoder,
]

class TrainOptions:
    def __init__(self, criterion: Optional[CriterionProtocol] = None, optimizer: Optional[OptimizerFactory] = None, encoder: Optional[Encoder] = None, max_epoch: int = 30):
        self.criterion = criterion
        self.optimizer = optimizer
        self.encoder = encoder
        self.max_epoch = max_epoch
        
    @classmethod
    def get_default_none(cls) -> "TrainOptions":
        return cls(
            criterion=None,
            optimizer=None,
            encoder=None,
        )
        
    @classmethod
    def get_default(cls) -> "TrainOptions":
        return cls(
            criterion=nn.CrossEntropyLoss(),
            optimizer=lambda params: torch.optim.AdamW(
                params,
                lr=0.001,
                weight_decay=0.0,
            ),
            encoder=encoding.PoissonEncoder()
        )
    
# # Set neural network hyperparameters
# # Set the loss function
# loss_fn = nn.CrossEntropyLoss()
# loss_fn= loss_fn.to (device) # Convert the loss function

# # Set up optimizer
# learning_rate = 0.001 # Set the learning rate
# lambda_val = 0.0 # Set the regularization coefficient Lambda
# optim = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=lambda_val)

# # Define input data encoder
# encoder = encoding.PoissonEncoder()

# # Set the simulation time T
# T = 20


def start_training_snn_loop(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,  config: Config, options: TrainOptions, T: int = 20, epoch: int = 5):
    # Set the number of training rounds epoch and define the temporary parameters required during the training process
    # Set the number of training rounds
    # epoch = 5
    
    loss_fn = (
        options.criterion if options.criterion is not None
        else nn.CrossEntropyLoss()
    )
    
    optim = (
        options.optimizer(model.parameters())
        if options.optimizer is not None
        else torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    )
    
    encoder = (
        options.encoder if options.encoder is not None
        else encoding.PoissonEncoder()
    )
    
    batch_size = config.batch_size
    train_dataset_size = len(train_dataloader.dataset)
    test_dataset_size = len(test_dataloader.dataset)
        
    
    # Define the training set loss and accuracy variables
    loss_train = 0.0 # Training loss value
    loss_train_sum = 0.0 # Cumulative value of training loss value
    loss_train_mean = 0.0 # The average value of the loss value of this round of training
    loss_train_store = np.zeros(epoch) # Training loss value storage array
    accuracy_train = 0 # Training accuracy
    accuracy_rate_train = 0.0 # Training accuracy
    accuracy_rate_train_max = 0.0 # Maximum training accuracy
    accuracy_rate_train_store = np.zeros(epoch) # Training accuracy storage prime array
    # Define cross-verification set loss and accuracy variables
    loss_cv = 0.0 # Verify the loss value
    loss_cv_sum = 0.0 # Verify the cumulative value of the loss value
    loss_cv_mean = 0.0 # The average value of the loss value in this round of verification
    loss_cv_store = np.zeros(epoch) # Verification loss value storage array
    accuracy_cv = 0 # Verify the exact number
    accuracy_rate_cv = 0.0 # Verification accuracy
    accuracy_rate_cv_max = 0.0 # Maximum verification accuracy
    accuracy_rate_cv_store = np.zeros(epoch) # Verify the accuracy of the stored prime array
    
    # Training network

    #########################Start training########################

    for i in range(epoch):
        print("----------- %d round of training begins------------" % i)
        # The accumulated value of each round of loss is cleared to zero
        loss_train_sum = 0
        loss_cv_sum = 0
        # Start training neural network
        model.train()
        # This round of training counts
        train_step = 0
        # Accuracy variable cleared
        accuracy_train = 0
        for data in train_dataloader:
            inputs, labels = data
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            label_onehot = F.one_hot(labels, 10).float() # Convert tags to independent thermal coding

            out_fr = 0. # Output neuron pulse emission rate
            #out_fr is the tensor of shape=[batch_size, 10]
            # Record the pulse emission rate of 10 neurons in the output layer during the entire simulation time
            for t in range(T):
                encoded_img = encoder(inputs) #First encode the input image into pulse data
                out_fr+=model(encoded_img) # Accumulate the pulse output of 10 neurons in the output layer
            out_fr = out_fr /T # Divide by time to obtain the pulse emission rate of 10 neurons in the output layer
            # Calculate the loss function
            loss_train = F.mse_loss(out_fr, label_onehot)
            # The loss function is the pulse emission frequency of neurons in the output layer, which is the same as the MSE of the real category
            # Such a loss function will make: when the label i is given, the pulse emission frequency of the i-th neuron in the output layer approaches 1, while the pulse emission frequency of other neurons approaches 0
            # Cumulative loss function
            loss_train_sum += loss_train.item()
            # Convert the classification probability into the corresponding label
            pred_label = out_fr.argmax(1)
            # Accuracy of cumulative calculation results
            accuracy_train += (pred_label == labels).sum()
            # Clear the cumulative gradient of the optimizer
            optim.zero_grad()
            # Calculate gradient
            loss_train.backward()
            # The optimizer starts to optimize
            optim.step()
            
            # After optimizing the parameters once, you need to reset the state of the network, because SNN neurons have "memory”
            functional.reset_net(model)

        # Calculate the average training set loss
        loss_train_mean = loss_train_sum / (train_dataset_size / batch_size)
        loss_train_store[i] = loss_train_mean # Store the loss for plotting the loss curve later
        # Calculate training set accuracy
        accuracy_rate_train = (accuracy_train / train_dataset_size) * 100
        accuracy_rate_train_store[i] = accuracy_rate_train


        # Start testing the neural network
        model.eval()
        # Test step counter for this round
        test_step = 0
        # Reset the accuracy variable
        accuracy_cv = 0
        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)
                label_onehot = F.one_hot(labels, 10).float()
                out_fr = 0.0
                for t in range(T):
                    encoded_img = encoder(inputs)
                    # Feed into the neural network and run
                    out_fr += model(encoded_img)
                out_fr = out_fr / T
                # Compute the loss function
                loss_cv = F.mse_loss(out_fr, label_onehot)
                # Accumulate the loss function
                loss_cv_sum += loss_cv.item()
                # Convert the classification probability to the corresponding label
                pred_label = out_fr.argmax(1)
                # Accumulate accuracy
                accuracy_cv += (pred_label == labels).sum()
                test_step += 1
                # After each parameter optimization, the network’s state must be reset because SNN neurons have “memory”
                functional.reset_net(model)

            # Calculate the cross-validation set loss (loss_cv)
            loss_cv_mean = loss_cv_sum / (test_dataset_size / batch_size)
            loss_cv_store[i] = loss_cv_mean # Record the loss for later plotting of the loss curve
            # Calculate the cross-validation accuracy
            accuracy_rate_cv = (accuracy_cv / test_dataset_size) * 100
            accuracy_rate_cv_store[i] = accuracy_rate_cv

        # Print the results of this training round
        print(f"loss train: {loss_train_mean: .4f}, accuracy rate train: {accuracy_rate_train.item(): .4f}%, loss cv: {loss_cv_mean: .4f}, accuracy rate cv: {accuracy_rate_cv.item(): .4f}%")
        # Automatically save neural network models whose accuracy during training exceeds that of previous models

        if (accuracy_rate_train + accuracy_rate_cv > accuracy_rate_train_max + accuracy_rate_cv_max):
            torch.save(model, config.model_auto_save_dir + "model_auto_save_acc%d.pth" %accuracy_rate_cv)
            accuracy_rate_train_max = accuracy_rate_train  # Record the maximum training accuracy
            accuracy_rate_cv_max = accuracy_rate_cv  # Record the maximum validation accuracy

    ######################### End of Training########################
    print('-------------------------------Training Complete-------------------------------')
    
    return (loss_train_store, loss_cv_store, accuracy_rate_train_store, accuracy_rate_cv_store, epoch)