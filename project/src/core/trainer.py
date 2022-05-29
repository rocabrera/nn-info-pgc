from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from core.information import make_discrete_information_plane, make_continuos_information_plane


def evaluate_model(model, criterion, valid_dl, device):

    valid_loss_batchs = []
    valid_auc_batchs = [] 

    model.eval()
    with torch.no_grad():
        for input, target in valid_dl:
            input_in_device = input.to(device)
            targets_in_device = target.to(device, dtype=torch.float32)
            yhat_valid = model(input_in_device)
            valid_loss = criterion(yhat_valid, targets_in_device)
            valid_auc_batchs.append(auroc(yhat_valid, targets_in_device.to(torch.int), pos_label=1)
                                    .cpu()
                                    .numpy())

            valid_loss_batchs.append(valid_loss.item())

    valid_auc = np.mean(valid_auc_batchs)
    valid_loss = np.mean(valid_loss_batchs)

    return valid_auc, valid_loss

def train_model(model, optimizer, criterion, train_dl, device):

    train_loss_batchs = []
    train_auc_batchs = []

    model.train()
    for _, (inputs, targets) in enumerate(train_dl):
        
        inputs_in_device = inputs.to(device)
        targets_in_device = targets.to(device, dtype=torch.float32)
        optimizer.zero_grad() # clear the gradients
        yhat = model(inputs_in_device) # compute the model output
        train_loss = criterion(yhat, targets_in_device) # calculate loss
        train_loss.backward() 
        optimizer.step()

        train_auc_batchs.append(auroc(yhat, targets_in_device.to(torch.int), pos_label=1)
                                .cpu()
                                .numpy())

        train_loss_batchs.append(train_loss.item())

    train_auc = np.mean(train_auc_batchs)
    train_loss = np.mean(train_loss_batchs)

    return train_auc, train_loss, inputs_in_device, targets_in_device

def run_experiment(train_dl, 
                   valid_dl,
                   model, 
                   n_epochs, 
                   learning_rate, 
                   rand_init_number, 
                   estimation_param, 
                   result_file_path, 
                   device,
                   discrete):

    # define the optimization
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    for epoch in tqdm(range(n_epochs), desc="Epochs", position=1, leave=False):

        valid_auc, valid_loss = evaluate_model(model, criterion, valid_dl, device)

        train_auc, train_loss, inputs_train, targets_train = train_model(model, optimizer, criterion, train_dl, device)

        if discrete:
            make_discrete_information_plane(model=model, 
                                            inputs=inputs_train, 
                                            targets=targets_train, 
                                            valid_auc=valid_auc,
                                            train_auc=train_auc,
                                            valid_loss=valid_loss,
                                            train_loss=train_loss,
                                            n_bin=estimation_param, 
                                            result_file_path=result_file_path, 
                                            epoch=epoch, 
                                            rand_init_number=rand_init_number,)
        else:
            make_continuos_information_plane(model=model, 
                                             inputs=inputs_train, 
                                             targets=targets_train, 
                                             valid_auc=valid_auc,
                                             train_auc=train_auc,
                                             valid_loss=valid_loss,
                                             train_loss=train_loss,
                                             kernel_size=estimation_param, 
                                             result_file_path=result_file_path, 
                                             epoch=epoch, 
                                             rand_init_number=rand_init_number)