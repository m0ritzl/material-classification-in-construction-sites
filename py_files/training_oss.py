
import neuralnets
import dataset_loader
#from neuralnets import ResidualBlock, ResNet
from dataset_loader import OpenSurfacesSmall 
from efficientnet_pytorch import EfficientNet
#from __future__ import print_function, division

import os
import time
import copy
from PIL import Image
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import Tensor, is_tensor
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

# WandB  Import the wandb library
import wandb
# Ignore excessive warnings
import logging 
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()
    
    # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
    for batch_idx, (data, target) in enumerate(train_loader):
        #if batch_idx > 20:
        #  break
        # Load the input features and labels from the training dataset
        data, target = data.to(device), target.to(device)
        
        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()
        
        # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
        output = model(data)
        
        # Define our loss function, and compute the loss
        loss = criterion(output, target)
        #loss = F.nll_loss(output, target)
        
        # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
        loss.backward()
        
        # Update the neural network weights
        optimizer.step()


def test(args, model, device, test_loader, criterion):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()
    global test_loss
    test_loss = 0
    total = 0
    correct = 0
    best_loss = 1
    num_classes = 4
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            # Load the input features and labels from the test dataset
            data, target = data.to(device), target.to(device)
            
            # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
            output = model(data)
            
            # Compute the loss sum up batch loss
            test_loss += criterion(output, target).item()
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # Get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1][:,0]
            per_class_count = torch.bincount(target, minlength=num_classes)
            #correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(num_classes):
              class_total[i] += per_class_count[i]
              mask = target == i
              p = pred[mask]
              t = target[mask]
              class_correct[i] += p.eq(t.view_as(p)).sum().item()            
            total += pred.eq(pred).sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # WandB  Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            #example_images.append(wandb.Image(
            #    data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))
    
    # WandB  wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
    # You can log anything by passing it to wandb.log, including histograms, custom matplotlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
    # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
    #wandb.log({
    #    "Examples": example_images,
    #    "Test Accuracy": 100. * correct / total,
    #    "Test Loss": test_loss})
    log_dict = {
        #"Examples": example_images,
	"Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss}
    for i in range(num_classes):
      log_dict["Test Accuracy - " + test_loader.dataset.classes[i]] = 100. * class_correct[i] / class_total[i]
    #print(log_dict)
    wandb.log(log_dict)
# WandB  Initialize a new run
wandb.init(project="resnet-classifier-opensurfaces")
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB  Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 32          # input batch size for training (default: 64)
config.test_batch_size = 32    # input batch size for testing (default: 1000)
config.epochs = 50             # number of epochs to train (default: 10)
#config.lr = 0.001               # learning rate (default: 0.01)
config.no_cuda = False         # disables CUDA training
config.gpu_id = 0
config.seed = 42               # random seed (default: 42)
config.log_interval = 10     # how many batches to wait before logging training status
config.model = 'wide_resnet50_2_224'
config.crop_size = 224
config.num_workers = 4
config.optimizer = 'SGD'     #set to Adam or SGD

def main():
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    torch.cuda.set_device(config.gpu_id)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if use_cuda else {}
    data_root="../datasets/sample_" + str(config.crop_size)
    crop_size_to_image_number = {128: 5530, 224: 1184}

    train_trans = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.48485812, 0.40377784, 0.32280155], [0.37216536, 0.349832,   0.37452201])
                ])
    eval_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.48485812, 0.40377784, 0.32280155], [0.37216536, 0.349832,   0.37452201])
            ])


    #use for training from the ground up
    train_set_os = OpenSurfacesSmall(root_dir=data_root, n=crop_size_to_image_number[config.crop_size], split=(0,0.8), transform=train_trans)
    val_set_os = OpenSurfacesSmall(root_dir=data_root, n=crop_size_to_image_number[config.crop_size], split=(0.8,1), transform=eval_trans)

    train_loader = torch.utils.data.DataLoader(dataset=train_set_os,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_workers,
                                              shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=val_set_os,
                                              batch_size=config.test_batch_size, 
                                              num_workers=config.num_workers,
                                              shuffle=False)
    #model = EfficientNet.from_pretrained(config.model, num_classes=4).to(device)   
    model = models.wide_resnet50_2(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.to(device)

    #set optimizer
    if config.optimizer == 'Adam':
        initial_lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    elif config.optimizer == 'SGD':
        initial_lr = 0.1
        optimizer = optim.SGD(model.parameters(), lr=initial_lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    #optimizer = optim.SGD(model.parameters(), lr=config.lr)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    #optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    '''optimizer = optim.SGD(model.parameters(), lr=config.lr,
                          momentum=config.momentum)'''
    
    # WandB  wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    wandb.watch(model, log="all")

    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, epoch, criterion)
        test(config, model, device, test_loader, criterion)
        if config.optimizer == 'SGD':
            scheduler.step(test_loss)
        
    # WandB  Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
    torch.save(model.state_dict(), "model_" + str(config.crop_size) + "_" + config.optimizer + "_new " + ".h5")
    wandb.save("model_" + str(config.crop_size) + "_" + config.optimizer + "_new " + ".h5")    


if __name__ == '__main__':
    main()
