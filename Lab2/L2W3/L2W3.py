# Rodrigo Caye Daudt
# rodrigo.cayedaudt@geod.baug.ethz.ch
# 04/2021

import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
import os
from glob import glob
import torch
from tqdm import tqdm

from EuroSAT_dataset import EuroSAT
from network import Net

if not os.path.exists('./outputs'):
    os.mkdir('./outputs')

print('Imports OK')


# Global parameters

# If USE_CUDA is True, computations will be done using the GPU (may not work in all systems)
# This will make the calculations happen faster
USE_CUDA = torch.cuda.is_available() 

DATASET_PATH = '../EuroSAT_data'

BATCH_SIZE = 32 # Number of images that are used for calculating gradients at each step

NUM_EPOCHS = 10 # Number of times we will go through all the training images. Do not go over 25

LEARNING_RATE = 1e-3 # Controls the step size
MOMENTUM = 0.0 # Momentum for the gradient descent
WEIGHT_DECAY = 1e-6 # Regularization factor to reduce overfitting


print('Parameters OK')


# Create datasets and data loaders
train_dataset = EuroSAT(DATASET_PATH, True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

test_dataset = EuroSAT(DATASET_PATH, False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


print('Dataloaders OK')


# Create network
net = Net()
if USE_CUDA:
    net = net.cuda()

print('Network OK')


# Criterion, optimizer, and scheduler

criterion = torch.nn.CrossEntropyLoss() # Do not change this
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Helper function to organize main loop
# This function is called for training and for testing at each epoch

def run_epoch(net, optimizer, dataloader, criterion, train=True, cuda=USE_CUDA):
    epoch_total_loss = 0
    epoch_total_samples = 0
    epoch_total_correct = 0

    for sample in tqdm(dataloader):
        img = sample['image']
        label = sample['label']

        if cuda:
            img, label = img.cuda(), label.cuda()

        if train:
            optimizer.zero_grad()
        out = net(img)
        loss = criterion(out, label)
        if train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            epoch_total_samples += img.size(0)
            epoch_total_loss += img.size(0) * loss
            epoch_total_correct += torch.sum(torch.argmax(out, dim=1) == label)

    return epoch_total_loss / epoch_total_samples, epoch_total_correct / epoch_total_samples



print('Ready to train')


# Main loop

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []
epochs = []

for epoch in range(1, NUM_EPOCHS+1):
    print(f'\n\nRunning epoch {epoch} of {NUM_EPOCHS}...\n')
    epochs.append(epoch)

    # Train
    net.train()
    loss, accuracy = run_epoch(net, optimizer, train_loader, criterion, train=True)
    train_loss.append(loss.cpu())
    train_accuracy.append(accuracy.cpu())


    # Test
    net.eval()
    with torch.no_grad():
        loss, accuracy = run_epoch(net, optimizer, test_loader, criterion, train=False)
    test_loss.append(loss.cpu())
    test_accuracy.append(accuracy.cpu())
    print(f'\nEpoch {epoch} validation results: Loss={loss.cpu()} | Accuracy={accuracy.cpu()}\n')


    # Plot and save
    plt.figure(figsize=(12, 8), num=1)
    plt.clf()
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, test_loss, label='Test')
    plt.legend()
    plt.grid()
    plt.title('Cross entropy loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('outputs/01-loss.pdf')

    plt.figure(figsize=(12, 8), num=2)
    plt.clf()
    plt.plot(epochs, train_accuracy, label='Train')
    plt.plot(epochs, test_accuracy, label='Test')
    plt.legend()
    plt.grid()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('outputs/02-accuracy.pdf')

    

print(f'Final train loss: {train_loss[-1]}')
print(f'Final test loss: {test_loss[-1]}')
print(f'Final train accuracy: {train_accuracy[-1]}')
print(f'Final test accuracy: {test_accuracy[-1]}')