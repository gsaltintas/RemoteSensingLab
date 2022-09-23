from torch import nn
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

classes = ['annual crop', 'forest', 'herbaceous vegetation', 'highway', 'industrial', 'pasture', 'permanent crop', 'residential', 'river', 'sea lake']

plt.rcParams["savefig.bbox"] = 'tight'
def visualize_batch(batch, labels=None, true_labels=None):
    imgs = batch['image'].detach().cpu()
    labels = batch['label'] if labels is None else labels
    plt.figure(figsize=(9, 15))
    label_str = [classes[i][:4] for i in labels]
    label_str = '\n'.join([' , '.join(label_str[i*8 : (i+1)*8]) for i in range(4)])    
    grid = make_grid(imgs, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0), )
    plt.show()
    print(label_str)
    print(true_labels)

    
class Net(pl.LightningModule):
    
    def __init__(self, n_classes=10, loss_func=nn.CrossEntropyLoss(),
                     lr: float = 0.001,
                b1: float = 0.5,
                b2: float = 0.999,
                momentum: float = 0.9,
                dropout: float = 0.1,
                weight_decay: float = 5e-4,
                hidden_layers =[6, 12, 24, 48, 96],):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_func'])

        prev_size = 3
        self.convolutions = nn.Sequential()
        output_res = 64
        for i in range(len(hidden_layers)):
            h = hidden_layers[i]
            self.convolutions.append(nn.Conv2d(kernel_size=3, padding=1, in_channels=prev_size, out_channels=h))
            self.convolutions.append(nn.BatchNorm2d(h))
            self.convolutions.append(nn.Dropout(p=dropout))
            self.convolutions.append(nn.LeakyReLU(inplace=True))
            if i == 1 or i == 3:
                self.convolutions.append(nn.MaxPool2d(kernel_size=2, stride=2))
                output_res /= 2
            prev_size = h
        output_res = int(output_res)

        self.fully_connected = nn.Sequential(
            nn.Linear(h*output_res*output_res, n_classes),
        )
        self.loss = loss_func

    def forward(self, img):

        # Apply convolution operations
        x = self.convolutions(img)

        # Reshape
        x = x.view(x.size(0), -1)

        # Apply fully connected operations
        x = self.fully_connected(x)

        return x

    def avg_accuracy(self, pred, label):
        return (pred.argmax(dim=1) == label).float().mean()

    def training_step(self, batch, batch_idx):
        img, label = batch['image'], batch['label']
        model_out = self(img)
        loss = self.loss(model_out, label)
        self.log('train_loss', loss)
        self.log('train/accuracy', self.avg_accuracy(model_out, label))
        # randomly plot some predictions
        if np.random.random() < 0.1:
            pred_labels = torch.argmax(self.softmax(model_out), axis=1)
            true_labels = [classes[i][:4] for i in label]
            true_labels = '\n'.join([' , '.join(true_labels[i*8 : (i+1)*8]) for i in range(4)])   
            visualize_batch(batch, pred_labels, true_labels)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch['image'], batch['label']
        model_out = self(img)
        loss = self.loss(model_out, label)
        self.log('val_loss', loss)
        self.log('val/accuracy', self.avg_accuracy(model_out, label))
        return loss
            
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self(batch)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        weight_decay = self.hparams.weight_decay
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2), weight_decay= weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=1, T_mult = 2,
                                                             eta_min=1e-7)

        return {"optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                },
               }