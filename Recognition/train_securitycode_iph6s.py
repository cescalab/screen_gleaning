from __future__ import division


import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from config import Config_eyedoctor, Config_securitycode_iph6s
from utils import data_transform_iph6s, evaluate, LeNet_EMAGE_iph6

model = LeNet_EMAGE_iph6()
for param in model.parameters():
    param.requires_grad = True

model.train().to(device)

data_transform = data_transform_iph6s

cfg = Config_securitycode_iph6s

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)


org_font_dataset = datasets.ImageFolder(root=cfg.train_data_root,
                                           transform=data_transform)

dataset_loader = torch.utils.data.DataLoader(org_font_dataset,
                                             batch_size=cfg.batch_size, shuffle=True,
                                             pin_memory = True,
                                             num_workers=cfg.num_workers)

org_font_dataset_val = datasets.ImageFolder(root=cfg.val_data_root,
                                           transform=data_transform)
dataset_loader_val = torch.utils.data.DataLoader(org_font_dataset_val,
                                             batch_size=64, shuffle=False,
                                             num_workers=cfg.num_workers)

# writer = SummaryWriter()
# writer_ct = 0
val_acc_best = 0
for epoch in tqdm(range(cfg.max_epoch)):
    
    running_loss = 0.0
    model.train()
    for i, data in enumerate(dataset_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
        
        # writer.add_scalar('runs/loss', loss.item(), writer_ct)
        # writer_ct += 1
        
    
    print(running_loss / len(org_font_dataset))    
    # writer.add_scalar('runs/val_loss', val_loss(dataset_loader_val, model), epoch)
    model.eval()
    val_acc = evaluate(dataset_loader_val, model, 'Validation ')    

    if val_acc > val_acc_best:
        val_acc_best = val_acc
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, cfg.ckpt_path + 'secpin_6s_best.tar')

print('Training finished.')




print('Testing...')


org_font_dataset_test = datasets.ImageFolder(root=cfg.test_data_root,
                                           transform=data_transform)
dataset_loader_test = torch.utils.data.DataLoader(org_font_dataset_test,
                                             batch_size=64, shuffle=False,
                                             num_workers=cfg.num_workers)

model = LeNet_EMAGE_iph6()


model.load_state_dict(torch.load(cfg.ckpt_path + 'secpin_6s_best.tar', map_location='cuda:0')['model_state_dict'])
model.eval().to(device)
evaluate(dataset_loader_test, model, 'Intra-Session test ')   

org_font_dataset_test = datasets.ImageFolder(root=cfg.inter_1_test_data_root,
                                           transform=data_transform)
dataset_loader_test = torch.utils.data.DataLoader(org_font_dataset_test,
                                             batch_size=64, shuffle=False,
                                             num_workers=cfg.num_workers)

evaluate(dataset_loader_test, model, 'Inter-Session-s8 test ')  

org_font_dataset_test = datasets.ImageFolder(root=cfg.inter_2_test_data_root,
                                           transform=data_transform)
dataset_loader_test = torch.utils.data.DataLoader(org_font_dataset_test,
                                             batch_size=64, shuffle=False,
                                             num_workers=cfg.num_workers)

evaluate(dataset_loader_test, model, 'Inter-Session-s9 test ')  