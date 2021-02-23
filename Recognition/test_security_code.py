from __future__ import division


import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
import numpy as np
from PIL import Image
from matplotlib.ticker import PercentFormatter


from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils import evaluate, LeNet_EMAGE_iph6, LeNet_EMAGE_honor, data_transform_iph6, data_transform_iph6s, data_transform_honor, security_test, security_test_honor


model = LeNet_EMAGE_iph6()
model.load_state_dict(torch.load('./checkpoints/secpin_6_best.pth', map_location='cuda:0'))
model.eval().to(device)

test_root = './data/security_code/simulated_security_code/cross_device_iphone6c/'
res = security_test(root=test_root, model=model, data_transforms_t=data_transform_iph6)
print('Cross device accuray (iphone 6) is %f' % (res))

test_root = './data/security_code/simulated_security_code/withnoise_iphone6/'
res = security_test(root=test_root, model=model, data_transforms_t=data_transform_iph6)
print('With noise accuray (iphone 6) is %f' % (res))


test_root = './data/security_code/simulated_security_code/inter_session_iphone6_1/'
res = security_test(root=test_root, model=model, data_transforms_t=data_transform_iph6)
print('Inter-session-1 (iphone 6) is %f' % (res))

test_root = './data/security_code/simulated_security_code/inter_session_iphone6_2/'
res = security_test(root=test_root, model=model, data_transforms_t=data_transform_iph6)
print('Inter-session-1 (iphone 6) is %f' % (res))


test_root = './data/security_code/simulated_security_code/cross_magazine_iphone6/'
res = security_test(root=test_root, model=model, data_transforms_t=data_transform_iph6)
print('Cross magazine accuray (iphone 6) is %f' % (res))

model = LeNet_EMAGE_honor()
model.load_state_dict(torch.load('./checkpoints/secpin_honor_best.pth', map_location='cuda:0'))
model.eval().to(device)

test_root = './data/security_code/simulated_security_code/cross_magazine_honor6x/'
res = security_test_honor(root=test_root, model=model, data_transforms_t=data_transform_honor)
print('Cross magazine accuray (honor 6x) is %f' % (res))

test_root = './data/security_code/simulated_security_code/withnoise_honor6x/'
res = security_test_honor(root=test_root, model=model, data_transforms_t=data_transform_honor)
print('With noise accuray (honor 6x) is %f' % (res))

test_root = './data/security_code/simulated_security_code/inter_session_honor_1/'
res = security_test_honor(root=test_root, model=model, data_transforms_t=data_transform_honor)
print('Inter-session-1 (honor 6x) is %f' % (res))


test_root = './data/security_code/simulated_security_code/inter_session_honor_2/'
res = security_test_honor(root=test_root, model=model, data_transforms_t=data_transform_honor)
print('Inter-session-2 (honor 6x) is %f' % (res))