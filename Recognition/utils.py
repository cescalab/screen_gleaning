from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform_eyed = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2968, 0.2968, 0.2968],
                             std=[0.1310, 0.1310, 0.1310])
    ])

data_transform_iph6s = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.1545],[0.0489]),
    ])

data_transform_iph6 = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.2933],[0.0546]),
    ])

data_transform_honor = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.4303],[0.0315]),
    ])

def evaluate(dloader, model, phrase):
    correct = 0
    total = 0
    for data in dloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(phrase + 'accuracy is: %d %%' % (100 * correct / total))
    return (100 * correct / total)


class LeNet_EMAGE_iph6(nn.Module):
    def __init__(self):
        super(LeNet_EMAGE_iph6, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(128, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNet_EMAGE_honor(nn.Module):
    def __init__(self):
        super(LeNet_EMAGE_honor, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out



def security_test(root, model, data_transforms_t):
    corr = 0
    base = 0
    res_ls = []
    zero_right = 0
    one_right = 0
    two_right = 0
    three_right = 0
    four_right = 0
    five_right = 0
    six_right = 0
    five_right = 0

    to_test = os.listdir(root)

    digit_base = []
    digit_predicted_properly = []

    for item in to_test:
        
        
        img_kou = Image.open(root + item)
        img_kou = img_kou.resize((120, 31))

        pin1 = img_kou.crop((0, 0, 20, 31))
        pin2 = img_kou.crop((20, 0, 40, 31))
        pin3 = img_kou.crop((40, 0, 60, 31))
        pin4 = img_kou.crop((60, 0, 80, 31))
        pin5 = img_kou.crop((80, 0, 100, 31))
        pin6 = img_kou.crop((100, 0, 120, 31))
        
        gt = [int(digit) for digit in item.split('_')[0].split('-')]
        digit_base.append(gt)
        
        pred = [int(torch.max(model(data_transforms_t(pin1).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin2).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin3).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin4).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin5).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin6).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())]
        digit_predicted_properly.append(pred)
        
        res = (int(gt[0] == int(torch.max(model(data_transforms_t(pin1).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[1] == int(torch.max(model(data_transforms_t(pin2).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[2] == int(torch.max(model(data_transforms_t(pin3).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[3] == int(torch.max(model(data_transforms_t(pin4).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[4] == int(torch.max(model(data_transforms_t(pin5).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[5] == int(torch.max(model(data_transforms_t(pin6).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())))
        res_ls.append(res)
            
        corr += res
        base += 6
        
        
        
    return (corr / base)


def security_test_honor(root, model, data_transforms_t):

    corr = 0
    base = 0
    res_ls = []
    zero_right = 0
    one_right = 0
    two_right = 0
    three_right = 0
    four_right = 0
    five_right = 0
    six_right = 0
    five_right = 0

    to_test = os.listdir(root)

    digit_base = []
    digit_predicted_properly = []

    for item in to_test:
        
        
        img_kou = Image.open(root + item)
        img_kou = img_kou.resize((126, 45))

        pin1 = img_kou.crop((0, 0, 21, 45))
        pin2 = img_kou.crop((21, 0, 42, 45))
        pin3 = img_kou.crop((42, 0, 63, 45))
        pin4 = img_kou.crop((63, 0, 84, 45))
        pin5 = img_kou.crop((84, 0, 105, 45))
        pin6 = img_kou.crop((105, 0, 126, 45))
        
        gt = [int(digit) for digit in item.split('_')[0].split('-')]
        digit_base.append(gt)
        
        pred = [int(torch.max(model(data_transforms_t(pin1).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin2).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin3).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin4).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin5).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())
        ,int(torch.max(model(data_transforms_t(pin6).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())]
        digit_predicted_properly.append(pred)
        
        res = (int(gt[0] == int(torch.max(model(data_transforms_t(pin1).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[1] == int(torch.max(model(data_transforms_t(pin2).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[2] == int(torch.max(model(data_transforms_t(pin3).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[3] == int(torch.max(model(data_transforms_t(pin4).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[4] == int(torch.max(model(data_transforms_t(pin5).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy()))
        +int(gt[5] == int(torch.max(model(data_transforms_t(pin6).unsqueeze(0).to(device)), 1)[1].cpu().data.numpy())))
        res_ls.append(res)
            
        corr += res
        base += 6
        
        
        
    return(corr / base)