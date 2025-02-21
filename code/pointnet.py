
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#
##
#%%
import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys

# Import functions to read and write ply files
from ply import write_ply, read_ply



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud
        

class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),RandomNoise(),ToTensor()])

def test_transforms():
    return transforms.Compose([ToTensor()])

class RandomDuplicate(object):
    def create_noised_rotated_matrix(self, pointcloud):
        theta = random.random() * 2. * math.pi * 0.1
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T

        noise = np.random.normal(0, 0.002, (rot_pointcloud.shape))
        noisy_pointcloud = rot_pointcloud + noise
        return noisy_pointcloud
    
    def __call__(self, pointcloud):
        noisy_pointcloud = self.create_noised_rotated_matrix(pointcloud)
        duplicate_pointcloud = np.vstack((pointcloud, noisy_pointcloud))
        return duplicate_pointcloud

def duplicate_transforms():
    return transforms.Compose([RandomDuplicate(), ToTensor()])

class RandomResize(object):
    """Make randomly bigger or smaller the pointcloud.
    The idea is to simulate the distance of capture. 
    If the dataset as been captured in differents set, it may make it better. 
    
    Not a really good idea because the model normalize the data.
    """
    def __call__(self, pointcloud):
        """With pointcloud of dim (n, 3)"""
        center = np.mean(pointcloud, axis=0)
        order_of_magnitude = np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0])
        variation_scale = 0.001*order_of_magnitude
        centered_pointcloud = pointcloud - center
        random_resize = np.random.normal(0.0, variation_scale)
        
        return random_resize*(pointcloud - centered_pointcloud) + centered_pointcloud

    
def resize_transforms():
    return transforms.Compose([RandomResize(), RandomRotation_z(), RandomNoise(),ToTensor()])



class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}

##
#%%




class MLP(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, classes)
        )
        

    def forward(self, input):
        return self.layers(input)



class PointNetBasic(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, classes)
        )

    def forward(self, input):
        x1 = self.mlp1(input)
        x2 = self.mlp2(x1)
        # print(x2.shape)
        gf = x2.max(axis=2).values
        # gf = F.max_pool1d(x2, kernel_size=1024)  # (B, 1024, 1)
        x3 = self.mlp3(gf)
        return x3
        
        
        
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(k, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k*k)
        )
        self.k = k
        self.identity = torch.eye(k)
        if torch.cuda.is_available():
            self.identity = self.identity.cuda()

    def forward(self, input):
        x = self.mlp1(input)
        x = x.max(axis=2).values
        x = self.mlp2(x)
        x = x.view(-1, self.k, self.k) 
        return x

class PointNetFull(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.tnet1 = Tnet(k=3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # self.tnet2 = Tnet(k=64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, classes)
        )
        self.identity = torch.eye(3)
        if torch.cuda.is_available():
            self.identity = self.identity.cuda()
    def forward(self, input):
        mRot = self.tnet1(input)
        x = (mRot + self.identity)@input 
        x = self.mlp1(x)
        # x = self.tnet2(x)
        x = self.mlp2(x)
        x = x.max(axis=2).values
        x = self.mlp3(x)
        return x, mRot
        
def basic_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.CrossEntropyLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)



def train(model, device, train_loader, test_loader=None, epochs=10, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    for epoch in range(epochs): 
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            # outputs = model(inputs.transpose(1,2))
            # loss = basic_loss(outputs, labels)
            outputs, m3x3 = model(inputs.transpose(1,2))
            loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    # outputs = model(inputs.transpose(1,2))
                    outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, test_acc))


 
if __name__ == '__main__':
    
    t0 = time.time()
    
    ROOT_DIR = "../data/ModelNet10_PLY"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    
    train_ds = PointCloudData_RAM(ROOT_DIR, folder='train', transform=duplicate_transforms())
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)
    classes=len(train_ds.classes)
    # model = MLP(classes)
    # model = PointNetBasic(classes)
    model = PointNetFull(classes)
    model = ExampleNetwork()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device)
    learning_rate = 0.001
    epochs = 10
    print(f"Training with learning rate : {learning_rate}; nb of epochs : {epochs}")
    train(model, device, train_loader, test_loader, epochs = epochs, learning_rate = learning_rate)
    
    t1 = time.time()
    print("Total time for training : ", t1-t0)

    
    


