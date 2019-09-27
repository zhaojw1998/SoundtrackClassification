from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch

def default_loader(path):
    x=np.load(path)
    #x=x.reshape(1,-1)
    #mu = np.mean(x, axis=0)
    #sigma = np.std(x, axis=0)
    #print(x.shape, mu.shape, sigma.shape)
    #x = (x - mu)/(sigma+1e-8)
    #x=x.reshape(64, 96)
    return torch.from_numpy(x)

class audioDataLoader(Dataset):
    def __init__(self, phase, X, Y, loader=default_loader):
        #定义好 image 的路径
        self.images = X
        self.target = Y
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)